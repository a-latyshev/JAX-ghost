# JAX-ghost

JIT-compatible ghost updates for JAX arrays in MPI-parallel FEM applications.
DOLFINx supplies ownership metadata; JAX and mpi4jax exchange owner values into
local ghost entries. Currently supports scalar and blocked vector P1/P2/P3 forward INSERT and reverse ADD updates on 1D, 2D, and 3D
meshes, with one local JAX device per MPI rank. CPU execution is tested.

## Current scope

1. One JAX device per MPI process. Each MPI rank manages one local JAX device,
   so the total number of participating devices equals the number of MPI
   processes.
2. Numerical data remains on JAX devices. During computation, vectors and
   communication buffers are JAX arrays. No numerical data is synchronized with
   DOLFINx data structures; DOLFINx is used only for initialization metadata
   and, optionally, validation.
3. DOLFINx defines the distributed layout. Its DoF map and IndexMap provide
   ownership, ghost indices, and process relationships. Preserving this layout
   makes the framework compatible with DOLFINx’s distributed indexing
   conventions and supports future integration with its data structures.

## Installation

Use an environment with DOLFINx, JAX, mpi4py, and `mpi4jax==0.9.1.post1`, built
against compatible MPI libraries. From the repository root:

```bash
python -m pip install --no-deps --no-build-isolation -e .
```

See [environment setup and troubleshooting](DOCUMENTATION.md#verified-environment-and-macos-setup)
for tested versions, ARM builds, and macOS MPI transport settings.

## Usage

Run an example:

```bash
export JAX_PLATFORMS=cpu
export JAX_NUM_CPU_DEVICES=1
mpirun -n 2 python examples/interval.py
mpirun -n 4 python examples/square.py --degree 2
mpirun -n 4 python examples/cube.py --degree 3
mpirun -n 4 python examples/irregular.py --degree 2
```

The interval, square, and cube examples accept `--degree 1`, `2`, or `3` (default: `1`).

In your application, create a plan from a DOLFINx function space `V` and
pass a JAX array containing `[owned | ghosts]`:

```python
import jax
from jaxghost import JAXGhost

with JAXGhost.from_index_map(
    V.dofmap.index_map, comm=V.mesh.comm,
    block_size=V.dofmap.index_map_bs,
) as ghost:
    forward = jax.jit(ghost.scatter_forward)
    x = forward(x)  # Preserve owned values and refresh ghosts.
```

For vector-valued spaces, pass `V.dofmap.index_map_bs` as above. Keep `x` flat,
with consecutive components for each DoF; `ghost.n_owned` and `ghost.n_ghost`
count scalar entries. For example, create a two-component P2 space with
`V = fem.functionspace(domain, ("Lagrange", 2, (2,)))`.

For assembly contributions, accumulate ghosts back into owners:

```python
# Within the same ghost context:
reverse = jax.jit(ghost.scatter_reverse)
x = reverse(x)  # Return updated owned values and unchanged ghosts.
```

Run `mpirun -n 4 python examples/reverse.py` for a DOLFINx comparison.
Reverse ADD does not clear or refresh ghosts; repeat calls add them again.

All ranks must call the update in the same sequence, using the same dtype.

Run the pytest suite on 1–4 MPI ranks with timeout protection:

```bash
python -m pip install pytest  # If not already installed.
python scripts/run_mpi_tests.py
mpirun -n 3 python -m pytest tests
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for the data-flow diagram, implementation
details, validation, limitations, and project goals. Contributor guidance is in
[AGENTS.md](AGENTS.md).


Scalar CSR matvec follows DOLFINx's owned-row `y += Ax` semantics:

```python
import jax.numpy as jnp
from jaxghost import JAXMatrixCSR

# A is a scalar DOLFINx matrix with numerical assembly finalized.
with JAXMatrixCSR.from_dolfinx(A, comm) as operator:
    values = jnp.array(A.data[:operator.nnz_owned], copy=True)
    y = jax.jit(operator.mult)(values, x, y)
```

Input arrays are preserved; output ghosts remain unchanged. Use the matrix's
column layout for `x` and row layout for `y`.
Run `mpirun -n 4 python examples/matvec.py` for a complete example.
See [TODO.md](TODO.md) for the operator roadmap.


For the experimental native JAX forward backend, initialize distributed JAX
before device queries, then use `ShardedJAXGhost`:

```python
jax.distributed.initialize(cluster_detection_method="mpi4py")
mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("rank",))
ghost = ShardedJAXGhost.from_index_map(
    V.dofmap.index_map, comm, mesh, block_size=V.dofmap.index_map_bs,
)
x = ghost.to_sharded(x_local)
x = jax.jit(ShardedJAXGhost.scatter_forward)(ghost, x)
x_local = ghost.local_array(x)
```

Import `numpy as np` and `ShardedJAXGhost` from `jaxghost`. This backend uses
padded `[owned | ghosts]` shards and native all-to-all; only forward INSERT is
supported. See [sharded setup and validation](DOCUMENTATION.md#sharded-forward-backend)
for the example, isolated tests, and macOS launcher workaround.

Sharded scalar CSR matvec uses the same distributed initialization and mesh:

```python
from jaxghost import ShardedJAXMatrixCSR

op = ShardedJAXMatrixCSR.from_dolfinx(A, comm, mesh)
values = op.to_sharded(jnp.array(A.data[:op.nnz_owned], copy=True), kind="values")
x = op.to_sharded(x_local, kind="x")
y = op.to_sharded(y_local, kind="y")
y = jax.jit(ShardedJAXMatrixCSR.mult)(op, values, x, y)
y_local = op.local_array(y)
```

Run `python scripts/run_sharding_tests.py --matvec-example --ranks 2 4`
(add `--local-cpu` for the documented single-host macOS workaround).
