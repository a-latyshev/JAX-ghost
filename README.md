# JAX-ghost

JIT-compatible ghost updates for JAX arrays in MPI-parallel FEM applications.
DOLFINx supplies ownership metadata; JAX and mpi4jax exchange owner values into
local ghost entries. Currently supports scalar P1/P2/P3 forward updates on 1D, 2D, and 3D
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
```

All examples accept `--degree 1`, `2`, or `3` (default: `1`).

In your application, create a plan from a scalar DOLFINx function space `V` and
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

All ranks must call the update in the same sequence, using the same dtype.

Run the pytest suite on 1–4 MPI ranks with timeout protection:

```bash
python -m pip install pytest  # If not already installed.
python scripts/run_mpi_tests.py
```

See [DOCUMENTATION.md](DOCUMENTATION.md) for the data-flow diagram, implementation
details, validation, limitations, and project goals. Contributor guidance is in
[AGENTS.md](AGENTS.md).
