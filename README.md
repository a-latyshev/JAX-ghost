# JAX-ghost

Distributed Ghost Updates for JAX used within MPI-parallelized FEM code

## Challenge

Implement a reusable JAX-compatible abstraction for synchronizing **owned and ghost entries** of vectors distributed across multiple MPI processes and JAX devices.

DOLFINx supplies the distributed mesh, degree-of-freedom map, and ownership metadata. JAX owns the numerical arrays used during execution.

Each process stores

\[
x_p = [x_p^{\mathrm{owned}} \mid x_p^{\mathrm{ghost}}].
\]

Every global entry has one owner, while other processes may hold ghost copies required for local computation.

## Core operations

### Forward scatter

```python
x_local = scatter_forward(x_owned, layout)
```

Copy current owner values into the corresponding ghost entries:

\[
x_{p,i}^{\mathrm{ghost}} = x_{\operatorname{owner}(i),i}^{\mathrm{owned}}.
\]

### Reverse scatter

```python
x_owned = scatter_reverse(x_local, layout, op="sum")
```

Accumulate ghost contributions back into their owners:

\[
x_i^{\mathrm{owned}} \mathrel{+}= \sum_{p:\,i\in G_p} x_{p,i}^{\mathrm{ghost}}.
\]

## DOLFINx and JAX responsibilities

DOLFINx provides:

- distributed mesh and function space;
- cell-to-DoF map;
- `IndexMap` ownership and ghost metadata;
- local sparse matrix structure for the demonstration.

JAX provides:

- persistent device arrays for vector and matrix data;
- packing and unpacking kernels;
- local sparse matrix-vector multiplication;
- JIT compilation and automatic differentiation.

Initial copies of static metadata and matrix data to JAX devices are acceptable. Repeated DOLFINx/PETSc-to-JAX synchronization and shared-memory interoperability are outside the hackathon scope.

## Minimum implementation

Build a static communication layout from a DOLFINx `IndexMap`, containing:

- number of locally owned entries;
- global indices and owners of ghost entries;
- source and destination ranks;
- local send and receive indices;
- packed message sizes and offsets.

Use `mpi4jax`, native JAX communication, or another suitable backend. Packing, communication, unpacking, and local numerical operations should work with `jax.jit` wherever possible.

A target interface is:

```python
scatterer = JAXScatterer.from_index_map(V.dofmap.index_map)
x_local = scatterer.forward(x_owned)
x_owned = scatterer.reverse_add(x_local)
y_owned = distributed_spmv(A_local, x_owned, scatterer)
```