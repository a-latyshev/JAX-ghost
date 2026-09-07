# Implementation strategy

Use DOLFINx's C++ and Python implementations as the reference for new work.
Read the corresponding implementation before designing a feature; reproduce
its ownership, packing, communication, and accumulation ideas, adapting them
for JAX's immutable device arrays, tracing, static shapes, and ordered effects.
Explain any intentional differences and validate numerical behavior against
DOLFINx. Do not replace working designs solely to use different JAX syntax.

Useful reference sources (prefer the installed DOLFINx version):

- `dolfinx/la/__init__.py`: Python `Vector.scatter_forward()` wrapper.
- `include/dolfinx/common/Scatterer.h`: IndexMap-derived communication plans,
  packing indices, neighbor communication, and reverse exchange.
- `include/dolfinx/la/Vector.h`: vector storage, forward begin/end, and unpacking.

Upstream equivalents are `python/dolfinx/la/__init__.py`,
`cpp/dolfinx/common/Scatterer.h`, and `cpp/dolfinx/la/Vector.h` in
https://github.com/FEniCS/dolfinx. Older versions may use `python/dolfinx/la.py`.

Keep host metadata preparation separate from repeated device-array operations.
A scalar field on a 2D mesh still has a flat `[owned | ghosts]` vector; derive
communication from IndexMap rather than spatial dimension or geometric guesses.

Validate changes with `python scripts/run_mpi_tests.py` in a compatible MPI/JAX/
DOLFINx environment with pytest installed. The runner uses `mpirun` and checks
1–4 ranks with timeouts. Use pytest assertions, `pytest.raises`, and pytest skip
markers for tests. All ranks must execute collective tests in the same order; do
not use pytest-xdist for these tests. For example changes, also run the affected
example under MPI. Preserve existing user edits.
