# Implementation validation — 2026-09-08

Local macOS ARM64 CPU validation used the existing `.venv` with DOLFINx 0.11.0,
JAX 0.10.2, mpi4jax 0.9.1.post1 and MPICH 5.0.1. MPI was launched with the
`fenicsx-0.11.0` conda environment's `bin/mpirun`, `FI_PROVIDER=tcp`,
`FI_TCP_IFACE=en0`, and one OpenMP/OpenBLAS thread per rank. These network
settings are specific to the development machine.

- Export and CPU replay passed separately with 1, 2, 3 and 4 MPI ranks:
  `--subdivisions 4 --warmup 1 --iterations 3 --repeats 2` (125 global DOFs).
  All saved arrays round-tripped through NPZ, and both the single application
  and accumulated batch matched the DOLFINx references. Output ghosts were
  preserved and JAX inputs remained unchanged. Replay verified no DOLFINx import.
- Negative integration checks copied the two-rank dataset to temporary folders:
  replaying with one rank rejected the rank mismatch; setting rank 0's first
  CSR column to 999999 rejected the CSR; setting rank 0's first column ghost
  owner to rank 0 rejected the layout. Each MPI job failed with the intended
  error and exited within a 45-second timeout, without hanging its peer.
- `python scripts/run_mpi_tests.py --mpirun <conda-env>/bin/mpirun` passed:
  two-rank compiled smoke; one rank 26 passed/2 skipped; two, three and four
  ranks each 28 passed per rank. No existing failures in this local run.
- Python compilation and `git diff --check` passed.

Temporary raw results: `/tmp/jaxghost-benchmark-validation/`.
Regression log: `/tmp/jaxghost-benchmark-suite.log`. These are local validation
artifacts, not portable repository dependencies or representative scaling data.

GPU execution, GPU transport and the default 103,823-DOF HPC sampling were not
run. The README provides the commands for those runs. Compatibility with the
HPC DOLFINx 0.10 installation still needs the documented small-mesh check there.

Timer update: DOLFINx sampling now uses named `dolfinx.common.Timer` instances,
with explicit start/stop, elapsed seconds and flush. A two-rank n4 export and
CPU replay passed with 1 warmup, 3 iterations and 2 repeats. The complete MPI
suite passed again (same pass/skip counts as above). Raw output is in
`/tmp/jaxghost-timer-validation/` and `/tmp/jaxghost-timer-suite.log`.
Python syntax checks and `git diff --check` passed after this update.

Shared-clock update: both implementations now use `MPI.Wtime()` and identify
that clock in their timing JSON. This supersedes the Timer update above.
Two-rank n4 export/replay passed again (1 warmup, 3 iterations, 2 repeats),
and the complete required MPI suite passed with the same counts. Syntax and
whitespace checks passed. Raw results: `/tmp/jaxghost-wtime-validation/`;
regression log: `/tmp/jaxghost-wtime-suite.log`.
