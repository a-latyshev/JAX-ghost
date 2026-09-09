# Validation — 2026-09-09

Environment: allocation **5878288**, iris-169, four V100-SXM2-16GB GPUs,
28 allocated CPU cores. DOLFINx 0.10.0.post4, JAX 0.11.1, mpi4py 4.1.1,
Open MPI 4.1.6; one disjoint CPU binding and GPU per tested rank.
The portable source contains no IRIS-specific launch paths or modules.

## Prepared fixtures

Before reducing the distributed bundle, all 24 n4/n46/n99 configurations were exported from DOLFINx on 1–8 CPU MPI
ranks. Export validated single/100-call references, serialized arrays, and MPI
ownership. A separate replay loaded every fixture, validated CSR structure and
ownership, and built the actual ghost communication plan with MPI. **24 passed**.
All **108 NPZ files** passed SHA-256 verification: **427,050,219 bytes** of
compressed rank data (about 407 MiB; filesystem allocation can be larger).
The compact distribution now retains n4 at all 1–8 ranks and n99 at 1, 2, 4, 8
ranks: 12 previously validated fixtures, 51 NPZ files, about 188 MiB.
The n46 and other n99 results below remain historical validation records.
Metadata carries its original DOLFINx version, MPI environment and checksums.

## Runtime without optional backends

A separate venv exposed only the installed JAX/CUDA, NumPy/SciPy, mpi4py and
related JAX dependencies through symlinks. No packages were downloaded or changed.
`find_spec('mpi4jax')` and `find_spec('dolfinx')` both returned None, with user-site
imports disabled and the parent Spack PYTHONPATH removed.

- Small n4 correctness/timing: **1, 2, 3, 4 GPUs passed**.
- Production n46 and n99 correctness/timing: **1, 2, 3, 4 GPUs passed** each.
- The standalone `check.py` entrypoint also passed on four GPUs.
- Each timed smoke used 100 calls per batch, one warmup, two measured batches,
  one independent launch. These validate the tools, not final performance claims.
- Every worker checks single and accumulated outputs, changed coefficients and
  inputs, stale ghosts, unchanged inputs, and output ghosts/padding. It also
  checks that neither DOLFINx nor mpi4jax was imported.

Artifacts: `results/isolated-smoke-v2/`, `results/isolated-production-smoke/`,
`results/validation/check4.json`. The first isolated setup attempt lacked the
JAX CUDA plugin package; the corrected environment passed. Its failed log is
retained in `results/isolated-smoke/`.

## Failure handling and reports

Two-rank launches correctly rejected a duplicated physical GPU, no visible GPU,
and overlapping CPU affinity before distributed JAX computation.
**14 developer tests passed**. Developer pytest checks cover optional imports, rank mismatch, checksum corruption,
GPU selection and duplicate placement, scaling arithmetic, incomplete campaigns,
mixed timer/source/CPU settings, inconsistent medians, wrong global sizes,
failed numerical validation, and bounded launcher failures/timeouts.

Run developer checks in an environment with pytest:

```bash
python -m pytest Jean-Zay/test_portable.py -q
```

SVG files parse as XML; CSV/Markdown/SVG regeneration runs using the Python
standard library. Output is rejected for incomplete campaigns. GPU/rank startup
checks use MPI shared-memory groups, CUDA UUIDs and PCI addresses, not ordinals
alone. Logs and results retain exact CPU/GPU placement and MPI.Wtime samples.

## Existing MPI backend regression

Ran the required `python scripts/run_mpi_tests.py` with the compatible full
MPI/JAX/DOLFINx environment and a launcher bound inside the allocation:

- Two-rank compiled mpi4jax smoke: passed.
- One rank: **26 passed, 2 skipped**.
- Two ranks: **27 passed, 1 failed per rank**. Failure is the previously recorded
  float32 `test_matrix_dolfinx` tolerance mismatch; numerical kernels were not
  changed. The runner stopped before three/four-rank tests.

The dependency/import refactor moves the existing CSR validator unchanged into
a backend-neutral module. The original MPI backend and tests are preserved.
Regression log: `results/validation/mpi-regression.log`.

## Not yet verified

**5–8 GPU execution and multi-node GPU execution have not been run here.**
The datasets and MPI metadata are validated at those rank counts; GPU execution
must be checked on your collaborator's allocation. Slurm examples need the site's
account, partition and supported MPI plugin. The full default three-launch,
ten-batch production study is intended to run there.

## Slurm batch launcher correction

Job 5880927 on iris-185 passed n4 with one rank but failed with two ranks
at the first NCCL all-to-all (CUDA error 101), after placement and compilation
had succeeded. Job 5880983 on the same node passed n4 correctness and timing
with 1, 2, and 4 ranks using `--gpus-per-node=4 --gpu-bind=none` instead of
`--gpus-per-task=1`. One trial, one warmup, two batches of 100 calls; CSV and
SVG generation also completed. Evidence: `results/sbatch-5880983/`. These are
smoke timings, not production scaling measurements. Multi-node and partner-HPC
execution remain unverified.

## GPU selection before MPI initialization

The partner's screenshot reported four CUDA devices after the worker attempted to
mask to one. GPU selection now runs before importing MPI, with subsequent MPI
verification of the launcher's local rank. Eight selection/error scenarios passed
using a standard-library assertion harness. The login Python lacked pytest, so
the new pytest file was not run there. IRIS job 5883793 (iris-178) passed n4
correctness, timing and plot generation on 1/2/4 GPUs after this change. Full MPI
regressions were not repeated for this entrypoint change. Partner-HPC and
multi-node validation of this change remain pending.
