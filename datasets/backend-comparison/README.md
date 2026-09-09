# Consistent matvec backend comparison

This folder runs fresh native DOLFINx CPU, mpi4jax GPU, and native JAX sharding
measurements on the same concrete 3D Poisson fixtures. It does not consume old
benchmark timings. Public `jaxghost` implementations are unchanged.

[Completed comparison from allocation 5878288](RESULTS.md) · [Validation record](VALIDATION.md)

From the repository root, on IRIS with an active four-GPU allocation:

```bash
BENCH_JOB_ID=5878288 bash datasets/backend-comparison/run.sh
```

The entrypoint joins the supplied allocation with a single enclosing Slurm step
(the noninteractive equivalent of `sjoin`), then runs all workers serially.
It expects one node with at least 28 allocated CPU cores and four GPUs, as in
5878288. `--smoke-only` runs the small validation and regression suite without
the production sizes. `--output /absolute/new/directory` selects a new result
folder; existing folders are never overwritten. `--timeout 600` is the default
per-launch timeout. `--skip-regression` is for follow-up campaigns only when a
regression result has already been recorded. Do not run other workloads on the
selected cores or GPUs during measurement.

## Workload and placement

- Scalar P1 tetrahedral unit cube, Poisson stiffness, no boundary conditions,
  float64. Subdivisions 46 and 99 give 103,823 and 1,000,000 DoFs.
- 1, 2, 3, 4 MPI ranks. Each rank is restricted to one hardware thread on one
  distinct physical CPU core. Numerical library thread counts are one.
- DOLFINx uses zero GPUs. Each JAX rank additionally sees one physical GPU,
  selected by UUID. Both JAX backends have identical CPU/GPU assignments.
- Within each size/rank/trial, DOLFINx constructs and times the native matrix,
  exports the exact owned-row CSR, maps, inputs and reference outputs, and then
  both JAX backends replay that fixture. SHA-256 fingerprints bind all three
  records to the same files. No DOLFINx import occurs in either JAX replay.
- Each batch uses fixed A and x, resets y to zero outside timing, and performs
  100 Python-loop calls to `y += A*x`. This is accumulated Ax, not successive
  powers of A. Each call includes halo exchange. No enclosing loop is JITed.

`launch.sh` activates Spack and then the associated venv in a clean environment:
`mpc-v10` / `$HOME/venvs/mpc-v10-jax` for DOLFINx and sharding;
`jax-mpi-gpu` / `$HOME/venvs/jax-mpi-gpu` for mpi4jax. Both use the same checkout.
Spack setup is `$HOME/spack/share/spack/setup-env.sh`. The mpi4jax stack uses
UCX with `self,sm,tcp,cuda_copy,cuda_ipc` and CUDA-enabled MPI. The mpc-v10 stack
uses its existing Open MPI host transports; sharding communicates using JAX
collectives. Physical GPU transport is not inferred from these settings.

## Measurement contract

All reported computation/startup durations use `MPI.Wtime()`. A barrier precedes
each measurement, outside the timed interval. GPU arrays and ordered effects
are completed before stopping the timer. There is no per-call synchronization
or per-call barrier: Python dispatch and completion of all 100 calls are included.

JAX explicitly lowers and compiles one matvec, then invokes the cached `jax.jit`
callable for execution on both backends. Direct AOT executable invocation in
JAX 0.11.1 failed during repeated mpi4jax calls with a missing ordered-effect
token; the cached JIT entrypoint preserves the original benchmark calling convention.
Smoke runs enable compilation logging to verify reuse. Tracing/lowering, compilation,
and first synchronized execution are reported separately. Processes are fresh
for each trial and persistent JAX compilation caching is disabled. Plan construction
and any helper compilation it triggers are setup, outside these phase timings.
These are operator startup phases, not total process startup cost.

After first execution, each full launch performs ten untimed warmup calls and
ten timed batches of 100 calls. Three independent launches are run per case.
For each batch, `max(rank_elapsed)/100` is its time per matvec. The report uses
the median of the three launch medians; CSV includes their minimum and maximum.
Startup phases also use maximum rank durations and median across launches.

Assembly, plan construction, uploads, output resets, validation and I/O are
excluded from steady-state timing. DOLFINx overlaps communication with local
SpMV and refreshes input ghosts in place; JAX preserves its input arrays.
Sharding's existing padding and collectives are retained. Those implementation
differences are part of the measured backend behavior.

The two JAX launches alternate order by trial. DOLFINx always runs first to
produce the fixture. This is a matched hardware comparison of the installed
stacks; their MPI/runtime differences remain part of the result.

## Validation and outputs

Before full sampling, all three backends run a subdivision-4 fixture at 1–4
ranks, including 100-call accumulation (one warmup, two batches). Full sampling
is withheld if these checks fail. The repository's `scripts/run_mpi_tests.py`
also runs, and its exit status/log are retained separately from benchmark cases.
A regression failure does not silently become a benchmark pass.

Single and accumulated outputs must be finite, preserve output ghosts, and meet
`global_owned_L2_error <= 1e-12 + 1e-10 * reference_norm`. JAX inputs and device
placement are checked. Workers reject overlapping physical CPU cores or GPUs;
the summarizer verifies exact placement, fixtures, settings and matching JAX
versions between successful runs. Run the summarizer's audit tests in an environment with pytest:

```bash
python -m pytest datasets/backend-comparison/test_summary.py -q
```

Each result directory contains `manifest.json` (allocation, physical topology,
CPU IDs, GPU UUIDs, source revision and source hashes), `allocation.txt`, GPU
inventory, `fixtures/`, `raw/`, worker `logs/`, and `launches.json`. Failed and
skipped launches stay explicit; no historical or partial-series values are
substituted. The campaign exits nonzero if any worker or regression launch fails,
after writing the available results and report. `summary.csv` includes successful trial counts and partial values;
only complete configurations enter comparisons and plots. `REPORT.md` contains
runtime and startup tables; PNG/PDF plots cover per-call/batch time, scaling,
DOLFINx/backend ratios and startup phases. The CSV additionally records
mpi4jax/backend ratios and scaling efficiency.

To regenerate summaries, activate mpc-v10 and its venv, then run:

```bash
python datasets/backend-comparison/summarize.py /absolute/path/to/results/RUN
```

## Curves including compilation

Runtime and DOLFINx-ratio plots include dashed JAX curves for
`tracing/lowering + compilation + one warmed 100-call batch`.
The batch plot uses total milliseconds; the per-call plot amortizes this total
over 100 calls. DOLFINx's total equals its main batch time. Each launch's costs
are added before taking the median across launches. These derived totals exclude
first-execution startup and other setup; they are not measured cold-run wall times.
CSV columns `total_ms_per_batch`, `total_amortized_us_per_matvec`, trial total
ranges and the ratios ending in `_total` expose the same values.
