# Distributed Poisson matvec benchmark

Compare native DOLFINx CPU `MatrixCSR.mult` with JIT-compiled JAXGhost
`JAXMatrixCSR.mult`. Both perform owned-row **y += Ax**, including forward ghost
communication on every call. No solver, RHS, or boundary conditions are needed.
The default tetrahedral unit cube has 46 subdivisions per direction, 584,016
cells and 103,823 scalar P1 DOFs. The matrix is the float64 Poisson stiffness
operator; its nullspace is irrelevant to this multiplication benchmark.

Run commands below from the JAX-ghost repository root. Activate a compatible
DOLFINx/MPI environment for export. See the [existing CPU environment notes](../../DOCUMENTATION.md)
and [GPU environment setup](../../HPC-GPU/spack-jax-mpi/README.md).
Use actual allocated CPU cores, without oversubscribing.

## Export and measure DOLFINx

Reserve at least four CPU cores. Use one thread per rank:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for n in 1 2 3 4; do
    mpirun -n "$n" python datasets/benchmark/export_dolfinx.py || break
done
```

Inside a Slurm allocation, replace `mpirun -n "$n"` with
`srun --mpi=pmix --exact -n "$n" -c 1 --cpu-bind=cores` if supported by your site.
Do not nest launchers. Defaults: `--subdivisions 46 --warmup 10 --iterations 100
--repeats 10`. `--output PATH` changes the data root. Re-running overwrites that
mesh/rank dataset; metadata.json is written last as its completion marker.

The options mean:

| Option | Meaning | Default |
| --- | --- | --- |
| `--subdivisions` | Number of intervals along each cube axis. Each little cube is split into six tetrahedra; scalar P1 has `(n+1)^3` DOFs. This sets mesh size, not process count. | 46 → 103,823 DOFs |
| `--warmup` | Extra matvec calls before collecting timings, to warm up execution. JAX compilation happens before these calls and is also excluded. | 10 calls |
| `--iterations` | Matvec calls within one timed batch. Divide the batch time by this count to get time per matvec. | 100 calls |
| `--repeats` | Number of separately timed batches. Report the median of their times per matvec to reduce sensitivity to timing noise. | 10 batches |

Thus the default sampling measures 10 batches of 100 matvecs, after warmup.
Reference construction and correctness checks are additional untimed work.
Keep subdivisions fixed for the entire 1–4-process strong-scaling series.

Outputs: `datasets/benchmark/data/n46/1ranks/` through `4ranks/`. Each contains
`metadata.json` and `rank-0000.npz`, etc. Keep the entire directory when copying
to another machine. Replay requires precisely the saved MPI rank count.

NPZ arrays:

| Keys | Contents |
| --- | --- |
| `indptr`, `indices`, `values` | Owned-row CSR, local column numbering, float64 coefficients |
| `row_owned_range`, `col_owned_range` | Global half-open owned ranges |
| `row_ghosts`, `col_ghosts` | Global ghost IDs, in local order |
| `row_owners`, `col_owners` | Owner ranks for those ghosts |
| `x`, `y_initial` | Local `[owned | ghosts]` input and zero output |
| `single_expected`, `batch_expected` | Independent DOLFINx executions of one and 100 accumulated applications |

Owned input values are `sin(2*pi*x) + y**2 + 0.5*z` evaluated at DOF coordinates;
input ghosts initially contain zero and are refreshed by each backend. Metadata
contains timing settings, per-rank timings, partition sizes, versions and runtime
information. Saved layouts preserve each concrete partition; numbering can change
between process counts or DOLFINx versions. No mesh reconstruction is needed for
JAX replay. Large generated data and default results are ignored by Git.

## Replay with JAXGhost

CPU debugging requires NumPy, mpi4py, JAX, mpi4jax and this checkout, but **does
not import DOLFINx**:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for n in 1 2 3 4; do
    mpirun -n "$n" python datasets/benchmark/replay_jax.py \
        "datasets/benchmark/data/n46/${n}ranks" --backend cpu || break
done
```

For GPU runs, activate the CUDA JAX/mpi4jax environment and reserve four GPUs.
Each rank must see one GPU. Inside that allocation:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPI4JAX_USE_CUDA_MPI=1
for n in 1 2 3 4; do
    srun --mpi=pmix --exact -n "$n" -c 1 --gpus-per-task=1 --cpu-bind=cores \
        python datasets/benchmark/replay_jax.py \
        "datasets/benchmark/data/n46/${n}ranks" --backend cuda || break
done
```

The environment variable alone does not configure CUDA-aware MPI. Follow the
[HPC transport notes](../../HPC-GPU/notes.md) for your installation, including the
recorded IRIS UCX/library setup. CUDA-aware MPI is not evidence of GPU-direct
transport; the recorded small-message IRIS diagnostic used internal host staging.
These scripts do not verify the physical transport route.

Results default to `datasets/benchmark/results/n46-Nranks-BACKEND.json`;
`--output FILE` selects another JSON outside the source dataset. Re-running
replaces that result. The dataset is unchanged.

## Timing and interpretation

Compile and warm up before timing. Each batch resets output outside timing, then
executes 100 Python-driven calls, accumulating `y += Ax`. Input and coefficients
remain fixed. JAX blocks on results and effects at batch boundaries. There is no
per-call barrier. DOLFINx uses a named `dolfinx.common.Timer` for each batch:
`start()` immediately before the loop, `stop()` immediately after it, and
`elapsed().total_seconds()` for the duration. `flush()` registers each named
measurement so it is also available through `dolfinx.common.timing(name)`.
`resume()` is unnecessary because each batch is one uninterrupted timed region.
The JAX-only replay uses `MPI.Wtime()` to retain its DOLFINx-free environment.
Both measure elapsed wall time in seconds.

Every batch begins with an MPI barrier; rank elapsed times are
gathered after timing. A sample is `max(rank_seconds)/iterations`; the reported
time is the median of ten samples. Setup, compilation, transfers, validation,
output reset and file I/O are excluded. Native DOLFINx and JAX may use different
communication/computation overlap strategies; those differences are measured.

Both single and batch references are checked with owned-entry global L2 norms:
`error <= 1e-12 + 1e-10 * reference_norm`. Outputs must be finite, output ghosts
preserved, and JAX inputs unchanged. Validation runs outside timing.

Compare absolute seconds/matvec and, separately for each backend, speedup
`T(1)/T(p)` and efficiency `T(1)/(p*T(p))`, for p=1,2,3,4 at the same mesh size.
`dolfinx_over_jax > 1` means JAX was faster than the stored CPU measurement.
One CPU rank and one GPU are different hardware allocations; record node/device
placement and do not describe their ratio as hardware-independent efficiency.
The 100K case may be overhead dominated on GPUs.

## Small validation run

Before full sampling, run both scripts for all four rank counts with
`export_dolfinx.py --subdivisions 4 --warmup 1 --iterations 3 --repeats 2`, then
replay `data/n4/${n}ranks`. Also run the repository regression suite:

```bash
python scripts/run_mpi_tests.py
```

GPU performance must be measured on the HPC allocation; local CPU verification
does not establish GPU performance or transport. See `VALIDATION.md` for checks
performed during implementation.
