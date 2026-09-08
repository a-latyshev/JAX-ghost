# Check GPU–CPU–GPU staging

Requires a configured CUDA-aware MPI/JAX/mpi4jax environment, GDB, and **two
MPI ranks with one visible GPU each**. No DOLFINx or dataset files are needed.
This tracer targets UCX/OpenMPI; it does not configure your cluster's transport.

Run from this directory inside a GPU allocation:

```bash
export MPI4JAX_USE_CUDA_MPI=1
export PYTHON="$(command -v python)"
set -o pipefail
timeout 180 srun --mpi=pmix -n 2 -c 1 --gpus-per-task=1 --cpu-bind=cores \
    bash trace.sh --count 8 2>&1 | tee transport-small.log

grep 'TRANSPORT_RESULT' transport-small.log
```

Adapt the launcher to your cluster. Use `--count 131072` and another log file
for 1 MiB messages instead of 64 bytes. Both ranks should report
`CORRECTNESS_PASS`, with launcher exit status 0.

| Marker | Meaning |
| --- | --- |
| `HOST_STAGING` | UCX host packing/unpacking; confirm the printed MPI exchange stack. |
| `CUDA_COPY_REVIEW_STACK` | Inspect the stack to identify the copy path. |
| `CUDA_IPC_OBSERVED` | An intra-node CUDA IPC transfer was entered; other transfers may still stage. |
| No markers | **Inconclusive**: symbols or transport may not be supported. |

Inspect activity between `EXCHANGE_BEGIN` and `EXCHANGE_END`; setup and
validation copies occur outside them. Numerical correctness does not prove
GPU-direct transfers, and GDB timings are not performance measurements.

Validated on two IRIS GPUs (2026-09-08): both message sizes passed; the small
case exposed host staging. Other clusters remain untested. The repository MPI
suite reproduced the existing two-rank float32 matrix-test failure.


If Python cannot import JAX, activate the GPU venv and reset `PYTHON` to
`"$VIRTUAL_ENV/bin/python"`. Copy all files in this directory, including
`trace.gdb.in` (named this way because the repository ignores `*.gdb`).


Each rank ends with `TRANSPORT_RESULT`:
- `COMMUNICATION_FAILED` / `INCOMPLETE`: crash, failed run, or missing correctness result; nonzero exit.
- `HOST_STAGING_DETECTED`: exchange passed, but UCX host pack/unpack was observed.
- `TRANSPORT_INCONCLUSIVE`: exchange passed without confirmed host pack/unpack;
  inspect copy-helper stacks and IPC counts. This never means “no staging.”

Successful communication retains exit status 0 even when staging is detected.
Counts include only marked exchanges. Missing/unsupported symbols cannot prove
direct transfers. If Slurm kills a rank abruptly, its final summary may be absent;
treat a failed launcher or missing rank summary as an incomplete/failed run.

For CUDA copy directions, run each GPU-bound rank under Nsight Systems:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o "transport-${SLURM_JOB_ID:-local}-${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}" \
    "$PYTHON" -u check.py --count 2048 --profile
```

Place this command inside your MPI launch wrapper after GPU assignment, replacing
`exec bash trace.sh ...`. Capture includes only the second, warmed-up exchange,
ending after GPU synchronization and before CPU validation. Inspect both reports
with `nsys stats --report cuda_gpu_mem_size_sum FILE.nsys-rep` and the timeline:
payload DtoH/HtoD copies indicate host staging; DtoD/peer copies indicate GPU copies.
Small control transfers are not by themselves evidence of payload staging. Missing
events or profiling errors are inconclusive. The profiler option requires Linux
and an identifiable loaded shared CUDA runtime; it prefers the Python environment's
NVIDIA runtime when native MPI/UCX loads another version too.

Nsight 2024.4 validated on IRIS allocation 5874886 (2026-09-08), two distinct
GPUs, OpenMPI 5.0.8 with `OMPI_MCA_pml=ucx` and UCX 1.19.0. Both ranks passed:
16 KiB captured one 16 KiB peer copy and two device copies (16 and 32 KiB),
with no recorded host copies. The 64-byte control captured both DtoH and HtoD
payload copies. Reports: `verified-5874886-rank{0,1}.nsys-rep` and
`verified-small-5874886-rank{0,1}.nsys-rep`; summaries: `verified-nsys-*-stats.log`.
This verifies the observed single-node paths, not inter-node GPUDirect RDMA
or optimal performance.
