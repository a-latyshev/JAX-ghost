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

grep -E 'TRANSPORT|CORRECTNESS_PASS' transport-small.log
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
