# Minimal CUDA example

`example_cuda.py` loads the saved four-rank 1D P1 fixture, initializes JAXGhost,
and checks one compiled GPU forward update against the DOLFINx reference.
It imports JAXGhost from this checkout; DOLFINx and an editable install are unnecessary.

Activate and configure your CUDA-aware MPI/JAX/mpi4jax environment first.
Inside an allocation with four GPUs, run from this directory:

```bash
export MPI4JAX_USE_CUDA_MPI=1
srun --mpi=pmix -n 4 -c 1 --gpus-per-task=1 --cpu-bind=cores \
    "$VIRTUAL_ENV/bin/python" -u example_cuda.py
```

The script calls `main()` automatically. Each MPI rank must see one GPU.
Expect one `forward PASS` line per rank. Edit `dataset` in `main()` to select
another fixture; if changing its rank count, also change the four-rank assertion
and launch size. Data loading, CUDA initialization and GPU usage are separate
sections in the script. There is no CPU fallback or transport setup in it.

See [DATASETS.md](DATASETS.md) for fixture details and generation logs.
A PASS verifies results and GPU placement, not absence of internal MPI staging.

Validated on iris-171 (2026-09-08): syntax check and four-GPU execution passed
with launcher status 0 using the existing configured `mpc-v10-jax` environment.
UCX `SYNC_MEMOPS` and JAX shutdown warnings remained. The required MPI suite
passed its smoke and one-rank checks, then stopped at the existing two-rank
`test_matrix_dolfinx` float32 tolerance failure; no unrelated code was changed.
