#!/usr/bin/env bash
# Invoked with a clean environment by run.py; never stack Spack activations.
set -euo pipefail
backend=$1
ranks=$2
shift 2
here=$(cd -- "$(dirname -- "$0")" && pwd)
source "$HOME/spack/share/spack/setup-env.sh"
if [[ $backend == mpi4jax ]]; then
    spack env activate jax-mpi-gpu
    source "$HOME/venvs/jax-mpi-gpu/bin/activate"
else
    spack env activate mpc-v10
    source "$HOME/venvs/mpc-v10-jax/bin/activate"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$here/../../src${PYTHONPATH:+:$PYTHONPATH}"
if [[ $backend == mpi4jax ]]; then
    export JAX_PLATFORMS=cuda MPI4JAX_USE_CUDA_MPI=1 OMPI_MCA_pml=ucx
    export UCX_TLS=self,sm,tcp,cuda_copy,cuda_ipc
else
    export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp
    if [[ $backend == sharding ]]; then export JAX_PLATFORMS=cuda; fi
fi
if [[ $backend == study-summary ]]; then exec python "$here/study_summary.py" "$@"; fi
if [[ $backend == summary ]]; then exec python "$here/summarize.py" "$@"; fi
if [[ $backend != dolfinx && $backend != regression ]]; then
    nvidia_lib_dirs=$(python -c 'from pathlib import Path; import sysconfig; print(":".join(str(p) for p in sorted((Path(sysconfig.get_path("purelib"))/"nvidia").glob("*/lib"))))')
    export LD_LIBRARY_PATH="$nvidia_lib_dirs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ $backend == regression ]]; then
    # The required runner selects CPU JAX and exercises 1–4 MPI ranks.
    export OMPI_MCA_ras=^slurm
    cd "$here/../.."
    exec python scripts/run_mpi_tests.py "$@"
fi
script=replay_jax.py
if [[ $backend == dolfinx ]]; then script=export_dolfinx.py; fi
exec mpirun --mca ras ^slurm --host "$(hostname):4" -n "$ranks" --map-by slot --bind-to none \
    "$VIRTUAL_ENV/bin/python" "$here/rank_exec.py" "$backend" "$here/$script" "$@"
