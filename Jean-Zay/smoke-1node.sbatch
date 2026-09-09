#!/bin/bash
#SBATCH --job-name=jz-smoke-1node
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=jz-smoke-1node-%j.out

# Submit from the repository root or Jean-Zay/. Default partition: IRIS gpu.
# Override the partition/account with sbatch options on another HPC.
# Activate the site's GPU JAX + MPI environment below, or set JAXGHOST_ENV_SETUP
# to a shell setup file (export this variable before calling sbatch).
set -eo pipefail
if [[ -n ${JAXGHOST_ENV_SETUP:-} ]]; then
    source "$JAXGHOST_ENV_SETUP"
fi
# Example: source /path/to/environment/bin/activate
set -u
cd "${SLURM_SUBMIT_DIR:?Submit this script using sbatch}"
if [[ -f scale.py && -f ../src/jaxghost/sharded_matrix.py ]]; then
    cd ..
fi
if [[ ! -f Jean-Zay/scale.py ]]; then
    echo 'Submit from the JAX-ghost repository root or its Jean-Zay directory.' >&2
    exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# One driver launches fresh MPI steps. Do not prefix this Python command with srun.
# Keep allocated peer GPUs accessible to NCCL; the worker selects and verifies
# exactly one distinct GPU per rank. Add the site-required --mpi option if needed.
# For a different GPU layout, change BOTH --gpus-per-node=4 settings below and
# the SBATCH allocation above; available prepared rank counts stop at eight.
python -u Jean-Zay/scale.py \
  --sizes 4 \
  --ranks 1 2 4 \
  --trials 1 --warmup 1 --repeats 2 \
  --gpus-per-node 4 \
  --timeout 600 \
  --output "Jean-Zay/results/smoke-1node-${SLURM_JOB_ID}" \
  --launcher 'srun --nodes={nodes} --exact --ntasks={ranks} --distribution=block:block --gpus-per-node=4 --gpu-bind=none --cpus-per-task=1 --cpu-bind=cores'
