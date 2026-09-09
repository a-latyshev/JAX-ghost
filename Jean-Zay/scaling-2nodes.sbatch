#!/bin/bash
#SBATCH --job-name=jz-scaling-2nodes
#SBATCH --constraint=h100
#SBATCH --hint=nomultithread
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --time=01:00:00
#SBATCH --output=jz-scaling-2nodes-%j.out

# Jean Zay H100: 4 GPUs and 96 physical CPU cores per node.
# Submit from JAX-ghost/ or Jean-Zay/:
#   export JAXGHOST_ENV_SETUP=/absolute/path/to/jax-mpi-environment.sh
#   sbatch --account=YOUR_PROJECT@h100 scaling-2nodes.sbatch
# The setup file must load your compatible JAX/Python/MPI modules and/or venv.
# Add the project's QoS at submission if required. No IRIS modules are used.
set -eo pipefail
: "${JAXGHOST_ENV_SETUP:?Set JAXGHOST_ENV_SETUP to your JAX/MPI shell setup file}"
if [[ ! -r $JAXGHOST_ENV_SETUP ]]; then
    echo "Cannot read environment setup: $JAXGHOST_ENV_SETUP" >&2
    exit 1
fi
module purge
module load arch/h100
source "$JAXGHOST_ENV_SETUP"
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

# One Python driver; srun launches each MPI configuration independently.
# All allocated GPUs stay accessible to NCCL. The worker selects and verifies
# exactly one GPU per rank before MPI/JAX initialization.
# Add --mpi=... to the launcher only if required by your site's MPI module.
module list
command -v python

python -u Jean-Zay/scale.py \
  --sizes 99 \
  --ranks 1 2 4 8 \
  --trials 3 --warmup 10 --repeats 10 \
  --gpus-per-node 4 \
  --timeout 600 \
  --output "Jean-Zay/results/scaling-2nodes-${SLURM_JOB_ID}" \
  --launcher "srun --nodes={nodes} --exact --ntasks={ranks} --distribution=block:block --gpus-per-node=4 --gpu-bind=none --cpus-per-task=${SLURM_CPUS_PER_TASK} --cpu-bind=cores --hint=nomultithread"
