#!/bin/bash
#SBATCH --job-name=jz-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=jz-smoke-%j.out

set -eo pipefail
cd /home/users/alatyshev/JAX-ghost

# Initialize Spack explicitly: sbatch does not source the interactive shell setup.
source /home/users/alatyshev/spack/share/spack/setup-env.sh
spack env activate mpc-v10
source /home/users/alatyshev/venvs/mpc-v10-jax/bin/activate

export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader,tcp
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Keep peer GPUs accessible to NCCL; placement.py selects one GPU per rank.
python -u Jean-Zay/scale.py \
  --sizes 4 \
  --ranks 1 2 4 \
  --trials 1 --warmup 1 --repeats 2 \
  --gpus-per-node 4 \
  --timeout 120 \
  --output "Jean-Zay/results/sbatch-${SLURM_JOB_ID}" \
  --launcher 'srun --nodes={nodes} --exact --ntasks={ranks} --gpus-per-node=4 --gpu-bind=none --cpus-per-task=1 --cpu-bind=cores'
