#!/bin/bash
#SBATCH --job-name=jax-sharded
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
# Add the site's account/partition/QoS directives above and activate its
# GPU JAX + mpi4py environment below. Adjust the node/GPU layout to the hardware.
set -euo pipefail
# cd /path/to/JAX-ghost
# source /path/to/your/environment/bin/activate
# This is one driver, not eight MPI drivers. The worker steps request their
# own task/GPU counts inside the allocation.
python Jean-Zay/scale.py --ranks 1 2 4 8 --gpus-per-node 4 \
  --launcher 'srun --nodes={nodes} --exact --ntasks={ranks} --gpus-per-task=1 --cpus-per-task=1 --cpu-bind=cores'
