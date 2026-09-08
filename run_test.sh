#!/bin/bash
#SBATCH --job-name=jax_dolfinx_gpu_test   # Nom du job
#SBATCH --output=out_%j.log           # Fichier de sortie (log)
#SBATCH --error=out_%j.log             # Fichier d'erreur (log)
#SBATCH --nodes=1                        # Nombre de nœuds
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:10:00                  # Temps maximal d'exécution (10 minutes)
#SBATCH --hint=nomultithread
#SBATCH --constraint=v100
#SBATCH --account=sos@v100

# Charger les modules nécessaires (à adapter selon votre cluster)
module purge
module load cuda/12.1.0                  # Charge CUDA (version à adapter)
module load python                       # Charge Python (version à adapter)
module unload intel-compilers/2021.9.0

conda activate fenicsx-env


echo "Début de l'exécution à $(date)"

# ============================================================
# 1. Environnement Conda
# ============================================================

echo "Python      : $(which python)"
echo "Python ver. : $(python --version)"
echo "Conda       : $CONDA_PREFIX"

# Exporter les variables pour JAX (forcer l'utilisation des GPU)
export JAX_PLATFORMS=cuda

cd $SCRATCH/Workshop_JAX/JAX-ghost
export PYTHONPATH=$PWD:$PYTHONPATH

# ============================================================
# 2. MPI
# ============================================================

export CC="$CONDA_PREFIX/bin/mpicc"
export CXX="$CONDA_PREFIX/bin/mpicxx"
export MPICC="$CONDA_PREFIX/bin/mpicc"
export MPICXX="$CONDA_PREFIX/bin/mpicxx"

echo "mpicc       : $(which mpicc)"
echo "mpicxx      : $(which mpicxx)"

# ============================================================
# CUDA / NVIDIA libraries fournies par pip
# ============================================================

unset CUDA_HOME
unset CUDA_PATH
unset JAX_PLATFORMS

export NVIDIA_LIBS="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia"

export LD_LIBRARY_PATH="$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/nvjitlink/lib:$NVIDIA_LIBS/nccl/lib"

echo "NVIDIA_LIBS=$NVIDIA_LIBS"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
# ============================================================
# 4. Vérification GPU / Slurm
# ============================================================

echo
echo "============================================================"
echo "GPU"
echo "============================================================"

echo "HOSTNAME            : $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi -L
nvidia-smi

# ============================================================
# 5. Test direct du driver CUDA
# ============================================================

echo
echo "============================================================"
echo "Test cuInit"
echo "============================================================"

python - <<'PY'
import ctypes

lib = ctypes.CDLL("libcuda.so.1")

lib.cuInit.argtypes = [ctypes.c_uint]
lib.cuInit.restype = ctypes.c_int

err = lib.cuInit(0)

print("cuInit(0) =", err)

if err != 0:
    raise RuntimeError(f"cuInit(0) failed with CUDA error {err}")
PY

# ============================================================
# 6. Vérification des bibliothèques NVIDIA pip
# ============================================================

echo
echo "============================================================"
echo "NVIDIA libraries"
echo "============================================================"

find "$CONDA_PREFIX/lib/python3.11/site-packages/nvidia" \
    -name 'libcusparse.so*' \
    -o -name 'libcublas.so*' \
    -o -name 'libcudart.so*' \
    | head -30

# ============================================================
# 7. Test JAX GPU
# ============================================================

echo
echo "============================================================"
echo "JAX"
echo "============================================================"

python - <<'PY'
import jax

print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
PY

# ============================================================
# 8. Test mpi4jax
# ============================================================

echo
echo "============================================================"
echo "mpi4jax"
echo "============================================================"

python - <<'PY'
import mpi4py
import mpi4jax
import jax

print("mpi4py :", mpi4py.__version__)
print("mpi4jax:", mpi4jax.__version__)
print("JAX    :", jax.__version__)
print("Devices:", jax.devices())
PY

python - <<'PY'
import jax

print("AVANT mpi4jax")
print(jax.devices())

import mpi4jax

print("APRES mpi4jax")
print(jax.devices())
PY



# ============================================================
# 9. Exécution du programme
# ============================================================

echo
echo "============================================================"
echo "Lancement interval_GPUs.py"
echo "============================================================"

srun --mpi=pmi2 python examples/interval_GPUs.py

echo
echo "Fin de l'exécution à $(date)"