# JAX and mpi4jax on IRIS

These instructions add pip packages in a virtual environment over the existing
Spack `mpc-v10` environment, without rebuilding or modifying its packages.

## Allocate an interactive GPU session

Run from the login node:

```bash
si-gpu -N 1 -n 4 -c 7 -G 4 -t 0-02:00:00
```

This reserves four tasks, seven CPU cores per task, and four GPUs on one node.
`-N` counts nodes; `-n` counts processes. Requesting GPUs alone does not reserve
multiple CPU tasks.

## Verify the existing MPI installation

Inside the allocation:

```bash
spack env activate mpc-v10
srun --mpi=list
srun --mpi=pmix -n 4 -c 7 --cpu-bind=cores \
    python -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size, MPI.Get_library_version(), flush=True)'
nvidia-smi
```

The MPI command was verified on `iris-185`: four ranks reported size 4 using
OpenMPI 4.1.6. Slurm listed `pmix` with plugin version `pmix_v6`.
Use `--mpi=pmix` explicitly: PMIx supplies the startup connection between Slurm
and OpenMPI. A plain Python MPI import in the interactive shell previously
failed during MPI initialization.

The inspected environment contains Python 3.12.12, pip 25.1.1, NumPy 2.3.4,
SciPy 1.16.3, and mpi4py 4.1.1. Its binaries target AION's AMD `zen2` CPUs and
reference AION external libraries. The MPI test establishes that MPI launches
on IRIS; it does not establish portability of every compiled dependency.

## Install the pip overlay once

Use CUDA 12 JAX wheels for IRIS's V100 GPUs. CUDA 13 wheels do not support
V100s. The tested node's NVIDIA driver was 580.159.04, which meets the CUDA 12
JAX driver requirement. The CUDA version printed by `nvidia-smi` describes
driver capability, not the toolkit installed in Python.

Run where package downloads are available, with `mpc-v10` activated:

```bash
python -m venv --system-site-packages "$HOME/venvs/mpc-v10-jax"
source "$HOME/venvs/mpc-v10-jax/bin/activate"

# Preserve the packages already used by the FEniCS/MPI stack.
cat > "$VIRTUAL_ENV/constraints.txt" <<'EOF'
numpy==2.3.4
scipy==1.16.3
mpi4py==4.1.1
EOF

python -m pip install --upgrade pip setuptools wheel nanobind

# Install JAX and its CUDA libraries before compiling mpi4jax.
python -m pip install \
    -c "$VIRTUAL_ENV/constraints.txt" 'jax[cuda12]'

# Reuse the installed mpi4py, MPI, JAX, and CUDA build dependencies.
MPICC="$(command -v mpicc)" \
python -m pip install --no-build-isolation --no-cache-dir \
    -c "$VIRTUAL_ENV/constraints.txt" 'jax[cuda12]' mpi4jax

python -m pip check
```

`--system-site-packages` exposes the existing Spack Python packages to the
virtual environment. Pip additions belong to the virtual environment; avoid
installing them directly into Spack's environment view or using `pip --user`.
`--no-build-isolation` lets mpi4jax build against the existing mpi4py and JAX.
The constraints make incompatible dependency requirements fail instead of
silently replacing the numerical/MPI stack. If resolution fails, inspect the
reported JAX/mpi4jax requirements before choosing compatible versions.

Leave `MPI4JAX_USE_CUDA_MPI` unset initially. mpi4jax normally stages GPU data
through CPU memory for MPI communication. Direct GPU-buffer communication
requires verified CUDA-aware MPI support; the successful CPU MPI test does
not establish that capability.

## Activate and test in each new GPU session

```bash
spack env activate mpc-v10
source "$HOME/venvs/mpc-v10-jax/bin/activate"

srun --mpi=pmix -n 4 -c 7 --gpus-per-task=1 --cpu-bind=cores \
    "$VIRTUAL_ENV/bin/python" -c '
from mpi4py import MPI
import jax
import mpi4jax

jax.config.update("jax_platforms", "cuda")
jax.distributed.initialize(local_device_ids=[0])
assert jax.process_count() == 4
assert jax.local_device_count() == 1
assert jax.device_count() == 4
print(
    f"rank={MPI.COMM_WORLD.rank} "
    f"jax={jax.__version__} "
    f"local={jax.local_devices()} "
    f"global_count={jax.device_count()}",
    flush=True,
)
jax.distributed.shutdown()
'
```

Expect four ranks, each with one local GPU and four global GPUs. Each task is
bound to one GPU by Slurm, so its process-local CUDA index is `0`.
Call `jax.distributed.initialize(local_device_ids=[0])` before device queries
or JAX computations. This check verifies imports and distributed GPU discovery;
it does not test an mpi4jax collective or numerical correctness.

On 2026-09-08, the user confirmed that this import/discovery check ran with
JAX 0.11.1 on all four ranks, each reporting one local CUDA device and four
global devices. The shell reported status 0. `WatchTasksAsync` connection-refused
warnings appeared at teardown; these remain under investigation. This result
does not yet verify GPU computation or mpi4jax collectives.

Run the separate collective diagnostic (syntax checked, not yet GPU-tested):

```bash
cd "$HOME/JAX-ghost/HPC-GPU"
unset MPI4JAX_USE_CUDA_MPI
srun --mpi=pmix -n 4 -c 7 --gpus-per-task=1 --cpu-bind=cores \
    "$VIRTUAL_ENV/bin/python" -u check-mpi4jax.py
echo "srun exit status: $?"
```

Expect `GPU sum=10.0 PASS` on every rank and exit status 0. Shutdown markers
help locate warnings relative to explicit shutdown; stdout/stderr merging can
still affect displayed ordering. The barriers synchronize application work,
but are not a proven fix for background watcher shutdown warnings.

For application runs, activate both environments and use the same launcher:

```bash
cd "$HOME/JAX-ghost/HPC-GPU"
srun --mpi=pmix -n 4 -c 7 --gpus-per-task=1 --cpu-bind=cores \
    "$VIRTUAL_ENV/bin/python" -u jax-gpu.py
```

Before using the current `jax-gpu.py`, change its initialization to
`jax.distributed.initialize(local_device_ids=[0])` and fix the print of
`local_device_ids` before assignment. Print `os.environ.get("CUDA_VISIBLE_DEVICES")`
directly; do not use its physical GPU identifiers as process-local JAX indices.

## References

- [ULHPC getting started](https://hpc-docs.uni.lu/getting-started/)
- [IRIS GPU nodes and allocation](https://hpc-docs.uni.lu/systems/iris/compute/)
- [OpenMPI launching with Slurm and PMIx](https://docs.open-mpi.org/en/main/launching-apps/slurm.html)
- [JAX installation](https://docs.jax.dev/en/latest/installation.html)
- [mpi4jax installation](https://mpi4jax.readthedocs.io/en/latest/installation.html)
- [mpi4jax CUDA-aware MPI precautions](https://mpi4jax.readthedocs.io/en/latest/sharp-bits.html#using-cuda-mpi)
