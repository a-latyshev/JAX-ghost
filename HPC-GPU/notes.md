# JAX + mpi4jax on HPC

This guide uses Slurm and NVIDIA GPUs, with one MPI process per GPU. Adapt the
partition, account, CPU count, environment activation, and launcher to your
cluster. Start on one node before testing multiple nodes.

## Allocate and verify MPI

Activate your site's Python/compiler/MPI stack (for example, a Spack environment).
Reserve CPU tasks as well as GPUs; `-N` counts nodes and `-n` counts processes.

```bash
# Replace the partition and resource counts with your site's values.
salloc -p your_gpu_partition -N 1 -n 4 -c 4 --gpus-per-task=1 -t 01:00:00
```

Use the site's compute-node shell or job steps. For OpenMPI with Slurm PMIx:

```bash
srun --mpi=list
srun --mpi=pmix -n 4 -c 4 python -c \
    'from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size, MPI.Get_library_version(), flush=True)'
```

Expect ranks 0–3, each reporting size 4. If this fails, resolve the site's MPI
launcher/plugin configuration before testing JAX. Other schedulers/MPI stacks
may require a different launcher. Shared filesystems do not guarantee that
CPU-specific binaries or external library paths work on another cluster.

## Install a pip overlay

With the base stack activated and NumPy, SciPy, and mpi4py already available:

```bash
python -m venv --system-site-packages "$HOME/venvs/hpc-jax"
source "$HOME/venvs/hpc-jax/bin/activate"

# Preserve the base numerical/MPI packages.
python - <<'PY' > "$VIRTUAL_ENV/constraints.txt"
from importlib.metadata import version
for name in ("numpy", "scipy", "mpi4py"):
    print(f"{name}=={version(name)}")
PY

python -m pip install --upgrade pip setuptools wheel nanobind
python -m pip install -c "$VIRTUAL_ENV/constraints.txt" 'jax[cuda12]'
MPI4JAX_BUILD_MPICC="$(command -v mpicc)" \
python -m pip install --no-build-isolation --no-cache-dir \
    -c "$VIRTUAL_ENV/constraints.txt" 'jax[cuda12]' mpi4jax
python -m pip check
```

Choose JAX wheels for your GPU and driver: CUDA 12 is needed for V100s; CUDA 13
JAX wheels do not support them. `nvidia-smi` reports driver capability, not the
Python CUDA toolkit. Install where downloads and compilation are permitted.
Activate the same base stack **before** the overlay in every job.

### Spack/venv CUDA-discovery fix: mpi4jax 0.9.1.post1

This release searches for NVIDIA wheels beside mpi4py. With mpi4py in Spack and
CUDA in a venv, it can print `CUDA path not found` and install only CPU support.
`--no-build-isolation` alone does not fix that layout.

Apply the [CUDA 12 discovery patch](mpi4jax-0.9.1.post1-cuda12-discovery.patch)
to a freshly extracted **0.9.1.post1** source distribution. From that directory:

```bash
CUDA_PATCH=/path/to/JAX-ghost/HPC-GPU/mpi4jax-0.9.1.post1-cuda12-discovery.patch
patch --dry-run -p1 < "$CUDA_PATCH"
# Continue only if the dry run succeeds.
patch -p1 < "$CUDA_PATCH"

set -o pipefail
MPI4JAX_BUILD_MPICC="$(command -v mpicc)" \
python -m pip install --force-reinstall --no-deps --no-build-isolation -v . \
    2>&1 | tee "$VIRTUAL_ENV/mpi4jax-build-fixed.log"
```

Confirm the build includes `mpi_xla_bridge_cuda`. This patch is specific to
that release and CUDA 12 wheels; retain it with your environment records.
Do not guess a `CUDA_ROOT`: use it only for an actual compatible local toolkit.

## Test GPU communication

From `HPC-GPU`, run the adjacent four-rank diagnostic:

```bash
# Explicitly use transfers staged through CPU memory initially.
export MPI4JAX_USE_CUDA_MPI=0
srun --mpi=pmix -n 4 -c 4 --gpus-per-task=1 --cpu-bind=cores \
    "$VIRTUAL_ENV/bin/python" -u check-mpi4jax.py
echo "srun exit status: $?"
```

Expect `GPU sum=10.0 PASS` on all four ranks and status 0. The script checks
GPU discovery, mpi4jax CUDA support, and a JIT-compiled MPI sum. It targets
mpi4jax 0.9.1.post1's array-returning API; older examples may return tokens too.

For application code:

- Initialize JAX before device queries/computation. Use
  `jax.distributed.initialize(local_device_ids=[0])` **only when each process
  sees its own single GPU**. Other visibility layouts need distinct assignments.
- Transfer local data with `jax.device_put`; pack only the entries to exchange.
- Owners send values for forward ghost updates; ghost holders send contributions
  to owners for reverse accumulation. Implement the ownership map explicitly.
- Separate mpi4jax and mpi4py communicators, and await results before shutdown.

### CUDA-aware MPI

JAX GPU support, mpi4jax GPU support, and CUDA-aware MPI are separate capabilities.
For OpenMPI, inspect the actual build rather than relying on external Spack metadata:

```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

If it reports `true`, rerun the diagnostic with `MPI4JAX_USE_CUDA_MPI=1` to test
device-buffer communication. This removes mpi4jax's CPU staging, but does not
guarantee GPUDirect RDMA or eliminate staging inside MPI's transport.

## Debug by stage

| Symptom | First check |
| --- | --- |
| More processors requested than permitted | Allocation task/core counts versus launch request |
| Missing script/package | Working directory, `sys.executable`, base/venv activation |
| MPI/PMIx startup failure | MPI-only test and site's supported launcher |
| Illegal instruction / missing library | CPU target and external library paths |
| CUDA backend failure / wrong GPU mapping | GPU allocation, per-rank visibility, driver/wheel compatibility |
| `has_cuda_support()` is false | CUDA extension build log; Spack/venv discovery issue above |
| Collective hangs or wrong results | Matching rank participation, shapes/types, ownership map, communicator ordering |

`WatchTasksAsync` warnings only at shutdown, after correct results and with
status 0, appear low severity in the observed test; their exact cause remains
unconfirmed. During computation, or with hangs/nonzero status, investigate them.
Barriers are not a proven fix. For validated runs, optional
`TF_CPP_MIN_LOG_LEVEL=2` suppresses C++ INFO/WARNING messages broadly, not the cause.

Share the full first error, allocation/launch commands, exit status, package
versions, modules/Spack lockfile, GPU/driver, and which checks passed. Multi-node
runs additionally need reachable coordination and working inter-node transports.

## UL-HPC: IRIS quick-start

Run from the IRIS login node. This requests four GPUs and seven CPU cores per
rank, matching IRIS's 28-core/four-GPU nodes:

```bash
si-gpu -N 1 -n 4 -c 7 -G 4 -t 0-02:00:00
```

Inside the allocated shell, activate the base stack first, then the pip overlay:

```bash
spack env activate mpc-v10
source "$HOME/venvs/mpc-v10-jax/bin/activate"
cd "$HOME/JAX-ghost/HPC-GPU"

srun --mpi=pmix -n 4 -c 7 --cpu-bind=cores \
    "$VIRTUAL_ENV/bin/python" -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size, MPI.Get_library_version(), flush=True)'

export MPI4JAX_USE_CUDA_MPI=0
srun --mpi=pmix -n 4 -c 7 --gpus-per-task=1 --cpu-bind=cores \
    "$VIRTUAL_ENV/bin/python" -u check-mpi4jax.py
echo "srun exit status: $?"
```

These environment names and checkout paths belong to the tested setup; other
UL-HPC users should substitute their own. For a new installation, follow the
pip-overlay instructions above, using `$HOME/venvs/mpc-v10-jax` as the venv path.
Use CUDA 12 wheels for IRIS V100s and apply the discovery patch if reusing
Spack's mpi4py with mpi4jax 0.9.1.post1.

The prepared patched source in this setup can be reinstalled without rebuilding
Spack or MPI:

```bash
MPI4JAX_BUILD_MPICC="$(command -v mpicc)" \
"$VIRTUAL_ENV/bin/python" -m pip install \
    --force-reinstall --no-deps --no-build-isolation \
    "$VIRTUAL_ENV/src/mpi4jax-0.9.1.post1-cuda-fix"
```

Use `--mpi=pmix` explicitly: it worked with IRIS's advertised `pmix_v6` plugin.
Do not infer MPI failure from a plain Python MPI import in the interactive shell.
To test CUDA-aware transfers, set `MPI4JAX_USE_CUDA_MPI=1` and repeat the GPU
command; keep the CPU-staged run as a baseline.

For application runs, replace `check-mpi4jax.py` with the application filename.
Before using the existing `jax-gpu.py`, fix its `local_device_ids` print before
assignment and use `jax.distributed.initialize(local_device_ids=[0])` with this
one-visible-GPU-per-task launch.

Verified on 2026-09-08: Python 3.12.12, OpenMPI 4.1.6, mpi4py 4.1.1,
JAX 0.11.1, patched mpi4jax 0.9.1.post1, V100 GPUs, driver 580.159.04.
All four GPU sum checks passed with CPU staging; shutdown watcher warnings
remained. Actual OpenMPI reports CUDA support despite Spack's external `~cuda`
metadata. Direct-buffer mode needs its own recorded runtime result.

The base environment references AION AMD `zen2` binaries and external paths;
these checks do not establish portability of every package onto IRIS.

## References

[JAX installation](https://docs.jax.dev/en/latest/installation.html) ·
[JAX initialization](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html) ·
[mpi4jax installation](https://mpi4jax.readthedocs.io/en/latest/installation.html) ·
[mpi4jax precautions](https://mpi4jax.readthedocs.io/en/latest/sharp-bits.html) ·
[OpenMPI/Slurm](https://docs.open-mpi.org/en/main/launching-apps/slurm.html) ·
[IRIS GPU nodes](https://hpc-docs.uni.lu/systems/iris/compute/)
