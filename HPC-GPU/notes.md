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

## Strict DOLFINx–JAXGhost GPU test

From `HPC-GPU`, with the base environment and overlay activated:

```bash
export MPI4JAX_USE_CUDA_MPI=1
srun --mpi=pmix -n 4 -c 1 --gpus-per-task=1 --cpu-bind=cores \
    "$VIRTUAL_ENV/bin/python" -u check-jaxghost.py
```

Use `-n 2` for a two-GPU run. This test requires OpenMPI runtime CUDA support,
mpi4jax CUDA support, and one visible GPU per rank; failures abort with no CPU
fallback. It checks DOLFINx host ownership and unchanged source data, GPU array
placement, and repeated compiled forward/reverse-add updates against DOLFINx.
Expected: one PASS line per rank and exit status 0. Host copies are used only
for setup and validation. CUDA-aware MPI does not prove GPUDirect or exclude
internal transport staging. The script imports JAXGhost from this checkout.

For a native crash, rerun `check-jaxghost.py --verbose` to print per-rank stages.
A PETSc SEGV report means PETSc caught the signal, not necessarily that PETSc
caused it. Keep the complete log and first failing stage. Python fault handling
is enabled, although native library signal handlers can intercept it.

### Verified IRIS transport configuration

On `iris-173` (2026-09-08), the strict four-rank JAXGhost test crashed during
its first GPU exchange with the default MPI transport. The same test passed
both forward and reverse updates for two value sets on all ranks with:

```bash
set -o pipefail
srun --mpi=pmix -n 4 -c 1 --gpus-per-task=1 --cpu-bind=cores \
    env MPI4JAX_USE_CUDA_MPI=1 OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,tcp \
    "$VIRTUAL_ENV/bin/python" -u check-jaxghost.py \
    2>&1 | tee jaxghost-ob1.log
```

This is a verified correctness configuration for the tested node/environment,
not a performance recommendation for all clusters. It bypasses UCX and
shared-memory transports while retaining GPU-buffer MPI calls. MPI can stage
internally; this does not prove GPU-direct transport. A subsequent GDB investigation confirmed selection of UCX and a crash in
its shared-memory CPU-copy path; UCX exposes no CUDA resources. The working
TCP path was confirmed to call OpenMPI's internal GPU-copy routine. See
[transport investigation](transport-investigation.md) for stacks and evidence.
DOLFINx host ownership, unchanged source arrays, GPU placement, and compiled
JAXGhost results all passed. Shutdown watcher warnings remained; no explicit
launcher exit status was included in the supplied successful log.

The subsequent instrumented TCP run passed all ranks with launcher status 0.

### UCX with CUDA plugins: verified workaround

On iris-173 (2026-09-08), loading UCX-CUDA alone was insufficient. With
`UCX_MEMTYPE_CACHE=y`, GDB showed a SIGSEGV in UCX's shared-memory CPU-copy
path even with CUDA plugins loaded. With `UCX_MEMTYPE_CACHE=n`, the strict
four-rank test passed under GDB and normally (launcher status 0).
This supports a GPU-memory classification problem; the precise cache/hook
failure remains unisolated. See the [UCX GPU-segfault FAQ](https://openucx.readthedocs.io/en/master/faq.html#i-m-running-ucx-with-gpu-memory-and-getting-a-segfault-why).

After activating Spack and the venv, run from `HPC-GPU`:

```bash
(
    module load lib/UCX-CUDA/1.15.0-GCCcore-13.2.0-CUDA-12.6.0 || exit
    NVIDIA_LIB_DIRS=$("$VIRTUAL_ENV/bin/python" - <<'PYCUDA'
from pathlib import Path
import sysconfig
root = Path(sysconfig.get_path("purelib")) / "nvidia"
print(":".join(str(p) for p in sorted(root.glob("*/lib")) if p.is_dir()))
PYCUDA
    )
    export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    unset OMPI_MCA_btl
    srun --mpi=pmix -n 4 -c 1 --gpus-per-task=1 --cpu-bind=cores \
        env MPI4JAX_USE_CUDA_MPI=1 OMPI_MCA_pml=ucx UCX_MEMTYPE_CACHE=n \
        "$VIRTUAL_ENV/bin/python" -u check-jaxghost.py --verbose
)
```

The library ordering preserves the venv's CUDA libraries after the module
loads CUDA 12.6; otherwise cuSPARSE can fail to load due to a conflicting
nvJitLink library. The subshell confines module/path changes to this run.
Disabling the memory-type cache retains CUDA-aware MPI and does not enable
mpi4jax CPU staging. Internal MPI staging, GPU-direct transfers, performance,
and multi-node behavior were not established by this test. Shutdown watcher
warnings remained after successful computation.

Logs and the reproduction script are retained in
`$HOME/jaxghost-transport-evidence-20260908/` (`jaxghost-ucx-cache-on.log`,
`jaxghost-ucx-cache-off.log`, and `probe-ucx.sh`).


Transport follow-up (iris-173, 2026-09-08): GDB confirmed that the working
UCX configuration internally stages this diagnostic's small ghost messages
through host buffers (24 CUDA pack copies and 24 unpack copies, no CUDA IPC
copy breakpoint hits). All four ranks passed and exited normally. This keeps
`MPI4JAX_USE_CUDA_MPI=1`; larger-message and multi-node routes remain untested.
See [the measured payload path](transport-investigation.md#actual-payload-path-with-the-working-ucx-configuration).
