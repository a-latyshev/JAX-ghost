# Minimal IRIS GPU communication environment

`spack.yaml` builds UCX 1.19.0 and OpenMPI 5.0.8 from source, using the
site CUDA 12.6 toolkit, GCC 13.2 compiler, and matching binutils 2.40 by
explicit paths. It overrides
inherited external MPI/UCX policies. The versions are available in this
checkout's Spack recipes; this is a candidate stack, not a proven direct-GPU
transport configuration. `cuda_arch=70` targets IRIS V100 GPUs.

Start in a fresh shell with no mpc-v10 environment, old venv, or MPI/UCX modules
activated. Source your Spack setup, then:

```bash
spack env create jax-mpi-gpu "$HOME/JAX-ghost/HPC-GPU/spack-jax-mpi/spack.yaml"
spack env activate jax-mpi-gpu
spack concretize
spack install -j 4
```

Run compilation on a site-approved build/compute allocation and adjust `-j`
to its allocated CPUs. Inspect the concrete DAG before installation: OpenMPI
and UCX must be source builds with CUDA enabled. Keep the resulting lockfile.
The site paths and Slurm version are IRIS-specific; adjust for other clusters.

## Recovering from the assembler failure

`as: unrecognized option '--gdwarf-5'` means the assembler selected by GCC
cannot handle its debug-information option. The external GCC configuration
now prepends the matching site binutils directory inside Spack builds.

A named environment has its own copy of `spack.yaml`; editing this repository's
file alone does not update an existing `jax-mpi-gpu` environment. After applying
the same compiler environment setting to that copy, run on an IRIS compute
allocation (the site Skylake binaries may fail on older login-node CPUs):

```bash
spack env activate jax-mpi-gpu
spack concretize --force
spack install -j 4
spack build-env readline -- sh -c 'command -v as; as --version'
```

The optional assembler check requires readline dependencies to be installed;
run it after installation and confirm the configured binutils 2.40 executable.
Re-concretization updates compiler metadata in the
lockfile and may change hashes; Spack reuses installed packages where hashes
still match. Full installation and GPU transport validation remain necessary.

## Python overlay

The installed Spack mpi4jax recipe is only 0.3.11.post3. Use an isolated venv
for the current Python GPU packages instead. Do not use `--system-site-packages`:
mpi4py and NVIDIA packages should reside together for mpi4jax CUDA discovery.

```bash
python -m venv "$HOME/venvs/jax-mpi-gpu"
source "$HOME/venvs/jax-mpi-gpu/bin/activate"
python -m pip install --upgrade pip setuptools wheel nanobind
MPICC="$(command -v mpicc)" python -m pip install \
    --no-cache-dir --no-binary=mpi4py 'mpi4py==4.1.1'
python -m pip install 'jax[cuda12]==0.11.1'
MPI4JAX_BUILD_MPICC="$(command -v mpicc)" python -m pip install \
    --no-build-isolation --no-cache-dir --no-binary=mpi4jax \
    'mpi4jax==0.9.1.post1'
python -m pip check
python -m pip freeze > "$VIRTUAL_ENV/requirements.lock.txt"
```

Activate Spack first, then this venv in every job. Do not load the old site
UCX-CUDA/OpenMPI modules into this environment. JAX's pip CUDA libraries must
resolve consistently; if site CUDA paths are in LD_LIBRARY_PATH, give the
venv NVIDIA library directories precedence before launching Python:

```bash
NVIDIA_LIB_DIRS=$(python - <<'PY'
from pathlib import Path
import sysconfig
root = Path(sysconfig.get_path('purelib')) / 'nvidia'
print(':'.join(str(p) for p in sorted(root.glob('*/lib')) if p.is_dir()))
PY
)
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MPI4JAX_USE_CUDA_MPI=1
```

Verify the actual MPI loaded by mpi4py and CUDA support on an allocated GPU
node. Validate minimal exchanges and trace their payload path before adding
DOLFINx or claiming absence of MPI-internal CPU staging. The existing
`check-jaxghost.py` still needs DOLFINx; a fixture-only replacement is separate
work. This file does not export any FEniCSx data.

Validation: full concretization succeeded on 2026-09-08 after exempting the
external Slurm package from the global compiler/CPU requirements and updating
its version to the installed 25.11.8. The active jax-mpi-gpu environment now has
a lockfile. Installation and GPU runtime validation have not been performed.
