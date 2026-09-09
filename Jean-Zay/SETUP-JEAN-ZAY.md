# Setting up on IDRIS Jean Zay

Use the site's software stack, not the IRIS Spack setup in `small-test.sh`.
The IDRIS H100 user-guide example starts from a clean module/Conda environment,
loads `arch/h100`, and uses an H100 project account and constraint. The exact
Python/JAX/MPI module versions must be checked on the actual machine; the guide's
PyTorch example does not establish a supported JAX module version.

Source: [IDRIS new-user guide](https://www.idris.fr/media/eng/ia/guide_nouvel_utilisateur_ia-eng.pdf).
Direct access to the new documentation landing page timed out during this review;
the indexed official guide was available. Its examples may predate current modules.

## Identify the software stack first

In a fresh shell, deactivate any inherited personal Conda/venv environment first.
For H100 nodes (use the corresponding site architecture for another GPU type):

```bash
module purge
module load arch/h100
module avail jax
module avail mpi4py
module avail openmpi
```

Choose the site's compatible GPU JAX and MPI/Python modules from that output.
Do not mix an unrelated Conda MPI with a module-provided MPI. Load the same modules
in the batch job, including the architecture module, before activating any personal
venv. Save these commands in an environment setup file and set:

```bash
export JAXGHOST_ENV_SETUP=/absolute/path/to/jean-zay-environment.sh
```

Our runtime needs GPU-enabled JAX, NumPy and mpi4py. It does not need DOLFINx,
mpi4jax or PyTorch. JAX 0.11.1 was tested on IRIS; other versions and the partner's
Jean Zay environment have not yet been validated. Do not reinstall CUDA/JAX solely
because the placement check fails: first collect the actual stack:

```bash
module list
command -v python
command -v mpicc
python -m pip show jax jaxlib mpi4py
python -c 'from mpi4py import MPI; print(MPI.Get_library_version())'
```

Run GPU checks in allocated GPU jobs. Keep the repository, prepared fixtures and
environment available on all compute nodes. Use the site's approved installation
procedure if a dependency is missing; compute nodes should not perform package
installation during the benchmark.

## Submit the small test before scaling

The smoke script defaults to the IRIS partition `gpu`; override it on Jean Zay
with the partition/account/QoS from a working job. The two scaling scripts now
target H100 directly, load `arch/h100` followed by `JAXGHOST_ENV_SETUP`, and
reserve 24 CPU cores per rank. Submit those with `--account=PROJECT@h100`.
For H100, the official guide uses `--constraint=h100` and an account of the form
`PROJECT@h100`. Do not copy an IRIS account, partition or Spack environment.

```bash
# From the repository root; replace all uppercase placeholders.
sbatch --partition=SITE_GPU_PARTITION --account=PROJECT@h100 \
  --constraint=h100 Jean-Zay/smoke-1node.sbatch
```

Add the site's required QoS and MPI launcher option, if any. The smoke script reserves four GPUs per node and one CPU core per rank; adjust
both its allocation and launcher CPU count if required. The scaling scripts
use 24 CPU cores per rank in both the allocation and worker steps.
After 1/2/4 ranks pass, submit `scaling-1node.sbatch`; test two nodes only after
single-node correctness and timings work.

## Why the screenshot's four-visible-GPUs error matters

The reported failure was before JAX initialization: the CUDA driver still saw four
GPUs although the worker had attempted to mask to one. Previously `worker.py`
imported mpi4py/MPI before changing `CUDA_VISIBLE_DEVICES`. A CUDA-aware MPI stack
can initialize CUDA during MPI startup, making that mask change too late. This is
a plausible explanation, not yet confirmed on the partner's machine.

The updated `gpu_bootstrap.py` selects the GPU before NumPy/MPI/JAX imports using
the launcher's local-rank environment. It preserves a scheduler-provided single
GPU mask and selects within a multi-GPU mask. MPI subsequently checks that the
launcher rank agrees with its shared-memory communicator, and the normal distinct
GPU UUID, CPU affinity and one-local-JAX-GPU checks still run.

Transfer `gpu_bootstrap.py`, `worker.py`, and `placement.py` together before retrying.
Do not delete the one-GPU check or accept four JAX GPUs per MPI rank. If it still
fails, provide the full worker log, batch script and software-stack output above.
