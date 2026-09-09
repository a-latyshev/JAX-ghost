# Jean-Zay: sharded JAX matvec on 1–8 GPUs

Run from the **JAX-ghost repository root**. The scripts use the actual sharded
backend in `src/jaxghost`; no alternative numerical implementation is bundled.

## Requirements

- Linux, NVIDIA GPUs, GPU-enabled JAX (tested: **0.11.1**), NumPy, mpi4py,
  and an MPI installation compatible with mpi4py. MPI is mandatory.
- No DOLFINx, mpi4jax, pytest, matplotlib or CUDA-aware MPI is needed for replay.
  MPI exchanges setup metadata and timing scalars; JAX handles GPU communication.
- A shared checkout and `Jean-Zay/data/` visible to every node. **Copy the prepared
  data directory along with the checkout**; a source-only checkout may omit data.
- One MPI process per GPU and disjoint CPU bindings. Use full GPUs; multiple MIG
  instances on one physical GPU are rejected. The CUDA driver is queried directly
  for UUID and PCI identity; nvidia-smi is not required.

Use the HPC site's working JAX/MPI environment. Do not replace its GPU/MPI stack
with a generic pip command. These scripts import the checkout directly. For a
package installation, `pip install -e . --no-deps` uses the already installed
requirements. The MPI numerical backend is optional: `pip install -e '.[mpi4jax]'`
requires an appropriately built mpi4jax. DOLFINx is only needed to prepare data.

## 1. Quick correctness test

Inside an allocation, with two GPUs visible and available:

```bash
timeout 600 mpirun -n 2 --bind-to core python Jean-Zay/check.py
```

With Slurm assigning one GPU to each task, use the site's supported MPI plugin:

```bash
timeout 600 srun --exact --ntasks=2 --gpus-per-node=4 --gpu-bind=none --cpus-per-task=1 \
  --cpu-bind=cores python Jean-Zay/check.py
```

The test selects the small n4 fixture for the MPI rank count. It checks single
and 100-call accumulated matvecs against DOLFINx references, changed values and
inputs, stale input ghosts, unchanged inputs, and output ghosts/padding. Every
rank is matched to its JAX process, and MPI shared-memory groups check UUID/PCI
uniqueness and non-overlapping CPU affinity on each node. It prints rank/GPU/CPU
assignments and PASS. Duplicate GPU assignments and wrong fixture ranks fail.

If the scheduler gives each rank one visible GPU, that selection is respected.
If it exposes all GPUs, the node-local MPI rank selects one. If visibility is
unset, local CUDA ordinals are used. Let the scheduler set CUDA_VISIBLE_DEVICES;
do not give every rank the same single GPU. JAX initialization needs network
connectivity between compute nodes; use site-approved network settings.

## 2. Strong scaling and plots

### Ready-to-submit batch scripts

Submit from the repository root or `Jean-Zay/` after configuring your site's
JAX/MPI environment. The scripts reserve four GPUs per node and default to the
IRIS `gpu` partition; override it using `sbatch --partition=...` on another HPC. Either edit their environment section or export `JAXGHOST_ENV_SETUP`
with the absolute path to your site's shell setup file.

| Script | Allocation | Cases |
| --- | --- | --- |
| [smoke-1node.sbatch](smoke-1node.sbatch) | 1 node, 4 GPUs | n4, 1/2/4 ranks; one trial, one warmup, two batches |
| [scaling-1node.sbatch](scaling-1node.sbatch) | 1 node, 4 GPUs | n99, 1/2/4 ranks; full timing defaults |
| [scaling-2nodes.sbatch](scaling-2nodes.sbatch) | 2 nodes, 4 GPUs each | n99, 1/2/4/8 ranks; full timing defaults |

```bash
# Replace partition/account with the site's values.
sbatch --partition=GPU_PARTITION --account=ACCOUNT Jean-Zay/smoke-1node.sbatch
# After the smoke test passes:
sbatch --partition=GPU_PARTITION --account=ACCOUNT Jean-Zay/scaling-1node.sbatch
sbatch --partition=GPU_PARTITION --account=ACCOUNT Jean-Zay/scaling-2nodes.sbatch
```

The two-node script runs the 1/2/4-rank points on one node and the 8-rank point
on two nodes. Its result therefore includes the cost of crossing nodes.
Two-node execution remains unverified. Each script writes a separate
`Jean-Zay/results/<script-name>-<job-id>/` directory with raw logs and results.
The IRIS-specific `small-test.sh` remains available with its local environment setup.


Allocate enough GPUs/CPUs first. Launch **one driver**, not one driver per rank:

```bash
python Jean-Zay/scale.py --ranks 1 2 4 8 \
  --launcher 'mpirun -n {ranks} --bind-to core'
```

For Slurm (works within a single-node or multi-node GPU allocation):

```bash
python Jean-Zay/scale.py --ranks 1 2 4 8 --gpus-per-node 4 \
  --launcher 'srun --nodes={nodes} --exact --ntasks={ranks} --gpus-per-node=4 --gpu-bind=none --cpus-per-task=1 --cpu-bind=cores'
```

Set both `--gpus-per-node` values to the number reserved on each node; `{nodes}`
requests only the nodes needed for each point. `--gpu-bind=none` keeps the
step’s allocated GPUs accessible for NCCL peer communication. The worker still
selects one GPU per rank and checks distinct UUIDs and one local JAX device.
On IRIS, per-task GPU isolation (`--gpus-per-task=1`) passed placement checks
but failed at the first NCCL exchange with CUDA error 101; this step-level
GPU allocation passed the 1/2/4-GPU batch smoke test.
An editable [Slurm batch example](validation/slurm-example.sbatch) is included.
Add the site's `--mpi=...` option if required. The quoted launcher is an argument
list, not shell code. It must contain `{ranks}`. Adjust CPU cores per task if
needed, keeping them fixed across points. MPI launchers must propagate the same
Python environment and access to the checkout on every node. GPU model, ranks,
node placement, affinity and software versions are stored in each result.

The default size is n99 (1,000,000 DoFs). Options: `--sizes 4`, `--ranks 1 2 4`,
`--output NEW_DIRECTORY`, `--timeout 600`. Any subset of 1–8 is accepted, but
rank 1 is required for the scaling baseline. Defaults are three independent
launches per point, ten warmups, ten batches of 100 calls. For a quick timing
smoke check: `--sizes 4 --trials 1 --warmup 1 --repeats 2`.

Each fresh launch compiles once and calls the cached JIT matvec from Python:

```python
y = y_initial
comm.Barrier()                 # outside timer; earlier GPU work is complete
start = MPI.Wtime()
for _ in range(100):
    y = mult(op, values, x, y)
y.block_until_ready()
elapsed = MPI.Wtime() - start
```

The implemented timer also completes ordered effects before stopping. Each
sample is maximum rank elapsed / 100. `A` and `x` stay fixed: this computes
**100 accumulated Ax terms**, not powers of A or an iterative solver. Every call
includes halo exchange. No outer loop is JIT-compiled. Setup, compilation,
uploads, reset, validation and I/O are excluded. Compile/first-call times are
recorded separately; y resets and validation occur outside each timed batch.

Results contain `REPORT.md`, `summary.csv`, `scaling.svg`, and per-launch JSON/logs.
The SVG shows latency, T(1)/T(p) speedup, and T(1)/(p*T(p)) efficiency, with
launch variation. Open it in a browser. Regenerate with standard Python only:

```bash
python Jean-Zay/plot.py Jean-Zay/results/YOUR_RUN
```

The summarizer rejects missing/duplicate trials, differing fixtures within a
point, changed software/source/settings, changed per-point placement, or failed
checks. Compare multi-node topology and GPU models explicitly when interpreting
scaling. Launch timeouts stop the MPI launcher; scheduler wall limits provide an
additional bound on remote tasks. Failed launches remain logged and no partial
scaling plot is presented as complete.

## Prepared data / regeneration

The compact bundle contains 12 fixtures (about 188 MiB of compressed data):

- `data/n4`: 125 DoFs, every rank count 1–8, for quick correctness checks.
- `data/n99`: 1,000,000 DoFs, rank counts 1, 2, 4, 8, for strong scaling.

Each matrix is a scalar P1 tetrahedral
unit-cube Poisson stiffness operator, float64, without boundary conditions.
Global size stays fixed across rank counts. Local NPZ files contain CSR values
and indices, owned/ghost maps, inputs and independent single/100-call DOLFINx
references. Metadata contains sizes, versions, partition information and SHA-256
checksums; replay verifies every rank file before use. Inputs use
sin(2*pi*x) + y² + 0.5*z. The correctness tolerance on owned global L2 error is
`1e-12 + 1e-10 * reference_norm`; output ghosts must agree exactly.

Check the complete transferred data tree without MPI or third-party packages:

```bash
python Jean-Zay/verify_data.py
```

Only the preparer needs DOLFINx and a compatible MPI environment. Eight CPU
ranks suffice; eight GPUs are not required:

```bash
python Jean-Zay/prepare_all.py --launcher 'mpirun -n {ranks} --bind-to core'
```

This prepares the 12 bundled fixtures and skips already completed ones. Optional
cases can be regenerated with, for example, `--sizes 46 --ranks 1 2 4 8` or
`--sizes 99 --ranks 3 5 6 7`; they require DOLFINx only during preparation.
Use a different
`--output` to regenerate from scratch. Preserve the entire data tree when sharing;
NPZ files are compressed. No user-site paths or IRIS modules are embedded in the
portable commands. See `VALIDATION.md` for what was actually exercised here.
