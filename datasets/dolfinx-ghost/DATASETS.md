# DOLFINx ghost reference datasets

Twelve scalar continuous P1 fixtures on unit-domain simplicial meshes. These
are distributed vector layouts and CPU reference results, not full meshes or
assembled finite-element systems. Each dataset must be replayed with exactly
its recorded MPI rank count. Local entries are ordered `[owned | ghosts]`.

## Dataset log

Generated on 2026-09-08 on IRIS iris-171 using the existing `mpc-v10` environment.
The meshes have 10,000 intervals, 9,800 triangles (70×70 square subdivisions),
and 10,368 tetrahedra (12×12×12 cube subdivisions). Counts below exclude ghost
cells; owned/ghost entries are listed in rank order.

| Dataset | Global cells | Global DOFs | Owned entries per rank | Ghost entries per rank |
| --- | ---: | ---: | --- | --- |
| `1d/1ranks` | 10000 | 10001 | 10001 | 0 |
| `1d/2ranks` | 10000 | 10001 | 4998, 5003 | 1, 2 |
| `1d/3ranks` | 10000 | 10001 | 3337, 3323, 3341 | 1, 1, 4 |
| `1d/4ranks` | 10000 | 10001 | 2476, 2479, 2510, 2536 | 3, 1, 1, 4 |
| `2d/1ranks` | 9800 | 5041 | 5041 | 0 |
| `2d/2ranks` | 9800 | 5041 | 2524, 2517 | 91, 90 |
| `2d/3ranks` | 9800 | 5041 | 1686, 1703, 1652 | 96, 92, 140 |
| `2d/4ranks` | 9800 | 5041 | 1252, 1255, 1277, 1257 | 63, 170, 93, 130 |
| `3d/1ranks` | 10368 | 2197 | 2197 | 0 |
| `3d/2ranks` | 10368 | 2197 | 1092, 1105 | 206, 203 |
| `3d/3ranks` | 10368 | 2197 | 742, 726, 729 | 234, 266, 212 |
| `3d/4ranks` | 10368 | 2197 | 570, 539, 540, 548 | 174, 236, 232, 177 |

## File format and references

Every dataset has `metadata.json` and `rank-0000.npz`, etc. NPZ files contain:

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `owned_range` | `(2,)`, int64 | Global interval `[start, stop)` |
| `ghost_global_indices` | `(num_ghosts,)`, int64 | Ghost IDs in original local order |
| `ghost_owner_ranks` | `(num_ghosts,)`, int32 | Owner of each ghost |
| `forward_input`, `forward_expected` | `(2, local_size)`, float64 | Two independent forward cases |
| `reverse_input`, `reverse_expected` | `(2, local_size)`, float64 | Two independent reverse-add cases |

Block size is 1. For case `k`, owned values are `1 + global_id + 100000*k`.
Forward ghosts start at `-(k+1)`; reverse ghosts start at `(k+1)*(rank+1)`.
Forward replaces ghosts and preserves owned entries. Reverse adds ghost
contributions to existing owner values and preserves ghosts. References are
computed with DOLFINx, never with JAXGhost. Integer-valued float64 inputs allow
exact comparisons. Single-rank cases have no ghosts and both operations are
no-ops. Metadata records package versions, MPI library, partition counts,
mesh parameters and validation checks. Partition numbering may change with
DOLFINx/partitioner versions; saved fixtures preserve one concrete partition.

## Reproduce and replay

From the repository root, activate a compatible DOLFINx/MPI environment for
export. Reserve at least four CPU slots. For example, with the existing stack:

```bash
spack env activate mpc-v10
source "$HOME/venvs/mpc-v10-jax/bin/activate"
for n in 1 2 3 4; do
    mpirun -n "$n" python scripts/export_ghost_datasets.py
done
```

The exporter writes all three dimensions per invocation, checks the metadata
and references, and reloads each NPZ to verify it. It overwrites the selected
rank-count fixtures; use `--output /another/directory` for a separate export.

Replay needs NumPy, mpi4py, JAX, mpi4jax and this checkout, with no DOLFINx import:

```bash
mpirun -n 4 python scripts/replay_ghost_dataset.py \
    datasets/dolfinx-ghost/3d/4ranks --backend cpu
```

For a configured CUDA-aware MPI environment and one visible GPU per rank:

```bash
export MPI4JAX_USE_CUDA_MPI=1
srun --mpi=pmix -n 4 -c 1 --gpus-per-task=1 --cpu-bind=cores \
    python scripts/replay_ghost_dataset.py datasets/dolfinx-ghost/3d/4ranks --backend cuda
```

CPU replay is the recorded validation. GPU replay additionally requires a
working CUDA JAX/mpi4jax/MPI runtime; fixture data do not configure that stack.
A PASS verifies values and placement, not GPU-direct transport or performance.
The replay script aborts MPI on failures so peer ranks do not hang.

## Loading snippet

Save the following as `replay_example.py` at the repository root and launch
`mpirun -n 4 python replay_example.py`. This CPU example shows the adapter and
both compiled updates explicitly; use the supplied replay script for CLI
backend selection and error handling.

```python
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from mpi4py import MPI
import jax

sys.path.insert(0, str(Path('src').resolve()))
from jaxghost import JAXGhost

jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)
comm = MPI.COMM_WORLD
try:
    assert comm.size == 4
    device, = jax.local_devices()
    path = Path('datasets/dolfinx-ghost/3d/4ranks')
    with np.load(path / f'rank-{comm.rank:04d}.npz', allow_pickle=False) as data:
        start, stop = map(int, data['owned_range'])
        index_map = SimpleNamespace(
            size_local=stop-start, local_range=(start, stop),
            ghosts=data['ghost_global_indices'], owners=data['ghost_owner_ranks'])
        with JAXGhost.from_index_map(index_map, comm, block_size=1) as ghost:
            for name in ('forward', 'reverse'):
                update = jax.jit(getattr(ghost, f'scatter_{name}'))
                for case in range(2):
                    x = jax.device_put(data[f'{name}_input'][case], device)
                    y = update(x)
                    y.block_until_ready()
                    jax.effects_barrier()
                    np.testing.assert_array_equal(
                        np.asarray(y), data[f'{name}_expected'][case])
    print(f'rank={comm.rank} PASS', flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    comm.Abort(1)
```


## Recorded validation

- All 12 exports passed ownership, reference, preserved-entry and NPZ reload checks.
- All 12 CPU replays passed both cases of both operations; replay processes
  explicitly checked that no DOLFINx module was imported.
- The loading snippet above passed on four MPI ranks.
- Required repository runner: two-rank smoke passed; one-rank suite had
  26 passes and 2 skips; three- and four-rank suites each had 28 passes per rank.
- The two-rank suite had 27 passes and one failure per rank in the existing
  `test_matrix_dolfinx` float32 comparison: maximum violating absolute difference
  `9.059906e-06` with `rtol=atol=7.62939e-06`. The standard runner stopped there;
  three/four-rank checks were then run separately with `--ranks 3` and `--ranks 4`.
  No matrix implementation or tolerance was changed for this dataset task.
- GPU replay and GPU transport/performance checks were not run for these fixtures.

Environment: Python 3.12.12, DOLFINx 0.10.0.post4, NumPy 2.3.4,
mpi4py 4.1.1, JAX/jaxlib 0.11.1, patched mpi4jax 0.9.1.post1,
OpenMPI 4.1.6. CPU validation used `OMPI_MCA_pml=ob1`,
`OMPI_MCA_btl=self,tcp`, one JAX CPU device per rank, and one OpenMP/OpenBLAS
thread per rank. Nested mpirun jobs ran inside allocation 5873886 on iris-171;
`OMPI_MCA_rmaps_base_oversubscribe=1` allowed four ranks inside the parent job step.

Raw logs: [generation/replay and initial suite](validation.log),
[snippet and remaining suites](validation-extra.log). Syntax checks and
`git diff --check` passed. There are 30 NPZ files (807,525 bytes total) and
12 JSON metadata files.
