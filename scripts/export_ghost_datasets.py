"""Export scalar P1 DOLFINx ghost references; launch with 1, 2, 3, or 4 MPI ranks."""
import argparse
import json
from pathlib import Path
import platform
import traceback

import numpy as np
import mpi4py
from mpi4py import MPI
import dolfinx
from dolfinx import fem, la, mesh

COMM = MPI.COMM_WORLD
ROOT = Path(__file__).resolve().parents[1] / 'datasets' / 'dolfinx-ghost'


def export(dimension, root):
    subdivisions = {1: [10000], 2: [70, 70], 3: [12, 12, 12]}[dimension]
    if dimension == 1:
        domain = mesh.create_unit_interval(COMM, *subdivisions)
    elif dimension == 2:
        domain = mesh.create_unit_square(COMM, *subdivisions, cell_type=mesh.CellType.triangle)
    else:
        domain = mesh.create_unit_cube(COMM, *subdivisions, cell_type=mesh.CellType.tetrahedron)
    space = fem.functionspace(domain, ('Lagrange', 1))
    imap = space.dofmap.index_map
    assert space.dofmap.index_map_bs == 1
    cells = domain.topology.index_map(dimension)
    global_cells = COMM.allreduce(cells.size_local)
    assert global_cells == {1: 10000, 2: 9800, 3: 10368}[dimension]
    start, stop = imap.local_range
    n = imap.size_local
    ghosts = np.asarray(imap.ghosts, dtype=np.int64)
    owners = np.asarray(imap.owners, dtype=np.int32)
    ranges = COMM.allgather((start, stop))
    assert stop - start == n
    ordered = sorted(ranges)
    assert ordered[0][0] == 0 and ordered[-1][1] == imap.size_global
    assert all(a[1] == b[0] for a, b in zip(ordered, ordered[1:]))
    assert len(ghosts) == len(owners) == imap.num_ghosts
    assert len(np.unique(ghosts)) == len(ghosts)
    for gid, owner in zip(ghosts, owners):
        assert 0 <= owner < COMM.size and owner != COMM.rank
        assert ranges[owner][0] <= gid < ranges[owner][1]
    total_ghosts = COMM.allreduce(len(ghosts))
    assert total_ghosts > 0 if COMM.size > 1 else total_ghosts == 0
    gids = np.concatenate((np.arange(start, stop, dtype=np.int64), ghosts))
    values = {'owned_range': np.array([start, stop], dtype=np.int64),
              'ghost_global_indices': ghosts, 'ghost_owner_ranks': owners}
    reference = fem.Function(space, dtype=np.float64)
    for operation in ('forward', 'reverse'):
        inputs, expected = [], []
        for case in range(2):
            x = (1 + gids + case * 100000).astype(np.float64)
            x[n:] = -(case + 1) if operation == 'forward' else (case + 1) * (COMM.rank + 1)
            reference.x.array[:] = x
            if operation == 'forward':
                reference.x.scatter_forward()
                np.testing.assert_array_equal(reference.x.array, 1 + gids + case * 100000)
                np.testing.assert_array_equal(reference.x.array[:n], x[:n])
            else:
                reference.x.scatter_reverse(la.InsertMode.add)
                np.testing.assert_array_equal(reference.x.array[n:], x[n:])
                # Independent ownership-based sum checks the saved CPU oracle.
                sums = np.zeros(imap.size_global, dtype=np.float64)
                for ids, contributions in COMM.allgather((ghosts, x[n:])):
                    np.add.at(sums, ids, contributions)
                np.testing.assert_array_equal(reference.x.array[:n], x[:n] + sums[start:stop])
            if COMM.size == 1:
                np.testing.assert_array_equal(reference.x.array, x)
            inputs.append(x.copy())
            expected.append(reference.x.array.copy())
        values[f'{operation}_input'] = np.stack(inputs)
        values[f'{operation}_expected'] = np.stack(expected)
    folder = root / f'{dimension}d' / f'{COMM.size}ranks'
    folder.mkdir(parents=True, exist_ok=True)
    filename = folder / f'rank-{COMM.rank:04d}.npz'
    np.savez_compressed(filename, **values)
    with np.load(filename, allow_pickle=False) as saved:
        for key, value in values.items():
            np.testing.assert_array_equal(saved[key], value)
            assert saved[key].dtype == value.dtype
    ranks = COMM.gather({'rank': COMM.rank, 'owned': n, 'ghosts': len(ghosts),
                         'owned_range': [start, stop], 'owned_cells': cells.size_local}, root=0)
    if COMM.rank == 0:
        metadata = dict(format_version=1, dimension=dimension, domain='unit interval/square/cube',
                        cell_type=domain.topology.cell_type.name, subdivisions=subdivisions,
                        global_cells=global_cells, global_dofs=imap.size_global,
                        rank_count=COMM.size, block_size=1, cases=2, function_space='scalar continuous Lagrange P1',
                        value_dtype='float64', global_index_dtype='int64', owner_rank_dtype='int32',
                        layout='[owned | ghosts]', ranks=ranks,
                        versions=dict(dolfinx=dolfinx.__version__, numpy=np.__version__,
                                      mpi4py=mpi4py.__version__, python=platform.python_version()),
                        mpi_library=MPI.Get_library_version().strip(),
                        validation='ownership, DOLFINx references, preserved entries, NPZ roundtrip passed')
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print(f'EXPORT PASS {dimension}d/{COMM.size}ranks cells={global_cells} dofs={imap.size_global} ghosts={total_ghosts}', flush=True)
    COMM.Barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    assert COMM.size in (1, 2, 3, 4)
    for dimension in (1, 2, 3):
        export(dimension, args.output)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        COMM.Abort(1)
