"""Export a distributed Poisson CSR operator and time DOLFINx y += Ax."""
import argparse
from pathlib import Path
import traceback

import numpy as np
from mpi4py import MPI
from common import StoredMatrix, check_result, environment, timings, write_json, placement, fingerprints

COMM = MPI.COMM_WORLD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--subdivisions', type=int, default=46,
                        help='mesh intervals per coordinate direction; P1 DOFs = (n+1)^3')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'data')
    parser.add_argument('--warmup', type=int, default=10,
                        help='untimed matvec calls before sampling')
    parser.add_argument('--iterations', type=int, default=100,
                        help='matvec calls in each timed batch')
    parser.add_argument('--repeats', type=int, default=10,
                        help='independent timed batches; report their median')
    args = parser.parse_args()
    if COMM.size not in (1, 2, 3, 4) or min(args.subdivisions, args.iterations, args.repeats) < 1 or args.warmup < 0:
        raise ValueError('use 1–4 ranks, positive sizes/counts, and nonnegative warmup')
    hardware = placement(COMM, 'dolfinx')
    import dolfinx
    from dolfinx import fem, la, mesh
    import ufl

    domain = mesh.create_unit_cube(COMM, *([args.subdivisions] * 3), cell_type=mesh.CellType.tetrahedron)
    V = fem.functionspace(domain, ('Lagrange', 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    A = fem.assemble_matrix(fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx, dtype=np.float64))
    A.scatter_reverse()
    # Match owned coordinates by global IDs, rather than assuming ghost layouts match.
    f = fem.Function(V, dtype=np.float64)
    f.interpolate(lambda p: np.sin(2 * np.pi * p[0]) + p[1]**2 + 0.5 * p[2])
    source_map = V.dofmap.index_map
    source_ids = np.arange(*source_map.local_range)
    col_map, row_map = A.index_map(1), A.index_map(0)
    if tuple(col_map.local_range) != tuple(source_map.local_range):
        raise ValueError('unexpected column owned layout')
    x, y = la.vector(col_map, dtype=np.float64), la.vector(row_map, dtype=np.float64)
    x.array[:] = 0
    x.array[:len(source_ids)] = f.x.array[:len(source_ids)]
    y.array[:] = 0
    data = dict(x=x.array.copy(), y_initial=y.array.copy())
    nr = row_map.size_local
    nnz = int(A.indptr[nr])
    data.update(indptr=A.indptr[:nr+1].copy(), indices=A.indices[:nnz].copy(), values=A.data[:nnz].copy())
    for axis, imap in [('row', row_map), ('col', col_map)]:
        data.update({f'{axis}_owned_range': np.asarray(imap.local_range, dtype=np.int64),
                     f'{axis}_ghosts': np.asarray(imap.ghosts, dtype=np.int64),
                     f'{axis}_owners': np.asarray(imap.owners, dtype=np.int32)})
    A.mult(x, y)
    data['single_expected'] = y.array.copy()
    y.array[:] = 0
    for _ in range(args.iterations):
        A.mult(x, y)
    data['batch_expected'] = y.array.copy()
    check_result(COMM, data['batch_expected'], args.iterations * data['single_expected'], nr)
    for _ in range(args.warmup):
        A.mult(x, y)
    errors = []
    samples = []
    for _ in range(args.repeats):
        y.array[:] = 0
        COMM.Barrier()
        start = MPI.Wtime()
        for _ in range(args.iterations):
            A.mult(x, y)
        elapsed = MPI.Wtime() - start
        samples.append(COMM.allgather(elapsed))
        errors.append(check_result(COMM, y.array, data['batch_expected'], nr))
    meta = dict(format_version=1, operator='Poisson stiffness, no boundary conditions',
                dtype='float64', rank_count=COMM.size, subdivisions=args.subdivisions,
                global_dofs=int(row_map.size_global), global_cells=6 * args.subdivisions**3,
                settings=dict(warmup=args.warmup, iterations=args.iterations, repeats=args.repeats),
                dolfinx_version=dolfinx.__version__, environment=environment(COMM),
                timing=dict(timer='MPI.Wtime', **timings(samples, args.iterations)))
    if meta['global_dofs'] != (args.subdivisions + 1)**3:
        raise ValueError('unexpected global DOF count')
    adapter = StoredMatrix(data, meta, COMM)
    adapter.validate_ownership(COMM, meta['global_dofs'])
    meta['partitions'] = COMM.allgather(dict(rank=COMM.rank, owned_rows=nr,
        owned_cols=col_map.size_local, ghost_rows=row_map.num_ghosts,
        ghost_cols=col_map.num_ghosts, nnz=nnz))
    folder = args.output / f'n{args.subdivisions}' / f'{COMM.size}ranks'
    folder.mkdir(parents=True, exist_ok=True)
    # Remove completion marker before replacing files in an existing dataset.
    if COMM.rank == 0:
        (folder / 'metadata.json').unlink(missing_ok=True)
    COMM.Barrier()
    path = folder / f'rank-{COMM.rank:04d}.npz'
    np.savez_compressed(path, **data)
    with np.load(path, allow_pickle=False) as saved:
        for key in data:
            np.testing.assert_array_equal(saved[key], data[key])
    COMM.Barrier()
    meta['fingerprints'] = fingerprints(folder, COMM)
    if COMM.rank == 0:
        write_json(folder / 'metadata.json', meta)
        write_json(args.result, dict(backend='dolfinx', trial=args.trial, subdivisions=args.subdivisions,
            global_dofs=meta['global_dofs'], ranks=COMM.size, settings=meta['settings'],
            dataset=str(folder.resolve()), fingerprints=meta['fingerprints'], placement=hardware,
            timing=meta['timing'], validation=errors, correctness_passed=True, phases={},
            versions=dict(dolfinx=dolfinx.__version__), environment=meta['environment']))
        print(f'EXPORT PASS {folder}: {meta["timing"]["median_seconds_per_matvec"]:.6g} s/matvec', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        COMM.Abort(1)
