"""Shared benchmark serialization and validation; no DOLFINx/JAX imports."""
import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mpi4py
from mpi4py import MPI


def environment(comm):
    return dict(python=platform.python_version(), numpy=np.__version__, mpi4py=mpi4py.__version__,
                mpi_library=MPI.Get_library_version().strip(),
                ranks=comm.allgather(dict(rank=comm.rank, hostname=platform.node(),
                    environment={k: v for k, v in os.environ.items() if k.startswith(
                        ('OMP_', 'OPENBLAS_', 'MKL_', 'OMPI_MCA_', 'UCX_', 'MPI4JAX_',
                         'JAX_', 'CUDA_VISIBLE_', 'SLURM_CPUS_', 'SLURM_JOB_ID'))})))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def timings(samples, iterations):
    per_call = np.max(samples, axis=1) / iterations
    return dict(rank_seconds=samples, seconds_per_matvec=per_call.tolist(),
                median_seconds_per_matvec=float(np.median(per_call)))


def check_result(comm, actual, expected, n_owned):
    if actual.shape != expected.shape:
        raise ValueError('result shape mismatch')
    valid = bool(np.all(np.isfinite(actual)) and
                 np.array_equal(actual[n_owned:], expected[n_owned:]))
    error2 = comm.allreduce(float(np.sum((actual[:n_owned] - expected[:n_owned])**2)))
    ref2 = comm.allreduce(float(np.sum(expected[:n_owned]**2)))
    error, reference = np.sqrt(error2), np.sqrt(ref2)
    valid = comm.allreduce(valid, op=MPI.LAND)
    if not valid or not error <= 1e-12 + 1e-10 * reference:
        raise ValueError(f'reference mismatch: error={error}, reference={reference}')
    return dict(error_l2=float(error), reference_l2=float(reference))


class StoredMatrix:
    """Host metadata adapter for the existing from_dolfinx constructor."""
    block_size = (1, 1)

    def __init__(self, data, meta, comm):
        self.maps = []
        for axis in ('row', 'col'):
            bounds = data[f'{axis}_owned_range']
            ghosts, owners = data[f'{axis}_ghosts'], data[f'{axis}_owners']
            if bounds.shape != (2,) or bounds.dtype.kind not in 'iu':
                raise ValueError('invalid owned range')
            start, stop = map(int, bounds)
            if not 0 <= start <= stop <= meta['global_dofs']:
                raise ValueError('owned range outside global layout')
            if (ghosts.ndim != 1 or owners.shape != ghosts.shape or
                    ghosts.dtype.kind not in 'iu' or owners.dtype.kind not in 'iu' or
                    len(np.unique(ghosts)) != len(ghosts) or
                    np.any((owners < 0) | (owners >= comm.size) | (owners == comm.rank))):
                raise ValueError('invalid ghost layout')
            self.maps.append(SimpleNamespace(size_local=stop-start, local_range=(start, stop),
                                             ghosts=ghosts, owners=owners))
        self.indptr, self.indices = data['indptr'], data['indices']
        nr, nc = self.maps
        if (self.indptr.shape != (nr.size_local + 1,) or self.indptr.dtype.kind not in 'iu'
                or self.indices.ndim != 1 or self.indices.dtype.kind not in 'iu'
                or self.indptr[0] != 0 or np.any(self.indptr[1:] < self.indptr[:-1])
                or self.indptr[-1] != len(self.indices)
                or np.any((self.indices < 0) | (self.indices >= nc.size_local + len(nc.ghosts)))):
            raise ValueError('invalid owned-row CSR')
        for key, size in [('values', len(self.indices)), ('x', nc.size_local + len(nc.ghosts)),
                          *[(k, nr.size_local + len(nr.ghosts)) for k in
                            ('y_initial', 'single_expected', 'batch_expected')]]:
            if data[key].shape != (size,) or data[key].dtype != np.float64 or not np.all(np.isfinite(data[key])):
                raise ValueError(f'invalid array: {key}')
        if np.any(data['y_initial'] != 0):
            raise ValueError('initial output must be zero')

    def validate_ownership(self, comm, global_dofs):
        for imap in self.maps:
            ranges = comm.allgather(imap.local_range)
            ordered = sorted(ranges)
            if (ordered[0][0] != 0 or ordered[-1][1] != global_dofs or
                    any(a[1] != b[0] for a, b in zip(ordered, ordered[1:]))):
                raise ValueError('owned ranges do not partition global DOFs')
            if any(not ranges[int(owner)][0] <= gid < ranges[int(owner)][1]
                   for gid, owner in zip(imap.ghosts, imap.owners)):
                raise ValueError('ghost owner mismatch')

    def index_map(self, axis):
        return self.maps[axis]
