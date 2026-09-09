"""Shared benchmark serialization and validation; no DOLFINx/JAX imports."""
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mpi4py
from mpi4py import MPI


def environment(comm):
    return dict(python=platform.python_version(), executable=sys.executable,
                spack_env=os.environ.get("SPACK_ENV"), venv=os.environ.get("VIRTUAL_ENV"),
                numpy=np.__version__, mpi4py=mpi4py.__version__,
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


def fingerprints(folder, comm):
    import hashlib
    path = Path(folder) / f'rank-{comm.rank:04d}.npz'
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return comm.allgather(digest.hexdigest())


def placement(comm, backend, device=None):
    """Validate actual physical CPU binding and physical GPU identity."""
    import subprocess
    affinity = sorted(os.sched_getaffinity(0))
    physical = []
    for cpu in affinity:
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        physical.append((int((root/'physical_package_id').read_text()),
                         int((root/'core_id').read_text())))
    record = dict(rank=comm.rank, hostname=platform.node(), cpu_affinity=affinity,
                  physical_cores=sorted(set(physical)), gpu_uuid=None)
    error = None
    expected = 1 if backend == 'dolfinx' else int(os.environ.get('BENCH_JAX_CPUS_PER_RANK', '1'))
    if len(affinity) != expected or len(set(physical)) != expected:
        error = f'Each rank must be bound to {expected} distinct physical cores (one hardware thread each)'
    if backend != 'dolfinx':
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible or ',' in visible or device is None or device.platform != 'gpu':
            error = 'Each JAX rank must see exactly one CUDA GPU'
        else:
            info = subprocess.check_output(['nvidia-smi', '-i', visible,
                '--query-gpu=uuid,name,pci.bus_id', '--format=csv,noheader'], text=True).strip().split(', ')
            record.update(gpu_uuid=info[0], gpu_name=info[1], gpu_pci_bus_id=info[2],
                          jax_device=str(device), cuda_visible_devices=visible)
    errors = comm.allgather(error)
    records = comm.allgather(record)
    cores = [(r['hostname'], tuple(c)) for r in records for c in r['physical_cores']]
    gpus = [r['gpu_uuid'] for r in records if r['gpu_uuid'] is not None]
    if any(errors) or len(cores) != len(set(cores)) or len(gpus) != len(set(gpus)):
        raise ValueError(f'Invalid or overlapping placement: {errors}; {records}')
    return records


def phase(comm, operation, finish=lambda value: None):
    comm.Barrier()
    start = MPI.Wtime()
    result = operation()
    finish(result)
    elapsed = MPI.Wtime() - start
    seconds = comm.allgather(elapsed)
    return result, dict(timer='MPI.Wtime', rank_seconds=seconds, max_seconds=max(seconds))
