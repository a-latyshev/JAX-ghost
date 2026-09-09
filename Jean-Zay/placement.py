"""One rank per NVIDIA GPU, checked using MPI shared-memory groups and CUDA UUIDs."""
import ctypes
import os
import platform
from common import collective
from mpi4py import MPI


def select_visible_gpu(local_rank):
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        # No scheduler mask: each local rank chooses its ordinal before CUDA starts.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    else:
        ids = [s.strip() for s in visible.split(',') if s.strip()]
        if not ids or ids == ['-1']:
            raise ValueError('No visible GPU; request GPUs from the scheduler')
        if len(ids) > 1:
            if local_rank >= len(ids):
                raise ValueError('More local MPI ranks than visible GPUs')
            os.environ['CUDA_VISIBLE_DEVICES'] = ids[local_rank]


def cuda_identity():
    """Query logical CUDA device 0 after masking; no nvidia-smi dependency."""
    cuda = ctypes.CDLL('libcuda.so.1')
    def call(name, *args):
        code = getattr(cuda, name)(*args)
        if code:
            raise ValueError(f'{name} failed with CUDA status {code}')
    call('cuInit', 0)
    count = ctypes.c_int()
    call('cuDeviceGetCount', ctypes.byref(count))
    if count.value != 1:
        raise ValueError(f'Expected exactly one visible CUDA GPU, found {count.value}; '
                         f'CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")!r}. '
                         'CUDA may have initialized before GPU selection.')
    device = ctypes.c_int()
    call('cuDeviceGet', ctypes.byref(device), 0)
    uuid = (ctypes.c_ubyte*16)()
    call('cuDeviceGetUuid', ctypes.byref(uuid), device)
    raw = bytes(uuid).hex()
    name = ctypes.create_string_buffer(256)
    bus = ctypes.create_string_buffer(32)
    call('cuDeviceGetName', name, len(name), device)
    call('cuDeviceGetPCIBusId', bus, len(bus), device)
    return dict(gpu_uuid=f'GPU-{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}',
                gpu_name=name.value.decode(), gpu_pci_bus_id=bus.value.decode())


def validate_records(records):
    for i, a in enumerate(records):
        if not a['cpu_affinity'] or not a['gpu_uuid']:
            raise ValueError('Missing CPU affinity or GPU identity')
        for b in records[:i]:
            if a['node_group'] != b['node_group']:
                continue
            if a['gpu_uuid'] == b['gpu_uuid'] or a['gpu_pci_bus_id'] == b['gpu_pci_bus_id']:
                raise ValueError('Two ranks share a physical GPU (MIG sharing is not supported)')
            if set(a['cpu_affinity']).intersection(b['cpu_affinity']):
                raise ValueError('Overlapping CPU affinity: bind each rank to disjoint CPUs')


def prepare(comm, expected_local_rank=None):
    local = comm.Split_type(MPI.COMM_TYPE_SHARED, key=comm.rank)
    local_rank = local.rank
    node_group = local.bcast(comm.rank if local_rank == 0 else None, root=0)
    local.Free()
    def select():
        if expected_local_rank is not None and expected_local_rank != local_rank:
            raise ValueError("Launcher local rank disagrees with MPI shared-memory rank")
        select_visible_gpu(local_rank)
    collective(comm, select)
    def inspect():
        record = dict(rank=comm.rank, node_local_rank=local_rank, node_group=node_group,
                      hostname=platform.node(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                      cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'])
        record.update(cuda_identity())
        return record
    record = collective(comm, inspect)
    records = comm.allgather(record)
    validate_records(records)
    return records
