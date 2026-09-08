"""One-time MPI ownership metadata shared by communication backends."""

import operator
from types import SimpleNamespace
import numpy as np
from mpi4py import MPI


def _collective_check(comm, error):
    """Raise on every rank before any rank enters the next setup collective."""
    errors = comm.allgather(error)
    failures = [f"rank {rank}: {msg}" for rank, msg in enumerate(errors) if msg]
    if failures:
        raise ValueError("Invalid ghost layout: " + "; ".join(failures))


def build_plan(index_map, comm, block_size=1):
    error = None
    try:
        if isinstance(block_size, (bool, np.bool_)):
            raise ValueError("block_size must be a positive integer")
        try:
            block_size = operator.index(block_size)
        except TypeError:
            raise ValueError("block_size must be a positive integer") from None
        if block_size < 1:
            raise ValueError("block_size must be a positive integer")
        n_owned = int(index_map.size_local)
        start, stop = map(int, index_map.local_range)
        ghosts = np.asarray(index_map.ghosts, dtype=np.int64)
        owners = np.asarray(index_map.owners, dtype=np.int32)
        if n_owned < 0 or start < 0 or stop - start != n_owned:
            raise ValueError("owned size and global range disagree")
        if ghosts.ndim != 1 or owners.shape != ghosts.shape:
            raise ValueError("ghosts and owners must be matching one-dimensional arrays")
        if np.any(ghosts < 0):
            raise ValueError("ghost global IDs must be nonnegative")
        if np.any((owners < 0) | (owners >= comm.size) | (owners == comm.rank)):
            raise ValueError("ghost owners must be valid remote ranks")
    except (AttributeError, TypeError, ValueError) as exc:
        error = str(exc)
    _collective_check(comm, error)

    block_sizes = comm.allgather(block_size)
    if len(set(block_sizes)) != 1:
        raise ValueError("block_size must be identical on all ranks")

    # A stable grouping preserves each requester's original order within
    # an owner. Its permutation also tells us where to unpack the answer.
    permutation = np.argsort(owners, kind="stable")
    request_ids = np.ascontiguousarray(ghosts[permutation])
    request_counts = np.bincount(owners, minlength=comm.size).astype(np.int64)
    incoming_counts = np.empty(comm.size, dtype=np.int64)
    comm.Alltoall([request_counts, MPI.INT64_T], [incoming_counts, MPI.INT64_T])
    request_offsets = np.concatenate(([0], np.cumsum(request_counts)))
    incoming_offsets = np.concatenate(([0], np.cumsum(incoming_counts)))
    incoming_ids = np.empty(int(incoming_offsets[-1]), dtype=np.int64)
    comm.Alltoallv(
        [request_ids, request_counts, request_offsets[:-1], MPI.INT64_T],
        [incoming_ids, incoming_counts, incoming_offsets[:-1], MPI.INT64_T],
    )
    error = None
    if np.any((incoming_ids < start) | (incoming_ids >= stop)):
        error = "a requested global ID is outside this owner's range"
    if max(n_owned + ghosts.size, incoming_ids.size) * block_size > np.iinfo(np.int32).max:
        error = "local indexing exceeds the supported int32 range"
    _collective_check(comm, error)

    self = SimpleNamespace()
    self.block_size = block_size
    self.n_owned = n_owned * block_size
    self.n_ghost = int(ghosts.size) * block_size
    self.peers = tuple(
        int(p) for p in np.flatnonzero(request_counts + incoming_counts)
    )
    # Reverse the request direction: received requests become value sends.
    self.send_counts = tuple(int(incoming_counts[p]) * block_size for p in self.peers)
    self.recv_counts = tuple(int(request_counts[p]) * block_size for p in self.peers)
    self.send_offsets = tuple(int(v) for v in np.cumsum((0,) + self.send_counts))
    self.recv_offsets = tuple(int(v) for v in np.cumsum((0,) + self.recv_counts))
    # DOLFINx Scatterer expands each block i into bs*i + component.
    # Setup requests remain block IDs; runtime buffers contain scalar entries.
    components = np.arange(block_size, dtype=np.int64)
    self.send_indices = np.asarray(
        ((incoming_ids - start)[:, None] * block_size + components).ravel(),
        dtype=np.int32,
    )
    self.receive_positions = np.asarray(
        ((n_owned + permutation)[:, None] * block_size + components).ravel(),
        dtype=np.int32,
    )
    return self
