"""Static host-side setup and device-side forward scatter.

Only construction communicates NumPy metadata through mpi4py. Numerical
values passed to scatter_forward stay in JAX arrays throughout the operation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import mpi4jax
from mpi4py import MPI
import numpy as np


def _collective_check(comm, error):
    """Raise on every rank before any rank enters the next setup collective."""
    errors = comm.allgather(error)
    failures = [f"rank {rank}: {msg}" for rank, msg in enumerate(errors) if msg]
    if failures:
        raise ValueError("Invalid ghost layout: " + "; ".join(failures))


class JAXGhost:
    """A fixed owner/ghost communication plan; construct with from_index_map.

    All ranks in ``comm`` must construct and execute their scatterers in the
    same sequence, using the same vector dtype. Each rank may have a different
    local vector length. Use ``jax.jit(ghost.scatter_forward)`` to compile the
    complete update. One local JAX device and scalar entries are supported.

    The plan and its runtime communicator must outlive every compiled call.
    Call ``close`` collectively after the last use, then discard compiled
    functions capturing this object. No communication is performed in __del__.
    """

    @classmethod
    def from_index_map(cls, index_map, comm, *, block_size=1):
        """Collectively derive the plan from a DOLFINx-compatible IndexMap.

        ``comm`` must have the same rank numbering as the index map. Pass
        ``V.dofmap.index_map_bs`` as block_size to check that the space is scalar.
        Metadata requests use one-time Alltoall/Alltoallv; runtime exchanges
        involve neighbors only. The index map itself is not retained.
        """
        error = None
        try:
            if block_size != 1:
                raise ValueError("only block_size=1 is supported")
            if jax.local_device_count() != 1:
                raise ValueError("exactly one local JAX device per MPI rank is required")
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
        if max(n_owned + ghosts.size, incoming_ids.size) > np.iinfo(np.int32).max:
            error = "local indexing exceeds the supported int32 range"
        _collective_check(comm, error)

        self = cls()
        self.n_owned = n_owned
        self.n_ghost = int(ghosts.size)
        self.peers = tuple(
            int(p) for p in np.flatnonzero(request_counts + incoming_counts)
        )
        # Reverse the request direction: received requests become value sends.
        self.send_counts = tuple(int(incoming_counts[p]) for p in self.peers)
        self.recv_counts = tuple(int(request_counts[p]) for p in self.peers)
        self.send_offsets = tuple(int(v) for v in np.cumsum((0,) + self.send_counts))
        self.recv_offsets = tuple(int(v) for v in np.cumsum((0,) + self.recv_counts))
        self.send_indices = jnp.asarray(incoming_ids - start, dtype=jnp.int32)
        self.receive_positions = jnp.asarray(n_owned + permutation, dtype=jnp.int32)
        self._comm = comm.Dup()
        self._closed = False
        return self

    def scatter_forward(self, x):
        """Return ``x`` with ghosts overwritten and owned values preserved.

        ``x`` is a one-dimensional JAX float32/float64 array with shape
        ``(n_owned + n_ghost,)``, already placed on the local JAX device.
        No numerical values are converted to host arrays. All ranks must use
        the same dtype. This method
        is JIT compatible; construction and close are not. Differentiation
        through the communication is outside the supported interface.
        """
        if self._closed:
            raise RuntimeError("scatterer is closed")
        if not isinstance(x, (jax.Array, jax.core.Tracer)):
            raise TypeError("scatter_forward expects a JAX array already on the device")
        if x.ndim != 1 or x.shape[0] != self.n_owned + self.n_ghost:
            raise ValueError(f"expected shape ({self.n_owned + self.n_ghost},), got {x.shape}")
        if x.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
            raise TypeError("scatter_forward supports float32 and float64")

        if not self.peers:
            return x

        packed = x[self.send_indices]
        received_parts = []
        for i, peer in enumerate(self.peers):
            send = packed[self.send_offsets[i] : self.send_offsets[i + 1]]
            template = jnp.empty((self.recv_counts[i],), dtype=x.dtype)
            # mpi4jax 0.9.1.post1 returns an array and uses ordered effects
            # internally. Even a send-only exchange must not be eliminated.
            received = mpi4jax.sendrecv(
                send, template, source=peer, dest=peer,
                sendtag=0, recvtag=0, comm=self._comm,
            )
            if self.recv_counts[i]:
                received_parts.append(received)

        # A send-only rank must still execute the exchanges above. Its array
        # needs no update; ordered MPI effects retain the otherwise-unused sends.
        if not received_parts:
            return x
        received = jnp.concatenate(received_parts)
        return x.at[self.receive_positions].set(received)

    def close(self):
        """Collectively release the communicator after all calls finish.

        Compiled functions capturing this plan must never be called afterward.
        effects_barrier also waits for exchanges on ranks with no output ghosts.
        """
        if not self._closed:
            jax.effects_barrier()
            self._comm.Free()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
