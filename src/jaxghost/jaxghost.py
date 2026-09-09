"""Static host-side setup and device-side forward/reverse scatter.

Only construction communicates NumPy metadata through mpi4py. Numerical
values passed to either scatter stay in JAX arrays throughout the operation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
try:
    import mpi4jax
except ModuleNotFoundError as exc:
    if exc.name != "mpi4jax":
        raise
    raise ImportError(
        "The mpi4jax backend requires mpi4jax. Install jaxghost[mpi4jax] "
        "with an MPI-compatible build, or use ShardedJAXGhost/ShardedJAXMatrixCSR."
    ) from exc


from ._metadata import build_plan, _collective_check


class JAXGhost:
    """A fixed owner/ghost communication plan; construct with from_index_map.

    All ranks in ``comm`` must construct and execute their scatterers in the
    same sequence, using the same vector dtype. Each rank may have a different
    local vector length. Use ``jax.jit(ghost.scatter_forward)`` to compile the
    complete update. One local JAX device and fixed-size blocks are supported.
    n_owned and n_ghost count scalar entries, including all block components.

    The plan and its runtime communicator must outlive every compiled call.
    Call ``close`` collectively after the last use, then discard compiled
    functions capturing this object. No communication is performed in __del__.
    """

    @classmethod
    def from_index_map(cls, index_map, comm, *, block_size=1):
        """Collectively derive the plan from a DOLFINx-compatible IndexMap.

        ``comm`` must have the same rank numbering as the index map. Pass
        ``V.dofmap.index_map_bs`` as block_size (a positive integer, identical
        on all ranks).
        Metadata requests use one-time Alltoall/Alltoallv; runtime exchanges
        involve neighbors only. The index map itself is not retained.
        """
        _collective_check(comm, None if jax.local_device_count() == 1 else
                          "exactly one local JAX device per MPI rank is required")
        plan = build_plan(index_map, comm, block_size)
        self = cls()
        self.__dict__.update(vars(plan))
        self.send_indices = jnp.asarray(plan.send_indices)
        self.receive_positions = jnp.asarray(plan.receive_positions)
        self._comm = comm.Dup()
        self._closed = False
        return self

    def _validate_vector(self, x, operation):
        if self._closed:
            raise RuntimeError("scatterer is closed")
        if not isinstance(x, (jax.Array, jax.core.Tracer)):
            raise TypeError(f"{operation} expects a JAX array already on the device")
        if x.ndim != 1 or x.shape[0] != self.n_owned + self.n_ghost:
            raise ValueError(f"expected shape ({self.n_owned + self.n_ghost},), got {x.shape}")
        if x.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
            raise TypeError(f"{operation} supports float32 and float64")

    def scatter_forward(self, x):
        """Return ``x`` with ghosts overwritten and owned values preserved.

        ``x`` is a one-dimensional JAX float32/float64 array with shape
        ``(n_owned + n_ghost,)``, already placed on the local JAX device.
        No numerical values are converted to host arrays. All ranks must use
        the same dtype. This method
        is JIT compatible; construction and close are not. Differentiation
        through the communication is outside the supported interface.
        """
        self._validate_vector(x, "scatter_forward")

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

    def scatter_reverse(self, x):
        """Add remote ghost contributions to owners; return the full local array.

        Matches DOLFINx scatter_reverse(InsertMode.add), with immutable JAX
        semantics: the input is preserved, and the returned array contains
        updated owned values and unchanged ghosts. Repeated calls add the ghost
        contributions again. Call scatter_forward separately to refresh ghosts.

        Accepts the same device-array shape and dtype as scatter_forward and
        supports jax.jit. INSERT and automatic differentiation are not supported.
        """
        self._validate_vector(x, "scatter_reverse")
        if not self.peers:
            return x

        # Reverse DOLFINx's packing direction: forward receive positions now
        # identify outgoing ghost contributions, grouped by their owners.
        packed = x[self.receive_positions]
        received_parts = []
        for i, peer in enumerate(self.peers):
            send = packed[self.recv_offsets[i] : self.recv_offsets[i + 1]]
            template = jnp.empty((self.send_counts[i],), dtype=x.dtype)
            received = mpi4jax.sendrecv(
                send, template, source=peer, dest=peer,
                sendtag=1, recvtag=1, comm=self._comm,
            )
            if self.send_counts[i]:
                received_parts.append(received)

        # Ordered effects retain sends even if this rank receives no additions.
        if not received_parts:
            return x
        received = jnp.concatenate(received_parts)
        # Several peers may contribute to the same owned index. Indexed ADD
        # accumulates every occurrence, rather than overwriting duplicate indices.
        return x.at[self.send_indices].add(received)

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
