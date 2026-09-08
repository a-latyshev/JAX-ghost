"""Forward halo exchange with sharded index tables and native JAX all-to-all."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ._metadata import build_plan, _collective_check


def _pack_tables(plan, nranks, width, length):
    """Host setup: one fixed-width packet per peer; length is a drop sentinel."""
    send = np.zeros((nranks, width), dtype=np.int32)
    positions = np.full((nranks, width), length, dtype=np.int32)
    valid = np.zeros((nranks, width), dtype=bool)
    for i, peer in enumerate(plan.peers):
        ns, nr = plan.send_counts[i], plan.recv_counts[i]
        send[peer, :ns] = plan.send_indices[plan.send_offsets[i]:plan.send_offsets[i + 1]]
        valid[peer, :ns] = True
        positions[peer, :nr] = plan.receive_positions[plan.recv_offsets[i]:plan.recv_offsets[i + 1]]
    return send, positions, valid


def _forward_shards(x, send, positions, valid):
    """Identical body on every device; all rank-specific indices are operands."""
    local = x[0]
    packed = jnp.where(valid[0], local[send[0]], jnp.zeros((), dtype=x.dtype))
    received = jax.lax.all_to_all(packed, 'rank', split_axis=0, concat_axis=0, tiled=True)
    updated = local.at[positions[0].ravel()].set(received.ravel(), mode='drop')
    return updated[None, :]


def _check_array(x, shape):
    if not isinstance(x, (jax.Array, jax.core.Tracer)):
        raise TypeError('expected a JAX device array')
    if x.shape != shape:
        raise ValueError(f'expected shape {shape}, got {x.shape}')
    if x.dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
        raise TypeError('forward supports float32 and float64')


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class ShardedJAXGhost:
    """Immutable forward-only plan; runtime communication is owned by JAX.

    Pass the plan explicitly: jax.jit(ShardedJAXGhost.scatter_forward)(plan, x).
    Index tables are dynamic pytree leaves; all static configuration is global
    and identical across ranks. There is no communicator to close. Reverse
    updates, differentiation and GPU performance are not validated interfaces.
    """

    mesh: Mesh
    counts: tuple  # Per-rank (owned, ghost) scalar counts, identical on all ranks.
    block_size: int
    padded_length: int
    max_message_length: int
    send_indices: jax.Array
    receive_positions: jax.Array
    send_valid: jax.Array

    def tree_flatten(self):
        return ((self.send_indices, self.receive_positions, self.send_valid),
                (self.mesh, self.counts, self.block_size, self.padded_length,
                 self.max_message_length))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*aux, *children)

    @classmethod
    def from_index_map(cls, index_map, comm, mesh, *, block_size=1):
        """Collective metadata setup after caller initializes distributed JAX."""
        error = None
        try:
            if not jax.distributed.is_initialized():
                raise ValueError('initialize distributed JAX before creating a sharded plan')
            if not isinstance(mesh, Mesh) or mesh.axis_names != ('rank',) or mesh.devices.ndim != 1:
                raise ValueError("mesh must be one-dimensional with axis 'rank'")
            if comm.size != jax.process_count() or comm.rank != jax.process_index():
                raise ValueError('communicator must span all JAX processes in process order')
            if jax.local_device_count() != 1 or len(jax.devices()) != comm.size:
                raise ValueError('exactly one JAX device per MPI process is required')
            if mesh.size != comm.size or any(d.process_index != i for i, d in enumerate(mesh.devices)):
                raise ValueError('mesh device positions must match MPI rank order')
            if tuple(mesh.devices) != tuple(jax.devices()):
                raise ValueError('mesh must contain all participating JAX devices')
        except (AttributeError, TypeError, ValueError) as exc:
            error = str(exc)
        _collective_check(comm, error)
        plan = build_plan(index_map, comm, block_size)
        counts = tuple(comm.allgather((plan.n_owned, plan.n_ghost)))
        length = max(1, max(n + g for n, g in counts))
        width = max(comm.allgather(max((1,) + plan.send_counts + plan.recv_counts)))
        tables = _pack_tables(plan, comm.size, width, length)
        device = mesh.devices[comm.rank]
        leaves = []
        sharding = NamedSharding(mesh, P('rank', None, None))
        for table in tables:
            local = jax.device_put(table[None], device)
            leaves.append(jax.make_array_from_single_device_arrays(
                (comm.size, comm.size, width), sharding, [local]))
        return cls(mesh, counts, plan.block_size, length, width, *leaves)

    @property
    def n_owned(self):
        return self.counts[jax.process_index()][0]

    @property
    def n_ghost(self):
        return self.counts[jax.process_index()][1]

    @property
    def sharding(self):
        return NamedSharding(self.mesh, P('rank', None))

    def to_sharded(self, x_local):
        """Outside JIT: pad a local device array and expose its global shape."""
        size = self.n_owned + self.n_ghost
        _check_array(x_local, (size,))
        if isinstance(x_local, jax.core.Tracer):
            raise TypeError('to_sharded must be called outside JIT')
        device = self.mesh.devices[jax.process_index()]
        if x_local.devices() != {device}:
            raise ValueError('local array must reside on this rank mesh device')
        padded = jnp.pad(x_local, (0, self.padded_length - size))[None, :]
        return jax.make_array_from_single_device_arrays(
            (self.mesh.size, self.padded_length), self.sharding, [padded])

    def _check_global(self, x):
        _check_array(x, (self.mesh.size, self.padded_length))
        if not isinstance(x, jax.core.Tracer) and not x.sharding.is_equivalent_to(self.sharding, 2):
            raise ValueError('array must use the plan row sharding')

    def local_array(self, x):
        """Outside JIT: return this process's unpadded shard without a host copy."""
        self._check_global(x)
        if isinstance(x, jax.core.Tracer):
            raise TypeError('local_array must be called outside JIT')
        return x.addressable_shards[0].data[0, :self.n_owned + self.n_ghost]

    def scatter_forward(self, x):
        """Refresh ghosts, preserving owned entries, padding and the input."""
        self._check_global(x)
        exchange = jax.shard_map(
            _forward_shards, mesh=self.mesh,
            in_specs=(P('rank', None),) + (P('rank', None, None),) * 3,
            out_specs=P('rank', None),
        )
        return exchange(x, self.send_indices, self.receive_positions, self.send_valid)
