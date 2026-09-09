"""Scalar owned-row CSR matvec with native JAX halo exchange."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jax.sharding import PartitionSpec as P
import numpy as np

from ._csr_metadata import _read_structure
from .sharded import ShardedJAXGhost, _check_array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class ShardedJAXMatrixCSR:
    """Fixed sparsity, dynamic coefficients; pass this plan explicitly to JIT.

    Global arrays have a rank axis and a padded local-storage axis. No runtime
    MPI communicator is created; distributed JAX lifetime is caller-owned.
    """
    ghost: ShardedJAXGhost
    row_counts: tuple
    nnz_counts: tuple
    row_length: int
    max_rows: int
    max_nnz: int
    indptr: jax.Array
    indices: jax.Array
    owned_mask: jax.Array

    def tree_flatten(self):
        return ((self.ghost, self.indptr, self.indices, self.owned_mask),
                (self.row_counts, self.nnz_counts, self.row_length, self.max_rows, self.max_nnz))

    @classmethod
    def tree_unflatten(cls, aux, children):
        ghost, pointers, columns, mask = children
        return cls(ghost, *aux, pointers, columns, mask)

    @classmethod
    def from_dolfinx(cls, A, comm, mesh):
        """Collective setup from finalized scalar CSR, reading metadata only."""
        _, cmap, nr, ngr, _, _, nnz, pointers, columns = _read_structure(A, comm)
        ghost = ShardedJAXGhost.from_index_map(cmap, comm, mesh)
        counts = tuple(comm.allgather((nr, ngr)))
        nonzeros = tuple(comm.allgather(nnz))
        length = max(1, max(n + g for n, g in counts))
        rows = max(1, max(n for n, _ in counts))
        width = max(1, max(nonzeros))
        def table(a):
            local = jax.device_put(a[None], mesh.devices[comm.rank])
            return jax.make_array_from_single_device_arrays(
                (comm.size, a.size), ghost.sharding, [local])
        return cls(ghost, counts, nonzeros, length, rows, width,
                   table(np.pad(pointers.astype(np.int32), (0, rows-nr), constant_values=nnz)),
                   table(np.pad(columns.astype(np.int32), (0, width-nnz))),
                   table(np.arange(rows) < nr))

    @property
    def nnz_owned(self):
        return self.nnz_counts[jax.process_index()]

    @property
    def n_owned_rows(self):
        return self.row_counts[jax.process_index()][0]

    @property
    def n_ghost_rows(self):
        return self.row_counts[jax.process_index()][1]

    @property
    def n_owned_cols(self):
        return self.ghost.n_owned

    @property
    def n_ghost_cols(self):
        return self.ghost.n_ghost

    def _layout(self, kind):
        if kind == 'x':
            return self.n_owned_cols + self.n_ghost_cols, self.ghost.padded_length
        if kind == 'y':
            return self.n_owned_rows + self.n_ghost_rows, self.row_length
        if kind == 'values':
            return self.nnz_owned, self.max_nnz
        raise ValueError("kind must be 'x', 'y', or 'values'")

    def to_sharded(self, local, *, kind):
        """Outside JIT: pad local x, y or values on this rank's device."""
        size, width = self._layout(kind)
        _check_array(local, (size,))
        if isinstance(local, jax.core.Tracer):
            raise TypeError('to_sharded must be called outside JIT')
        if local.devices() != {self.ghost.mesh.devices[jax.process_index()]}:
            raise ValueError('local array must reside on this rank mesh device')
        piece = jnp.pad(local, (0, width-size))[None]
        return jax.make_array_from_single_device_arrays(
            (self.ghost.mesh.size, width), self.ghost.sharding, [piece])

    def _check(self, array, width):
        _check_array(array, (self.ghost.mesh.size, width))
        if not isinstance(array, jax.core.Tracer) and not array.sharding.is_equivalent_to(self.ghost.sharding, 2):
            raise ValueError('array must use the plan row sharding')

    def local_array(self, array, *, kind='y'):
        """Outside JIT: expose the unpadded addressable shard without host copy."""
        size, width = self._layout(kind)
        self._check(array, width)
        if isinstance(array, jax.core.Tracer):
            raise TypeError('local_array must be called outside JIT')
        return array.addressable_shards[0].data[0, :size]

    def mult(self, values, x, y):
        """Return owned y += Ax, preserving ghosts, padding and inputs.

        Refresh x with native all-to-all before local CSR multiplication. No
        overlap, blocked matrices, transpose or differentiation is promised.
        """
        for a, width in ((values, self.max_nnz), (x, self.ghost.padded_length), (y, self.row_length)):
            self._check(a, width)
        if not (values.dtype == x.dtype == y.dtype):
            raise TypeError('values, x and y must have matching dtypes')
        refreshed = self.ghost.scatter_forward(x)
        def body(v, xx, yy, ptr, col, mask):
            matrix = sparse.CSR((v[0], col[0], ptr[0]),
                                shape=(self.max_rows, self.ghost.padded_length))
            product = sparse.csr_matvec(matrix, xx[0])
            updated = jnp.where(mask[0], yy[0, :self.max_rows] + product, yy[0, :self.max_rows])
            return yy.at[0, :self.max_rows].set(updated)
        return jax.shard_map(body, mesh=self.ghost.mesh,
                             in_specs=(P('rank', None),)*6, out_specs=P('rank', None))(
            values, refreshed, y, self.indptr, self.indices, self.owned_mask)
