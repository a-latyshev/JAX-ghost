"""Fixed scalar CSR metadata with device-side distributed y += Ax."""

import jax
import jax.numpy as jnp
from jax.experimental import sparse

from .jaxghost import JAXGhost
from ._csr_metadata import _read_structure


class JAXMatrixCSR:
    """Owned-row matvec; construct collectively from a finalized DOLFINx matrix.

    The plan is immutable and must outlive compiled calls. Matrix coefficients
    are dynamic operands, never read from DOLFINx during construction or mult.
    All ranks must call mult in the same order with matching float dtypes.
    """

    @classmethod
    def from_dolfinx(cls, A, comm):
        """Read scalar CSR structure and IndexMaps, retaining only owned rows.

        A must have finalized sparsity. Its numerical ghost-row contributions
        must be accumulated before the caller transfers coefficients to JAX.
        DOLFINx MatrixCSR exposes finalized CSR, but no numerical-finalization
        flag; the latter is a caller responsibility. comm must use its rank order.
        """
        row_map, col_map, nr, ngr, nc, ngc, nnz, pointers, columns = _read_structure(A, comm)
        self = cls()
        self.n_owned_rows, self.n_ghost_rows = nr, ngr
        self.n_owned_cols, self.n_ghost_cols = nc, ngc
        self.nnz_owned = nnz
        self.indptr = jnp.asarray(pointers, dtype=jnp.int32)
        self.indices = jnp.asarray(columns, dtype=jnp.int32)
        self._ghost = JAXGhost.from_index_map(col_map, comm)
        self._closed = False
        return self

    def mult(self, values, x, y):
        """Return y + Ax on owned rows; preserve y ghosts and every input.

        values contains only owned-row nonzeros. x and y are full local
        [owned | ghosts] device arrays for the column and row maps respectively.
        Input ghosts are refreshed internally; input x itself is unchanged.
        JIT compatible; differentiation is outside the supported interface.
        """
        if self._closed:
            raise RuntimeError("matrix operator is closed")
        for name, array, size in (
            ('values', values, self.nnz_owned),
            ('x', x, self.n_owned_cols + self.n_ghost_cols),
            ('y', y, self.n_owned_rows + self.n_ghost_rows),
        ):
            if not isinstance(array, (jax.Array, jax.core.Tracer)):
                raise TypeError(f"{name} must be a JAX array on the device")
            if array.shape != (size,):
                raise ValueError(f"{name}: expected shape ({size},), got {array.shape}")
            if array.dtype not in (jnp.dtype('float32'), jnp.dtype('float64')):
                raise TypeError("mult supports float32 and float64")
        if not (values.dtype == x.dtype == y.dtype):
            raise TypeError("values, x and y must have matching dtypes")
        refreshed = self._ghost.scatter_forward(x)
        # Preserve ordered sends even on ranks with no owned matrix entries.
        if not self.nnz_owned:
            return y
        matrix = sparse.CSR((values, self.indices, self.indptr),
                            shape=(self.n_owned_rows, self.n_owned_cols + self.n_ghost_cols))
        product = sparse.csr_matvec(matrix, refreshed)
        return y.at[:self.n_owned_rows].add(product)

    def close(self):
        """Collectively wait for communication and release the owned plan."""
        if not self._closed:
            self._ghost.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
