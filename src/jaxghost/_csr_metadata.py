"""Scalar CSR metadata validation shared by both execution backends."""
import numpy as np
from ._metadata import _collective_check


def _read_structure(A, comm):
    """Collectively validate and copy owned-row CSR metadata only."""
    error = None
    try:
        if tuple(A.block_size) != (1, 1):
            raise ValueError("only scalar matrix block sizes [1, 1] are supported")
        row_map, col_map = A.index_map(0), A.index_map(1)
        nr, ngr = int(row_map.size_local), len(row_map.ghosts)
        nc, ngc = int(col_map.size_local), len(col_map.ghosts)
        pointers = np.asarray(A.indptr)
        columns = np.asarray(A.indices)
        if min(nr, nc) < 0 or pointers.ndim != 1 or pointers.size < nr + 1:
            raise ValueError("invalid owned-row CSR shape")
        if columns.ndim != 1 or pointers.dtype.kind not in 'iu' or columns.dtype.kind not in 'iu':
            raise ValueError("CSR indices and pointers must be integer vectors")
        pointers = pointers[:nr + 1].copy()
        if pointers[0] != 0 or np.any(pointers[1:] < pointers[:-1]):
            raise ValueError("invalid CSR row pointers")
        nnz = int(pointers[-1])
        if nnz > columns.size:
            raise ValueError("CSR pointers exceed column storage")
        columns = columns[:nnz].copy()
        if np.any((columns < 0) | (columns >= nc + ngc)):
            raise ValueError("CSR column outside column IndexMap")
        if max(nnz, nr + ngr, nc + ngc) > np.iinfo(np.int32).max:
            raise ValueError("CSR indexing exceeds int32 range")
    except (AttributeError, TypeError, ValueError, IndexError) as exc:
        error = str(exc)
    _collective_check(comm, error)
    return row_map, col_map, nr, ngr, nc, ngc, nnz, pointers, columns

