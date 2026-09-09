"""Run after distributed JAX initialization via scripts/sharding_worker.py."""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from mpi4py import MPI
from dolfinx import la
from examples.matvec import create_matrix
from jaxghost import ShardedJAXMatrixCSR


def main():
    comm = MPI.COMM_WORLD
    A = create_matrix(comm)
    mesh = Mesh(np.asarray(jax.devices()), ('rank',))
    op = ShardedJAXMatrixCSR.from_dolfinx(A, comm, mesh)
    values = op.to_sharded(jnp.array(A.data[:op.nnz_owned], copy=True), kind='values')
    local_x = jnp.full((op.n_owned_cols + op.n_ghost_cols,), jnp.nan)
    local_x = local_x.at[:op.n_owned_cols].set(10. + jnp.asarray(np.arange(*A.index_map(1).local_range)))
    local_y = jnp.full((op.n_owned_rows + op.n_ghost_rows,), 3.)
    x = op.to_sharded(local_x, kind='x')
    y = op.to_sharded(local_y, kind='y')
    result = jax.jit(ShardedJAXMatrixCSR.mult)(op, values, x, y)
    result.block_until_ready()
    xr, yr = (la.vector(A.index_map(i), dtype=np.float64) for i in (1, 0))
    xr.array[:] = np.asarray(local_x)
    yr.array[:] = np.asarray(local_y)
    A.mult(xr, yr)
    actual = np.asarray(op.local_array(result))
    if not comm.allreduce(bool(np.allclose(actual, yr.array, rtol=1e-12, atol=1e-12)), op=MPI.LAND):
        raise AssertionError('sharded matvec differs from DOLFINx')
    error = np.max(np.abs(actual-yr.array), initial=0.)
    reports = comm.gather(f'rank {comm.rank}: rows={op.n_owned_rows}, nnz={op.nnz_owned}, error={error:.3e}', root=0)
    if comm.rank == 0:
        print('Sharded owned-row y += Ax matches DOLFINx\n' + '\n'.join(reports))


if __name__ == '__main__':
    main()
