"""Scalar mass-plus-diffusion matvec: mpirun -n 4 python examples/matvec.py."""

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from dolfinx import fem, la, mesh
import ufl
from jaxghost import JAXMatrixCSR


def create_matrix(comm, dtype=np.float64):
    domain = mesh.create_unit_square(comm, 4, 4, dtype=dtype)
    V = fem.functionspace(domain, ('Lagrange', 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    form = fem.form((u * v + ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx, dtype=dtype)
    A = fem.assemble_matrix(form)
    A.scatter_reverse()
    return A


def main():
    jax.config.update('jax_enable_x64', True)
    comm = MPI.COMM_WORLD
    A = create_matrix(comm)
    with JAXMatrixCSR.from_dolfinx(A, comm) as operator:
        values = jnp.array(A.data[:operator.nnz_owned], copy=True)
        ids = jnp.asarray(np.arange(*A.index_map(1).local_range))
        x = jnp.full((operator.n_owned_cols + operator.n_ghost_cols,), jnp.nan)
        x = x.at[:operator.n_owned_cols].set(10.0 + ids)
        y = jnp.full((operator.n_owned_rows + operator.n_ghost_rows,), 2.0)
        result = jax.jit(operator.mult)(values, x, y)
        result.block_until_ready()
        jax.effects_barrier()
        # Reference data transfers are confined to validation.
        xr = la.vector(A.index_map(1), dtype=np.float64)
        yr = la.vector(A.index_map(0), dtype=np.float64)
        xr.array[:] = np.asarray(x)
        yr.array[:] = np.asarray(y)
        A.mult(xr, yr)
        actual = np.asarray(result)
        correct = np.allclose(actual, yr.array, rtol=1e-12, atol=1e-12)
        if not comm.allreduce(bool(correct), op=MPI.LAND):
            raise AssertionError('Matvec differs from DOLFINx')
        error = np.max(np.abs(actual - yr.array), initial=0.0)
        rows = comm.gather(f'rank {comm.rank}: rows={operator.n_owned_rows}, '
                           f'nnz={operator.nnz_owned}, max_error={error:.3e}', root=0)
        if comm.rank == 0:
            print('Owned-row y += Ax matches DOLFINx', flush=True)
            print('\n'.join(rows), flush=True)


if __name__ == '__main__':
    main()
