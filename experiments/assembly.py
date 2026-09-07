"""Assemble integral(v dx) on eight P1 interval cells entirely in JAX.

Run: mpirun -n 4 python experiments/assembly.py
"""

from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
import ufl
from dolfinx import fem, la, mesh

from jaxghost import JAXGhost


def run_assembly(comm, *, compiled=True, report=False):
    jax.config.update("jax_enable_x64", True)
    domain = mesh.create_unit_interval(comm, 8, dtype=np.float64)
    space = fem.functionspace(domain, ("Lagrange", 1))
    index_map = space.dofmap.index_map
    n_cells = domain.topology.index_map(1).size_local
    # One-time initialization: only owned cells contribute, avoiding double
    # assembly of ghost cells. Transfer connectivity and geometry once.
    cell_dofs = jnp.asarray(
        np.asarray([space.dofmap.cell_dofs(c) for c in range(n_cells)],
                   dtype=np.int32).reshape(-1, 2)
    )
    coordinates = jnp.asarray(space.tabulate_dof_coordinates()[:, 0])

    with JAXGhost.from_index_map(index_map, comm) as ghost:
        def assemble(coords):
            # For constant load 1 and interval P1 basis functions, each cell
            # contributes [h/2, h/2]. Compute h and accumulate on the device.
            endpoints = coords[cell_dofs]
            h = jnp.abs(endpoints[:, 1] - endpoints[:, 0])
            cell_values = jnp.broadcast_to(h[:, None] / 2, cell_dofs.shape)
            local = jnp.zeros((ghost.n_owned + ghost.n_ghost,), dtype=coords.dtype)
            local = local.at[cell_dofs.ravel()].add(cell_values.ravel())
            accumulated = ghost.scatter_reverse(local)
            refreshed = ghost.scatter_forward(accumulated)
            return local, accumulated, refreshed

        execute = jax.jit(assemble) if compiled else assemble
        local, accumulated, refreshed = execute(coordinates)
        refreshed.block_until_ready()
        jax.effects_barrier()

        # Independent FEM assembly is validation only: no reference values
        # enter the JAX computation. Check every stage against DOLFINx.
        reference = fem.assemble_vector(fem.form(ufl.TestFunction(space) * ufl.dx))
        expected_local = reference.array.copy()
        reference.scatter_reverse(la.InsertMode.add)
        expected_accumulated = reference.array.copy()
        reference.scatter_forward()
        expected_refreshed = reference.array.copy()
        actual = [np.asarray(v) for v in (local, accumulated, refreshed)]
        expected = [expected_local, expected_accumulated, expected_refreshed]
        correct = all(np.allclose(a, b, rtol=1e-12, atol=1e-12)
                      for a, b in zip(actual, expected))
        correct &= np.array_equal(actual[0][ghost.n_owned:], actual[1][ghost.n_owned:])
        correct &= np.array_equal(actual[1][:ghost.n_owned], actual[2][:ghost.n_owned])
        correct &= all(v.devices() == coordinates.devices()
                       for v in (local, accumulated, refreshed))
        # Analytical check: endpoints get h/2, interior nodes h. Owned values
        # alone sum to the integral of 1 over the unit interval, namely 1.
        coords_host = np.asarray(coordinates)
        analytic = np.where(np.isclose(coords_host, 0, atol=1e-12) | np.isclose(coords_host, 1, atol=1e-12), 1 / 16, 1 / 8)
        correct &= np.allclose(actual[2], analytic, rtol=1e-12, atol=1e-12)
        total = comm.allreduce(float(actual[2][:ghost.n_owned].sum()), op=MPI.SUM)
        correct &= np.isclose(total, 1.0, rtol=1e-12, atol=1e-12)
        if not comm.allreduce(bool(correct), op=MPI.LAND):
            raise AssertionError("Assembly pipeline differs from DOLFINx or analytical load")
        error = max(np.max(np.abs(a - b), initial=0.0) for a, b in zip(actual, expected))
        if report:
            rows = comm.gather(
                f"rank {comm.rank}: cells={n_cells}, owned={ghost.n_owned}, "
                f"ghosts={ghost.n_ghost}, max_error={error:.3e}", root=0,
            )
            if comm.rank == 0:
                print("JAX cell assembly -> reverse ADD -> forward INSERT matches DOLFINx")
                print(f"Sum over owned entries: {total:.16g}")
                print("\n".join(rows), flush=True)


if __name__ == "__main__":
    run_assembly(MPI.COMM_WORLD, report=True)
