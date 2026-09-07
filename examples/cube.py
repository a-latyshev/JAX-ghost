"""Scalar P1 forward updates on a tetrahedral unit-cube mesh.

Run with: JAX_PLATFORMS=cpu mpirun -n 2 python examples/cube.py
"""

from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
from dolfinx import fem, mesh

from jaxghost import JAXGhost


def main():
    jax.config.update("jax_enable_x64", True)
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_cube(
        comm, 4, 4, 4, cell_type=mesh.CellType.tetrahedron, dtype=np.float64
    )
    space = fem.functionspace(domain, ("Lagrange", 1))
    index_map = space.dofmap.index_map
    n = index_map.size_local
    owned_ids = np.arange(*index_map.local_range, dtype=np.int64)
    local_ids = np.concatenate((owned_ids, index_map.ghosts))
    owned_ids_device = jnp.asarray(owned_ids, dtype=jnp.int64)
    reference = fem.Function(space, dtype=np.float64)

    # The same IndexMap-based scatterer works for interval, square, and cube meshes.
    with JAXGhost.from_index_map(
        index_map, comm, block_size=space.dofmap.index_map_bs
    ) as ghost:
        forward = jax.jit(ghost.scatter_forward)
        x = jnp.full((n + ghost.n_ghost,), jnp.nan, dtype=jnp.float64)
        # Numerical updates stay on the device.
        owned = 10.0 + owned_ids_device
        x = x.at[:n].set(owned)
        x = forward(x)
        x.block_until_ready()
        jax.effects_barrier()

        # Host transfers and DOLFINx synchronization below are validation only.
        reference.x.array[:n] = np.asarray(owned)
        reference.x.scatter_forward()
        actual = np.asarray(x)
        expected = 10.0 + local_ids
        correct = np.array_equal(actual, expected) and np.array_equal(
            actual, reference.x.array
        )
        if not comm.allreduce(correct, op=MPI.LAND):
            raise AssertionError("JAXGhost differs from owner values or DOLFINx")
        max_error = max(
            np.max(np.abs(actual - expected), initial=0.0),
            np.max(np.abs(actual - reference.x.array), initial=0.0),
        )
        summaries = comm.gather(
            f"rank {comm.rank}: owned={n}, ghosts={ghost.n_ghost}, "
            f"peers={ghost.peers}, max_error={max_error:.3e}",
            root=0,
        )
        if comm.rank == 0:
            print("Cube field matches owners and DOLFINx", flush=True)
            print("\n".join(summaries), flush=True)


if __name__ == "__main__":
    main()
