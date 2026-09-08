"""Scalar P1/P2/P3 forward updates on a clipped L-shaped mesh.

Run with: JAX_PLATFORMS=cpu mpirun -n 2 python examples/irregular.py
"""

import argparse

from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
from dolfinx import fem, mesh
from basix.ufl import element
import ufl

from jaxghost import JAXGhost

jax.config.update("jax_enable_x64", True)


def create_irregular_mesh(comm):
    """Create 53 triangles; let DOLFINx partition the connected L-shaped domain."""
    if comm.rank == 0:
        points = np.array([(i / 6, j / 6) for j in range(7) for i in range(7)],
                          dtype=np.float64)
        triangles = []
        for j in range(6):
            for i in range(6):
                if i >= 3 and j >= 3:
                    continue  # Remove the upper-right quarter.
                a = j * 7 + i
                # Clip the lower-right exterior corner by removing one triangle.
                if (i, j) != (5, 0):
                    triangles.append((a, a + 1, a + 8))
                triangles.append((a, a + 8, a + 7))
        cells = np.asarray(triangles, dtype=np.int64)
        used, inverse = np.unique(cells, return_inverse=True)
        cells = inverse.reshape(-1, 3).astype(np.int64)
        points = points[used]
    else:
        cells = np.empty((0, 3), dtype=np.int64)
        points = np.empty((0, 2), dtype=np.float64)
    coordinate_domain = ufl.Mesh(element("Lagrange", "triangle", 1, shape=(2,)))
    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.shared_facet, 2)
    return mesh.create_mesh(comm, cells, coordinate_domain, points, partitioner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, choices=(1, 2, 3), default=1,
                        help="scalar Lagrange polynomial degree (default: 1)")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    comm = MPI.COMM_WORLD
    domain = create_irregular_mesh(comm)
    cell_counts = comm.allgather(domain.topology.index_map(2).size_local)
    assert sum(cell_counts) == 53
    if 2 <= comm.size <= 4:
        assert len(set(cell_counts)) > 1
    space = fem.functionspace(domain, ("Lagrange", args.degree))
    index_map = space.dofmap.index_map
    n = index_map.size_local
    owned_ids = np.arange(*index_map.local_range, dtype=np.int64)
    local_ids = np.concatenate((owned_ids, index_map.ghosts))
    owned_ids_device = jnp.asarray(owned_ids, dtype=jnp.int64)
    reference = fem.Function(space, dtype=np.float64)

    # The scatterer uses the resulting DoF ownership, independently of geometry.
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
            f"rank {comm.rank}: cells={cell_counts[comm.rank]}, owned={n}, ghosts={ghost.n_ghost}, "
            f"peers={ghost.peers}, max_error={max_error:.3e}",
            root=0,
        )
        if comm.rank == 0:
            print(f"Clipped L-shape P{args.degree} field matches owners and DOLFINx", flush=True)
            print("\n".join(summaries), flush=True)


if __name__ == "__main__":
    main()
