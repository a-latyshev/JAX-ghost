"""Run with: JAX_PLATFORMS=cpu mpiexec -n 2 python examples/interval.py."""

from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
from dolfinx import fem, mesh

from jaxghost import JAXGhost


def main():
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_interval(comm, max(8, 4 * comm.size))
    space = fem.functionspace(domain, ("Lagrange", 1))
    index_map = space.dofmap.index_map
    n = index_map.size_local
    owned_ids = np.arange(*index_map.local_range, dtype=np.int64)
    local_ids = np.concatenate((owned_ids, index_map.ghosts))
    reference = fem.Function(space, dtype=np.float32)

    with JAXGhost.from_index_map(
        index_map, comm, block_size=space.dofmap.index_map_bs
    ) as ghost:
        forward = jax.jit(ghost.scatter_forward)
        x = jnp.full((n + len(index_map.ghosts),), jnp.nan, dtype=jnp.float32)
        for step in range(2):
            # Only owners change their values. Ghosts remain stale until forward.
            owned = 10 + owned_ids.astype(np.float32) + 100 * step
            x = x.at[:n].set(jnp.asarray(owned))
            x = forward(x)
            x.block_until_ready()
            jax.effects_barrier()

            # Host copies below are for validation/output, not the ghost update.
            reference.x.array[:n] = owned
            reference.x.scatter_forward()
            actual = np.asarray(x)
            expected = 10 + local_ids.astype(np.float32) + 100 * step
            correct = np.array_equal(actual, expected) and np.array_equal(
                actual, reference.x.array
            )
            if not comm.allreduce(correct, op=MPI.LAND):
                raise AssertionError("JAXGhost differs from owner values or DOLFINx")
            rows = comm.gather(
                f"rank {comm.rank}: IDs={local_ids.tolist()}, "
                f"owned={actual[:n].tolist()}, ghosts={actual[n:].tolist()}", root=0,
            )
            if comm.rank == 0:
                print(f"Step {step}: matches DOLFINx", flush=True)
                print("\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
