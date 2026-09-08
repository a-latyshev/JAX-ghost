"""One reverse ADD update; run with mpirun -n 4 python examples/reverse.py."""

from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
from dolfinx import fem, la, mesh

from jaxghost import JAXGhost


def main():
    jax.config.update("jax_enable_x64", True)
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, 4, 4, cell_type=mesh.CellType.triangle)
    space = fem.functionspace(domain, ("Lagrange", 1))
    index_map = space.dofmap.index_map
    owned_ids = jnp.asarray(np.arange(*index_map.local_range), dtype=jnp.int64)
    reference = fem.Function(space, dtype=np.float64)

    with JAXGhost.from_index_map(
        index_map, comm, block_size=space.dofmap.index_map_bs
    ) as ghost:
        # Owners already have local contributions; ghosts hold contributions
        # destined for remote owners. All numerical preparation is in JAX.
        owned = 10.0 + owned_ids
        contributions = jnp.full((ghost.n_ghost,), comm.rank + 1.0)
        x = jnp.concatenate((owned, contributions))
        reverse = jax.jit(ghost.scatter_reverse)
        accumulated = reverse(x)
        accumulated.block_until_ready()
        jax.effects_barrier()

        # Host copies and DOLFINx data are used only as a reference.
        reference.x.array[:] = np.asarray(x)
        reference.x.scatter_reverse(la.InsertMode.add)
        actual = np.asarray(accumulated)
        correct = np.array_equal(actual, reference.x.array) and np.array_equal(
            actual[ghost.n_owned:], np.asarray(contributions)
        )
        if not comm.allreduce(correct, op=MPI.LAND):
            raise AssertionError("Reverse ADD differs from DOLFINx or changed ghosts")
        error = np.max(np.abs(actual - reference.x.array), initial=0.0)
        rows = comm.gather(
            f"rank {comm.rank}: owned={ghost.n_owned}, ghosts={ghost.n_ghost}, "
            f"peers={ghost.peers}, max_error={error:.3e}", root=0,
        )
        if comm.rank == 0:
            print("Reverse ADD matches DOLFINx; ghost contributions are unchanged", flush=True)
            print("\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
