"""Run with: sbatch run_test.sh"""

from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
import time
from dolfinx import fem, mesh


from src.jaxghost import JAXGhost

def main():
    comm = MPI.COMM_WORLD

    jax.distributed.initialize('localhost:10000', comm.size, comm.rank)

    # Configurer JAX pour utiliser les GPU
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "gpu")

    print(jax.devices())

    domain = mesh.create_unit_interval(comm, 10000)
    space = fem.functionspace(domain, ("Lagrange", 1))
    index_map = space.dofmap.index_map
    n = index_map.size_local
    owned_ids = np.arange(*index_map.local_range, dtype=np.int64)
    local_ids = np.concatenate((owned_ids, index_map.ghosts))
    reference = fem.Function(space)

    # Initialiser les temps
    jax_times = []
    dolfinx_times = []

    with JAXGhost.from_index_map(
        index_map, comm, block_size=space.dofmap.index_map_bs
    ) as ghost:
        forward = jax.jit(ghost.scatter_forward)

        for step in range(2):
            # Mesurer le temps pour JAX (GPU)
            start_jax = time.perf_counter()
            x = jnp.full((n + len(index_map.ghosts),), jnp.nan)
            owned = 10 + owned_ids + 100 * step
            x = x.at[:n].set(jnp.asarray(owned))
            x = forward(x)
            x.block_until_ready()
            jax.effects_barrier()
            end_jax = time.perf_counter()
            jax_times.append(end_jax - start_jax)

            # Mesurer le temps pour DOLFINx (CPU)
            start_dolfinx = time.perf_counter()
            reference.x.array[:n] = owned
            reference.x.scatter_forward()
            end_dolfinx = time.perf_counter()
            dolfinx_times.append(end_dolfinx - start_dolfinx)

            # Validation
            actual = np.asarray(x)
            expected = 10 + local_ids + 100 * step
            correct = np.array_equal(actual, expected) and np.array_equal(
                actual, reference.x.array
            )
            if not comm.allreduce(correct, op=MPI.LAND):
                raise AssertionError("JAXGhost differs from owner values or DOLFINx")

            # Afficher les résultats pour chaque étape
            rows = comm.gather(
                f"rank {comm.rank}: IDs={local_ids.tolist()}, "
                f"owned={actual[:n].tolist()}, ghosts={actual[n:].tolist()}", root=0,
            )
            if comm.rank == 0:
                print(f"Step {step}: matches DOLFINx", flush=True)
                print("\n".join(rows), flush=True)
            comm.Barrier()

    # Calculer les temps moyens
    if comm.rank == 0:
        avg_jax_time = np.mean(jax_times[1:])
        avg_dolfinx_time = np.mean(dolfinx_times[1:])
        total_time = avg_jax_time + avg_dolfinx_time
        print("\n--- Performance Results ---")
        print(f"Average JAX (GPU) time: {avg_jax_time:.6f} seconds")
        print(f"Average DOLFINx (CPU) time: {avg_dolfinx_time:.6f} seconds")
        print(f"Total time: {total_time:.6f} seconds")

if __name__ == "__main__":
    main()