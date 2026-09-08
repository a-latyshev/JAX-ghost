"""One sharded P2 vector-field forward update on an irregular 2D mesh.

Run: mpirun -n 2 python examples/sharded.py
Local macOS workaround: python scripts/run_sharding_tests.py --local-cpu --example --ranks 2 4
"""

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI
from jax.sharding import Mesh
from dolfinx import fem
from jaxghost import ShardedJAXGhost


def main():
    # Initialize before importing the mesh helper (which enables x64) or
    # querying devices. A test worker may already own distributed initialization.
    owned_runtime = not jax.distributed.is_initialized()
    if owned_runtime:
        jax.distributed.initialize(cluster_detection_method='mpi4py')
    try:
        jax.config.update('jax_enable_x64', True)
        # Works for direct script execution and module/runpy execution.
        from pathlib import Path
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from examples.irregular import create_irregular_mesh
        comm = MPI.COMM_WORLD
        domain = create_irregular_mesh(comm)
        V = fem.functionspace(domain, ('Lagrange', 2, (2,)))
        index_map, bs = V.dofmap.index_map, V.dofmap.index_map_bs
        mesh = Mesh(np.asarray(jax.devices()), ('rank',))
        ghost = ShardedJAXGhost.from_index_map(index_map, comm, mesh, block_size=bs)
        ids = jnp.asarray(np.arange(*index_map.local_range))
        owned = (10.0 + ids[:, None] * bs + jnp.arange(bs)).ravel()
        local = jnp.concatenate((owned, jnp.full((ghost.n_ghost,), jnp.nan)))
        x = ghost.to_sharded(local)
        updated = jax.jit(ShardedJAXGhost.scatter_forward)(ghost, x)
        updated.block_until_ready()
        actual = np.asarray(ghost.local_array(updated))
        reference = fem.Function(V)
        reference.x.array[:ghost.n_owned] = np.asarray(owned)
        reference.x.scatter_forward()
        if not comm.allreduce(bool(np.array_equal(actual, reference.x.array)), op=MPI.LAND):
            raise AssertionError('Sharded forward differs from DOLFINx')
        rows = comm.gather(f'rank {comm.rank}: owned={ghost.n_owned}, ghosts={ghost.n_ghost}, '
                           f'max_error={np.max(np.abs(actual-reference.x.array),initial=0):.3e}', root=0)
        if comm.rank == 0:
            print(f'Sharded forward matches DOLFINx; global buffer shape={updated.shape}')
            print('\n'.join(rows), flush=True)
    finally:
        if owned_runtime:
            jax.distributed.shutdown()


if __name__ == '__main__':
    main()
