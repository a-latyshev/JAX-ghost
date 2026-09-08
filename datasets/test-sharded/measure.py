"""One MPI job: compare DOLFINx and ShardedJAXGhost forward INSERT on CPU."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subdivisions', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--local-cpu', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    from mpi4py import MPI
    import mpi4py
    import jax
    import jaxlib
    import psutil
    comm = MPI.COMM_WORLD
    if args.local_cpu:
        address = None
        if comm.rank == 0:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                address = f'127.0.0.1:{sock.getsockname()[1]}'
        address = comm.bcast(address, root=0)
        jax.distributed.initialize(coordinator_address=address, coordinator_bind_address=address,
                                  num_processes=comm.size, process_id=comm.rank,
                                  initialization_timeout=30)
        # Same opt-in, single-host workaround as scripts/sharding_worker.py.
        from jax._src import distributed, xla_bridge
        from jaxlib import _jax
        from jax.extend import backend
        collectives = _jax.make_gloo_tcp_collectives(
            distributed_client=distributed.global_state.client, hostname='127.0.0.1')
        backend.register_backend_factory('cpu', partial(xla_bridge.make_cpu_client,
                                                       collectives=collectives),
                                         priority=0, fail_quietly=False)
    else:
        jax.distributed.initialize(cluster_detection_method='mpi4py', initialization_timeout=30)
    try:
        import jax.numpy as jnp
        from jax.sharding import Mesh
        import dolfinx
        from dolfinx import fem, mesh
        from jaxghost import ShardedJAXGhost
        jax.config.update('jax_enable_x64', True)
        comm.Barrier()
        start = MPI.Wtime()
        domain = mesh.create_unit_square(comm, args.subdivisions, args.subdivisions,
                                         cell_type=mesh.CellType.triangle, dtype=np.float64)
        V = fem.functionspace(domain, ('Lagrange', 1))
        reference = fem.Function(V, dtype=np.float64)
        imap = V.dofmap.index_map
        setup_dolfinx = comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)
        start = MPI.Wtime()
        plan = ShardedJAXGhost.from_index_map(imap, comm, Mesh(np.asarray(jax.devices()), ('rank',)))
        setup_sharded = comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)
        n = imap.size_local
        ids = np.concatenate((np.arange(*imap.local_range), imap.ghosts))
        expected = 10.0 + ids
        local = jnp.concatenate((jnp.asarray(expected[:n]), jnp.full((plan.n_ghost,), jnp.nan)))
        x = plan.to_sharded(local)
        x.block_until_ready()
        comm.Barrier()
        start = MPI.Wtime()
        forward = jax.jit(ShardedJAXGhost.scatter_forward).lower(plan, x).compile()
        compile_seconds = comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)
        start = MPI.Wtime()
        result = forward(plan, x)
        result.block_until_ready()
        first_seconds = comm.allreduce(MPI.Wtime() - start, op=MPI.MAX)

        def validate(result, target):
            actual = np.asarray(plan.local_array(result))
            correct = bool(np.array_equal(actual, target) and np.array_equal(actual, reference.x.array))
            if not comm.allreduce(correct, op=MPI.LAND):
                raise AssertionError('forward output differs from global IDs or DOLFINx')
            return comm.allreduce(float(np.max(np.abs(actual-target), initial=0)), op=MPI.MAX)

        # Changed values and stale ghosts are verified before timings.
        max_error = 0.0
        for step in (0, 1):
            target = expected + 100 * step
            reference.x.array[:] = np.nan
            reference.x.array[:n] = target[:n]
            reference.x.scatter_forward()
            local = jnp.concatenate((jnp.asarray(target[:n]), jnp.full((plan.n_ghost,), jnp.nan)))
            x = plan.to_sharded(local)
            before = np.asarray(x.addressable_shards[0].data).copy()
            result = forward(plan, x)
            result.block_until_ready()
            max_error = max(max_error, validate(result, target))
            if not comm.allreduce(bool(np.array_equal(np.asarray(x.addressable_shards[0].data),
                                                      before, equal_nan=True)), op=MPI.LAND):
                raise AssertionError('sharded forward mutated its input')
            x = result
        target = expected + 100
        for _ in range(args.warmup):
            reference.x.scatter_forward()
            x = forward(plan, x)
        x.block_until_ready()
        samples = {'dolfinx': [], 'sharded': []}
        process = psutil.Process()
        for repeat in range(args.repeats):
            # Alternate order to reduce drift/thermal bias; never run backends concurrently.
            order = ('dolfinx', 'sharded') if (repeat + args.trial) % 2 == 0 else ('sharded', 'dolfinx')
            for name in order:
                x.block_until_ready()
                comm.Barrier()
                cpu_start = time.process_time()
                start = MPI.Wtime()
                if name == 'dolfinx':
                    for _ in range(args.iterations):
                        reference.x.scatter_forward()
                else:
                    for _ in range(args.iterations):
                        x = forward(plan, x)
                    x.block_until_ready()
                elapsed = MPI.Wtime() - start
                cpu_elapsed = time.process_time() - cpu_start
                times = comm.allgather(elapsed)
                cpu_times = comm.allgather(cpu_elapsed)
                samples[name].append(dict(rank_seconds=times, rank_cpu_seconds=cpu_times,
                                          seconds_per_update=max(times)/args.iterations))
                max_error = max(max_error, validate(x, target))
        layout = comm.allgather(dict(rank=comm.rank, owned=int(n), ghosts=int(plan.n_ghost),
                                     local_length=n+plan.n_ghost, threads=process.num_threads(),
                                     hostname=platform.node(), device=str(jax.local_devices()[0])))
        output = dict(subdivisions=args.subdivisions, ranks=comm.size, trial=args.trial,
                      global_dofs=int(imap.size_global), global_cells=2*args.subdivisions**2,
                      dtype='float64', element='scalar continuous P1 triangles',
                      iterations=args.iterations, repeats=args.repeats, warmup=args.warmup,
                      compile_seconds=compile_seconds, first_call_seconds=first_seconds,
                      mesh_space_setup_seconds=setup_dolfinx, sharded_plan_setup_seconds=setup_sharded,
                      max_error=max_error, partitions=layout,
                      padded_length=plan.padded_length, max_message_length=plan.max_message_length,
                      vector_bytes_per_shard=8*plan.padded_length,
                      logical_send_buffer_bytes_per_rank=8*comm.size*plan.max_message_length,
                      total_useful_halo_bytes=8*sum(p['ghosts'] for p in layout),
                      samples=samples,
                      versions=dict(python=platform.python_version(), jax=jax.__version__,
                                    jaxlib=jaxlib.__version__, dolfinx=dolfinx.__version__,
                                    numpy=np.__version__, mpi4py=mpi4py.__version__),
                      platform=platform.platform(), mpi=MPI.Get_library_version(),
                      local_cpu_workaround=args.local_cpu,
                      environment={k:v for k,v in os.environ.items() if k.startswith(
                          ('JAX_', 'XLA_', 'OMP_', 'OPENBLAS_', 'MKL_', 'TF_', 'FI_'))})
        if comm.rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2)+'\n')
            med = {k:float(np.median([r['seconds_per_update'] for r in v])) for k,v in samples.items()}
            print(f'n={args.subdivisions}, ranks={comm.size}, trial={args.trial}, '
                  f'median_us={ {k:round(v*1e6,2) for k,v in med.items()} }, error={max_error}', flush=True)
    finally:
        jax.distributed.shutdown()


if __name__ == '__main__':
    main()
