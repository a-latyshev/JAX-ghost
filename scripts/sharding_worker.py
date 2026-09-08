"""Initialize distributed JAX before loading an example or pytest module."""

import argparse
from functools import partial
from pathlib import Path
import runpy
import socket
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-cpu', action='store_true',
                        help='single-host CPU validation with explicit loopback/Gloo binding')
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / 'src')]
    from mpi4py import MPI
    import jax
    comm = MPI.COMM_WORLD
    if args.local_cpu:
        address = None
        if comm.rank == 0:
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0))
                address = f'127.0.0.1:{sock.getsockname()[1]}'
        address = comm.bcast(address, root=0)
        jax.distributed.initialize(coordinator_address=address,
                                  coordinator_bind_address=address,
                                  num_processes=comm.size, process_id=comm.rank,
                                  initialization_timeout=20)
        # Opt-in compatibility workaround for the tested JAX 0.10.2/macOS
        # Gloo hostname-resolution failure. Kept out of the library API.
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
        jax.config.update('jax_enable_x64', True)
        if args.example:
            runpy.run_path(str(root / 'examples/sharded.py'), run_name='__main__')
        else:
            import pytest
            raise SystemExit(pytest.main([str(root / 'tests/sharding_checks.py'), '-v']))
    finally:
        jax.distributed.shutdown()


if __name__ == '__main__':
    main()
