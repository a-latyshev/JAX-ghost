"""Two-rank CUDA ghost exchange; use trace.sh to inspect MPI's payload path."""
import argparse
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
comm = MPI.COMM_WORLD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=8, help='float64 ghost entries per rank')
    args = parser.parse_args()
    assert comm.size == 2, 'Launch two ranks, each with one visible GPU'
    assert args.count > 0
    assert os.environ.get('MPI4JAX_USE_CUDA_MPI') == '1'
    import jax
    import mpi4jax
    from jaxghost import JAXGhost
    assert mpi4jax.has_cuda_support()
    jax.config.update('jax_platforms', 'cuda')
    jax.config.update('jax_enable_x64', True)
    jax.distributed.initialize(local_device_ids=[0])
    device, = jax.local_devices()
    assert device.platform == 'gpu' and jax.process_count() == 2
    n = args.count
    rank = comm.rank
    other = 1-rank
    imap = SimpleNamespace(size_local=n, local_range=(rank*n, (rank+1)*n),
        ghosts=np.arange(other*n, (other+1)*n, dtype=np.int64),
        owners=np.full(n, other, dtype=np.int32))
    print(f'rank={rank} host={MPI.Get_processor_name()} bytes_per_message={8*n} '
          f'jax={jax.__version__} mpi4jax={mpi4jax.__version__} '
          f'MPI={MPI.Get_library_version().strip()!r}', flush=True)
    with JAXGhost.from_index_map(imap, comm) as ghost:
        forward = jax.jit(ghost.scatter_forward)
        for step in range(2):
            host = np.concatenate((np.full(n, rank+1+step, dtype=np.float64), np.zeros(n)))
            x = jax.device_put(host, device)
            x.block_until_ready()
            comm.Barrier()
            print(f'rank={rank} EXCHANGE_BEGIN step={step}', flush=True)
            y = forward(x)
            y.block_until_ready()
            jax.effects_barrier()
            print(f'rank={rank} EXCHANGE_END step={step}', flush=True)
            assert y.devices() == {device}
            # Copies outside the marked exchange are setup/validation, not MPI staging.
            expected = np.concatenate((host[:n], np.full(n, other+1+step)))
            np.testing.assert_array_equal(np.asarray(y), expected)
    print(f'rank={rank} CORRECTNESS_PASS (transport verdict requires trace)', flush=True)
    comm.Barrier()
    jax.distributed.shutdown()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        comm.Abort(1)
