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
    def stage(message):
        print(f'rank={comm.rank} stage={message}', flush=True)

    stage('MPI initialized')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=8, help='float64 ghost entries per rank')
    parser.add_argument('--profile', action='store_true',
                        help='capture only the second exchange with Nsight cudaProfilerApi')
    args = parser.parse_args()
    assert comm.size == 2, 'Launch two ranks, each with one visible GPU'
    assert args.count > 0
    assert os.environ.get('MPI4JAX_USE_CUDA_MPI') == '1'
    stage('importing JAX')
    import jax
    stage('importing mpi4jax and JAXGhost')
    import mpi4jax
    from jaxghost import JAXGhost
    assert mpi4jax.has_cuda_support()
    jax.config.update('jax_platforms', 'cuda')
    jax.config.update('jax_enable_x64', True)
    # Use the launched MPI world, not a possibly larger Slurm allocation.
    stage(f'initializing distributed JAX via mpi4py: size={comm.size}')
    jax.distributed.initialize(cluster_detection_method='mpi4py',
                               local_device_ids=[0], initialization_timeout=60)
    stage('distributed JAX initialized; initializing CUDA backend')
    device, = jax.local_devices()
    assert device.platform == 'gpu' and jax.process_count() == 2
    if args.profile:
        import ctypes
        import sysconfig
        # Native MPI/UCX and pip JAX can load different CUDA runtimes. Prefer
        # the already-loaded Python environment runtime for profiler control.
        runtime_paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                         if '/libcudart.so' in line}
        python_runtime_dir = (Path(sysconfig.get_path('purelib')) /
                              'nvidia' / 'cuda_runtime' / 'lib').resolve()
        preferred_paths = {path for path in runtime_paths
                           if Path(path).resolve().parent == python_runtime_dir}
        candidates = preferred_paths or runtime_paths
        if len(candidates) != 1:
            raise RuntimeError('Cannot select CUDA runtime for profiling: '
                               f'loaded={sorted(runtime_paths)}, preferred={sorted(preferred_paths)}')
        runtime_path = next(iter(candidates))
        stage(f'profiler runtime={runtime_path}')
        cudart = ctypes.CDLL(runtime_path)
        for name in ('cudaProfilerStart', 'cudaProfilerStop', 'cudaDeviceSynchronize'):
            func = getattr(cudart, name)
            func.argtypes = []
            func.restype = ctypes.c_int

        def cuda_call(name):
            status = getattr(cudart, name)()
            if status != 0:
                raise RuntimeError(f'{name} failed with CUDA error {status}')
    n = args.count
    rank = comm.rank
    other = 1-rank
    imap = SimpleNamespace(size_local=n, local_range=(rank*n, (rank+1)*n),
        ghosts=np.arange(other*n, (other+1)*n, dtype=np.int64),
        owners=np.full(n, other, dtype=np.int32))
    print(f'rank={rank} host={MPI.Get_processor_name()} bytes_per_message={8*n} '
          f'jax={jax.__version__} mpi4jax={mpi4jax.__version__} '
          f'MPI={MPI.Get_library_version().strip()!r}', flush=True)
    stage('creating JAXGhost plan')
    with JAXGhost.from_index_map(imap, comm) as ghost:
        stage('JAXGhost plan ready')
        forward = jax.jit(ghost.scatter_forward)
        for step in range(2):
            host = np.concatenate((np.full(n, rank+1+step, dtype=np.float64), np.zeros(n)))
            stage(f'step={step} host-to-GPU copy')
            x = jax.device_put(host, device)
            x.block_until_ready()
            comm.Barrier()
            capture = args.profile and step == 1
            if capture:
                cuda_call('cudaDeviceSynchronize')
                cuda_call('cudaProfilerStart')
                comm.Barrier()  # Both profilers must be active before exchanging.
            print(f'rank={rank} EXCHANGE_BEGIN step={step}', flush=True)
            y = forward(x)
            y.block_until_ready()
            jax.effects_barrier()
            if capture:
                cuda_call('cudaDeviceSynchronize')
                comm.Barrier()
                cuda_call('cudaProfilerStop')
            print(f'rank={rank} EXCHANGE_END step={step}', flush=True)
            assert y.devices() == {device}
            # Copies outside the marked exchange are setup/validation, not MPI staging.
            expected = np.concatenate((host[:n], np.full(n, other+1+step)))
            np.testing.assert_array_equal(np.asarray(y), expected)
    print(f'rank={rank} CORRECTNESS_PASS (transport verdict requires trace)', flush=True)
    comm.Barrier()
    stage('shutting down distributed JAX')
    jax.distributed.shutdown()
    stage('shutdown complete')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        comm.Abort(1)
