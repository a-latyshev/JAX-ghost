"""Validate an exported dataset using JAXGhost, without importing DOLFINx."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np

COMM = MPI.COMM_WORLD
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    os.environ['JAX_PLATFORMS'] = args.backend
    os.environ['JAX_NUM_CPU_DEVICES'] = '1'
    import jax
    import mpi4jax
    from jaxghost import JAXGhost
    jax.config.update('jax_enable_x64', True)
    if args.backend == 'cuda':
        assert os.environ.get('MPI4JAX_USE_CUDA_MPI') == '1'
        assert mpi4jax.has_cuda_support()
        jax.distributed.initialize(local_device_ids=[0])
    device, = jax.local_devices()
    assert device.platform == ('gpu' if args.backend == 'cuda' else 'cpu')
    meta = json.loads((args.dataset / 'metadata.json').read_text())
    assert meta['format_version'] == 1 and meta['rank_count'] == COMM.size
    assert meta['block_size'] == 1 and meta['cases'] == 2
    with np.load(args.dataset / f'rank-{COMM.rank:04d}.npz', allow_pickle=False) as data:
        start, stop = map(int, data['owned_range'])
        index_map = SimpleNamespace(size_local=stop-start, local_range=(start, stop),
                                   ghosts=data['ghost_global_indices'], owners=data['ghost_owner_ranks'])
        with JAXGhost.from_index_map(index_map, COMM, block_size=1) as ghost:
            for operation in ('forward', 'reverse'):
                update = jax.jit(getattr(ghost, f'scatter_{operation}'))
                for case in range(meta['cases']):
                    source = data[f'{operation}_input'][case]
                    assert source.dtype == np.float64
                    assert source.shape == (stop-start + len(index_map.ghosts),)
                    x = jax.device_put(source, device)
                    result = update(x)
                    result.block_until_ready()
                    jax.effects_barrier()
                    assert result.devices() == x.devices() == {device}
                    np.testing.assert_array_equal(np.asarray(result), data[f'{operation}_expected'][case])
                    np.testing.assert_array_equal(np.asarray(x), source)
    assert not any(name == 'dolfinx' or name.startswith('dolfinx.') for name in sys.modules)
    COMM.Barrier()
    if args.backend == 'cuda':
        jax.distributed.shutdown()
    if COMM.rank == 0:
        print(f'REPLAY PASS {args.dataset} ranks={COMM.size} backend={args.backend} DOLFINx not imported', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        COMM.Abort(1)
