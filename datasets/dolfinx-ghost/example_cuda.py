"""Load the saved four-rank 1D fixture and perform one CUDA ghost update."""
import os
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

from mpi4py import MPI
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def main():
    # 1. Load this rank's saved DOLFINx data (no DOLFINx import).
    comm = MPI.COMM_WORLD
    
    dataset = Path(__file__).parent / '1d' / '4ranks' # <----- CHANGE THE DATA

    assert comm.size == 4, 'Launch this fixture with four MPI ranks'
    with np.load(dataset / f'rank-{comm.rank:04d}.npz', allow_pickle=False) as data:
        start, stop = map(int, data['owned_range'])
        index_map = SimpleNamespace(
            size_local=stop-start, local_range=(start, stop),
            ghosts=data['ghost_global_indices'], owners=data['ghost_owner_ranks'])
        values = data['forward_input'][0]
        expected = data['forward_expected'][0]

    # 2. Initialize CUDA and the owner/ghost communication plan.
    assert os.environ.get('MPI4JAX_USE_CUDA_MPI') == '1', 'Set MPI4JAX_USE_CUDA_MPI=1'
    import jax
    import mpi4jax
    from jaxghost import JAXGhost

    assert mpi4jax.has_cuda_support(), 'mpi4jax requires CUDA support'
    jax.config.update('jax_platforms', 'cuda')
    jax.config.update('jax_enable_x64', True)
    jax.distributed.initialize(local_device_ids=[0])
    device, = jax.local_devices()
    assert device.platform == 'gpu'
    assert jax.process_count() == comm.size and jax.device_count() == comm.size
    with JAXGhost.from_index_map(index_map, comm, block_size=1) as ghost:
        # 3. Move values to the GPU and run a compiled forward update.
        x = jax.device_put(values, device)
        y = jax.jit(ghost.scatter_forward)(x)
        y.block_until_ready()
        jax.effects_barrier()
        assert y.devices() == x.devices() == {device}
        np.testing.assert_array_equal(np.asarray(y), expected)  # Validation only.
    print(f'rank={comm.rank} GPU={device} forward PASS', flush=True)
    comm.Barrier()
    jax.distributed.shutdown()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
