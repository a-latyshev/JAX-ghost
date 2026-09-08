"""Replay and time a saved distributed CSR operator without DOLFINx."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from mpi4py import MPI
from common import StoredMatrix, check_result, environment, timings, write_json

COMM = MPI.COMM_WORLD
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--backend', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    meta = json.loads((args.dataset / 'metadata.json').read_text())
    if meta['format_version'] != 1 or meta['rank_count'] != COMM.size or meta['dtype'] != 'float64':
        raise ValueError('unsupported dataset or wrong MPI rank count')
    settings = meta['settings']
    if any(type(settings[k]) is not int or settings[k] < minimum for k, minimum in
           [('warmup', 0), ('iterations', 1), ('repeats', 1)]):
        raise ValueError('invalid timing settings')
    with np.load(args.dataset / f'rank-{COMM.rank:04d}.npz', allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    adapter = StoredMatrix(data, meta, COMM)
    adapter.validate_ownership(COMM, meta['global_dofs'])
    os.environ['JAX_PLATFORMS'] = args.backend
    os.environ['JAX_NUM_CPU_DEVICES'] = '1'
    import jax
    import mpi4jax
    from jaxghost import JAXMatrixCSR
    jax.config.update('jax_enable_x64', True)
    if args.backend == 'cuda':
        if os.environ.get('MPI4JAX_USE_CUDA_MPI') != '1' or not mpi4jax.has_cuda_support():
            raise ValueError('CUDA replay requires CUDA-enabled mpi4jax and MPI4JAX_USE_CUDA_MPI=1')
        jax.distributed.initialize(local_device_ids=[0])
        if jax.process_count() != COMM.size:
            raise ValueError('JAX/MPI process counts differ')
    device, = jax.local_devices()
    if device.platform != ('gpu' if args.backend == 'cuda' else 'cpu'):
        raise ValueError('unexpected device backend')
    values, x, zero = [jax.device_put(data[k], device) for k in ('values', 'x', 'y_initial')]

    def finish(y):
        y.block_until_ready()
        jax.effects_barrier()

    with JAXMatrixCSR.from_dolfinx(adapter, COMM) as operator:
        mult = jax.jit(operator.mult)
        result = mult(values, x, zero)
        finish(result)
        errors = dict(single=check_result(COMM, np.asarray(result), data['single_expected'], operator.n_owned_rows))
        for _ in range(settings['warmup']):
            result = mult(values, x, result)
        finish(result)
        samples = []
        for _ in range(settings['repeats']):
            result = zero
            finish(result)
            COMM.Barrier()
            start = MPI.Wtime()
            for _ in range(settings['iterations']):
                result = mult(values, x, result)
            finish(result)
            elapsed = MPI.Wtime() - start
            samples.append(COMM.allgather(elapsed))
            errors['batch'] = check_result(COMM, np.asarray(result), data['batch_expected'], operator.n_owned_rows)
        for key, array in [('values', values), ('x', x), ('y_initial', zero)]:
            np.testing.assert_array_equal(np.asarray(array), data[key])
            if array.devices() != {device}:
                raise ValueError('unexpected input device placement')
        if result.devices() != {device}:
            raise ValueError('unexpected output device placement')
    if any(k == 'dolfinx' or k.startswith('dolfinx.') for k in sys.modules):
        raise RuntimeError('replay imported DOLFINx')
    timing = dict(timer='MPI.Wtime', **timings(samples, settings['iterations']))
    baseline = meta['timing']['median_seconds_per_matvec']
    report = dict(dataset=str(args.dataset.resolve()), rank_count=COMM.size,
        global_dofs=meta['global_dofs'], backend=args.backend, settings=settings,
        timing=timing, dolfinx_seconds_per_matvec=baseline,
        dolfinx_over_jax=baseline / timing['median_seconds_per_matvec'], validation=errors,
        jax_version=jax.__version__, mpi4jax_version=mpi4jax.__version__,
        devices=COMM.allgather(str(device.device_kind)), environment=environment(COMM))
    output = args.output or (Path(__file__).parent / 'results' /
        f'n{meta["subdivisions"]}-{COMM.size}ranks-{args.backend}.json')
    if output.resolve().is_relative_to(args.dataset.resolve()):
        raise ValueError('results must be written outside the source dataset')
    if COMM.rank == 0:
        write_json(output, report)
        print(f'REPLAY PASS: {timing["median_seconds_per_matvec"]:.6g} s/matvec; '
              f'DOLFINx/JAX={report["dolfinx_over_jax"]:.3g}; {output}', flush=True)
    COMM.Barrier()
    if args.backend == 'cuda':
        jax.distributed.shutdown()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        COMM.Abort(1)
