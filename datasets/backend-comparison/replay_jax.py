"""Time individual compiled matvecs in a Python loop on a shared DOLFINx fixture."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from mpi4py import MPI
from common import StoredMatrix, check_result, environment, fingerprints, phase, placement, timings, write_json

from memory_probe import mark

COMM = MPI.COMM_WORLD
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--backend', choices=['mpi4jax', 'sharding'], required=True)
    parser.add_argument('--trial', type=int, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    mark('fixture_loading')
    meta = json.loads((args.dataset/'metadata.json').read_text())
    if meta['rank_count'] != COMM.size or meta['dtype'] != 'float64' or meta['format_version'] != 1:
        raise ValueError('Fixture rank count, dtype or format mismatch')
    hashes = fingerprints(args.dataset, COMM)
    if hashes != meta['fingerprints']:
        raise ValueError('Fixture fingerprint mismatch')
    with np.load(args.dataset/f'rank-{COMM.rank:04d}.npz', allow_pickle=False) as saved:
        data = {k: saved[k] for k in saved.files}
    adapter = StoredMatrix(data, meta, COMM)
    adapter.validate_ownership(COMM, meta['global_dofs'])
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    jax.config.update('jax_log_compiles', meta['subdivisions'] == 4)
    mark('runtime_initialization')
    jax.distributed.initialize(cluster_detection_method='mpi4py', local_device_ids=[0], initialization_timeout=120)
    try:
        device, = jax.local_devices()
        if jax.process_count() != COMM.size or jax.device_count() != COMM.size:
            raise ValueError('JAX process/device count must equal MPI rank count')
        hardware = placement(COMM, args.backend, device)
        versions = dict(jax=jax.__version__)
        import jaxlib
        versions['jaxlib'] = jaxlib.__version__
        mark('device_upload', device)
        local = [jax.device_put(data[k], device) for k in ('values', 'x', 'y_initial')]
        def finish(value):
            jax.block_until_ready(value)
            jax.effects_barrier()
        finish(local)
        mark('plan_construction', device)
        with ExitStack() as stack:
            if args.backend == 'mpi4jax':
                import mpi4jax
                from jaxghost import JAXMatrixCSR
                versions['mpi4jax'] = mpi4jax.__version__
                if not mpi4jax.has_cuda_support() or os.environ.get('MPI4JAX_USE_CUDA_MPI') != '1':
                    raise ValueError('CUDA-enabled mpi4jax/MPI is required')
                plan = stack.enter_context(JAXMatrixCSR.from_dolfinx(adapter, COMM))
                operands = local
                function = jax.jit(plan.mult)
                lower_args = operands
                extract = lambda a, kind='y': a
            else:
                from jax.sharding import Mesh
                from jaxghost import ShardedJAXMatrixCSR
                plan = ShardedJAXMatrixCSR.from_dolfinx(adapter, COMM, Mesh(np.asarray(jax.devices()), ('rank',)))
                operands = [plan.to_sharded(a, kind=k) for a,k in zip(local, ('values','x','y'))]
                function = jax.jit(ShardedJAXMatrixCSR.mult)
                lower_args = [plan, *operands]
                extract = lambda a, kind='y': plan.local_array(a, kind=kind)
            finish(lower_args if args.backend == 'sharding' else [*operands, plan.indptr, plan.indices, plan._ghost.send_indices, plan._ghost.receive_positions])
            mark('lowering', device)
            lowered, lowering = phase(COMM, lambda: function.lower(*lower_args))
            mark('compilation', device)
            compiled, compilation = phase(COMM, lowered.compile)
            # Compile one matvec only. Use the cached JIT callable on both backends:
            # JAX 0.11.1 direct AOT calls fail on repeated ordered MPI effects.
            call = function if args.backend == 'mpi4jax' else lambda v,x,y: function(plan,v,x,y)
            v, x, zero = operands
            mark('first_execution', device)
            result, first = phase(COMM, lambda: call(v,x,zero), finish)
            mark('validation', device)
            errors = [check_result(COMM, np.asarray(extract(result)), data['single_expected'], plan.n_owned_rows)]
            settings = meta['settings']
            mark('warmup', device)
            for _ in range(settings['warmup']):
                result = call(v,x,result)
            finish(result)
            mark('timed_computation', device)
            samples = []
            for _ in range(settings['repeats']):
                result = zero
                finish(result)
                COMM.Barrier()
                start = MPI.Wtime()
                for _ in range(settings['iterations']):
                    result = call(v,x,result)
                finish(result)
                elapsed = MPI.Wtime() - start
                samples.append(COMM.allgather(elapsed))
                errors.append(check_result(COMM, np.asarray(extract(result)), data['batch_expected'], plan.n_owned_rows))
            mark('final_validation', device)
            for a,key,kind in zip(operands, ('values','x','y_initial'), ('values','x','y')):
                view = extract(a, kind)
                valid = np.array_equal(np.asarray(view), data[key]) and view.devices() == {device}
                if not COMM.allreduce(bool(valid), op=MPI.LAND):
                    raise ValueError('Inputs changed or placed on unexpected devices')
            if not COMM.allreduce(extract(result).devices() == {device}, op=MPI.LAND):
                raise ValueError('Unexpected output placement')
            if any(k == 'dolfinx' or k.startswith('dolfinx.') for k in sys.modules):
                raise RuntimeError('JAX replay imported DOLFINx')
            if os.environ.get('BENCH_MEMORY_DIR') and args.backend == 'sharding':
                details = dict(max_rows=plan.max_rows, max_nnz=plan.max_nnz,
                    padded_x_length=plan.ghost.padded_length, padded_y_length=plan.row_length,
                    max_message_length=plan.ghost.max_message_length,
                    padded_send_values_per_rank=COMM.size*plan.ghost.max_message_length,
                    valid_send_values=int(np.asarray(plan.ghost.send_valid.addressable_shards[0].data).sum()),
                    addressable_operand_bytes=sum(a.addressable_shards[0].data.nbytes for a in operands),
                    addressable_metadata_bytes=sum(a.addressable_shards[0].data.nbytes
                        for a in jax.tree_util.tree_leaves(plan)))
                mark('workload_done', device, details)
                text_bytes = len(lowered.as_text().encode('utf-8'))
                mark('diagnostics', device, dict(lowered_text_bytes=text_bytes))
            report = dict(backend=args.backend, trial=args.trial, subdivisions=meta['subdivisions'],
                global_dofs=meta['global_dofs'], ranks=COMM.size, dataset=str(args.dataset.resolve()),
                fingerprints=hashes, settings=settings, placement=hardware, versions=versions,
                phases=dict(lowering=lowering, compilation=compilation, first_execution=first),
                timing=dict(timer='MPI.Wtime', **timings(samples, settings['iterations'])),
                validation=errors, correctness_passed=True, environment=environment(COMM))
            if COMM.rank == 0:
                write_json(args.result, report)
                print(f"PASS {args.backend}: {report['timing']['median_seconds_per_matvec']*1e6:.2f} us/matvec", flush=True)
        COMM.Barrier()
    finally:
        jax.distributed.shutdown()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        COMM.Abort(1)
