"""Shared sharding-only correctness and timing worker; launch through MPI."""
import argparse
from pathlib import Path
import sys
import traceback
import os
import numpy as np
from mpi4py import MPI
from common import collective, load_fixture, check_result, environment, timings, write_json, sha256
from placement import prepare

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'src'))


def main(check_only=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=HERE/'data')
    p.add_argument('--subdivisions', type=int, default=4 if check_only else 99)
    p.add_argument('--output', type=Path)
    p.add_argument('--trial', type=int, default=0)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeats', type=int, default=10)
    a = p.parse_args()
    comm = MPI.COMM_WORLD
    if min(a.repeats, a.subdivisions) < 1 or a.warmup < 0 or a.trial < 0:
        raise ValueError('Invalid counts')
    if not check_only and a.output is None:
        raise ValueError('Timing requires --output FILE')
    folder = a.data/f'n{a.subdivisions}/{comm.size}ranks'
    meta, data, adapter = load_fixture(folder, comm)
    placements = prepare(comm)  # Mask and query CUDA before JAX initializes a backend.
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from jaxghost import ShardedJAXMatrixCSR
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    jax.distributed.initialize(cluster_detection_method='mpi4py', local_device_ids=[0], initialization_timeout=60)
    try:
        def devices():
            device, = jax.local_devices()
            if device.platform != 'gpu' or jax.process_index() != comm.rank or jax.process_count() != comm.size or jax.device_count() != comm.size:
                raise ValueError('MPI/JAX ranks or GPU count disagree')
            return device
        device = collective(comm, devices)
        op = ShardedJAXMatrixCSR.from_dolfinx(adapter, comm, Mesh(np.asarray(jax.devices()), ('rank',)))
        local_x = data['x'].copy()
        local_x[op.n_owned_cols:] = np.nan  # Require the matvec's own halo refresh.
        source = (data['values'], local_x, data['y_initial'])
        arrays = [op.to_sharded(jax.device_put(v,device),kind=k) for v,k in zip(source,('values','x','y'))]
        # Nonzero padding exposes accidental writes to inactive rows.
        y = arrays[2]
        piece = y.addressable_shards[0].data.at[:,len(data['y_initial']):].set(-987.)
        arrays[2] = jax.make_array_from_single_device_arrays(y.shape,y.sharding,[piece])
        values, x, zero = arrays
        before = [np.asarray(v.addressable_shards[0].data).copy() for v in arrays]
        def finish(v):
            v.block_until_ready()
            jax.effects_barrier()
        mult = jax.jit(ShardedJAXMatrixCSR.mult)
        jax.block_until_ready(arrays)
        comm.Barrier()
        start = MPI.Wtime()
        mult.lower(op, values, x, zero).compile()
        compile_seconds = max(comm.allgather(MPI.Wtime()-start))
        comm.Barrier()
        start = MPI.Wtime()
        result = mult(op, values, x, zero)
        finish(result)
        first_seconds = max(comm.allgather(MPI.Wtime()-start))
        errors = []
        def validate(y, expected):
            actual = np.asarray(op.local_array(y))
            errors.append(check_result(comm,actual,expected,op.n_owned_rows))
            padding = np.asarray(y.addressable_shards[0].data)[0,len(expected):]
            collective(comm, lambda: np.testing.assert_array_equal(padding,before[2][0,len(expected):]))
        validate(result,data['single_expected'])
        # Dynamic values and changed owned x, while x ghosts remain stale.
        changed = mult(op,values*2,x*2,zero)
        finish(changed)
        validate(changed,data['single_expected']*4)
        result = zero
        for _ in range(100):
            result = mult(op,values,x,result)
        finish(result)
        validate(result,data['batch_expected'])
        samples = []
        if not check_only:
            result = zero
            for _ in range(a.warmup):
                result = mult(op,values,x,result)
            finish(result)
            for _ in range(a.repeats):
                result = zero
                finish(result)
                comm.Barrier()
                start = MPI.Wtime()
                for _ in range(100):
                    result = mult(op,values,x,result)
                finish(result)
                elapsed = MPI.Wtime()-start
                samples.append(comm.allgather(elapsed))
                validate(result,data['batch_expected'])
        for v,snapshot in zip(arrays,before):
            collective(comm,lambda v=v,snapshot=snapshot: np.testing.assert_array_equal(np.asarray(v.addressable_shards[0].data),snapshot))
            if v.addressable_shards[0].data.devices() != {device}:
                raise ValueError('Unexpected array placement')
        if any(k.split('.')[0] in ('dolfinx','mpi4jax') for k in sys.modules):
            raise RuntimeError('Sharded replay imported an optional backend')
        source_files = sorted((ROOT/'src/jaxghost').glob('*.py')) + [HERE/name for name in ('worker.py','common.py','placement.py')]
        source_hashes = {str(f.relative_to(ROOT)):sha256(f) for f in source_files}
        record = dict(mode='check' if check_only else 'time',correctness_passed=True,
            subdivisions=a.subdivisions,global_dofs=meta['global_dofs'],ranks=comm.size,trial=a.trial,
            settings=dict(iterations=100,repeats=a.repeats,warmup=a.warmup,loop='python',dtype='float64'),
            fixture_fingerprints=meta['fingerprints'],source_hashes=source_hashes,
            versions=dict(jax=jax.__version__),placement=placements,environment=environment(comm),
            validation=errors,compile_seconds=compile_seconds,first_seconds=first_seconds,
            timing=dict(timer='MPI.Wtime',**timings(samples,100)) if samples else None)
        if comm.rank == 0:
            if a.output:
                if a.output.exists(): raise ValueError('Output already exists')
                write_json(a.output,record)
            message = f'PASS n={a.subdivisions}, ranks={comm.size}'
            if samples: message += f", {record['timing']['median_seconds_per_matvec']*1e6:.2f} us/matvec"
            print(message,flush=True)
            for item in placements:
                print(f"  rank {item['rank']} on {item['hostname']}: {item['gpu_uuid']}, CPUs {item['cpu_affinity']}",flush=True)
        comm.Barrier()
    finally:
        jax.distributed.shutdown()


def entry(check_only=False):
    try:
        main(check_only)
    except BaseException:
        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)


if __name__ == '__main__':
    entry()
