"""Serial campaign orchestration; each worker launch starts in a clean shell."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

HERE = Path(__file__).resolve().parent


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def command_output(command):
    return subprocess.check_output(command, universal_newlines=True, timeout=30).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sizes', type=int, nargs='+', default=[46,99])
    parser.add_argument('--ranks', type=int, nargs='+', choices=[1,2,3,4], default=[1,2,3,4])
    parser.add_argument('--jax-only-from', type=Path, help='Reuse fixtures and DOLFINx records from a completed campaign')
    parser.add_argument('--jax-cpus-per-rank', type=int, choices=[1,7], default=1)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--skip-regression', action='store_true', help='Only for reruns after regression checks have been recorded')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout per launch in seconds')
    args = parser.parse_args()
    if min(args.sizes) < 1 or len(set(args.sizes)) != len(args.sizes) or len(set(args.ranks)) != len(args.ranks):
        parser.error('sizes must be positive and size/rank lists must contain no duplicates')
    out = (args.output or HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'raw').mkdir()
    (out/'logs').mkdir()
    baseline = args.jax_only_from.resolve() if args.jax_only_from else None
    old = json.loads((baseline/'manifest.json').read_text()) if baseline else None
    job = os.environ['SLURM_JOB_ID']
    (out/'allocation.txt').write_text(command_output(['scontrol','show','job',job])+'\n')
    gpu_text = command_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id','--format=csv,noheader'])
    (out/'gpus.csv').write_text(gpu_text+'\n')
    gpu_rows = [[x.strip() for x in row] for row in csv.reader(gpu_text.splitlines())]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    selected = [r for r in gpu_rows if r[0] in visible or r[1] in visible]
    if len(selected) < 4:
        raise RuntimeError(f'Four allocated GPUs required; visible={visible}, inventory={gpu_rows}')
    affinity = sorted(os.sched_getaffinity(0))
    topology = {}
    for cpu in affinity:
        p = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = (int((p/'physical_package_id').read_text()), int((p/'core_id').read_text()))
        topology.setdefault(key, cpu)
    # Stable placement reused for every backend and rank-count subset.
    cpus = list(topology.values())[:4]
    if len(cpus) != 4:
        raise RuntimeError('Four distinct allocated physical CPU cores required')
    cpu_sets = [[c] for c in cpus]
    if args.jax_cpus_per_rank == 7:
        available = set(topology.values()) - set(cpus)
        for i,anchor in enumerate(cpus):
            socket_id = next(k[0] for k,v in topology.items() if v == anchor)
            extra = sorted(v for k,v in topology.items() if k[0] == socket_id and v in available)[:6]
            if len(extra) != 6: raise RuntimeError('Insufficient non-overlapping cores on the rank socket')
            cpu_sets[i] = sorted([anchor, *extra])
            available.difference_update(extra)
    if old:
        assert old['hostname'] == socket.gethostname(), 'Use the original benchmark node'
        assert old['gpu_uuids'] == [r[1] for r in selected[:4]], 'GPU assignment changed'
        assert old['cpu_ids'] == cpus, 'CPU anchors changed'
    settings = dict(sizes=sorted(args.sizes), ranks=sorted(args.ranks), trials=3, warmup=10, iterations=100, repeats=10)
    if old and any(settings[k] != old['settings'][k] for k in settings):
        raise ValueError('Reused campaign settings must match the requested settings')
    (out/'memory-inventory.csv').write_text(command_output(['nvidia-smi',
        '--query-gpu=uuid,memory.total,memory.free', '--format=csv'])+'\n')
    (out/'host-memory.txt').write_text(Path('/proc/meminfo').read_text())
    manifest = dict(settings=settings, job=job, hostname=socket.gethostname(),
        jax_cpus_per_rank=args.jax_cpus_per_rank, jax_cpu_sets=cpu_sets,
        baseline=str(baseline) if baseline else None, dolfinx_reused=bool(baseline),
        cpu_ids=cpus, gpu_uuids=[r[1] for r in selected[:4]], cpu_topology=[dict(socket=k[0],core=k[1],cpu=v) for k,v in topology.items()],
        source_revision=command_output(['git','-C',str(HERE),'rev-parse','HEAD']),
        source_status=command_output(['git','-C',str(HERE),'status','--short']),
        smoke_only=args.smoke_only, regression_skipped=args.skip_regression or bool(baseline))
    import hashlib
    manifest['source_sha256'] = {str(p.relative_to(HERE.parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*HERE.glob('*.py'), *HERE.glob('*.sh'), *(HERE.parent.parent/'src/jaxghost').glob('*.py')]}
    # Conservative planning estimate, not a peak-memory guarantee: 27 local
    # neighbors per P1 grid vertex, 2x partition allowance, eight CSR-sized
    # buffers for padding, temporaries and compilation, plus 1 GiB runtime.
    manifest['memory_estimates'] = [dict(subdivisions=n, ranks=p,
        estimated_gpu_bytes=int(8 * 2 * ((n+1)**3 / p) * (27*12 + 4 + 32) + 2**30),
        assumptions='27 nonzeros/row; 2x partition allowance; 8 buffers; 1 GiB runtime; estimate only')
        for n in settings['sizes'] for p in settings['ranks']]
    write(out/'manifest.json',manifest)
    env = dict(HOME=os.environ['HOME'], USER=os.environ['USER'], LOGNAME=os.environ.get('LOGNAME',os.environ['USER']),
        PATH='/usr/local/bin:/usr/bin:/bin', LANG='C.UTF-8', SLURM_JOB_ID=job,
        BENCH_JAX_CPUS_PER_RANK=str(args.jax_cpus_per_rank), BENCH_CPU_SETS=json.dumps(cpu_sets),
        BENCH_CPU_IDS=','.join(map(str,cpus)), BENCH_GPU_UUIDS=','.join(manifest['gpu_uuids']))
    launches = []
    print(f'Campaign: {out}\nCPU IDs: {cpus}\nGPU UUIDs: {manifest["gpu_uuids"]}',flush=True)
    def launch(backend, ranks, name, options, result=None, timeout=None):
        cmd = ['bash','--noprofile','--norc',str(HERE/'launch.sh'),backend,str(ranks),*map(str,options)]
        log = out/'logs'/f'{name}.log'
        print(f'Start {name}',flush=True)
        began = time.time()
        timed_out = False
        with log.open('w') as stream:
            proc = subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                status = proc.wait(timeout=timeout or args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
                status = 124
        record = dict(name=name,backend=backend,ranks=ranks,command=cmd,exit_code=status,
            timed_out=timed_out,wall_seconds=time.time()-began,log=str(log),result=str(result) if result else None)
        if result and status == 0 and not result.is_file():
            record.update(exit_code=1,error='Worker exited without a result')
        launches.append(record)
        write(out/'launches.json',launches)
        print(f'End {name}: exit={record["exit_code"]}',flush=True)
        return record['exit_code'] == 0
    def configuration(n,p,trial,smoke=False):
        label = f'{"smoke" if smoke else "full"}-n{n}-p{p}-trial{trial}'
        data_root = out/'fixtures'/label
        fixture = data_root/f'n{n}'/f'{p}ranks'
        counts = dict(warmup=1,iterations=100,repeats=2) if smoke else {k:settings[k] for k in ('warmup','iterations','repeats')}
        result = out/'raw'/f'{label}-dolfinx.json'
        opts = ['--subdivisions',n,'--trial',trial,'--output',data_root,'--result',result]
        for k,v in counts.items(): opts.extend([f'--{k}',v])
        if baseline:
            original = baseline/'raw'/f'{label}-dolfinx.json'
            native = json.loads(original.read_text())
            fixture = Path(native['dataset'])
            write(result,native)
            launches.append(dict(name=f'{label}-dolfinx',backend='dolfinx',ranks=p,exit_code=0,
                reused_result=True,source=str(original),result=str(result)))
            write(out/'launches.json',launches)
            native_ok = True
        else:
            native_ok = launch('dolfinx',p,f'{label}-dolfinx',opts,result)
        if not native_ok:
            for backend in ('mpi4jax','sharding'):
                launches.append(dict(name=f'{label}-{backend}',backend=backend,ranks=p,
                    exit_code=None,skipped_reason='DOLFINx fixture generation failed'))
            write(out/'launches.json',launches)
            return False
        success = True
        order = ('mpi4jax','sharding') if trial%2 == 0 else ('sharding','mpi4jax')
        for backend in order:
            result = out/'raw'/f'{label}-{backend}.json'
            ok = launch(backend,p,f'{label}-{backend}',[fixture,'--backend',backend,'--trial',trial,'--result',result],result)
            success = success and ok
        return success
    smoke_passed = True
    for p in settings['ranks']:
        smoke_passed = configuration(4,p,0,smoke=True) and smoke_passed
    if not args.skip_regression and not baseline:
        launch('regression',4,'regression',[],timeout=900)
    if smoke_passed and not args.smoke_only:
        for trial in range(settings['trials']):
            for n in settings['sizes']:
                for p in (settings['ranks'] if trial%2 == 0 else settings['ranks'][::-1]):
                    configuration(n,p,trial)
    elif not smoke_passed:
        print('Full campaign withheld because smoke validation failed.',flush=True)
    write(out/'completion.json',dict(smoke_passed=smoke_passed,
        full_campaign_attempted=smoke_passed and not args.smoke_only,
        failures=[r['name'] for r in launches if r.get('exit_code') != 0]))
    subprocess.run(['bash','--noprofile','--norc',str(HERE/'launch.sh'),'summary','1',str(out)],env=env,check=True)
    print(f'Results: {out}',flush=True)
    if any(r.get('exit_code') != 0 for r in launches):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
