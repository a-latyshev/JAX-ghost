"""Sequential, repeated strong-scaling jobs with bounded wall time."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 1024])
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--local-cpu', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cpu', JAX_NUM_CPU_DEVICES='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
               TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1',
               XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
    if args.local_cpu:
        for key in list(env):
            if key.lower() in ('http_proxy','https_proxy','all_proxy'):
                del env[key]
    settings = dict(vars(args)); settings['output'] = str(output)
    (output/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    for trial in range(args.trials):
        ranks = args.ranks if trial % 2 == 0 else args.ranks[::-1]
        sizes = args.sizes if trial % 2 == 0 else args.sizes[::-1]
        for n in sizes:
            for rank in ranks:
                name = f'n{n}-p{rank}-trial{trial}'
                command = [args.mpirun, '-n', str(rank), sys.executable, '-u', str(HERE/'measure.py'),
                           '--subdivisions', str(n), '--trial', str(trial),
                           '--iterations', str(args.iterations), '--repeats', str(args.repeats),
                           '--output', str(output/'raw'/f'{name}.json')]
                if args.local_cpu:
                    command.append('--local-cpu')
                print('Running',name,flush=True)
                with (output/f'{name}.log').open('w') as log:
                    proc = subprocess.Popen(command, env=env, cwd=HERE.parents[1],
                                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        status=proc.wait(timeout=args.timeout)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt):
                        os.killpg(proc.pid,signal.SIGKILL); proc.wait(); raise
                if status:
                    raise SystemExit(f'{name} failed ({status}); see {output/name}.log')
                last = (output/f'{name}.log').read_text().splitlines()
                print(last[-1] if last else 'Completed', flush=True)
    print('Results:',output,flush=True)


if __name__ == '__main__':
    main()
