"""Run native JAX distributed tests separately from mpi4jax tests, with timeouts."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mpirun', default='mpirun')
    parser.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--local-cpu', action='store_true')
    parser.add_argument('--example', action='store_true')
    args = parser.parse_args()
    env = os.environ.copy()
    env['JAX_PLATFORMS'] = 'cpu'
    env['JAX_NUM_CPU_DEVICES'] = '1'
    env.setdefault('OMP_NUM_THREADS', '1')
    if args.local_cpu:
        for key in list(env):
            if key.lower() in ('http_proxy', 'https_proxy', 'all_proxy'):
                del env[key]
    root = Path(__file__).resolve().parents[1]
    for n in args.ranks:
        command = [args.mpirun, '-n', str(n), sys.executable, '-u', 'scripts/sharding_worker.py']
        command += ['--local-cpu'] if args.local_cpu else []
        command += ['--example'] if args.example else []
        print('Running:', ' '.join(command), flush=True)
        proc = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
        try:
            status = proc.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise SystemExit('Sharding job interrupted or timed out')
        if status:
            raise SystemExit(status)


if __name__ == '__main__':
    main()
