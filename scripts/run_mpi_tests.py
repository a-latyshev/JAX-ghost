"""Launch MPI checks with a timeout per job, including all child processes."""

import argparse
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpiexec", default="mpiexec", help="MPI launcher executable")
    parser.add_argument("--timeout", type=float, default=120, help="seconds per MPI job")
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_NUM_CPU_DEVICES"] = "1"
    env.setdefault("OMP_NUM_THREADS", "1")
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    jobs = [(2, ["tests/mpi_smoke.py"])]
    jobs.extend((n, ["-m", "unittest", "discover", "-s", "tests", "-p", "test_mpi.py", "-v"])
                for n in args.ranks)
    for n, script in jobs:
        command = [args.mpiexec, "-n", str(n), sys.executable, *script]
        print("Running:", shlex.join(command), flush=True)
        proc = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
        try:
            status = proc.wait(timeout=args.timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise SystemExit(f"MPI job interrupted or exceeded {args.timeout} seconds")
        if status:
            raise SystemExit(status)


if __name__ == "__main__":
    main()
