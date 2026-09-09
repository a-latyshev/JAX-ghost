"""Bound MPI launches, including their local process group."""
import os
from pathlib import Path
import signal
import subprocess


def launch(command, log, timeout):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.open('w') as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                env=env, stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            status = proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise RuntimeError(f'Launch interrupted or timed out: {log}')
    if status:
        raise RuntimeError(f'Launch failed ({status}); see {log}')
