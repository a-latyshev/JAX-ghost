"""Prepare all portable partitions in an allocated CPU/MPI environment."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
from runner import launch

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--launcher', default='mpirun -n {ranks}', help='MPI command template')
p.add_argument('--sizes', nargs='+', type=int, default=[4,99])
p.add_argument('--ranks', nargs='+', type=int, default=None, help='Default: 1–8 for n4; 1,2,4,8 for larger matrices')
p.add_argument('--output', type=Path, default=HERE/'data')
p.add_argument('--timeout', type=int, default=600)
a = p.parse_args()
for n in a.sizes:
    for ranks in (a.ranks if a.ranks is not None else (range(1,9) if n == 4 else (1,2,4,8))):
        if ranks not in range(1,9): raise ValueError('Use 1–8 ranks')
        folder = a.output/f'n{n}/{ranks}ranks'
        if (folder/'metadata.json').exists():
            print('Already prepared:',folder,flush=True)
            continue
        cmd = shlex.split(a.launcher.format(ranks=ranks)) + [sys.executable,str(HERE/'prepare.py'),
                '--subdivisions',str(n),'--output',str(a.output)]
        launch(cmd, a.output/f'prepare-n{n}-p{ranks}.log',a.timeout)
        print('Prepared:',folder,flush=True)
