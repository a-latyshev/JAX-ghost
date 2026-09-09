"""Sequential fresh MPI launches at fixed global size, then SVG scaling plots."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from runner import launch
from plot import summarize

HERE = Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launcher',default='mpirun -n {ranks} --bind-to core',
                   help='command template with {ranks}; shell operators are not supported')
    p.add_argument('--gpus-per-node',type=int,default=None,help='Reserved GPUs per node, required for {nodes} in launcher')
    p.add_argument('--data',type=Path,default=HERE/'data')
    p.add_argument('--ranks',type=int,nargs='+',default=[1,2,4,8])
    p.add_argument('--sizes',type=int,nargs='+',default=[99])
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--timeout',type=int,default=600)
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    if 1 not in a.ranks or len(set(a.ranks))!=len(a.ranks) or any(n not in range(1,9) for n in a.ranks):
        raise ValueError('Choose unique rank counts 1–8, including 1 for strong scaling')
    if len(set(a.sizes))!=len(a.sizes) or min(a.trials,a.repeats,a.timeout,*a.sizes)<1 or a.warmup<0:
        raise ValueError('Invalid sizes or counts')
    if '{nodes}' in a.launcher and (a.gpus_per_node is None or a.gpus_per_node<1):
        raise ValueError('A {nodes} launcher requires positive --gpus-per-node')
    if '{ranks}' not in a.launcher:
        raise ValueError('Launcher must contain {ranks}')
    for size in a.sizes:
        for ranks in a.ranks:
            if not (a.data/f'n{size}/{ranks}ranks/metadata.json').exists():
                raise ValueError(f'Missing fixture n{size}/{ranks}ranks')
    out=a.output or HERE/'results'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    settings=dict(vars(a));settings.update(data=str(a.data.resolve()),output=str(out.resolve()),iterations=100)
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    for trial in range(a.trials):
        for size in a.sizes:
            for ranks in (a.ranks if trial%2==0 else a.ranks[::-1]):
                name=f'n{size}-p{ranks}-t{trial}'
                cmd=shlex.split(a.launcher.format(ranks=ranks,nodes=(ranks+a.gpus_per_node-1)//a.gpus_per_node if a.gpus_per_node else 1))+[sys.executable,str(HERE/'worker.py'),
                    '--data',str(a.data.resolve()),'--subdivisions',str(size),'--trial',str(trial),
                    '--warmup',str(a.warmup),'--repeats',str(a.repeats),'--output',str((out/'raw'/f'{name}.json').resolve())]
                launch(cmd,out/'logs'/f'{name}.log',a.timeout)
                print('Completed',name,flush=True)
    summarize(out)
    print('Results:',out,flush=True)


if __name__=='__main__':
    main()
