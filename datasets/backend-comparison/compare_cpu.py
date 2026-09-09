"""Compare fresh seven-core JAX timings against the preserved one-core campaign."""
import csv
import json
from pathlib import Path
import sys


def compare(out):
    manifest=json.loads((out/'manifest.json').read_text())
    baseline=Path(manifest['baseline'])
    old=list(csv.DictReader((baseline/'summary.csv').open()))
    new=list(csv.DictReader((out/'summary.csv').open()))
    key=lambda r:(r['subdivisions'],r['ranks'],r['backend'])
    original={key(r):r for r in old}
    paired=[]
    for r in new:
        b=original[key(r)]
        if r['backend']=='dolfinx':
            assert r['ms_per_batch']==b['ms_per_batch']
            continue
        row=dict(dofs=r['dofs'],ranks=r['ranks'],backend=r['backend'],
                 one_core_trials=b['successful_trials'],seven_core_trials=r['successful_trials'])
        if b['status']=='complete' and r['status']=='complete':
            for field in ('us_per_matvec','total_ms_per_batch','lowering_ms','compilation_ms','first_execution_ms'):
                row['one_core_'+field]=float(b[field])
                row['seven_core_'+field]=float(r[field])
            row['main_speedup']=float(b['us_per_matvec'])/float(r['us_per_matvec'])
            row['total_speedup']=float(b['total_ms_per_batch'])/float(r['total_ms_per_batch'])
        paired.append(row)
    with (out/'cpu-comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in paired for k in r)))
        writer.writeheader();writer.writerows(paired)
    lines=['# One versus seven CPU cores per GPU rank','',
           'Both JAX backends use one distinct V100 per rank. The seven-core masks include each original CPU core and do not overlap. The node, GPU assignments, fixtures, dtype, MPI.Wtime protocol and Python loop are unchanged. Numerical-library thread limits remain one; extra cores are available to JAX/runtime helper threads.',
           'DOLFINx is the unchanged saved one-core-per-rank baseline. The two JAX campaigns were measured at different times on the same node.', '',
           '| DoFs | GPU ranks | Backend | 1 core µs/matvec | 7 cores µs/matvec | Speedup |',
           '| ---: | ---: | :--- | ---: | ---: | ---: |']
    for r in paired:
        vals=[f"{r[k]:.2f}" if k in r else 'unavailable' for k in ('one_core_us_per_matvec','seven_core_us_per_matvec','main_speedup')]
        lines.append(f"| {int(r['dofs']):,} | {r['ranks']} | {r['backend']} | "+' | '.join(vals)+' |')
    lines += ['', 'Speedup = one-core time / seven-core time. Unavailable pairs are retained in CSV; see the run report for failures.',
              'Total in the figure means lowering + compilation + one warmed 100-call batch, excluding first-execution startup and other setup.', '',
              '![CPU allocation comparison](cpu-comparison.png)', '',
              '[Seven-core report](REPORT.md) · [Detailed comparison CSV](cpu-comparison.csv)', '']
    (out/'CPU-COMPARISON.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,n in enumerate(('46','99')):
        for line,(field,ylabel) in enumerate((('us_per_matvec','µs per warmed matvec'),('total_ms_per_batch','ms: compilation + 100 calls'))):
            ax=axes[line,col]
            native=[r for r in new if r['subdivisions']==n and r['backend']=='dolfinx' and r['status']=='complete']
            ax.plot([int(r['ranks']) for r in native],[float(r[field]) for r in native],'o-',color='C0',label='DOLFINx · 1 CPU/rank (saved)')
            for i,backend in enumerate(('mpi4jax','sharding'),1):
                for source,cores,style in ((old,1,'s--'),(new,7,'o-')):
                    rr=[r for r in source if r['subdivisions']==n and r['backend']==backend and r['status']=='complete']
                    ax.plot([int(r['ranks']) for r in rr],[float(r[field]) for r in rr],style,color=f'C{i}',label=f'{backend} · {cores} CPU/GPU rank')
            ax.set(title=f'{(int(n)+1)**3:,} DoFs',xlabel='MPI ranks / GPUs for JAX',ylabel=ylabel,xticks=[1,2,3,4],yscale='log');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1))
    for ext in ('png','pdf'):fig.savefig(out/f'cpu-comparison.{ext}',dpi=180)
    plt.close(fig)
    print(out/'CPU-COMPARISON.md')


if __name__=='__main__':compare(Path(sys.argv[1]).resolve())
