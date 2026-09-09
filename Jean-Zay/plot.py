"""Validate a complete campaign and write CSV, Markdown and SVG; standard library only."""
import csv
from html import escape
import json
import math
from pathlib import Path
import statistics as st
import sys


def aggregate(folder):
    folder=Path(folder)
    settings=json.loads((folder/'settings.json').read_text())
    expected={(n,p,t) for n in settings['sizes'] for p in settings['ranks'] for t in range(settings['trials'])}
    records=[json.loads(p.read_text()) for p in sorted((folder/'raw').glob('*.json'))]
    actual={(r['subdivisions'],r['ranks'],r['trial']) for r in records}
    if actual!=expected or len(records)!=len(expected):
        raise ValueError('Incomplete or duplicated campaign; no partial scaling plot is produced')
    if not records or 1 not in settings['ranks']:
        raise ValueError('A one-rank baseline is required')
    first=records[0]
    signature=lambda r:(r['source_hashes'],r['versions'],r['environment']['mpi_library'],
                        r['environment']['numpy'],r['environment']['mpi4py'],
                        sorted({p['gpu_name'] for p in r['placement']}),
                        sorted({len(p['cpu_affinity']) for p in r['placement']}))
    groups={}
    for r in records:
        wanted=dict(iterations=100,repeats=settings['repeats'],warmup=settings['warmup'],loop='python',dtype='float64')
        if not r['correctness_passed'] or r['mode']!='time' or r['settings']!=wanted or signature(r)!=signature(first):
            raise ValueError('Mixed settings, versions, source, GPU models, or failed validation')
        if r['timing']['timer']!='MPI.Wtime': raise ValueError('Wrong timer')
        ss=r['timing']['rank_seconds']
        if len(ss)!=settings['repeats'] or any(len(s)!=r['ranks'] or any(not math.isfinite(v) or v<=0 for v in s) for s in ss):
            raise ValueError('Invalid timing samples')
        if not r['validation'] or any(e['error_l2']>1e-12+1e-10*e['reference_l2'] or not math.isfinite(e['error_l2']) for e in r['validation']):
            raise ValueError('Invalid numerical validation')
        median=st.median(max(s)/100 for s in ss)
        if not math.isclose(median,r['timing']['median_seconds_per_matvec'],rel_tol=1e-12):
            raise ValueError('Stored median disagrees with samples')
        groups.setdefault((r['subdivisions'],r['ranks']),[]).append((r,median))
    rows=[]
    for (n,p), group in sorted(groups.items()):
        r0=group[0][0]
        placement=lambda r:[(p['hostname'],p['gpu_uuid'],tuple(p['cpu_affinity'])) for p in r['placement']]
        if any(r['fixture_fingerprints']!=r0['fixture_fingerprints'] or placement(r)!=placement(r0) or r['global_dofs']!=r0['global_dofs'] for r,_ in group):
            raise ValueError('Fixture or placement changed between trials')
        tt=[v*1e6 for _,v in group]
        rows.append(dict(subdivisions=n,global_dofs=r0['global_dofs'],ranks=p,
            median_us=st.median(tt),min_us=min(tt),max_us=max(tt),trials=len(tt)))
    for r in rows:
        base=next(b for b in rows if b['subdivisions']==r['subdivisions'] and b['ranks']==1)
        if base['global_dofs']!=r['global_dofs']: raise ValueError('Global workload changed with rank count')
        r.update(speedup=base['median_us']/r['median_us'],efficiency=base['median_us']/r['median_us']/r['ranks'])
        r['speedup_min']=base['min_us']/r['max_us'] if r['ranks']!=1 else 1.
        r['speedup_max']=base['max_us']/r['min_us'] if r['ranks']!=1 else 1.
    return rows,settings


def draw_svg(rows):
    sizes=sorted({r['subdivisions'] for r in rows})
    width,height=1140,80+310*len(sizes)
    out=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:sans-serif;fill:#20252b;font-size:12px}</style>',
         '<text x="30" y="28" style="font-size:20px">Sharded GPU matvec strong scaling · fixed global workload</text>',
         '<text x="30" y="50">Python loop · float64 · halo exchange included · compilation excluded</text>']
    for row_index,n in enumerate(sizes):
        rr=[r for r in rows if r['subdivisions']==n]
        for col,kind in enumerate(('time','speedup','efficiency')):
            left,top=65+375*col,100+310*row_index
            w,h=290,190
            vals=[]
            for r in rr:
                if kind=='time': v,lo,hi=r['median_us'],r['min_us'],r['max_us']
                else:
                    div=r['ranks'] if kind=='efficiency' else 1
                    v,lo,hi=r['speedup']/div,r['speedup_min']/div,r['speedup_max']/div
                vals.append((r['ranks'],v,lo,hi))
            maxx=max(r['ranks'] for r in rr)
            ymax=max([v[3] for v in vals]+([maxx] if kind=='speedup' else [1] if kind=='efficiency' else []))*1.1
            X=lambda p:left+(p-1)/max(1,maxx-1)*w
            Y=lambda v:top+h-v/ymax*h
            title={'time':'Time (µs/matvec)','speedup':'Speedup T(1)/T(p)','efficiency':'Efficiency T(1)/(p T(p))'}[kind]
            out.append(f'<text x="{left}" y="{top-15}">{rr[0]["global_dofs"]:,} DoFs · {escape(title)}</text>')
            for tick in range(5):
                v=ymax*tick/4;y=Y(v)
                out.extend([f'<path d="M {left} {y} h {w}" stroke="#ddd"/>',f'<text x="{left-8}" y="{y+4}" text-anchor="end">{v:.2g}</text>'])
            for p,_,_,_ in vals:
                out.append(f'<text x="{X(p)}" y="{top+h+20}" text-anchor="middle">{p}</text>')
            if kind!='time':
                ideal=[(p,p if kind=='speedup' else 1) for p,_,_,_ in vals]
                points=' '.join(f'{X(p)},{Y(v)}' for p,v in ideal)
                out.append(f'<polyline points="{points}" fill="none" stroke="#999" stroke-dasharray="4 4"/>')
            points=' '.join(f'{X(p)},{Y(v)}' for p,v,_,_ in vals)
            out.append(f'<polyline points="{points}" fill="none" stroke="#2166ac" stroke-width="2"/>')
            for p,v,lo,hi in vals:
                x=X(p)
                out.append(f'<path d="M {x} {Y(lo)} V {Y(hi)} M {x-4} {Y(lo)} h 8 M {x-4} {Y(hi)} h 8" stroke="#2166ac"/>')
                out.append(f'<circle cx="{x}" cy="{Y(v)}" r="4" fill="#2166ac"/>')
            out.append(f'<text x="{left+w/2}" y="{top+h+43}" text-anchor="middle">MPI ranks / GPUs</text>')
    out.append(f'<text x="30" y="{height-15}">Bars: launch-median ranges (scaling bounds combine baseline and point ranges); dashed: ideal scaling.</text></svg>')
    return '\n'.join(out)


def summarize(folder):
    folder=Path(folder)
    rows,settings=aggregate(folder)
    with (folder/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (folder/'scaling.svg').write_text(draw_svg(rows))
    table=['| DoFs | GPUs | µs/matvec | Launch range µs | Speedup | Efficiency |','| ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        table.append(f"| {r['global_dofs']:,} | {r['ranks']} | {r['median_us']:.2f} | {r['min_us']:.2f}–{r['max_us']:.2f} | {r['speedup']:.2f} | {100*r['efficiency']:.1f}% |")
    (folder/'REPORT.md').write_text('# Sharded strong scaling\n\n'+ '\n'.join(table)+f'''\n
{settings['trials']} independent launches per point; {settings['repeats']} batches
of 100 Python-driven accumulating matvecs after {settings['warmup']} warmups.
MPI.Wtime samples include GPU completion; maximum rank time / 100, then median
across batches and launches. A and x remain fixed. Halo exchange is included.
Setup, compilation, transfers, reset, validation and I/O are excluded.
Raw JSON records exact GPU placement, CPU affinity, environment and fixture/source
hashes. Compare runs only with the placement and node topology in mind.

![Strong scaling](scaling.svg)
''')
    print('\n'.join(table))


if __name__=='__main__':
    summarize(sys.argv[1])
