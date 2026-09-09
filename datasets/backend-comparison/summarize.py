"""Recompute timings, audit matched fixtures/placement and plot fresh results only."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st

BACKENDS = ('dolfinx','mpi4jax','sharding')


def summarize(out):
    manifest = json.loads((out/'manifest.json').read_text())
    launches = json.loads((out/'launches.json').read_text())
    records = []
    audit = []
    for launch in launches:
        if launch.get('exit_code') != 0 or not launch.get('result'):
            continue
        d = json.loads(Path(launch['result']).read_text())
        d['smoke'] = launch['name'].startswith('smoke-')
        samples = d['timing']['rank_seconds']
        assert d['correctness_passed'] and d['timing']['timer'] == 'MPI.Wtime'
        assert len(samples) == d['settings']['repeats']
        assert all(len(s) == d['ranks'] and all(t > 0 for t in s) for s in samples)
        med = st.median(max(s)/d['settings']['iterations'] for s in samples)
        assert abs(med-d['timing']['median_seconds_per_matvec']) < 1e-12
        assert len(d['placement']) == d['ranks']
        assert [p['cpu_affinity'] for p in d['placement']] == [[c] for c in manifest['cpu_ids'][:d['ranks']]]
        assert all(p['hostname'] == manifest['hostname'] for p in d['placement'])
        gpus = [p['gpu_uuid'] for p in d['placement']]
        assert gpus == ([None]*d['ranks'] if d['backend']=='dolfinx' else manifest['gpu_uuids'][:d['ranks']])
        records.append(d)
    for key in sorted({(d['subdivisions'],d['ranks'],d['trial']) for d in records}):
        group = [d for d in records if (d['subdivisions'],d['ranks'],d['trial']) == key]
        assert len({d['backend'] for d in group}) == len(group)
        reference = group[0]
        for d in group[1:]:
            assert d['fingerprints'] == reference['fingerprints']
            assert d['dataset'] == reference['dataset'] and d['settings'] == reference['settings']
        jax = [d for d in group if d['backend'] != 'dolfinx']
        if len(jax) == 2:
            assert jax[0]['versions']['jax'] == jax[1]['versions']['jax']
            assert jax[0]['versions']['jaxlib'] == jax[1]['versions']['jaxlib']
        audit.append(dict(subdivisions=key[0],ranks=key[1],trial=key[2],backends=[d['backend'] for d in group],matched=True))
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    rows = []
    for n in manifest['settings']['sizes']:
        for p in manifest['settings']['ranks']:
            for backend in BACKENDS:
                group = [d for d in records if not d['smoke'] and d['subdivisions']==n and d['ranks']==p and d['backend']==backend]
                row = dict(subdivisions=n,dofs=(n+1)**3,ranks=p,backend=backend,cpus=p,
                           gpus=0 if backend=='dolfinx' else p,successful_trials=len(group),
                           status='complete' if len(group)==manifest['settings']['trials'] else 'incomplete')
                if group:
                    times = [d['timing']['median_seconds_per_matvec'] for d in group]
                    med = st.median(times)
                    row.update(us_per_matvec=med*1e6,min_trial_us=min(times)*1e6,max_trial_us=max(times)*1e6,
                               ms_per_batch=med*manifest['settings']['iterations']*1e3)
                    if backend != 'dolfinx':
                        for name in ('lowering','compilation','first_execution'):
                            row[name+'_ms'] = 1e3*st.median(d['phases'][name]['max_seconds'] for d in group)
                        # Rank-wise sum retains the critical rank of the combined compile phase.
                        row['lowering_plus_compilation_ms'] = 1e3*st.median(max(a+b for a,b in zip(
                            d['phases']['lowering']['rank_seconds'],d['phases']['compilation']['rank_seconds'])) for d in group)
                rows.append(row)
    for r in rows:
        if r['status'] != 'complete': continue
        base = next(x for x in rows if x['subdivisions']==r['subdivisions'] and x['backend']==r['backend'] and x['ranks']==1)
        if base['status'] == 'complete':
            r['speedup'] = base['us_per_matvec']/r['us_per_matvec']
            r['efficiency'] = r['speedup']/r['ranks']
        for other, name in [('dolfinx','dolfinx_over_backend'),('mpi4jax','mpi4jax_over_backend')]:
            b = next(x for x in rows if x['subdivisions']==r['subdivisions'] and x['backend']==other and x['ranks']==r['ranks'])
            if b['status'] == 'complete': r[name] = b['us_per_matvec']/r['us_per_matvec']
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# Matvec backend comparison','',
        f"Allocation {manifest['job']} on {manifest['hostname']}. One physical CPU core per rank; JAX additionally uses one distinct GPU per rank.",
        '', 'Float64 3D Poisson; fixed A and x; 100 Python-driven `y += A*x` calls per batch, including halo exchange. No outer-loop JIT.',
        'MPI.Wtime measures maximum rank elapsed time, including final GPU/effect synchronization. Setup, compilation, uploads, resets, validation and I/O are excluded.',
        'Ten warmups and ten batches per launch; three fresh launches per configuration. Times below are median launch medians.', '',
        '| DoFs | Ranks | Backend | CPU cores | GPUs | Trials | µs/matvec | ms/100 calls | Speedup vs 1 rank |',
        '| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        value = lambda k: f'{r[k]:.2f}' if k in r and r['status']=='complete' else '—'
        lines.append(f"| {r['dofs']:,} | {r['ranks']} | {r['backend']} | {r['cpus']} | {r['gpus']} | {r['successful_trials']}/3 | {value('us_per_matvec')} | {value('ms_per_batch')} | {value('speedup')} |")
    lines += ['', '## JAX startup phases', '',
        'Separate MPI.Wtime measurements, maximum across ranks; median across launches. First execution includes runtime initialization remaining after explicit compilation. Compilation caching is disabled.', '',
        '| DoFs | Ranks | Backend | Lowering ms | Compilation ms | First execution ms |',
        '| ---: | ---: | :--- | ---: | ---: | ---: |']
    for r in rows:
        if r['backend']=='dolfinx' or r['status']!='complete':continue
        lines.append(f"| {r['dofs']:,} | {r['ranks']} | {r['backend']} | {r['lowering_ms']:.2f} | {r['compilation_ms']:.2f} | {r['first_execution_ms']:.2f} |")
    lines += ['', '## Audit and limitations','',
        'Successful paired launches passed exact fixture fingerprints, CPU/GPU placement and JAX-version checks; timing medians were recomputed from raw rank durations. Numerical checks use global owned-entry L2 error <= 1e-12 + 1e-10 * reference norm.',
        'DOLFINx refreshes input ghosts in place. JAX preserves inputs. DOLFINx overlaps halo exchange with local work; sharding retains its existing padded representation and collective implementation. These backend behaviors are included.',
        'Both JAX backends have the same host-core allowance and GPU assignment. They use different installed MPI/runtime stacks; this is a comparison of those stacks, not hardware-independent backend efficiency.',
        'DOLFINx runs first to produce each trial fixture; JAX launch order alternates by trial. Partial configurations are retained in CSV but excluded from comparisons and plots.', '', '## Launch failures', '']
    failed = [x for x in launches if x.get('exit_code') != 0]
    lines += [f"- {x['name']}: exit={x.get('exit_code')}; {x.get('skipped_reason',x.get('log',''))}" for x in failed] or ['None.']
    lines += ['', 'Raw records: `raw/`; fixture data: `fixtures/`; placement/source manifest: `manifest.json`; every launch and failure: `launches.json`.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    complete = [r for r in rows if r['status']=='complete']
    if complete:
        plot(out,complete)
    print(out/'REPORT.md')


def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field, ylabel, name, log in [
        ('us_per_matvec','µs per matvec','matvec',True),
        ('ms_per_batch','ms per 100 calls','batch',True),
        ('speedup','Speedup relative to one rank','scaling',False),
        ('dolfinx_over_backend','DOLFINx time / backend time','ratios',True)]:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,n in zip(axes,(46,99)):
            for backend in BACKENDS:
                rr=[r for r in rows if r['subdivisions']==n and r['backend']==backend and field in r]
                ax.plot([r['ranks'] for r in rr],[r[field] for r in rr],'o-',label=backend)
            ax.set(title=f'{(n+1)**3:,} DoFs',xlabel='MPI ranks (CPU cores; also GPUs for JAX)',ylabel=ylabel,xticks=[1,2,3,4])
            if log: ax.set_yscale('log')
            ax.grid(alpha=.2);ax.legend()
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for i,n in enumerate((46,99)):
        for j,backend in enumerate(('mpi4jax','sharding')):
            ax=axes[i,j];rr=[r for r in rows if r['subdivisions']==n and r['backend']==backend]
            for phase in ('lowering','compilation','first_execution'):
                ax.plot([r['ranks'] for r in rr],[r[phase+'_ms'] for r in rr],'o-',label=phase)
            ax.set(title=f'{backend} · {(n+1)**3:,} DoFs',xlabel='GPUs / ranks',ylabel='ms',yscale='log',xticks=[1,2,3,4]);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'startup.{ext}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    summarize(parser.parse_args().output.resolve())
