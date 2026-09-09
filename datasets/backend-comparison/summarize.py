"""Recompute timings, audit matched fixtures/placement and plot fresh results only."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st

BACKENDS = ('dolfinx','mpi4jax','sharding')


def total_timing_fields(group):
    """Compilation once plus one warmed batch, paired within each launch.

    Each separately synchronized phase contributes its maximum rank duration.
    First execution and other setup are deliberately outside this derived total.
    """
    totals = []
    counts = {d['settings']['iterations'] for d in group}
    assert len(counts) == 1
    iterations, = counts
    for d in group:
        batch = d['timing']['median_seconds_per_matvec'] * iterations
        overhead = (sum(d['phases'][phase]['max_seconds']
                        for phase in ('lowering', 'compilation'))
                    if d['backend'] != 'dolfinx' else 0.0)
        totals.append(overhead + batch)
    median = st.median(totals)
    return dict(total_ms_per_batch=median*1e3,
                total_amortized_us_per_matvec=median/iterations*1e6,
                min_trial_total_ms=min(totals)*1e3,
                max_trial_total_ms=max(totals)*1e3)


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
        expected_cpus = ([[c] for c in manifest['cpu_ids']] if d['backend']=='dolfinx'
                         else manifest.get('jax_cpu_sets', [[c] for c in manifest['cpu_ids']]))
        assert [p['cpu_affinity'] for p in d['placement']] == expected_cpus[:d['ranks']]
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
    baseline_ranks = min(manifest['settings']['ranks'])
    rows = []
    for n in manifest['settings']['sizes']:
        for p in manifest['settings']['ranks']:
            for backend in BACKENDS:
                group = [d for d in records if not d['smoke'] and d['subdivisions']==n and d['ranks']==p and d['backend']==backend]
                row = dict(subdivisions=n,dofs=(n+1)**3,ranks=p,backend=backend,cpus=p*(1 if backend=='dolfinx' else manifest.get('jax_cpus_per_rank',1)),
                           gpus=0 if backend=='dolfinx' else p,successful_trials=len(group),
                           status='complete' if len(group)==manifest['settings']['trials'] else 'incomplete')
                if group:
                    times = [d['timing']['median_seconds_per_matvec'] for d in group]
                    med = st.median(times)
                    row.update(us_per_matvec=med*1e6,min_trial_us=min(times)*1e6,max_trial_us=max(times)*1e6,
                               ms_per_batch=med*manifest['settings']['iterations']*1e3)
                    row.update(total_timing_fields(group))
                    if backend != 'dolfinx':
                        for name in ('lowering','compilation','first_execution'):
                            row[name+'_ms'] = 1e3*st.median(d['phases'][name]['max_seconds'] for d in group)
                        # Rank-wise sum retains the critical rank of the combined compile phase.
                        row['lowering_plus_compilation_ms'] = 1e3*st.median(max(a+b for a,b in zip(
                            d['phases']['lowering']['rank_seconds'],d['phases']['compilation']['rank_seconds'])) for d in group)
                rows.append(row)
    for r in rows:
        if r['status'] != 'complete': continue
        base = next(x for x in rows if x['subdivisions']==r['subdivisions'] and x['backend']==r['backend'] and x['ranks']==baseline_ranks)
        if base['status'] == 'complete':
            r['speedup'] = base['us_per_matvec']/r['us_per_matvec']
            r['efficiency'] = baseline_ranks*r['speedup']/r['ranks']
            r['scaling_baseline_ranks'] = baseline_ranks
        for other, name in [('dolfinx','dolfinx_over_backend'),('mpi4jax','mpi4jax_over_backend')]:
            b = next(x for x in rows if x['subdivisions']==r['subdivisions'] and x['backend']==other and x['ranks']==r['ranks'])
            if b['status'] == 'complete':
                r[name] = b['us_per_matvec']/r['us_per_matvec']
                r[name+'_total'] = b['total_ms_per_batch']/r['total_ms_per_batch']
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    lines=['# Matvec backend comparison','',
        f"Allocation {manifest['job']} on {manifest['hostname']}. DOLFINx: one physical CPU core per rank. JAX: {manifest.get('jax_cpus_per_rank',1)} physical CPU cores and one distinct GPU per rank.",
        ('DOLFINx measurements and fixtures are reused unchanged from '+manifest['baseline']+'. Only JAX was rerun.' if manifest.get('baseline') else 'All backends were measured in this campaign.'),
        '', 'Float64 3D Poisson; fixed A and x; 100 Python-driven `y += A*x` calls per batch, including halo exchange. No outer-loop JIT.',
        'MPI.Wtime measures maximum rank elapsed time, including final GPU/effect synchronization. Setup, compilation, uploads, resets, validation and I/O are excluded.',
        'Ten warmups and ten batches per launch; three fresh launches per configuration. Times below are median launch medians.', '',
        f'| DoFs | Ranks | Backend | CPU cores | GPUs | Trials | µs/matvec | ms/100 calls | Speedup vs {baseline_ranks} ranks |',
        '| ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        value = lambda k: f'{r[k]:.2f}' if k in r and r['status']=='complete' else '—'
        lines.append(f"| {r['dofs']:,} | {r['ranks']} | {r['backend']} | {r['cpus']} | {r['gpus']} | {r['successful_trials']}/{manifest['settings']['trials']} | {value('us_per_matvec')} | {value('ms_per_batch')} | {value('speedup')} |")
    lines += ['', '## Compilation plus one 100-call batch', '',
        'Derived total = tracing/lowering + compilation + one warmed 100-call batch. DOLFINx has no JAX compilation cost, so its total equals its batch time. First-execution startup, assembly, uploads, warmups and other setup remain excluded; this is not a measured cold-run wall time.',
        'Costs are added within each launch before taking the median across launches. Each separately timed compilation phase uses its maximum rank duration. Dashed curves add these totals to the runtime and DOLFINx-ratio plots; the per-call plot divides the total by 100.', '',
        '| DoFs | Ranks | Backend | Main batch ms | Compilation + batch ms | DOLFINx / total |',
        '| ---: | ---: | :--- | ---: | ---: | ---: |']
    for r in rows:
        if r['status'] != 'complete': continue
        ratio = f"{r['dolfinx_over_backend_total']:.2f}" if 'dolfinx_over_backend_total' in r else '—'
        lines.append(f"| {r['dofs']:,} | {r['ranks']} | {r['backend']} | {r['ms_per_batch']:.2f} | {r['total_ms_per_batch']:.2f} | {ratio} |")
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
        'DOLFINx fixtures are produced before JAX replay; JAX launch order alternates by trial. Partial configurations are retained in CSV but excluded from comparisons and plots.', '', '## Launch failures', '']
    failed = [x for x in launches if x.get('exit_code') != 0]
    lines += [f"- {x['name']}: exit={x.get('exit_code')}; {x.get('skipped_reason',x.get('log',''))}" for x in failed] or ['None.']
    lines += ['', 'Raw records: `raw/`; fixture data: `fixtures/`; placement/source manifest: `manifest.json`; every launch and failure: `launches.json`.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    complete = [r for r in rows if r['status']=='complete']
    if complete:
        plot(out,complete,manifest.get('jax_cpus_per_rank',1), manifest['settings']['sizes'], manifest['settings']['ranks'])
    print(out/'REPORT.md')


def plot(out,rows,jax_cpus=1,sizes=None,ranks=None):
    sizes = sorted(sizes or {r['subdivisions'] for r in rows})
    ranks = sorted(ranks or {r['ranks'] for r in rows})
    baseline_ranks = min(ranks)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for field, ylabel, name, log in [
        ('us_per_matvec','µs per call (100-call batch)','matvec',True),
        ('ms_per_batch','ms per 100 calls','batch',True),
        ('speedup',f'Speedup relative to {baseline_ranks} ranks','scaling',False),
        ('dolfinx_over_backend','DOLFINx time / backend time','ratios',True)]:
        total_field = {'us_per_matvec':'total_amortized_us_per_matvec',
                       'ms_per_batch':'total_ms_per_batch',
                       'dolfinx_over_backend':'dolfinx_over_backend_total'}.get(field)
        fig,grid=plt.subplots(1,len(sizes),squeeze=False,figsize=(max(8,6*len(sizes)),5.2 if total_field else 4.8))
        axes=grid[0]
        for ax,n in zip(axes,sizes):
            for index,backend in enumerate(BACKENDS):
                rr=[r for r in rows if r['subdivisions']==n and r['backend']==backend and field in r]
                label = backend + (' · main' if total_field else '')
                ax.plot([r['ranks'] for r in rr],[r[field] for r in rr],'o-',color=f'C{index}',label=label)
                if total_field and backend != 'dolfinx':
                    tt=[r for r in rr if total_field in r]
                    ax.plot([r['ranks'] for r in tt],[r[total_field] for r in tt],
                            's--',color=f'C{index}',label=backend+' · main + compilation')
            ax.set(title=f'{(n+1)**3:,} DoFs',xlabel=f'MPI ranks (JAX: {jax_cpus} CPU cores + 1 GPU/rank)',ylabel=ylabel,xticks=ranks)
            if field == 'speedup':
                ax.plot(ranks,[p/baseline_ranks for p in ranks],'k:',label='ideal')
            if log: ax.set_yscale('log')
            ax.grid(alpha=.2)
            if not total_field: ax.legend(fontsize=8)
        if total_field:
            handles,labels=axes[0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,fontsize=8)
            fig.text(.5,.01,'Total = lowering + compilation + one warmed 100-call batch; first-execution startup excluded.',
                     ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.18 if total_field else 0,1,1))
        for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(len(sizes),2,squeeze=False,figsize=(10,3.5*len(sizes)))
    for i,n in enumerate(sizes):
        for j,backend in enumerate(('mpi4jax','sharding')):
            ax=axes[i,j];rr=[r for r in rows if r['subdivisions']==n and r['backend']==backend]
            for phase in ('lowering','compilation','first_execution'):
                ax.plot([r['ranks'] for r in rr],[r[phase+'_ms'] for r in rr],'o-',label=phase)
            ax.set(title=f'{backend} · {(n+1)**3:,} DoFs',xlabel='GPUs / ranks',ylabel='ms',yscale='log',xticks=ranks);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'startup.{ext}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    summarize(parser.parse_args().output.resolve())
