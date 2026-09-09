"""Audit sharding timing trials and separately summarize memory diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import statistics as st


def dump_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def maximum(values):
    return max((v for v in values if v is not None),default=None)


def memory_summary(directory):
    states = []
    for path in directory.glob('rank-*.jsonl'):
        for line in path.read_text().splitlines():
            try: row = json.loads(line)
            except ValueError: continue
            if row['phase'] not in ('diagnostics','done'): states.append(row)
    samples = list(csv.DictReader((directory/'samples.csv').open())) if (directory/'samples.csv').exists() else []
    def number(row,key):
        return int(row[key]) if row.get(key) else None
    simultaneous = {}
    phase_rows = []
    for row in samples:
        value = number(row,'rss_bytes')
        if value is not None:
            simultaneous[row['timestamp']] = simultaneous.get(row['timestamp'],0)+value
    timestamps=sorted({float(r['timestamp']) for r in samples})
    intervals=[b-a for a,b in zip(timestamps,timestamps[1:])]
    for phase in sorted({r['phase'] for r in samples}):
        rr = [r for r in samples if r['phase']==phase]
        phase_rows.append(dict(phase=phase,sampled_rank_rss_peak_bytes=maximum(number(r,'rss_bytes') for r in rr),
            sampled_gpu_process_peak_bytes=maximum(number(r,'gpu_process_bytes') for r in rr)))
    allocator = [r.get('allocator_stats') or {} for r in states]
    result = dict(sample_count=len(samples),sample_interval_median_ms=st.median(intervals)*1000 if intervals else None,
        sample_interval_max_ms=max(intervals)*1000 if intervals else None,sampled_gpu_process_peak_bytes=maximum(number(r,'gpu_process_bytes') for r in samples),
        rank_host_peak_bytes=maximum([r.get('lifetime_host_peak_bytes') for r in states]+[number(r,'lifetime_host_peak_bytes') for r in samples]),
        sampled_simultaneous_rss_peak_bytes=maximum(simultaneous.values()),
        allocator_lifetime_peak_bytes=maximum(r.get('peak_bytes_in_use') for r in allocator),
        allocator_snapshot_in_use_max_bytes=maximum(r.get('bytes_in_use') for r in allocator),
        allocator_reserved_max_bytes=maximum(r.get('bytes_reserved') for r in allocator),
        allocator_pool_max_bytes=maximum(r.get('pool_bytes') for r in allocator),
        allocator_peak_pool_bytes=maximum(r.get('peak_pool_bytes') for r in allocator),
        allocator_limit_bytes=maximum(r.get('bytes_limit') for r in allocator))
    for phase in ('workload_done','diagnostics'):
        details=[]
        for path in directory.glob('rank-*.jsonl'):
            for line in path.read_text().splitlines():
                try: row=json.loads(line)
                except ValueError:continue
                if row['phase']==phase and row.get('details'):details.append(row['details'])
        for key in {k for d in details for k in d}:
            result['max_'+key] = maximum(d.get(key) for d in details)
    return result, phase_rows


def summarize(out):
    m=json.loads((out/'manifest.json').read_text())
    launches=json.loads((out/'launches.json').read_text())
    settings=m['settings']; rows=[]; memory=[]; phases=[]; failures=[]
    for launch in launches:
        if launch.get('memory_directory'):
            metrics, pp=memory_summary(Path(launch['memory_directory']))
            memory.append(dict(name=launch['name'],**metrics))
            phases.extend(dict(name=launch['name'],**p) for p in pp)
        if launch.get('exit_code') != 0:
            log=Path(launch['log']).read_text(errors='replace') if launch.get('log') else ''
            failure='timeout' if launch.get('timed_out') else 'unclassified failure'
            if launch.get('skipped_reason'):failure=launch['skipped_reason']
            elif any(s in log.lower() for s in ('out of memory','out_of_memory','cuda_error_out_of_memory','oom-kill')):failure='OOM evidence in log; inspect last phase'
            elif 'Check failed:' in log or 'Compilation failed' in log:failure='compiler assertion/error'
            last=[]
            if launch.get('memory_directory'):
                for p in Path(launch['memory_directory']).glob('rank-*.json'):
                    last.append(json.loads(p.read_text())['phase'])
            failures.append(dict(name=launch['name'],classification=failure,last_phases=last,log=launch.get('log')))
    records=[]
    for launch in launches:
        if not launch['name'].startswith(('timing-','profile-','smoke-sharding-')) or launch.get('exit_code') != 0:continue
        d=json.loads(Path(launch['result']).read_text());p=d['ranks']
        assert d['correctness_passed'] and d['timing']['timer']=='MPI.Wtime'
        assert len(d['timing']['rank_seconds'])==d['settings']['repeats']
        assert all(len(s)==p and all(t>0 for t in s) for s in d['timing']['rank_seconds'])
        med=st.median(max(s)/d['settings']['iterations'] for s in d['timing']['rank_seconds'])
        assert abs(med-d['timing']['median_seconds_per_matvec'])<1e-12
        assert [r['cpu_affinity'] for r in d['placement']]==[[c] for c in m['cpu_ids'][:p]]
        assert [r['gpu_uuid'] for r in d['placement']]==m['gpu_uuids'][:p]
        assert all(r['hostname']==m['hostname'] for r in d['placement'])
        d['kind']=launch['name'].split('-')[0]; records.append(d)
    for n in settings['sizes']:
        for p in settings['ranks']:
            all_records=[d for d in records if d['subdivisions']==n and d['ranks']==p]
            group=[d for d in all_records if d['kind']=='timing']
            if all_records:
                assert len({d['dataset'] for d in all_records})==1
                assert all(d['fingerprints']==all_records[0]['fingerprints'] and d['settings']==all_records[0]['settings'] for d in all_records)
                assert len({json.dumps(d['versions'],sort_keys=True) for d in all_records})==1
            row=dict(subdivisions=n,dofs=(n+1)**3,ranks=p,successful_trials=len(group),status='complete' if len(group)==3 else 'incomplete')
            native=out/'raw'/('fixture-n{}-p{}.json'.format(n,p))
            if native.exists():
                nd=json.loads(native.read_text()); meta=json.loads((Path(nd['dataset'])/'metadata.json').read_text())
                row.update(nnz=sum(r['nnz'] for r in meta['partitions']),
                    max_owned_rows=max(r['owned_rows'] for r in meta['partitions']),
                    max_ghost_cols=max(r['ghost_cols'] for r in meta['partitions']),
                    total_fixture_array_bytes=sum(sum(r['array_bytes'].values()) for r in meta['partitions']))
            if group:
                assert len({d['trial'] for d in group})==len(group)
                times=[d['timing']['median_seconds_per_matvec'] for d in group]
                row.update(ms_per_matvec=st.median(times)*1000,min_trial_ms=min(times)*1000,max_trial_ms=max(times)*1000,ms_per_batch=st.median(times)*100000)
                for name in ('lowering','compilation','first_execution'):
                    values=[d['phases'][name]['max_seconds']*1000 for d in group]
                    row.update({name+'_ms':st.median(values),name+'_min_ms':min(values),name+'_max_ms':max(values)})
                totals=[sum(d['phases'][k]['max_seconds'] for k in ('lowering','compilation'))+100*t for d,t in zip(group,times)]
                row['compilation_plus_batch_ms']=st.median(totals)*1000
            for kind in ('profile','fixture'):
                name='profile-n{}-p{}-trial0'.format(n,p) if kind=='profile' else 'fixture-n{}-p{}'.format(n,p)
                mm=next((x for x in memory if x['name']==name),{})
                row.update({kind+'_'+k:v for k,v in mm.items() if k!='name'})
            rows.append(row)
    for r in rows:
        baseline=next(x for x in rows if x['subdivisions']==r['subdivisions'] and x['ranks']==min(settings['ranks']))
        if r['status']==baseline['status']=='complete':
            r['speedup']=baseline['ms_per_matvec']/r['ms_per_matvec'];r['efficiency']=r['speedup']*min(settings['ranks'])/r['ranks']
    dump_csv(out/'summary.csv',rows);dump_csv(out/'memory-summary.csv',memory);dump_csv(out/'memory-phases.csv',phases)
    (out/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    (out/'audit.json').write_text(json.dumps(dict(validated_records=len(records),matched_timing_and_profile_fixtures=True),indent=2)+'\n')
    lines=['# Sharding scaling and memory study','',
        'Allocation {} on {}. One physical CPU core and one GPU per rank; current whole-matvec JIT.'.format(m['job'],m['hostname']),
        'Fresh float64 P1 fixtures; DOLFINx only generates and validates the operator. Three independent JAX timing trials, ten warmups, ten batches of 100 Python-driven calls; halo exchange every call.',
        'MPI.Wtime with array/effect completion, maximum rank time, median of trial medians. Compilation and first execution are separate. Total is lowering + compilation + one warmed batch; other setup and first execution are excluded.','',
        '| DoFs | GPUs | Trials | ms/call | Compile ms | First execution ms | Compile + batch ms | Speedup vs 2 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        value=lambda k: '{:.3f}'.format(r[k]) if r.get(k) is not None and r['status']=='complete' else '—'
        lines.append('| {:,} | {} | {}/3 | {} | {} | {} | {} | {} |'.format(r['dofs'],r['ranks'],r['successful_trials'],*(value(k) for k in ('ms_per_matvec','compilation_ms','first_execution_ms','compilation_plus_batch_ms','speedup'))))
    lines+=['','## Memory','',
        'Memory profiles are separate launches; their timings are excluded from performance comparisons. Raw rank snapshots retain all available allocator fields. Preallocation is disabled; allocator pool reservation, live buffers and NVIDIA process memory are different measurements.',
        '100 ms target sampling can miss brief peaks. GPU process usage includes runtime/context overhead; allocator peaks are lifetime peaks, not reset per phase. Host aggregate RSS is the maximum simultaneous sampled sum and may double-count shared pages. Diagnostic program-text extraction happens after workload memory collection.',
        '| DoFs | GPUs | GPU process peak GiB | Allocator lifetime peak GiB | JAX rank host peak GiB | JAX aggregate RSS GiB | Fixture aggregate RSS GiB |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        value=lambda k:'{:.3f}'.format(r[k]/2**30) if r.get(k) is not None else '—'
        lines.append('| {:,} | {} | {} | {} | {} | {} | {} |'.format(r['dofs'],r['ranks'],*(value(k) for k in ('profile_sampled_gpu_process_peak_bytes','profile_allocator_lifetime_peak_bytes','profile_rank_host_peak_bytes','profile_sampled_simultaneous_rss_peak_bytes','fixture_sampled_simultaneous_rss_peak_bytes'))))
    lines+=['','## Failures and limits','']+[ '- {}: {}; last phases {}'.format(f['name'],f['classification'],f['last_phases']) for f in failures]
    successful_profiles=sum(r.get('profile_sample_count',0)>0 for r in rows)
    lines+=['','Successful timing configurations: {}/{}. Profiles with samples: {}. A capacity limit is unmeasured unless supported by a recorded memory failure; do not interpret compiler errors as OOM.'.format(sum(r['status']=='complete' for r in rows),len(rows),successful_profiles),
        'See summary.csv, memory-summary.csv, memory-phases.csv, memory/*/samples.csv, raw/, logs/, manifest.json, and failures.json.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    if any(r['status']=='complete' for r in rows):plot(out,rows,m)
    print(out/'REPORT.md')


def plot(out,rows,m):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=160)
        plt.close(fig)
    fields=('ms_per_matvec','ms_per_batch','lowering_ms','compilation_ms','first_execution_ms','compilation_plus_batch_ms')
    for x in ('dofs','nnz'):
        fig,axes=plt.subplots(2,3,figsize=(14,8))
        for ax,field in zip(axes.flat,fields):
            for p in m['settings']['ranks']:
                rr=[r for r in rows if r['ranks']==p and r['status']=='complete']
                ax.plot([r[x] for r in rr],[r[field] for r in rr],'o-',label='{} GPUs'.format(p))
                lower=field.replace('_ms','_min_ms');upper=field.replace('_ms','_max_ms')
                if field=='ms_per_matvec':lower,upper='min_trial_ms','max_trial_ms'
                if rr and lower in rr[0]:ax.fill_between([r[x] for r in rr],[r[lower] for r in rr],[r[upper] for r in rr],alpha=.15)
            ax.set(xlabel=x,ylabel=field,xscale='log',yscale='log');ax.grid(alpha=.2);ax.legend()
        save(fig,'timing-vs-'+x)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,n in zip(axes.flat,m['settings']['sizes']):
        rr=[r for r in rows if r['subdivisions']==n and 'speedup' in r]
        ax.plot([r['ranks'] for r in rr],[r['speedup'] for r in rr],'o-',label='sharding')
        ax.plot(m['settings']['ranks'],[p/min(m['settings']['ranks']) for p in m['settings']['ranks']],'k:',label='ideal')
        ax.set(title='{:,} DoFs'.format((n+1)**3),xlabel='GPUs',ylabel='Speedup vs 2 GPUs',xticks=m['settings']['ranks']);ax.grid(alpha=.2);ax.legend()
    save(fig,'scaling')
    keys=('profile_sampled_gpu_process_peak_bytes','profile_allocator_lifetime_peak_bytes','profile_rank_host_peak_bytes','profile_sampled_simultaneous_rss_peak_bytes','fixture_sampled_simultaneous_rss_peak_bytes','profile_max_lowered_text_bytes')
    inventory=list(csv.DictReader((out/'memory-inventory.csv').open()))
    gpu_capacity=min(float(next(v for k,v in row.items() if 'memory.total' in k).split()[0])*1024**2 for row in inventory)
    limits=[int(v) for v in m.get('host_memory_limits',{}).values() if v.isdigit() and int(v)<2**60]
    if m.get('allocation_host_memory_bytes'):limits.append(m['allocation_host_memory_bytes'])
    for x in ('dofs','nnz'):
        fig,axes=plt.subplots(2,3,figsize=(15,8))
        for ax,key in zip(axes.flat,keys):
            for p in m['settings']['ranks']:
                rr=[r for r in rows if r['ranks']==p and r.get(key) is not None and x in r]
                ax.plot([r[x] for r in rr],[r[key]/2**30 for r in rr],'o-',label='{} GPUs'.format(p))
            cap=gpu_capacity if key in keys[:2] else min(limits) if limits and key in keys[2:5] else None
            if key == keys[1]:
                allocator_limit=maximum(r.get('profile_allocator_limit_bytes') for r in rows)
                if allocator_limit:ax.axhline(allocator_limit/2**30,color='r',ls='--',label='allocator limit')
            if cap:ax.axhline(cap/2**30,color='k',ls=':',label='capacity / host limit')
            ax.set(xlabel=x,ylabel='GiB',title=key.replace('profile_','').replace('_',' '),xscale='log',yscale='log');ax.grid(alpha=.2);ax.legend(fontsize=8)
        save(fig,'memory-vs-'+x)


    phase_data=list(csv.DictReader((out/'memory-phases.csv').open()))
    fig,axes=plt.subplots(2,len(m['settings']['ranks']),squeeze=False,figsize=(15,8))
    for column,p in enumerate(m['settings']['ranks']):
        for row_index,metric in enumerate(('sampled_gpu_process_peak_bytes','sampled_rank_rss_peak_bytes')):
            ax=axes[row_index,column]
            for phase in ('fixture_loading','plan_construction','compilation','first_execution','timed_computation','final_validation'):
                points=[]
                for n in m['settings']['sizes']:
                    name='profile-n{}-p{}-trial0'.format(n,p)
                    found=next((r for r in phase_data if r['name']==name and r['phase']==phase),None)
                    if found and found.get(metric):points.append(((n+1)**3,int(found[metric])/2**30))
                if points:ax.plot([v[0] for v in points],[v[1] for v in points],'o-',label=phase)
            ax.set(title='{} GPUs: {}'.format(p,'GPU process' if row_index==0 else 'rank host RSS'),xlabel='DoFs',ylabel='sampled peak GiB',xscale='log',yscale='log')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    save(fig,'memory-phases')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    summarize(parser.parse_args().output.resolve())
