"""Aggregate independent launches and plot forward-update strong-scaling results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = {'dolfinx': '#2563a6', 'sharded': '#d36b23'}
LABELS = {'dolfinx': 'DOLFINx (in-place)', 'sharded': 'ShardedJAXGhost (functional)'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    files = sorted((args.results/'raw').glob('*.json'))
    data = [json.loads(p.read_text()) for p in files]
    settings = json.loads((args.results/'settings.json').read_text())
    expected = {(n, p, trial) for n in settings['sizes'] for p in settings['ranks']
                for trial in range(settings['trials'])}
    actual = {(d['subdivisions'],d['ranks'],d['trial']) for d in data}
    if actual != expected or len(data) != len(expected):
        raise ValueError('Incomplete or duplicated benchmark results')
    if any(d['max_error'] != 0 for d in data):
        raise ValueError('Correctness validation failed')
    groups = {}
    for d in data:
        groups.setdefault((d['subdivisions'],d['ranks']), []).append(d)
    summary = []
    for (n,p), runs in sorted(groups.items()):
        row = dict(subdivisions=n, ranks=p, global_dofs=runs[0]['global_dofs'],
                   max_error=max(d['max_error'] for d in runs))
        for backend in COLORS:
            trials = [float(np.median([s['seconds_per_update'] for s in d['samples'][backend]])) for d in runs]
            row[backend+'_median_us'] = float(np.median(trials))*1e6
            row[backend+'_min_us'] = min(trials)*1e6
            row[backend+'_max_us'] = max(trials)*1e6
            row[backend+'_trial_medians_us'] = [t*1e6 for t in trials]
        ratios = [np.median([s['seconds_per_update'] for s in d['samples']['sharded']]) /
                  np.median([s['seconds_per_update'] for s in d['samples']['dolfinx']]) for d in runs]
        row['sharded_over_dolfinx'] = float(np.median(ratios))
        row['ratio_min'], row['ratio_max'] = float(min(ratios)), float(max(ratios))
        row['useful_halo_bytes'] = runs[0]['total_useful_halo_bytes']
        row['padded_alltoall_bytes'] = p*runs[0]['logical_send_buffer_bytes_per_rank']
        row['global_padded_vector_bytes'] = p*runs[0]['vector_bytes_per_shard']
        row['compile_seconds'] = float(np.median([d['compile_seconds'] for d in runs]))
        summary.append(row)
    fields = [k for k in summary[0] if 'trial_medians' not in k]
    with (args.results/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in summary)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold',
                         'savefig.dpi':200,'pdf.fonttype':42,'svg.fonttype':'none'})
    sizes = sorted(settings['sizes'])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    fig.suptitle('CPU forward ghost updates: fixed-workload scaling', fontsize=18, fontweight='bold', y=.98)
    fig.text(.5,.94,'Apple M2 Pro • float64 • one MPI rank per JAX CPU device • no CPU affinity',
             ha='center', color='#444444',fontsize=10)
    for ax,n in zip(axes[0],sizes):
        rows=[r for r in summary if r['subdivisions']==n]
        ranks=np.array([r['ranks'] for r in rows])
        for backend,color in COLORS.items():
            med=np.array([r[backend+'_median_us'] for r in rows])
            low=np.array([r[backend+'_min_us'] for r in rows])
            high=np.array([r[backend+'_max_us'] for r in rows])
            ax.errorbar(ranks,med,yerr=[med-low,high-med],fmt='o-',capsize=4,
                        color=color,label=LABELS[backend],linewidth=1.8)
        ax.set_title(f"{rows[0]['global_dofs']:,} DoFs · {n} × {n} subdivisions",pad=12)
        ax.set_yscale('log'); ax.set_ylabel('Wall time per update (µs, log scale)')
        ax.axvspan(.8,1.25,color='#aaaaaa',alpha=.12)
        ax.set_xticks([1,2,3,4]); ax.set_xlim(.8,4.2)
        ax.set_xlabel('MPI ranks / JAX CPU devices')
        ax.grid(axis='y',alpha=.22,which='both'); ax.legend(fontsize=9,loc='best')
    ax=axes[1,0]
    for n,color,marker in zip(sizes,['#52616b','#874b9b'],['o','s']):
        rows=[r for r in summary if r['subdivisions']==n]
        med=np.array([r['sharded_over_dolfinx'] for r in rows])
        low=np.array([r['ratio_min'] for r in rows]); high=np.array([r['ratio_max'] for r in rows])
        ax.errorbar([r['ranks'] for r in rows],med,yerr=[med-low,high-med],fmt=marker+'-',
                    color=color,capsize=4,label=f"{rows[0]['global_dofs']:,} DoFs")
    ax.axhline(1,color='#555555',linestyle=':',linewidth=1)
    ax.set_yscale('log'); ax.set_ylabel('Sharded time / DOLFINx time (log scale)')
    ax.set_title('Backend time ratio · lower is better for JAX',pad=12)
    ax.set_xticks([1,2,3,4]); ax.set_xlabel('MPI ranks / JAX CPU devices'); ax.grid(axis='y',alpha=.22,which='both'); ax.legend()
    ax=axes[1,1]
    for n,style in zip(sizes,['-','--']):
        rows=[r for r in summary if r['subdivisions']==n and r['ranks']>=2]
        for backend,color in COLORS.items():
            baseline=next(r[backend+'_median_us'] for r in rows if r['ranks']==2)
            ax.plot([r['ranks'] for r in rows],[baseline/r[backend+'_median_us'] for r in rows],
                    marker='o',linestyle=style,color=color,
                    label=f"{'DOLFINx' if backend=='dolfinx' else 'Sharded'} · {rows[0]['global_dofs']:,}")
    ax.axhline(1,color='#555555',linestyle=':',linewidth=1)
    ax.set_title('Communication scaling relative to 2 ranks',pad=12)
    ax.set_ylabel('T(2) / T(p) · higher is better'); ax.set_xticks([2,3,4])
    ax.set_xlabel('MPI ranks / JAX CPU devices'); ax.grid(axis='y',alpha=.22); ax.legend(fontsize=8)
    fig.text(.06,.045,'Points: median of 3 independent launch medians. Bars: range of launch medians.\n'
             'Each launch: 7 batches × 100 updates; sample = maximum rank time / 100. Setup and compilation excluded.\n'
             '1 rank has no remote ghosts; no T(1)-based speedup or ideal 1/p line is asserted.',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.085,right=.97,top=.875,bottom=.15,hspace=.42,wspace=.3)
    for ext in ('png','pdf','svg'):
        fig.savefig(args.results/f'scaling.{ext}',facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
    for ax,n in zip(axes,sizes):
        rows=[r for r in summary if r['subdivisions']==n and r['ranks']>=2]
        ranks=np.array([r['ranks'] for r in rows])
        ax.bar(ranks-.16,[r['useful_halo_bytes']/1024 for r in rows],width=.32,
               label='Useful ghost values',color=COLORS['dolfinx'])
        ax.bar(ranks+.16,[r['padded_alltoall_bytes']/1024 for r in rows],width=.32,
               label='Padded all-to-all send buffers',color=COLORS['sharded'])
        ax.set_title(f"{rows[0]['global_dofs']:,} DoFs"); ax.set_xticks([2,3,4]); ax.grid(axis='y',alpha=.2)
        ax.set_xlabel('MPI ranks / JAX CPU devices'); ax.set_ylabel('Aggregate logical payload (KiB)'); ax.legend(fontsize=8)
    fig.suptitle('Communication layout at fixed global mesh size',fontweight='bold',fontsize=15)
    fig.text(.07,.025,'Logical buffer sizes, not measured wire traffic. Padded sizes include unused/self slots; runtime may optimize them.',fontsize=9)
    fig.tight_layout(rect=(0,.075,1,.92))
    fig.savefig(args.results/'communication.png',facecolor='white')
    fig.savefig(args.results/'communication.pdf',facecolor='white')
    plt.close(fig)
    rows_text=['| Global DoFs | Ranks/devices | DOLFINx µs | Sharded µs | Sharded / DOLFINx |',
               '| ---: | ---: | ---: | ---: | ---: |']
    for r in summary:
        rows_text.append(f"| {r['global_dofs']:,} | {r['ranks']} | {r['dolfinx_median_us']:.2f} | {r['sharded_median_us']:.2f} | {r['sharded_over_dolfinx']:.2f}× |")
    communicating = [r for r in summary if r['ranks'] > 1]
    findings = (f"On 2–4 devices, the sharded backend took {min(r['sharded_over_dolfinx'] for r in communicating):.1f}–"
                f"{max(r['sharded_over_dolfinx'] for r in communicating):.1f}× the DOLFINx update time. "
                "Both backends' median latency increased from 2 to 4 devices for each mesh. "
                "This implementation shows no latency benefit from adding CPU devices in these measurements; "
                "the timings alone do not isolate the cause.\n\n")
    report='''# CPU sharded forward-update scaling

![Measured scaling](scaling.png)

## Measured results

'''+ findings + '\n'.join(rows_text)+'''

All changed-value checks and per-batch checks matched DOLFINx and the independent
global-ID oracle exactly (maximum absolute error 0). The sharded input-preservation
check also passed. Each point uses the same fixed global mesh for every rank count.

## Method

- Two triangular unit-square meshes: 256² and 1024² subdivisions; scalar continuous P1, float64.
- Ranks/devices: 1, 2, 3, 4; three independent MPI launches per point.
- Ten warmup updates; seven timed batches of 100 Python-driven calls per backend.
- MPI barrier before each batch, outside timing. MPI.Wtime measures each rank.
  The sharded output is blocked before stopping the timer; no per-call barrier.
- Each batch sample is the maximum rank elapsed time divided by 100.
  Plots report the median of the three launch medians, with their min/max range.
  Ratios are computed within launches, then their median/range is plotted.
- Alternate backend order between batches and reverse rank/mesh order on alternate
  trials. Launches run sequentially, not concurrently.
- Mesh/space setup, metadata preparation, transfers, compilation, first execution,
  correctness checks and result I/O are excluded from steady-state timing.
- Numerical inputs stay fixed during a timed batch. Each Python call still applies
  the forward update; repeated calls are not fused into an outer compiled loop.
- Raw JSON preserves all rank timings, CPU times, partition sizes, versions,
  compilation times and thread counts; summary.csv contains plotted numbers.

## Interpretation and limits

This is strong scaling of a **ghost-update operation**, not a PDE solver or matvec.
A one-rank forward update has no remote ghosts, so its latency is a control, not a
useful parallel-speedup baseline. The comparison therefore shows T(2)/T(p) for
communication scaling and omits an ideal 1/p line. As partitions change, boundary
sizes and communication graphs change even though total mesh size stays fixed.

DOLFINx updates a NumPy-backed vector in place. ShardedJAXGhost preserves its input
and returns a padded global JAX buffer; its timings include that API's dispatch,
packing, native all-to-all and output update costs. This is an end-to-end backend
comparison, not a controlled measurement of MPI versus Gloo transport alone.
Logical packet sizes do not establish physical wire traffic or identify bottlenecks.

Hardware: Apple M2 Pro, 10 physical/logical cores (6 performance + 4 efficiency),
16 GiB memory, macOS. One MPI process manages one JAX CPU device. These are **not
pinned single-core measurements**: macOS process affinity is unavailable and JAX
has runtime/helper threads. Thread-limit environment variables are recorded but
do not imply one total OS thread per rank. Runs share a workstation with the OS;
no cluster isolation or hardware-independent speedup claim is made.

MPI metadata and DOLFINx use MPICH with FI_PROVIDER=tcp and FI_TCP_IFACE=en0.
Native JAX CPU communication uses Gloo with the opt-in loopback workaround, verified
on JAX/jaxlib 0.10.2; it is not the same transport as DOLFINx. No system settings were
changed. Compile/setup costs are available in raw results but not counted as update latency.

![Communication buffers](communication.png)
'''
    (args.results/'REPORT.md').write_text(report)
    print('\n'.join(rows_text))


if __name__ == '__main__':
    main()
