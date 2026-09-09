"""Audit regressions: refuse apparently comparable but mismatched results."""
import contextlib
import io
import json
from pathlib import Path

import pytest
from summarize import summarize, total_timing_fields


def run_case(out, change=None):
    manifest = dict(job='test',hostname='node',cpu_ids=[2,4],gpu_uuids=['GPU-A','GPU-B'],
                    settings=dict(sizes=[46,99],ranks=[1,2,3,4],trials=3,iterations=100))
    (out/'manifest.json').write_text(json.dumps(manifest))
    launches=[]
    for backend in ('dolfinx','mpi4jax','sharding'):
        # The critical rank changes between batches: maximum must precede median.
        d=dict(backend=backend,subdivisions=4,ranks=2,trial=0,settings=dict(iterations=100,repeats=2),
            correctness_passed=True,fingerprints=['a','b'],dataset='same',versions=dict(jax='v',jaxlib='v'),
            timing=dict(timer='MPI.Wtime',rank_seconds=[[1.,9.],[8.,2.]],median_seconds_per_matvec=.085),
            placement=[dict(cpu_affinity=[cpu],hostname='node',gpu_uuid=None if backend=='dolfinx' else gpu)
                       for cpu,gpu in zip((2,4),('GPU-A','GPU-B'))])
        if change and backend=='sharding':change(d)
        result=out/(backend+'.json');result.write_text(json.dumps(d))
        launches.append(dict(name='smoke-'+backend,exit_code=0,result=str(result)))
    (out/'launches.json').write_text(json.dumps(launches))
    with contextlib.redirect_stdout(io.StringIO()):summarize(out)
    assert len(json.loads((out/'audit.json').read_text())) == 1


def test_rank_max_before_median(tmp_path):
    run_case(tmp_path)


@pytest.mark.parametrize('change', [
    lambda d:d.update(fingerprints=['other','b']),
    lambda d:d['placement'][1].update(gpu_uuid='GPU-A'),
    lambda d:d['placement'][0].update(cpu_affinity=[3]),
    lambda d:d['timing'].update(median_seconds_per_matvec=.05),
])
def test_reject_mismatched_comparison(tmp_path, change):
    with pytest.raises(AssertionError):
        run_case(tmp_path,change)


def test_total_pairs_compilation_and_batch_before_median():
    # Median of the totals is 10 s; adding independent medians would give 18 s.
    group = [dict(backend='mpi4jax',settings=dict(iterations=100),
                  timing=dict(median_seconds_per_matvec=batch/100),
                  phases=dict(lowering=dict(max_seconds=compile_seconds/4),
                              compilation=dict(max_seconds=3*compile_seconds/4),
                              first_execution=dict(max_seconds=1000)))
             for compile_seconds,batch in [(1,9),(9,1),(9,9)]]
    fields = total_timing_fields(group)
    assert fields['total_ms_per_batch'] == pytest.approx(10000)
    assert fields['total_amortized_us_per_matvec'] == pytest.approx(100000)
    assert fields['min_trial_total_ms'] == pytest.approx(10000)
    assert fields['max_trial_total_ms'] == pytest.approx(18000)


def test_dolfinx_total_has_no_compilation_charge():
    fields = total_timing_fields([dict(backend='dolfinx',settings=dict(iterations=100),
                                      timing=dict(median_seconds_per_matvec=.002),phases={})])
    assert fields['total_ms_per_batch'] == pytest.approx(200)
    assert fields['total_amortized_us_per_matvec'] == pytest.approx(2000)


def test_seven_core_jax_placement_keeps_native_baseline(tmp_path):
    run_case(tmp_path)
    manifest_path = tmp_path/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    masks = [[2,6,8,10,12,14,16],[4,18,20,22,24,26,28]]
    manifest.update(jax_cpus_per_rank=7,jax_cpu_sets=masks)
    manifest_path.write_text(json.dumps(manifest))
    for backend in ('mpi4jax','sharding'):
        path = tmp_path/(backend+'.json')
        record = json.loads(path.read_text())
        for rank,mask in enumerate(masks):record['placement'][rank]['cpu_affinity']=mask
        path.write_text(json.dumps(record))
    with contextlib.redirect_stdout(io.StringIO()):summarize(tmp_path)
    # The original native record must still be pinned to one core.
    assert json.loads((tmp_path/'dolfinx.json').read_text())['placement'][0]['cpu_affinity']==[2]
    path=tmp_path/'sharding.json'
    record=json.loads(path.read_text());record['placement'][1]['cpu_affinity']=masks[0]
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError):summarize(tmp_path)


def test_two_rank_baseline_and_single_size_plots(tmp_path):
    import csv
    run_case(tmp_path)
    path = tmp_path/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['settings'].update(sizes=[159],ranks=[2,3,4],trials=1)
    path.write_text(json.dumps(manifest))
    launches = json.loads((tmp_path/'launches.json').read_text())
    for launch in launches:
        launch['name'] = launch['name'].replace('smoke-', 'full-')
        path = Path(launch['result'])
        d = json.loads(path.read_text())
        d.update(subdivisions=159,global_dofs=4096000,phases={})
        if d['backend'] != 'dolfinx':
            d['phases'] = {k:dict(max_seconds=.1,rank_seconds=[.1,.05])
                           for k in ('lowering','compilation','first_execution')}
        path.write_text(json.dumps(d))
    (tmp_path/'launches.json').write_text(json.dumps(launches))
    summarize(tmp_path)
    rows = list(csv.DictReader((tmp_path/'summary.csv').open()))
    for row in rows:
        if row['ranks'] == '2':
            assert float(row['speedup']) == 1
            assert float(row['efficiency']) == 1
            assert row['scaling_baseline_ranks'] == '2'
        else:
            assert row['status'] == 'incomplete'
            assert not row['speedup']
    assert 'Speedup vs 2 ranks' in (tmp_path/'REPORT.md').read_text()
    for name in ('matvec','batch','scaling','ratios','startup'):
        assert (tmp_path/(name+'.png')).stat().st_size > 1000
