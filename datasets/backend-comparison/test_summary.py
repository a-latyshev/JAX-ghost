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
