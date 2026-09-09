"""Audit regressions: refuse apparently comparable but mismatched results."""
import contextlib
import io
import json
from pathlib import Path

import pytest
from summarize import summarize


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
