import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from memory_probe import mark, proc_memory
from sharding_study import campaign
from study_summary import memory_summary


def test_missing_allocator_stats_and_phase_memory(tmp_path,monkeypatch):
    monkeypatch.setenv('BENCH_MEMORY_DIR',str(tmp_path))
    class Unsupported:
        def memory_stats(self):return None
    mark('compilation',Unsupported())
    mark('workload_done',Unsupported(),dict(max_nnz=12))
    mark('diagnostics',Unsupported(),dict(lowered_text_bytes=42))
    result,phases=memory_summary(tmp_path)
    assert result['allocator_lifetime_peak_bytes'] is None
    assert result['sampled_gpu_process_peak_bytes'] is None
    assert result['rank_host_peak_bytes']>0
    assert result['max_lowered_text_bytes']==42


def test_sampler_stops_and_preserves_dead_worker(tmp_path):
    (tmp_path/'rank-0.json').write_text(json.dumps(dict(pid=999999999,rank=0,phase='compilation')))
    cpu=min(os.sched_getaffinity(0))
    script=Path(__file__).with_name('memory_probe.py')
    process=subprocess.Popen([sys.executable,str(script),str(tmp_path),'--cpu',str(cpu)])
    try:
        time.sleep(.5)
        (tmp_path/'stop').touch()
        assert process.wait(timeout=10)==0
        rows=list(csv.DictReader((tmp_path/'samples.csv').open()))
        assert rows and rows[0]['phase']=='compilation'
        assert rows[0]['rss_bytes']==''
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_fixture_reuse_and_failed_launch_schedule(tmp_path):
    (tmp_path/'raw').mkdir()
    launches=[]
    def launch(backend,ranks,name,options,result=None,**kwargs):
        launches.append(dict(name=name,backend=backend,exit_code=1 if name=='profile-n46-p2-trial0' else 0))
        (tmp_path/'launches.json').write_text(json.dumps(launches))
        if backend=='dolfinx':assert '--fixture-only' in options
        return launches[-1]['exit_code']==0
    manifest=dict(settings=dict(sizes=[46],ranks=[2,3,4]))
    campaign(tmp_path,manifest,launch,SimpleNamespace(skip_regression=True,smoke_only=False))
    assert sum(r['name'].startswith('fixture-') for r in launches)==3
    assert sum(r['name'].startswith('timing-') for r in launches)==9
    assert not any(r['backend']=='mpi4jax' for r in launches)
    assert 'profile-n46-p2-trial0' in json.loads((tmp_path/'completion.json').read_text())['failures']


def test_incomplete_study_reporting(tmp_path):
    from study_summary import summarize
    (tmp_path/'raw').mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(job='test',hostname='node',settings=dict(sizes=[199],ranks=[2,3,4]))))
    (tmp_path/'launches.json').write_text(json.dumps([dict(name='fixture-n199-p2',exit_code=None,skipped_reason='Insufficient allocation time remaining')]))
    summarize(tmp_path)
    rows=list(csv.DictReader((tmp_path/'summary.csv').open()))
    assert len(rows)==3 and all(r['status']=='incomplete' for r in rows)
    assert json.loads((tmp_path/'failures.json').read_text())[0]['classification']=='Insufficient allocation time remaining'
