"""Developer checks; pytest is not required by the portable runtime."""
import copy
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys

import pytest
from mpi4py import MPI
from common import load_fixture
from placement import select_visible_gpu, validate_records
from plot import aggregate

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def test_optional_backends_not_imported():
    code = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mpi4jax','dolfinx'):
            raise ModuleNotFoundError('blocked for dependency isolation', name=fullname)
sys.meta_path.insert(0,Block())
from jaxghost import ShardedJAXGhost, ShardedJAXMatrixCSR
assert 'mpi4jax' not in sys.modules and 'dolfinx' not in sys.modules
try:
    from jaxghost import JAXMatrixCSR
except ImportError as e:
    assert 'jaxghost[mpi4jax]' in str(e)
else:
    raise AssertionError('missing optional backend did not fail')
'''
    import os
    env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run([sys.executable,'-c',code],env=env,check=True,timeout=60)


def test_gpu_selection(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-a,GPU-b')
    select_visible_gpu(1)
    assert __import__('os').environ['CUDA_VISIBLE_DEVICES']=='GPU-b'
    select_visible_gpu(7)  # scheduler already masked this process to one GPU
    assert __import__('os').environ['CUDA_VISIBLE_DEVICES']=='GPU-b'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    with pytest.raises(ValueError,match='No visible GPU'):select_visible_gpu(0)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(ValueError,match='More local'):select_visible_gpu(2)


def test_duplicate_placement():
    a=dict(node_group=0,gpu_uuid='a',gpu_pci_bus_id='01',cpu_affinity=[0])
    b=dict(node_group=0,gpu_uuid='b',gpu_pci_bus_id='02',cpu_affinity=[1])
    validate_records([a,b])
    with pytest.raises(ValueError,match='physical GPU'):
        validate_records([a,dict(b,gpu_uuid='a')])
    with pytest.raises(ValueError,match='physical GPU'):
        validate_records([a,dict(b,gpu_pci_bus_id='01')])
    with pytest.raises(ValueError,match='CPU affinity'):
        validate_records([a,dict(b,cpu_affinity=[0])])
    validate_records([a,dict(a,node_group=1)])


def test_wrong_rank_count(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps(dict(format_version=1,rank_count=2,dtype='float64')))
    with pytest.raises(ValueError,match='wrong MPI rank'):
        load_fixture(tmp_path,MPI.COMM_SELF)


def test_bad_checksum(tmp_path):
    (tmp_path/'metadata.json').write_text(json.dumps(dict(format_version=1,rank_count=1,dtype='float64',settings=dict(iterations=100),fingerprints=['bad'])))
    (tmp_path/'rank-0000.npz').write_bytes(b'broken')
    with pytest.raises(ValueError,match='Checksum mismatch'):
        load_fixture(tmp_path,MPI.COMM_SELF)


def campaign(tmp_path):
    config=dict(sizes=[4],ranks=[1,2],trials=1,repeats=2,warmup=10)
    (tmp_path/'settings.json').write_text(json.dumps(config))
    (tmp_path/'raw').mkdir()
    for p in (1,2):
        r=dict(subdivisions=4,ranks=p,trial=0,global_dofs=125,mode='time',correctness_passed=True,
            settings=dict(iterations=100,repeats=2,warmup=10,loop='python',dtype='float64'),
            source_hashes={'a':'b'},versions={'jax':'0.11.1'},environment=dict(mpi_library='MPI',numpy='N',mpi4py='M'),
            placement=[dict(hostname='n',gpu_name='V100',gpu_uuid=str(i),cpu_affinity=[i]) for i in range(p)],
            fixture_fingerprints=[str(i) for i in range(p)],validation=[dict(error_l2=0,reference_l2=1)],
            timing=dict(timer='MPI.Wtime',rank_seconds=[[.1/p]*p]*2,median_seconds_per_matvec=.001/p))
        (tmp_path/'raw'/f'p{p}.json').write_text(json.dumps(r))
    return tmp_path


def test_scaling_math(tmp_path):
    rows,_=aggregate(campaign(tmp_path))
    assert rows[1]['median_us']==500
    assert rows[1]['speedup']==2
    assert rows[1]['efficiency']==1


@pytest.mark.parametrize('bad',['missing','timer','hash','median','dofs','validation','cores'])
def test_reject_mixed_or_incomplete(tmp_path,bad):
    campaign(tmp_path)
    p=tmp_path/'raw/p2.json'
    if bad=='missing':p.unlink()
    else:
        r=json.loads(p.read_text())
        if bad=='timer':r['timing']['timer']='other'
        if bad=='hash':r['source_hashes']['a']='different'
        if bad=='median':r['timing']['median_seconds_per_matvec']=1
        if bad=='dofs':r['global_dofs']=200
        if bad=='cores':r['placement'][0]['cpu_affinity']=[0,2]
        if bad=='validation':r['validation'][0]['error_l2']=1
        p.write_text(json.dumps(r))
    with pytest.raises(ValueError):aggregate(tmp_path)


def test_launcher_failures_are_bounded(tmp_path):
    from runner import launch
    with pytest.raises(RuntimeError,match='Launch failed'):
        launch([sys.executable,'-c','raise SystemExit(3)'],tmp_path/'failed.log',10)
    with pytest.raises(RuntimeError,match='timed out'):
        launch([sys.executable,'-c','import time; time.sleep(30)'],tmp_path/'timeout.log',.2)
