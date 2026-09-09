"""GPU selection checks that need neither MPI nor CUDA."""
import os
import pytest
from gpu_bootstrap import bootstrap


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for key in ('OMPI_COMM_WORLD_LOCAL_RANK', 'MPI_LOCALRANKID',
                'SLURM_LOCALID', 'CUDA_VISIBLE_DEVICES'):
        monkeypatch.delenv(key, raising=False)


def test_slurm_selection(monkeypatch):
    monkeypatch.setenv('SLURM_LOCALID', '1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,5,6,7')
    assert bootstrap() == (1, None)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '5'


def test_scheduler_single_gpu(monkeypatch):
    monkeypatch.setenv('SLURM_LOCALID', '3')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-allocated-uuid')
    assert bootstrap() == (3, None)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-allocated-uuid'


def test_mpirun_precedes_inherited_slurm(monkeypatch):
    monkeypatch.setenv('OMPI_COMM_WORLD_LOCAL_RANK', '1')
    monkeypatch.setenv('SLURM_LOCALID', '0')
    assert bootstrap() == (1, None)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'


@pytest.mark.parametrize('rank,visible', [('4','0,1'), ('-1','0,1'), ('bad','0,1'), ('0','')])
def test_deferred_failure(monkeypatch,rank,visible):
    monkeypatch.setenv('SLURM_LOCALID', rank)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    selected,error = bootstrap()
    assert selected is None
    assert error is not None


def test_unknown_launcher_keeps_mask(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    assert bootstrap() == (None,None)
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
