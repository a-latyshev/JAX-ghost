"""Optional assembly experiment; excluded from the essential suite."""

from importlib.util import find_spec
import pytest
from mpi4py import MPI

COMM = MPI.COMM_WORLD


@pytest.mark.skipif(find_spec("dolfinx") is None, reason="DOLFINx is not installed")
def test_assembly_pipeline():
    from experiments.assembly import run_assembly

    for compiled in (False, True):
        run_assembly(COMM, compiled=compiled)
