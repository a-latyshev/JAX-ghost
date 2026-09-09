"""Load a selected exported partition set and validate MPI ownership metadata."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from common import load_fixture
from jaxghost._metadata import build_plan
from jaxghost._csr_metadata import _read_structure
from mpi4py import MPI

comm=MPI.COMM_WORLD
try:
    meta,data,adapter=load_fixture(Path(sys.argv[1]),comm)
    _read_structure(adapter,comm)
    build_plan(adapter.index_map(1),comm)
    if comm.rank==0:print(f"PASS metadata n={meta['subdivisions']}, ranks={comm.size}",flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    comm.Abort(1)
