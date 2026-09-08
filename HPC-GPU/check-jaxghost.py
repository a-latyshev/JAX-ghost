"""Strict DOLFINx CPU / JAXGhost GPU check; run with >=2 MPI ranks.

Requires one visible GPU per process and MPI4JAX_USE_CUDA_MPI=1.
CUDA-aware buffers do not imply GPUDirect or exclude internal MPI staging.
"""
import argparse
import ctypes
import faulthandler
import os
from pathlib import Path
import sys
import traceback

from mpi4py import MPI

# Test this checkout, without installing it or modifying the environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Print per-rank diagnostic stages")
    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    def stage(message):
        if args.verbose:
            print(f"rank={comm.rank} stage={message}", flush=True)

    stage("MPI initialized; checking CUDA-aware support")
    assert comm.size >= 2, "Launch with at least two MPI ranks"
    assert os.environ.get("MPI4JAX_USE_CUDA_MPI") == "1", (
        "Set MPI4JAX_USE_CUDA_MPI=1; CPU staging is not allowed in this test"
    )
    assert MPI.Is_initialized(), "MPI must be initialized before querying CUDA support"
    vendor, _ = MPI.get_vendor()
    if vendor != "Open MPI":
        raise RuntimeError(f"CUDA-aware MPI verification unsupported for {vendor}")
    # Resolve through mpi4py's extension dependency scope, not a guessed libmpi
    # path that could load a different MPI installation.
    mpi_extension = ctypes.CDLL(MPI.__file__)
    try:
        query = mpi_extension.MPIX_Query_cuda_support
    except AttributeError as exc:
        raise RuntimeError("Loaded MPI lacks MPIX_Query_cuda_support") from exc
    query.argtypes = []
    query.restype = ctypes.c_int
    assert query() == 1, "Loaded OpenMPI reports no runtime CUDA support"

    stage("OpenMPI CUDA query passed; importing JAX/mpi4jax")
    import jax
    import mpi4jax
    import numpy as np
    stage("importing DOLFINx")
    from dolfinx import fem, la, mesh
    from jaxghost import JAXGhost

    stage("imports complete; initializing distributed JAX")
    assert mpi4jax.has_cuda_support(), "mpi4jax CUDA extension is unavailable"
    jax.config.update("jax_platforms", "cuda")
    jax.config.update("jax_enable_x64", True)
    jax.distributed.initialize(local_device_ids=[0])
    assert jax.process_count() == comm.size
    assert jax.local_device_count() == 1
    assert jax.device_count() == comm.size
    device = jax.local_devices()[0]
    assert device.platform == "gpu"

    def on_gpu(x):
        assert isinstance(x, jax.Array)
        assert x.devices() == {device}, f"Unexpected placement: {x.devices()}"

    stage(f"GPU ready: {device}; creating DOLFINx mesh")
    domain = mesh.create_unit_square(
        comm, max(4, 2 * comm.size), 4, cell_type=mesh.CellType.triangle
    )
    stage("mesh created; creating function space")
    space = fem.functionspace(domain, ("Lagrange", 1))
    imap = space.dofmap.index_map
    assert space.dofmap.index_map_bs == 1
    start, stop = imap.local_range
    assert stop - start == imap.size_local
    ranges = comm.allgather((start, stop))
    ordered = sorted(ranges)
    assert ordered[0][0] == 0 and ordered[-1][1] == imap.size_global
    assert all(a[1] == b[0] for a, b in zip(ordered, ordered[1:]))
    assert len(imap.ghosts) == len(imap.owners) == imap.num_ghosts
    for gid, owner in zip(imap.ghosts, imap.owners):
        assert 0 <= owner < comm.size and owner != comm.rank
        assert ranges[owner][0] <= gid < ranges[owner][1]
    assert comm.allreduce(imap.num_ghosts, op=MPI.SUM) > 0, "No ghost exchange exercised"

    stage("ownership checked; creating host functions")
    source = fem.Function(space, dtype=np.float64)
    reference = fem.Function(space, dtype=np.float64)
    assert isinstance(source.x.array, np.ndarray), "Expected host DOLFINx storage"
    n = imap.size_local
    gids = np.concatenate((np.arange(start, stop), imap.ghosts))
    stage("creating JAXGhost plan")
    with JAXGhost.from_index_map(imap, comm, block_size=1) as ghost:
        stage("JAXGhost plan ready")
        on_gpu(ghost.send_indices)
        on_gpu(ghost.receive_positions)
        forward = jax.jit(ghost.scatter_forward)
        reverse = jax.jit(ghost.scatter_reverse)
        for step in range(2):
            for operation in ("forward", "reverse"):
                source.x.array[:n] = 10 + 100 * step + np.arange(start, stop)
                source.x.array[n:] = -999 if operation == "forward" else (step + 1) * (comm.rank + 1)
                before = source.x.array.copy()
                reference.x.array[:] = before
                stage(f"step={step} {operation}: host-to-GPU copy")
                x = jax.device_put(before, device)
                on_gpu(x)
                stage(f"step={step} {operation}: GPU exchange starting")
                updated = (forward if operation == "forward" else reverse)(x)
                updated.block_until_ready()
                jax.effects_barrier()
                stage(f"step={step} {operation}: GPU exchange completed")
                on_gpu(updated)
                np.testing.assert_array_equal(source.x.array, before)

                stage(f"step={step} {operation}: DOLFINx reference starting")
                # Reference communication follows completed GPU communication.
                if operation == "forward":
                    reference.x.scatter_forward()
                else:
                    reference.x.scatter_reverse(la.InsertMode.add)
                stage(f"step={step} {operation}: reference completed; comparing")
                actual = np.asarray(updated)  # Validation-only device-to-host copy.
                np.testing.assert_array_equal(actual, reference.x.array)
                np.testing.assert_array_equal(np.asarray(x), before)
                if operation == "forward":
                    np.testing.assert_array_equal(actual[:n], before[:n])
                    np.testing.assert_array_equal(actual, 10 + 100 * step + gids)
                else:
                    np.testing.assert_array_equal(actual[n:], before[n:])
                np.testing.assert_array_equal(source.x.array, before)
    stage("checks completed; shutting down")
    comm.Barrier()
    jax.distributed.shutdown()
    comm.Barrier()
    print(
        f"rank={comm.rank} owned={n} ghosts={imap.num_ghosts} GPU={device} "
        "CPU ownership / GPU placement / forward / reverse PASS", flush=True,
    )


if __name__ == "__main__":
    faulthandler.enable(all_threads=True)
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)
