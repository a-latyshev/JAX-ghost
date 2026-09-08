"""Independent two-rank test of the pinned mpi4jax array-return API."""

from importlib.metadata import version
import platform

from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax
import numpy as np


comm = MPI.COMM_WORLD
if comm.size != 2:
    raise RuntimeError("mpi_smoke.py requires exactly two ranks")
runtime_comm = comm.Dup()


@jax.jit
def exchange(x):
    return mpi4jax.sendrecv(
        x, jnp.empty_like(x), source=1 - comm.rank, dest=1 - comm.rank,
        sendtag=0, recvtag=0, comm=runtime_comm,
    )


for step in range(2):
    received = exchange(jnp.asarray([comm.rank + 10 * step], dtype=jnp.float32))
    received.block_until_ready()
    jax.effects_barrier()
    correct = np.array_equal(np.asarray(received), [1 - comm.rank + 10 * step])
    if not comm.allreduce(correct, op=MPI.LAND):
        raise AssertionError("compiled mpi4jax sendrecv failed")
runtime_comm.Free()
if comm.rank == 0:
    print("Two-rank compiled mpi4jax smoke test passed", flush=True)
    print("Python:", platform.python_version(), "Platform:", platform.platform())
    print({name: version(name) for name in ("jax", "jaxlib", "mpi4jax", "mpi4py", "numpy")})
    print("MPI:", MPI.Get_library_version().strip(), flush=True)
