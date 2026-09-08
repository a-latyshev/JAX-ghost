"""Four-rank GPU collective check; launch with srun --mpi=pmix."""
import sys
from importlib.metadata import version

from mpi4py import MPI
import jax
import jax.numpy as jnp
import mpi4jax

world = MPI.COMM_WORLD
try:
    jax.config.update("jax_platforms", "cuda")
    jax.distributed.initialize(local_device_ids=[0])
    assert world.size == 4, "Launch with four MPI ranks"
    assert jax.process_count() == world.size
    assert jax.local_device_count() == 1
    assert jax.device_count() == world.size
    assert mpi4jax.has_cuda_support(), "mpi4jax was built without CUDA support"

    # Keep mpi4jax operations separate from mpi4py control collectives.
    comm = world.Dup()

    @jax.jit
    def reduce_sum(x):
        # mpi4jax 0.9.1.post1 returns an array, not an (array, token) pair.
        return mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)

    x = jax.device_put(float(world.rank + 1), jax.local_devices()[0])
    result = reduce_sum(x)
    result.block_until_ready()
    assert float(result) == 10.0, result
    print(
        f"rank={world.rank} jax={jax.__version__} "
        f"mpi4jax={version('mpi4jax')} GPU sum={float(result)} PASS",
        flush=True,
    )
    jax.effects_barrier()
    world.Barrier()
    comm.Free()
    print(f"rank={world.rank} entering JAX shutdown", flush=True)
    jax.distributed.shutdown()
    print(f"rank={world.rank} returned from JAX shutdown", flush=True)
    world.Barrier()
except Exception:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    world.Abort(1)
