"""Explicitly launched after distributed initialization; not default pytest discovery."""

from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mpi4py import MPI
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxghost import ShardedJAXGhost

COMM = MPI.COMM_WORLD
MESH = Mesh(np.asarray(jax.devices()), ('rank',))


def assert_equal(actual, expected):
    error = None
    try:
        np.testing.assert_array_equal(actual, expected)
    except AssertionError as exc:
        error = f'rank {COMM.rank}: {exc}'
    errors = COMM.allgather(error)
    assert not any(errors), '\n'.join(e for e in errors if e)


def layout(kind):
    sizes = np.arange(2, COMM.size + 2)
    if kind == 'empty':
        sizes[0] = 0
    if kind == 'all_empty':
        sizes[:] = 0
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    requests = []
    if kind == 'irregular':
        requests = [(int(offsets[p + 1] - 1), p) for p in reversed(range(COMM.size)) if p != COMM.rank]
    elif kind == 'one_way' and COMM.rank != 0:
        requests = [(1, 0), (0, 0)]
    elif kind == 'empty' and COMM.rank == 0 and COMM.size > 1:
        requests = [(int(offsets[1]), 1)]
    return SimpleNamespace(size_local=int(sizes[COMM.rank]),
                           local_range=tuple(offsets[COMM.rank:COMM.rank + 2]),
                           ghosts=np.array([g for g, _ in requests], dtype=np.int64),
                           owners=np.array([p for _, p in requests], dtype=np.int32))


def exercise(index_map, block_size=1, reference=None):
    ghost = ShardedJAXGhost.from_index_map(index_map, COMM, MESH, block_size=block_size)
    ids = np.concatenate((np.arange(*index_map.local_range), index_map.ghosts))
    ids = (ids[:, None] * block_size + np.arange(block_size)).ravel()
    for compiled in (False, True):
        forward = jax.jit(ShardedJAXGhost.scatter_forward) if compiled else ShardedJAXGhost.scatter_forward
        for dtype in (np.float32, np.float64):
            for step in (0, 1):
                expected = (10 + ids + 100 * step).astype(dtype)
                local = jnp.full((len(ids),), jnp.nan, dtype=dtype)
                local = local.at[:ghost.n_owned].set(jnp.asarray(expected[:ghost.n_owned]))
                x = ghost.to_sharded(local)
                # Nonzero padding detects accidental writes by invalid packets.
                piece = x.addressable_shards[0].data.at[:, len(ids):].set(-987)
                x = jax.make_array_from_single_device_arrays(x.shape, x.sharding, [piece])
                before = np.asarray(piece).copy()
                updated = forward(ghost, x)
                updated.block_until_ready()
                assert_equal(np.asarray(ghost.local_array(updated)), expected)
                assert_equal(np.asarray(x.addressable_shards[0].data), before)
                assert_equal(np.asarray(updated.addressable_shards[0].data)[0, len(ids):],
                             before[0, len(ids):])
                assert ghost.local_array(updated).devices() == local.devices()
                if reference is not None:
                    reference.x.array[:ghost.n_owned] = expected[:ghost.n_owned]
                    reference.x.scatter_forward()
                    assert_equal(np.asarray(ghost.local_array(updated)), reference.x.array)


def test_native_alltoall():
    nranks = COMM.size
    local = jax.device_put(np.full((1, nranks, 1), COMM.rank, dtype=np.float32), jax.local_devices()[0])
    sharding = NamedSharding(MESH, P('rank', None, None))
    x = jax.make_array_from_single_device_arrays((nranks, nranks, 1), sharding, [local])
    def body(piece):
        return jax.lax.all_to_all(piece[0], 'rank', split_axis=0, concat_axis=0, tiled=True)[None]
    exchange = jax.jit(jax.shard_map(body, mesh=MESH, in_specs=P('rank', None, None),
                                    out_specs=P('rank', None, None)))
    result = exchange(x)
    result.block_until_ready()
    assert_equal(np.asarray(result.addressable_shards[0].data).ravel(), np.arange(nranks))


@pytest.mark.parametrize('kind', ('irregular', 'one_way', 'empty', 'all_empty'))
def test_synthetic(kind):
    exercise(layout(kind), block_size=2)


def test_dolfinx_irregular():
    from dolfinx import fem
    from examples.irregular import create_irregular_mesh
    domain = create_irregular_mesh(COMM)
    space = fem.functionspace(domain, ('Lagrange', 2, (2,)))
    exercise(space.dofmap.index_map, space.dofmap.index_map_bs, fem.Function(space))


def test_validation():
    m = layout('one_way')
    ghost = ShardedJAXGhost.from_index_map(m, COMM, MESH)
    local = jnp.zeros((ghost.n_owned + ghost.n_ghost,), dtype=jnp.float32)
    with pytest.raises(TypeError, match='JAX'):
        ghost.to_sharded(np.asarray(local))
    with pytest.raises(ValueError, match='shape'):
        ghost.to_sharded(local[:, None])
    with pytest.raises(TypeError, match='float32'):
        ghost.to_sharded(local.astype(jnp.int32))
    x = ghost.to_sharded(local)
    with pytest.raises(ValueError, match='shape'):
        ghost.scatter_forward(local)
    with pytest.raises(TypeError, match='outside JIT'):
        jax.jit(ghost.local_array)(x)
    with pytest.raises(ValueError, match='block_size'):
        ShardedJAXGhost.from_index_map(m, COMM, MESH, block_size=0 if COMM.rank == 0 else 1)
    with pytest.raises(ValueError, match='rank'):
        ShardedJAXGhost.from_index_map(m, COMM, Mesh(np.asarray(jax.devices()), ('wrong',)))
    if COMM.size > 1:
        with pytest.raises(ValueError, match='rank order'):
            ShardedJAXGhost.from_index_map(m, COMM, Mesh(np.asarray(jax.devices())[::-1], ('rank',)))
        with pytest.raises(ValueError, match='communicator'):
            ShardedJAXGhost.from_index_map(m, MPI.COMM_SELF, MESH)
        replicated = jax.device_put(np.zeros(x.shape, dtype=np.float32), NamedSharding(MESH, P()))
        with pytest.raises(ValueError, match='sharding'):
            ghost.scatter_forward(replicated)
