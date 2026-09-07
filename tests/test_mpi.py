"""Collective pytest cases; run via scripts/run_mpi_tests.py."""

from importlib.util import find_spec
from types import SimpleNamespace
import pytest

from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np

from jaxghost import JAXGhost


COMM = MPI.COMM_WORLD
jax.config.update("jax_enable_x64", True)


def expand_ids(ids, block_size):
    return (np.asarray(ids)[:, None] * block_size + np.arange(block_size)).ravel()


def synthetic_map(kind):
    # Unequal owned sizes exercise global-to-local translation.
    sizes = np.arange(2, COMM.size + 2)
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    rank = COMM.rank
    requests = []
    if kind == "irregular":
        for owner in reversed(range(COMM.size)):
            if owner != rank:
                requests.append((int(offsets[owner + 1] - 1), owner))
                if (rank + owner) % 2 == 0 or rank == 0:
                    requests.append((int(offsets[owner]), owner))
    elif kind == "one_way" and rank != 0:
        requests = [(1, 0), (0, 0)]  # Rank 0 has no ghosts, but must send.
    elif kind == "empty_owner" and rank == 0 and COMM.size > 1:
        requests = [(int(offsets[1]), 1)]
    if kind == "empty_owner" and rank == 0:
        start, stop = 0, 0
    else:
        start, stop = int(offsets[rank]), int(offsets[rank + 1])
    return SimpleNamespace(
        size_local=stop - start,
        local_range=(start, stop),
        ghosts=np.array([g for g, _ in requests], dtype=np.int64),
        owners=np.array([p for _, p in requests], dtype=np.int32),
    )


class TestForward:
    def assert_collective_equal(self, actual, expected):
        error = None
        try:
            np.testing.assert_array_equal(actual, expected)
        except AssertionError as exc:
            error = f"rank {COMM.rank}: {exc}"
        errors = COMM.allgather(error)
        assert not any(errors), "\n".join(e for e in errors if e)

    def exercise(self, index_map, reference=None, block_size=1):
        n = index_map.size_local * block_size
        owned_ids = expand_ids(np.arange(*index_map.local_range), block_size)
        global_ids = np.concatenate((owned_ids, expand_ids(index_map.ghosts, block_size)))
        with JAXGhost.from_index_map(index_map, COMM, block_size=block_size) as ghost:
            for compiled in (False, True):
                forward = jax.jit(ghost.scatter_forward) if compiled else ghost.scatter_forward
                for dtype in (np.float32, np.float64):
                    x = jnp.full((len(global_ids),), jnp.nan, dtype=dtype)
                    for step in range(3):
                        owned = owned_ids.astype(dtype) * 2 + 10 + 100 * step
                        x = x.at[:n].set(jnp.asarray(owned))
                        before = np.asarray(x).copy()
                        updated = forward(x)
                        updated.block_until_ready()
                        jax.effects_barrier()
                        assert isinstance(updated, jax.Array)
                        assert updated.devices() == x.devices()
                        self.assert_collective_equal(
                            np.asarray(updated), global_ids.astype(dtype) * 2 + 10 + 100 * step
                        )
                        self.assert_collective_equal(np.asarray(x), before)
                        if reference is not None:
                            reference.x.array[:n] = owned
                            reference.x.scatter_forward()
                            self.assert_collective_equal(np.asarray(updated), reference.x.array)
                        x = updated

    def test_irregular_multiple_neighbors(self):
        self.exercise(synthetic_map("irregular"))

    def test_zero_ghost_sender_and_asymmetric_messages(self):
        self.exercise(synthetic_map("one_way"))

    def test_no_communication(self):
        self.exercise(synthetic_map("none"))

    def test_no_owned_entries(self):
        self.exercise(synthetic_map("empty_owner"))

    @pytest.mark.skipif(find_spec("dolfinx") is None, reason="DOLFINx is not installed")
    @pytest.mark.parametrize("degree", (1, 2, 3), ids=("P1", "P2", "P3"))
    def test_dolfinx_interval(self, degree):
        from dolfinx import fem, mesh

        domain = mesh.create_unit_interval(COMM, 8)
        space = fem.functionspace(domain, ("Lagrange", degree))
        assert space.dofmap.index_map_bs == 1
        if degree > 1:
            # Higher-order spaces include DoFs beyond mesh vertices.
            assert space.dofmap.index_map.size_global > domain.topology.index_map(0).size_global
        self.exercise(space.dofmap.index_map, fem.Function(space, dtype=np.float64))

    @pytest.mark.skipif(find_spec("dolfinx") is None, reason="DOLFINx is not installed")
    @pytest.mark.parametrize("degree", (1, 2, 3), ids=("P1", "P2", "P3"))
    def test_dolfinx_square(self, degree):
        from dolfinx import fem, mesh

        domain = mesh.create_unit_square(
            COMM, 8, 8, cell_type=mesh.CellType.triangle, dtype=np.float64
        )
        space = fem.functionspace(domain, ("Lagrange", degree))
        assert space.dofmap.index_map_bs == 1
        if degree > 1:
            # Higher-order spaces include DoFs beyond mesh vertices.
            assert space.dofmap.index_map.size_global > domain.topology.index_map(0).size_global
        self.exercise(space.dofmap.index_map, fem.Function(space, dtype=np.float64))

    @pytest.mark.skipif(find_spec("dolfinx") is None, reason="DOLFINx is not installed")
    @pytest.mark.parametrize("degree", (1, 2, 3), ids=("P1", "P2", "P3"))
    def test_dolfinx_cube(self, degree):
        from dolfinx import fem, mesh

        domain = mesh.create_unit_cube(
            COMM, 4, 4, 4, cell_type=mesh.CellType.tetrahedron, dtype=np.float64
        )
        space = fem.functionspace(domain, ("Lagrange", degree))
        assert space.dofmap.index_map_bs == 1
        if degree > 1:
            # Higher-order spaces include DoFs beyond mesh vertices.
            assert space.dofmap.index_map.size_global > domain.topology.index_map(0).size_global
        self.exercise(space.dofmap.index_map, fem.Function(space, dtype=np.float64))

    @pytest.mark.parametrize("block_size", (0, -1, 1.5, True, "2"))
    def test_invalid_block_size(self, block_size):
        with pytest.raises(ValueError, match="block_size"):
            JAXGhost.from_index_map(synthetic_map("none"), COMM, block_size=block_size)

    def test_invalid_metadata_is_collective(self):
        index_map = synthetic_map("none")
        if COMM.rank == 0:
            index_map.ghosts = np.array([0], dtype=np.int64)
            index_map.owners = np.array([COMM.size], dtype=np.int32)
        with pytest.raises(ValueError, match="ghost owners"):
            JAXGhost.from_index_map(index_map, COMM)

    @pytest.mark.skipif(COMM.size == 1, reason="requires a remote owner")
    def test_requested_id_outside_owner_range(self):
        index_map = synthetic_map("none")
        if COMM.rank == 0:
            index_map.ghosts = np.array([999999], dtype=np.int64)
            index_map.owners = np.array([1], dtype=np.int32)
        with pytest.raises(ValueError, match="outside"):
            JAXGhost.from_index_map(index_map, COMM)

    @pytest.mark.parametrize("operation", ("scatter_forward", "scatter_reverse"))
    def test_shape_dtype_and_close(self, operation):
        ghost = JAXGhost.from_index_map(synthetic_map("none"), COMM)
        x = jnp.zeros((ghost.n_owned,), dtype=jnp.float32)
        with pytest.raises(TypeError, match="JAX array"):
            getattr(ghost, operation)(np.zeros((ghost.n_owned,), dtype=np.float32))
        with pytest.raises(ValueError, match="shape"):
            getattr(ghost, operation)(x[:, None])
        with pytest.raises(TypeError, match="float32 and float64"):
            getattr(ghost, operation)(x.astype(jnp.int32))
        ghost.close()
        ghost.close()
        with pytest.raises(RuntimeError, match="closed"):
            getattr(ghost, operation)(x)


def assert_collective_close(actual, expected, *, rtol=0, atol=0):
    error = None
    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as exc:
        error = f"rank {COMM.rank}: {exc}"
    errors = COMM.allgather(error)
    assert not any(errors), "\n".join(e for e in errors if e)


class TestReverse:
    def exercise(self, index_map, reference=None, block_size=1):
        n = index_map.size_local * block_size
        start, stop = (v * block_size for v in index_map.local_range)
        owned_ids = np.arange(start, stop)
        ghost_ids = expand_ids(index_map.ghosts, block_size)
        with JAXGhost.from_index_map(index_map, COMM, block_size=block_size) as ghost:
            for compiled in (False, True):
                reverse = jax.jit(ghost.scatter_reverse) if compiled else ghost.scatter_reverse
                forward = jax.jit(ghost.scatter_forward) if compiled else ghost.scatter_forward
                for dtype in (np.float32, np.float64):
                    initial_owned = (10 + owned_ids * 0.1).astype(dtype)
                    contributions = ((COMM.rank + 1) * 0.3 + ghost_ids * 0.01).astype(dtype)
                    expected = np.concatenate((initial_owned, contributions))
                    x = jnp.asarray(expected)
                    # Independent, global-ID-based oracle, used only by tests.
                    requests = COMM.allgather((ghost_ids, contributions))
                    if reference is not None:
                        reference.x.array[:] = expected
                    tolerance = 32 * np.finfo(dtype).eps
                    for _ in range(2):
                        before = np.asarray(x).copy()
                        updated = reverse(x)
                        updated.block_until_ready()
                        jax.effects_barrier()
                        for ids, values in requests:
                            mask = (ids >= start) & (ids < stop)
                            np.add.at(expected[:n], ids[mask] - start, values[mask])
                        assert isinstance(updated, jax.Array)
                        assert updated.devices() == x.devices()
                        assert_collective_close(np.asarray(x), before)
                        assert_collective_close(np.asarray(updated)[n:], contributions)
                        assert_collective_close(np.asarray(updated), expected,
                                                rtol=tolerance, atol=tolerance)
                        if reference is not None:
                            from dolfinx import la
                            reference.x.scatter_reverse(la.InsertMode.add)
                            assert_collective_close(np.asarray(updated), reference.x.array,
                                                    rtol=tolerance, atol=tolerance)
                        x = updated

                    # Ghosts contain contributions until an explicit forward refresh.
                    all_owned = COMM.allgather((owned_ids, expected[:n].copy()))
                    owner_values = {int(g): v for ids, values in all_owned
                                    for g, v in zip(ids, values)}
                    expected[n:] = [owner_values[int(g)] for g in ghost_ids]
                    refreshed = forward(x)
                    refreshed.block_until_ready()
                    jax.effects_barrier()
                    assert_collective_close(np.asarray(refreshed), expected,
                                            rtol=tolerance, atol=tolerance)
                    if reference is not None:
                        reference.x.scatter_forward()
                        assert_collective_close(np.asarray(refreshed), reference.x.array,
                                                rtol=tolerance, atol=tolerance)

    @pytest.mark.parametrize("kind", ("irregular", "one_way", "none", "empty_owner"))
    def test_synthetic(self, kind):
        # With >=3 ranks, one_way has multiple senders adding to rank 0's same
        # owned entries. Rank 0 has no ghosts; other ranks only send in reverse.
        self.exercise(synthetic_map(kind))

    @pytest.mark.skipif(find_spec("dolfinx") is None, reason="DOLFINx is not installed")
    @pytest.mark.parametrize("dimension", (1, 2, 3))
    @pytest.mark.parametrize("degree", (1, 2, 3), ids=("P1", "P2", "P3"))
    def test_dolfinx(self, dimension, degree):
        from dolfinx import fem, mesh
        if dimension == 1:
            domain = mesh.create_unit_interval(COMM, 8)
        elif dimension == 2:
            domain = mesh.create_unit_square(COMM, 8, 8, cell_type=mesh.CellType.triangle)
        else:
            domain = mesh.create_unit_cube(COMM, 4, 4, 4, cell_type=mesh.CellType.tetrahedron)
        space = fem.functionspace(domain, ("Lagrange", degree))
        assert space.dofmap.index_map_bs == 1
        self.exercise(space.dofmap.index_map, fem.Function(space, dtype=np.float64))


@pytest.mark.parametrize("block_size", (2, 3))
@pytest.mark.parametrize("kind", ("irregular", "one_way", "none", "empty_owner"))
def test_blocked_synthetic(kind, block_size):
    index_map = synthetic_map(kind)
    TestForward().exercise(index_map, block_size=block_size)
    TestReverse().exercise(index_map, block_size=block_size)


@pytest.mark.skipif(find_spec("dolfinx") is None, reason="DOLFINx is not installed")
@pytest.mark.parametrize("dimension", (1, 2, 3))
@pytest.mark.parametrize("degree", (1, 2, 3))
@pytest.mark.parametrize(
    "value_shape",
    ((2,), (3,), (2, 2), (3, 3), (2, 2, 2), (3, 3, 3),
     (2, 2, 2, 2), (3, 3, 3, 3)),
    ids=("vector2", "vector3", "tensor2-order2", "tensor3-order2",
         "tensor2-order3", "tensor3-order3", "tensor2-order4", "tensor3-order4"),
)
def test_dolfinx_blocked(dimension, degree, value_shape):
    from dolfinx import fem, mesh
    if dimension == 1:
        domain = mesh.create_unit_interval(COMM, 8)
    elif dimension == 2:
        domain = mesh.create_unit_square(COMM, 4, 4)
    else:
        domain = mesh.create_unit_cube(COMM, 2, 2, 2)
    space = fem.functionspace(domain, ("Lagrange", degree, value_shape))
    # Full tensors use one consecutive block of components per scalar DoF.
    # Use the actual DOLFINx space, rather than a vector with the same size.
    block_size = int(np.prod(value_shape))
    assert tuple(space.element.value_shape) == value_shape
    assert space.dofmap.index_map_bs == block_size
    reference = fem.Function(space, dtype=np.float64)
    TestForward().exercise(space.dofmap.index_map, reference, block_size)
    TestReverse().exercise(space.dofmap.index_map, reference, block_size)


@pytest.mark.skipif(COMM.size == 1, reason="requires multiple ranks")
def test_inconsistent_block_size():
    with pytest.raises(ValueError, match="identical"):
        JAXGhost.from_index_map(synthetic_map("none"), COMM, block_size=COMM.rank + 1)


@pytest.mark.parametrize("operation", ("scatter_forward", "scatter_reverse"))
def test_blocked_shape(operation):
    index_map = synthetic_map("none")
    with JAXGhost.from_index_map(index_map, COMM, block_size=2) as ghost:
        assert ghost.block_size == 2
        assert ghost.n_owned == 2 * index_map.size_local
        with pytest.raises(ValueError, match="shape"):
            getattr(ghost, operation)(jnp.zeros((index_map.size_local,)))


@pytest.mark.skipif(find_spec("dolfinx") is None, reason="DOLFINx is not installed")
@pytest.mark.parametrize("degree", (1, 2, 3), ids=("P1", "P2", "P3"))
@pytest.mark.parametrize("value_shape", ((), (2,), (2, 2)),
                         ids=("scalar", "vector", "tensor"))
def test_irregular_geometry(degree, value_shape):
    from dolfinx import fem
    from examples.irregular import create_irregular_mesh

    domain = create_irregular_mesh(COMM)
    counts = COMM.allgather(domain.topology.index_map(2).size_local)
    assert sum(counts) == 53
    if 2 <= COMM.size <= 4:
        assert len(set(counts)) > 1, f"Expected uneven cell counts, got {counts}"
    space = fem.functionspace(domain, ("Lagrange", degree, value_shape))
    reference = fem.Function(space, dtype=np.float64)
    block_size = space.dofmap.index_map_bs
    TestForward().exercise(space.dofmap.index_map, reference, block_size)
    TestReverse().exercise(space.dofmap.index_map, reference, block_size)
