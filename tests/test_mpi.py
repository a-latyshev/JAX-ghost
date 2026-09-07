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

    def exercise(self, index_map, reference=None):
        n = index_map.size_local
        owned_ids = np.arange(*index_map.local_range)
        global_ids = np.concatenate((owned_ids, index_map.ghosts))
        with JAXGhost.from_index_map(index_map, COMM) as ghost:
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
    def test_dolfinx_interval(self):
        from dolfinx import fem, mesh

        domain = mesh.create_unit_interval(COMM, max(8, 4 * COMM.size))
        space = fem.functionspace(domain, ("Lagrange", 1))
        assert space.dofmap.index_map_bs == 1
        self.exercise(space.dofmap.index_map, fem.Function(space, dtype=np.float64))

    @pytest.mark.skipif(find_spec("dolfinx") is None, reason="DOLFINx is not installed")
    def test_dolfinx_square(self):
        from dolfinx import fem, mesh

        domain = mesh.create_unit_square(
            COMM, 8, 8, cell_type=mesh.CellType.triangle, dtype=np.float64
        )
        space = fem.functionspace(domain, ("Lagrange", 1))
        assert space.dofmap.index_map_bs == 1
        self.exercise(space.dofmap.index_map, fem.Function(space, dtype=np.float64))

    @pytest.mark.skipif(find_spec("dolfinx") is None, reason="DOLFINx is not installed")
    def test_dolfinx_cube(self):
        from dolfinx import fem, mesh

        domain = mesh.create_unit_cube(
            COMM, 4, 4, 4, cell_type=mesh.CellType.tetrahedron, dtype=np.float64
        )
        space = fem.functionspace(domain, ("Lagrange", 1))
        assert space.dofmap.index_map_bs == 1
        self.exercise(space.dofmap.index_map, fem.Function(space, dtype=np.float64))

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            JAXGhost.from_index_map(synthetic_map("none"), COMM, block_size=2)

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

    def test_shape_dtype_and_close(self):
        ghost = JAXGhost.from_index_map(synthetic_map("none"), COMM)
        x = jnp.zeros((ghost.n_owned,), dtype=jnp.float32)
        with pytest.raises(TypeError, match="JAX array"):
            ghost.scatter_forward(np.zeros((ghost.n_owned,), dtype=np.float32))
        with pytest.raises(ValueError, match="shape"):
            ghost.scatter_forward(x[:, None])
        with pytest.raises(TypeError, match="float32 and float64"):
            ghost.scatter_forward(x.astype(jnp.int32))
        ghost.close()
        ghost.close()
        with pytest.raises(RuntimeError, match="closed"):
            ghost.scatter_forward(x)
