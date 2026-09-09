"""Sharded JAX operations, with an optional mpi4jax execution backend."""
from importlib import import_module

from .sharded import ShardedJAXGhost
from .sharded_matrix import ShardedJAXMatrixCSR

__all__ = ["JAXGhost", "JAXMatrixCSR", "ShardedJAXGhost", "ShardedJAXMatrixCSR"]


def __getattr__(name):
    modules = {"JAXGhost": ".jaxghost", "JAXMatrixCSR": ".matrix"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
