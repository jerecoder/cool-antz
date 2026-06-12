"""Compatibility shim for legacy JAX MAPPO checkpoints.

Old checkpoints were pickled when the JAX core lived at this top-level module
name. Keep the import target available so ``pickle.load`` can resolve those
NamedTuple classes after the trainer refactor.
"""

from ant_byte_env.training.jax_mappo.core import *  # noqa: F401,F403
