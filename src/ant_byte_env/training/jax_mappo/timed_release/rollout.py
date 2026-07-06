"""Rollout entrypoint for timed-release MAPPO experiments."""

from __future__ import annotations

from ant_byte_env.training.jax_mappo.rollout import collect_rollout

__all__ = ["collect_rollout"]
