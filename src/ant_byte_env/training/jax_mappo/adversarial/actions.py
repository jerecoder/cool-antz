"""Action selection helpers for adversarial MAPPO policies."""

from __future__ import annotations

import jax
import jax.numpy as jnp

ADVERSARIAL_ACTION_MODES = (
    "deterministic",
    "sampled",
    "greedy_move_greedy_write",
    "greedy_move_sampled_write",
    "sampled_move_greedy_write",
    "sampled_move_sampled_write",
    "greedy_move_zero_write",
    "sampled_move_zero_write",
)


def actions_from_logits(
    move_logits: jax.Array,
    write_logits: jax.Array,
    key: jax.Array,
    *,
    action_mode: str,
    move_temperature: float = 1.0,
    write_temperature: float = 1.0,
) -> jax.Array:
    move_mode, write_mode = _split_action_mode(validate_action_mode(action_mode))
    move_key, write_key = jax.random.split(key)
    move_actions = _head_actions(
        move_logits,
        move_key,
        mode=move_mode,
        temperature=move_temperature,
    )
    write_actions = _head_actions(
        write_logits,
        write_key,
        mode=write_mode,
        temperature=write_temperature,
    )
    return jnp.stack([move_actions, write_actions], axis=-1).astype(jnp.int32)


def validate_action_mode(action_mode: str) -> str:
    if action_mode not in ADVERSARIAL_ACTION_MODES:
        choices = ", ".join(ADVERSARIAL_ACTION_MODES)
        raise ValueError(f"unknown adversarial action mode {action_mode!r}; choices: {choices}")
    return action_mode


def _split_action_mode(action_mode: str) -> tuple[str, str]:
    if action_mode == "deterministic":
        return "greedy", "greedy"
    if action_mode == "sampled":
        return "sampled", "sampled"
    if action_mode == "greedy_move_greedy_write":
        return "greedy", "greedy"
    if action_mode == "greedy_move_sampled_write":
        return "greedy", "sampled"
    if action_mode == "sampled_move_greedy_write":
        return "sampled", "greedy"
    if action_mode == "sampled_move_sampled_write":
        return "sampled", "sampled"
    if action_mode == "greedy_move_zero_write":
        return "greedy", "zero"
    if action_mode == "sampled_move_zero_write":
        return "sampled", "zero"
    raise ValueError(f"unknown adversarial action mode {action_mode!r}.")


def _head_actions(
    logits: jax.Array,
    key: jax.Array,
    *,
    mode: str,
    temperature: float,
) -> jax.Array:
    if mode == "greedy":
        return jnp.argmax(logits, axis=-1)
    if mode == "sampled":
        resolved_temperature = float(temperature)
        if resolved_temperature <= 0.0:
            raise ValueError("action temperature must be positive.")
        return jax.random.categorical(key, logits / resolved_temperature, axis=-1)
    if mode == "zero":
        return jnp.zeros(logits.shape[:-1], dtype=jnp.int32)
    raise ValueError(f"unknown action head mode {mode!r}.")
