#!/usr/bin/env python3
"""Pure JAX MAPPO/PPO trainer for the ant byte foraging curriculum."""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ant_byte_env import DEFAULT_WRITE_BITS, MAX_WRITE_BITS, write_value_count
from ant_byte_env.jax_env import JaxAntByteForagingEnv, JaxAntState, JaxObs
from train_mappo_jax_core import (
    AdamState,
    JaxMAPPOParams,
    Rollout,
    Transition,
    UpdateMetrics,
    build_actor_observations,
    build_central_observations,
    compute_forage_curriculum_rewards,
    compute_gae,
    flatten_agent_actions,
    get_action_and_value,
    get_value,
    init_adam_state,
    init_agent_params,
    update_agent,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JAX MAPPO on AntByte forage.")
    parser.add_argument("--exp-name", type=str, default="jax_mappo_forage")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)

    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--obs-width", type=int, default=None)
    parser.add_argument("--obs-height", type=int, default=None)
    parser.add_argument("--actor-vision-radius", type=int, default=2)
    parser.add_argument("--num-ants", type=int, default=2)
    parser.add_argument("--food-count", type=int, default=4)
    parser.add_argument("--food-sources", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--step-penalty", type=float, default=0.0)
    parser.add_argument("--write-penalty", type=float, default=0.0)
    parser.add_argument("--write-bits", type=int, default=DEFAULT_WRITE_BITS)
    parser.add_argument("--cookie-distance", type=int, default=1)
    parser.add_argument("--random-food", action="store_true")
    parser.add_argument("--random-hub", action="store_true")
    parser.add_argument("--pickup-bonus", type=float, default=0.25)
    parser.add_argument("--distance-bonus", type=float, default=0.02)
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--load-model", type=Path, default=None)

    args = parser.parse_args(argv)
    rollout_batch_size = args.num_envs * args.num_steps
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.num_steps <= 0:
        raise ValueError("--num-steps must be positive.")
    if args.num_minibatches <= 0:
        raise ValueError("--num-minibatches must be positive.")
    if rollout_batch_size < args.num_minibatches:
        raise ValueError("--num-minibatches cannot exceed rollout batch size.")
    if rollout_batch_size % args.num_minibatches != 0:
        raise ValueError("--num-minibatches must evenly divide rollout batch size.")
    if args.update_epochs <= 0:
        raise ValueError("--update-epochs must be positive.")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive.")
    if args.cookie_distance <= 0:
        raise ValueError("--cookie-distance must be positive.")
    if args.food_count > 0 and args.width * args.height <= 1:
        raise ValueError("food_count requires at least one non-hub tile.")
    if args.obs_width is not None and args.obs_width < args.width:
        raise ValueError("--obs-width must be at least --width.")
    if args.obs_height is not None and args.obs_height < args.height:
        raise ValueError("--obs-height must be at least --height.")
    if args.actor_vision_radius < 0:
        raise ValueError("--actor-vision-radius must be non-negative.")
    if args.write_bits <= 0 or args.write_bits > MAX_WRITE_BITS:
        raise ValueError(f"--write-bits must be an integer from 1 to {MAX_WRITE_BITS}.")
    return args


def _fixed_cookie_positions(
    hub_positions: jax.Array,
    *,
    width: int,
    height: int,
    distance: int,
) -> jax.Array:
    offsets = jnp.asarray(
        [[distance, 0], [-distance, 0], [0, distance], [0, -distance]],
        dtype=jnp.int32,
    )
    candidates = hub_positions[:, None, :] + offsets[None, :, :]
    valid = (
        (0 <= candidates[..., 0])
        & (candidates[..., 0] < width)
        & (0 <= candidates[..., 1])
        & (candidates[..., 1] < height)
    )
    selected = candidates[jnp.arange(hub_positions.shape[0]), jnp.argmax(valid, axis=1)]
    fallback = jnp.where(
        jnp.any(hub_positions != jnp.asarray([0, 0], dtype=jnp.int32), axis=1)[:, None],
        jnp.asarray([0, 0], dtype=jnp.int32),
        jnp.asarray([min(width - 1, 1), 0 if width > 1 else min(height - 1, 1)], dtype=jnp.int32),
    )
    return jnp.where(jnp.any(valid, axis=1)[:, None], selected, fallback)[:, None, :]


def reset_batch(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    key: jax.Array,
) -> tuple[JaxAntState, JaxObs]:
    reset_keys = jax.random.split(key, args.num_envs)
    if args.random_hub:
        hub_key_x, hub_key_y = jax.random.split(jax.random.fold_in(key, 11))
        hub_positions = jnp.stack(
            [
                jax.random.randint(hub_key_x, (args.num_envs,), 0, args.width),
                jax.random.randint(hub_key_y, (args.num_envs,), 0, args.height),
            ],
            axis=-1,
        ).astype(jnp.int32)
    else:
        hub_positions = jnp.broadcast_to(
            jnp.asarray([args.width // 2, args.height // 2], dtype=jnp.int32),
            (args.num_envs, 2),
        )

    if args.random_food:
        states, obs, _ = jax.vmap(lambda reset_key, hub: env.reset(reset_key, hub_pos=hub))(
            reset_keys,
            hub_positions,
        )
        return states, obs

    food_positions = _fixed_cookie_positions(
        hub_positions,
        width=args.width,
        height=args.height,
        distance=args.cookie_distance,
    )
    states, obs, _ = jax.vmap(
        lambda reset_key, hub, food: env.reset(
            reset_key,
            hub_pos=hub,
            food_positions=food,
        )
    )(reset_keys, hub_positions, food_positions)
    return states, obs


def collect_rollout(
    *,
    args: argparse.Namespace,
    env: JaxAntByteForagingEnv,
    params: JaxMAPPOParams,
    states: JaxAntState,
    obs: JaxObs,
    key: jax.Array,
) -> tuple[JaxAntState, JaxObs, Rollout]:
    def scan_step(
        carry: tuple[JaxAntState, JaxObs, jax.Array],
        _: Any,
    ) -> tuple[tuple[JaxAntState, JaxObs, jax.Array], Transition]:
        current_states, current_obs, current_key = carry
        action_key, next_key = jax.random.split(current_key)
        central_obs = build_central_observations(
            current_obs,
            food_scale=args.food_count,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        actor_obs = build_actor_observations(
            current_obs,
            food_scale=args.food_count,
            actor_vision_radius=args.actor_vision_radius,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        actions, logprobs, _, values = get_action_and_value(
            params,
            actor_obs,
            central_obs,
            action_key,
        )
        next_states, next_obs, env_rewards, terminated, truncated, _ = jax.vmap(env.step)(
            current_states,
            flatten_agent_actions(actions),
        )
        dones = jnp.logical_or(terminated, truncated)
        rewards = compute_forage_curriculum_rewards(
            previous_obs=current_obs,
            next_obs=next_obs,
            env_rewards=env_rewards,
            pickup_bonus=args.pickup_bonus,
            distance_bonus=args.distance_bonus,
        )
        transition = Transition(
            actor_obs=actor_obs,
            central_obs=central_obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones=dones,
            values=values,
            env_rewards=env_rewards,
        )
        return (next_states, next_obs, next_key), transition

    (final_states, final_obs, _), transitions = jax.lax.scan(
        scan_step,
        (states, obs, key),
        None,
        length=args.num_steps,
    )
    next_central_obs = build_central_observations(
        final_obs,
        food_scale=args.food_count,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    rollout = Rollout(
        actor_obs=transitions.actor_obs,
        central_obs=transitions.central_obs,
        actions=transitions.actions,
        logprobs=transitions.logprobs,
        rewards=transitions.rewards,
        dones=transitions.dones,
        values=transitions.values,
        env_rewards=transitions.env_rewards,
        next_value=get_value(params, next_central_obs),
    )
    return final_states, final_obs, rollout


def checkpoint_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def _numpy_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: np.asarray(value), tree)


def _jax_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.asarray, tree)


def save_checkpoint(
    path: Path,
    *,
    params: JaxMAPPOParams,
    opt_state: AdamState,
    args: argparse.Namespace,
    central_obs_dim: int,
    actor_obs_dim: int,
    run_name: str,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as checkpoint_file:
        pickle.dump(
            {
                "params": _numpy_tree(params),
                "opt_state": _numpy_tree(opt_state),
                "args": checkpoint_args(args),
                "central_obs_dim": int(central_obs_dim),
                "actor_obs_dim": int(actor_obs_dim),
                "run_name": run_name,
                "metrics": metrics,
            },
            checkpoint_file,
        )


def load_checkpoint(
    path: Path,
    *,
    central_obs_dim: int,
    actor_obs_dim: int,
) -> dict[str, Any]:
    with path.open("rb") as checkpoint_file:
        checkpoint = pickle.load(checkpoint_file)
    if int(checkpoint["central_obs_dim"]) != central_obs_dim:
        raise ValueError("Checkpoint central observation dimension does not match this run.")
    if int(checkpoint["actor_obs_dim"]) != actor_obs_dim:
        raise ValueError("Checkpoint actor observation dimension does not match this run.")
    checkpoint["params"] = _jax_tree(checkpoint["params"])
    checkpoint["opt_state"] = _jax_tree(checkpoint["opt_state"])
    return checkpoint


def _metrics_to_float(metrics: UpdateMetrics) -> dict[str, float]:
    return {
        "loss": float(metrics.loss),
        "policy_loss": float(metrics.policy_loss),
        "value_loss": float(metrics.value_loss),
        "entropy": float(metrics.entropy),
        "approx_kl": float(metrics.approx_kl),
        "clipfrac": float(metrics.clipfrac),
        "grad_norm": float(metrics.grad_norm),
    }


def _rollout_stats(rollout: Rollout) -> dict[str, float]:
    return {
        "episode_return": float(jnp.mean(jnp.sum(rollout.rewards, axis=0))),
        "env_return": float(jnp.mean(jnp.sum(rollout.env_rewards, axis=0))),
        "completed_episodes": float(jnp.sum(rollout.dones)),
    }


def main(
    argv: list[str] | None = None,
    *,
    progress_callback: Any | None = None,
) -> dict[str, float]:
    args = parse_args(argv)
    key = jax.random.PRNGKey(args.seed)
    run_name = f"{args.exp_name}__seed_{args.seed}__{int(time.time())}"

    env = JaxAntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        random_food=args.random_food,
        step_penalty=args.step_penalty,
        write_penalty=args.write_penalty,
        write_bits=args.write_bits,
    )

    key, reset_key, init_key = jax.random.split(key, 3)
    states, obs = reset_batch(args=args, env=env, key=reset_key)
    central_obs = build_central_observations(
        obs,
        food_scale=args.food_count,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs,
        food_scale=args.food_count,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    central_obs_dim = central_obs.shape[-1]
    actor_obs_dim = actor_obs.shape[-1]
    params = init_agent_params(
        init_key,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
        hidden_size=args.hidden_size,
        write_value_count=write_value_count(args.write_bits),
    )
    opt_state = init_adam_state(params)
    if args.load_model is not None:
        checkpoint = load_checkpoint(
            args.load_model,
            central_obs_dim=central_obs_dim,
            actor_obs_dim=actor_obs_dim,
        )
        params = checkpoint["params"]
        opt_state = checkpoint["opt_state"]

    reset_fn = jax.jit(lambda reset_key: reset_batch(args=args, env=env, key=reset_key))
    rollout_fn = jax.jit(
        lambda current_params, current_states, current_obs, rollout_key: collect_rollout(
            args=args,
            env=env,
            params=current_params,
            states=current_states,
            obs=current_obs,
            key=rollout_key,
        )
    )
    update_fn = jax.jit(
        lambda current_params, current_opt_state, rollout, learning_rate: update_agent(
            args=args,
            params=current_params,
            opt_state=current_opt_state,
            rollout=rollout,
            learning_rate=learning_rate,
        )
    )

    steps_per_update = args.num_envs * args.num_steps
    num_updates = max(1, args.total_timesteps // steps_per_update)
    global_step = 0
    final_metrics: dict[str, float] = {
        "global_step": 0.0,
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "episode_return": 0.0,
    }

    for update in range(1, num_updates + 1):
        key, reset_key, rollout_key = jax.random.split(key, 3)
        states, obs = reset_fn(reset_key)
        states, obs, rollout = rollout_fn(params, states, obs, rollout_key)
        learning_rate = args.learning_rate
        if args.anneal_lr:
            learning_rate *= 1.0 - (update - 1.0) / num_updates

        params, opt_state, update_metrics = update_fn(params, opt_state, rollout, learning_rate)
        global_step += steps_per_update
        final_metrics = {
            **_metrics_to_float(update_metrics),
            **_rollout_stats(rollout),
            "global_step": float(global_step),
            "learning_rate": float(learning_rate),
        }
        if progress_callback is not None:
            progress_callback(update, num_updates, final_metrics)
        if not args.quiet:
            print(
                "update={update}/{num_updates} step={step} loss={loss:.4f} "
                "return={episode_return:.3f} entropy={entropy:.3f}".format(
                    update=update,
                    num_updates=num_updates,
                    step=global_step,
                    **final_metrics,
                )
            )

    if args.save_model is not None:
        save_checkpoint(
            args.save_model,
            params=params,
            opt_state=opt_state,
            args=args,
            central_obs_dim=central_obs_dim,
            actor_obs_dim=actor_obs_dim,
            run_name=run_name,
            metrics=final_metrics,
        )
    return final_metrics


if __name__ == "__main__":
    main()
