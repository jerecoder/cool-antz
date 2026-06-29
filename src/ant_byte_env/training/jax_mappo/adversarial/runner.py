"""Executable adversarial frozen-opponent JAX MAPPO loop."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp

from ant_byte_env.runs import append_metrics, ensure_run_structure, write_json
from ant_byte_env.training.jax_mappo.adversarial.cli import parse_args
from ant_byte_env.training.jax_mappo.adversarial.env import reset_batch
from ant_byte_env.training.jax_mappo.adversarial.evaluation import evaluate_matrix
from ant_byte_env.training.jax_mappo.adversarial.observations import (
    build_team_actor_observations,
    build_team_central_observations,
)
from ant_byte_env.training.jax_mappo.adversarial.rollout import collect_rollout
from ant_byte_env.training.jax_mappo.adversarial.setup import (
    init_adversarial_params,
    make_env,
)
from ant_byte_env.training.jax_mappo.adversarial.transfer import warm_start_actor_params
from ant_byte_env.training.jax_mappo.adversarial.types import AdversarialRollout
from ant_byte_env.training.jax_mappo.checkpointing import (
    checkpoint_args,
    load_checkpoint,
    save_checkpoint,
)
from ant_byte_env.training.jax_mappo.observations import food_observation_scale
from ant_byte_env.training.jax_mappo.updates import init_adam_state, update_agent


def _rollout_stats(rollout: AdversarialRollout, *, args: Any) -> dict[str, float]:
    learner_team = int(args.learner_team)
    opponent_team = 1 - learner_team
    delivery_events = jnp.sum(rollout.delivery_events, axis=(0, 1))
    pickup_events = jnp.sum(rollout.pickup_events, axis=(0, 1))
    return {
        "episode_return": float(jnp.mean(jnp.sum(rollout.rewards, axis=0))),
        "completed_episodes": float(jnp.sum(rollout.dones)),
        "terminated_episodes": float(jnp.sum(rollout.terminations)),
        "own_delivery_events": float(delivery_events[learner_team]),
        "opponent_delivery_events": float(delivery_events[opponent_team]),
        "delivery_event_difference": float(
            delivery_events[learner_team] - delivery_events[opponent_team]
        ),
        "own_pickup_events": float(pickup_events[learner_team]),
        "opponent_pickup_events": float(pickup_events[opponent_team]),
        "final_mean_own_delivered_food": float(
            jnp.mean(rollout.delivered_food[-1, :, learner_team])
        ),
        "final_mean_opponent_delivered_food": float(
            jnp.mean(rollout.delivered_food[-1, :, opponent_team])
        ),
        "final_mean_remaining_food": float(jnp.mean(rollout.remaining_food[-1])),
    }


def _metrics_to_float(metrics: Any) -> dict[str, float]:
    return {
        "loss": float(metrics.loss),
        "policy_loss": float(metrics.policy_loss),
        "value_loss": float(metrics.value_loss),
        "entropy": float(metrics.entropy),
        "approx_kl": float(metrics.approx_kl),
        "clipfrac": float(metrics.clipfrac),
        "grad_norm": float(metrics.grad_norm),
    }


def main(
    argv: list[str] | None = None,
    *,
    progress_callback: Any | None = None,
) -> dict[str, float]:
    args = parse_args(argv)
    key = jax.random.PRNGKey(args.seed)
    run_name = f"{args.exp_name}__seed_{args.seed}__{int(time.time())}"
    metrics_path = None
    summary_path = None
    if args.run_dir is not None:
        ensure_run_structure(args.run_dir)
        if args.save_model is None:
            args.save_model = args.run_dir / "checkpoints" / "model.pkl"
        metrics_path = args.run_dir / "metrics.jsonl"
        summary_path = args.run_dir / "summary.json"
        write_json(
            args.run_dir / "config.json",
            {"backend": "jax", "workflow": "adversarial_frozen_opponent", "args": checkpoint_args(args)},
        )

    env = make_env(args)
    reset_fn = jax.jit(lambda reset_key: reset_batch(args=args, env=env, key=reset_key))
    key, reset_key, learner_init_key, opponent_init_key = jax.random.split(key, 4)
    states, obs = reset_fn(reset_key)
    food_scale = food_observation_scale(
        food_count=args.food_count,
        food_sources=getattr(args, "food_sources", None),
    )
    central_obs = build_team_central_observations(
        obs,
        team=args.learner_team,
        num_ants_per_team=args.num_ants_per_team,
        food_scale=food_scale,
        write_bits=args.write_bits,
    )
    actor_obs = build_team_actor_observations(
        obs,
        team=args.learner_team,
        num_ants_per_team=args.num_ants_per_team,
        food_scale=food_scale,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
    )
    central_obs_dim = central_obs.shape[-1]
    actor_obs_dim = actor_obs.shape[-1]
    params = init_adversarial_params(
        learner_init_key,
        args=args,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
    )
    opponent_params = init_adversarial_params(
        opponent_init_key,
        args=args,
        central_obs_dim=central_obs_dim,
        actor_obs_dim=actor_obs_dim,
    )
    opt_state = None
    if args.resume_model is not None:
        checkpoint = load_checkpoint(
            args.resume_model,
            central_obs_dim=central_obs_dim,
            actor_obs_dim=actor_obs_dim,
        )
        params = checkpoint["params"]
        opt_state = checkpoint["opt_state"]
    elif args.learner_load_model is not None:
        params = warm_start_actor_params(
            params,
            args.learner_load_model,
            actor_obs_dim=actor_obs_dim,
            target_write_bits=args.write_bits,
        )
    opponent_load_model = args.opponent_load_model or args.learner_load_model
    if opponent_load_model is not None:
        opponent_params = warm_start_actor_params(
            opponent_params,
            opponent_load_model,
            actor_obs_dim=actor_obs_dim,
            target_write_bits=args.write_bits,
        )
    if opt_state is None:
        opt_state = init_adam_state(params)

    rollout_fn = jax.jit(
        lambda current_params, current_states, current_obs, rollout_key: collect_rollout(
            args=args,
            env=env,
            learner_params=current_params,
            opponent_params=opponent_params,
            states=current_states,
            obs=current_obs,
            key=rollout_key,
        ),
        donate_argnums=(1, 2),
    )
    update_fn = jax.jit(
        lambda current_params, current_opt_state, rollout, learning_rate, update_key: update_agent(
            args=args,
            params=current_params,
            opt_state=current_opt_state,
            rollout=rollout,
            learning_rate=learning_rate,
            key=update_key,
        ),
        donate_argnums=(0, 1),
    )

    steps_per_update = args.num_envs * args.num_steps
    num_updates = max(1, args.total_timesteps // steps_per_update)
    final_metrics: dict[str, float] = {"global_step": 0.0, "loss": 0.0, "episode_return": 0.0}
    global_step = 0
    for update in range(1, num_updates + 1):
        key, rollout_key, update_key = jax.random.split(key, 3)
        states, obs, rollout = rollout_fn(params, states, obs, rollout_key)
        params, opt_state, update_metrics = update_fn(
            params,
            opt_state,
            rollout,
            args.learning_rate,
            update_key,
        )
        global_step += steps_per_update
        if update == 1 or update == num_updates or update % args.log_interval == 0:
            final_metrics = {
                **_metrics_to_float(update_metrics),
                **_rollout_stats(rollout, args=args),
                "global_step": float(global_step),
                "learning_rate": float(args.learning_rate),
            }
            logged_metrics = {"update": float(update), "num_updates": float(num_updates), **final_metrics}
            if progress_callback is not None:
                progress_callback(update, num_updates, final_metrics)
            if metrics_path is not None:
                append_metrics(metrics_path, logged_metrics)
            if not args.quiet:
                print(
                    "update={update}/{num_updates} step={step} loss={loss:.4f} "
                    "return={episode_return:.3f}".format(
                        update=update,
                        num_updates=num_updates,
                        step=global_step,
                        **final_metrics,
                    )
                )

    if args.eval_episodes > 0:
        final_metrics.update(
            evaluate_matrix(params=params, opponent_params=opponent_params, args=args, env=env)
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
    if summary_path is not None:
        write_json(
            summary_path,
            {
                "backend": "jax",
                "workflow": "adversarial_frozen_opponent",
                "run_name": run_name,
                "metrics": final_metrics,
                "checkpoint_path": args.save_model,
            },
        )
    return final_metrics


if __name__ == "__main__":
    main()
