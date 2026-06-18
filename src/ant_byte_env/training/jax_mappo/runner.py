"""Executable JAX MAPPO training loop."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp

from ant_byte_env import write_value_count
from ant_byte_env.jax_autocurriculum_env import JaxAntByteAutoCurriculumEnv
from ant_byte_env.jax_env import JaxAntByteForagingEnv
from ant_byte_env.runs import append_metrics, ensure_run_structure, write_json
from ant_byte_env.wandb_tracking import WandbTracker
from ant_byte_env.training.jax_mappo.checkpointing import (
    checkpoint_args,
    save_checkpoint,
)
from ant_byte_env.training.jax_mappo.cli import parse_args
from ant_byte_env.training.jax_mappo.core import (
    Rollout,
    UpdateMetrics,
    build_actor_observations,
    build_central_observations,
    init_adam_state,
    init_agent_params,
    update_agent,
)
from ant_byte_env.training.jax_mappo.curriculum import reset_batch
from ant_byte_env.training.jax_mappo.rollout import collect_rollout
from ant_byte_env.training.jax_mappo.transfer import load_checkpoint_for_training


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
    write_values = rollout.actions[..., 1].astype(jnp.float32)
    nonzero_write_actions = write_values > 0.0
    return {
        "episode_return": float(jnp.mean(jnp.sum(rollout.rewards, axis=0))),
        "env_return": float(jnp.mean(jnp.sum(rollout.env_rewards, axis=0))),
        "completed_episodes": float(jnp.sum(rollout.dones)),
        "terminated_episodes": float(jnp.sum(rollout.terminations)),
        "truncated_episodes": float(jnp.sum(rollout.truncations)),
        "pickup_events": float(jnp.sum(rollout.pickup_events)),
        "delivery_events": float(jnp.sum(rollout.delivery_events)),
        "mean_carrying_ants": float(jnp.mean(rollout.carrying_ants)),
        "final_mean_remaining_food": float(jnp.mean(rollout.remaining_food[-1])),
        "write_action_nonzero_rate": float(jnp.mean(nonzero_write_actions)),
        "mean_write_action_value": float(jnp.mean(write_values)),
        "mean_nonzero_byte_tiles": float(jnp.mean(rollout.nonzero_byte_tiles)),
        "final_mean_nonzero_byte_tiles": float(jnp.mean(rollout.nonzero_byte_tiles[-1])),
        "mean_nonzero_byte_fraction": float(jnp.mean(rollout.nonzero_byte_fraction)),
        "final_mean_nonzero_byte_fraction": float(
            jnp.mean(rollout.nonzero_byte_fraction[-1])
        ),
    }


def _make_env(args: Any) -> JaxAntByteForagingEnv | JaxAntByteAutoCurriculumEnv:
    common_kwargs = {
        "width": args.width,
        "height": args.height,
        "num_ants": args.num_ants,
        "food_count": args.food_count,
        "food_source_count": args.food_sources,
        "max_steps": args.max_steps,
        "random_food": args.random_food,
        "random_hub": args.random_hub,
        "step_penalty": args.step_penalty,
        "write_penalty": args.write_penalty,
        "write_bits": args.write_bits,
        "write_while_moving": args.write_while_moving,
    }
    if bool(getattr(args, "autocurriculum", False)):
        return JaxAntByteAutoCurriculumEnv(
            **common_kwargs,
            start_size=args.autocurriculum_start_size,
            success_cookies=args.autocurriculum_success_cookies,
            actor_vision_radius=args.actor_vision_radius,
        )
    return JaxAntByteForagingEnv(**common_kwargs)


def _autocurriculum_state_stats(states: Any) -> dict[str, float]:
    if not hasattr(states, "active_size"):
        return {}
    return {
        "autocurriculum_mean_active_size": float(jnp.mean(states.active_size.astype(jnp.float32))),
        "autocurriculum_max_active_size": float(jnp.max(states.active_size)),
        "autocurriculum_mean_stage_delivered_food": float(
            jnp.mean(states.stage_delivered_food.astype(jnp.float32))
        ),
        "autocurriculum_completed_stages": float(jnp.sum(states.completed_stages)),
    }


def _should_report_update(*, update: int, num_updates: int, log_interval: int) -> bool:
    return update == 1 or update == num_updates or update % log_interval == 0


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
            {
                "backend": "jax",
                "run_name": run_name,
                "args": checkpoint_args(args),
            },
        )

    env = _make_env(args)

    reset_fn = jax.jit(lambda reset_key: reset_batch(args=args, env=env, key=reset_key))

    key, reset_key, init_key = jax.random.split(key, 3)
    states, obs = reset_fn(reset_key)
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
        checkpoint = load_checkpoint_for_training(
            args.load_model,
            central_obs_dim=central_obs_dim,
            actor_obs_dim=actor_obs_dim,
            target_write_bits=args.write_bits,
            actor_vision_radius=args.actor_vision_radius,
            write_head_transfer=args.write_head_transfer,
        )
        params = checkpoint["params"]
        opt_state = checkpoint["opt_state"]

    tracker = WandbTracker(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_run_name or run_name,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        run_dir=args.run_dir,
        config=checkpoint_args(args),
        notes=args.wandb_notes,
    )
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
        lambda current_params, current_opt_state, rollout, learning_rate, update_key: update_agent(
            args=args,
            params=current_params,
            opt_state=current_opt_state,
            rollout=rollout,
            learning_rate=learning_rate,
            key=update_key,
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

    try:
        for update in range(1, num_updates + 1):
            key, rollout_key, update_key = jax.random.split(key, 3)
            states, obs, rollout = rollout_fn(params, states, obs, rollout_key)
            learning_rate = args.learning_rate
            if args.anneal_lr:
                learning_rate *= 1.0 - (update - 1.0) / num_updates

            params, opt_state, update_metrics = update_fn(
                params,
                opt_state,
                rollout,
                learning_rate,
                update_key,
            )
            global_step += steps_per_update
            if _should_report_update(
                update=update,
                num_updates=num_updates,
                log_interval=args.log_interval,
            ):
                final_metrics = {
                    **_metrics_to_float(update_metrics),
                    **_rollout_stats(rollout),
                    **_autocurriculum_state_stats(states),
                    "global_step": float(global_step),
                    "learning_rate": float(learning_rate),
                }
                logged_metrics = {
                    "update": update,
                    "num_updates": num_updates,
                    **final_metrics,
                }
                if progress_callback is not None:
                    progress_callback(update, num_updates, final_metrics)
                tracker.log_metrics(logged_metrics, step=global_step)
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
                if metrics_path is not None:
                    append_metrics(metrics_path, logged_metrics)

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
            tracker.log_artifact(
                "jax-mappo-checkpoint",
                args.save_model,
                artifact_type="model",
                aliases=["latest"],
            )
        if summary_path is not None:
            write_json(
                summary_path,
                {
                    "backend": "jax",
                    "run_name": run_name,
                    "metrics": final_metrics,
                    "checkpoint_path": args.save_model,
                },
            )
        return final_metrics
    finally:
        tracker.finish()





if __name__ == "__main__":
    main()
