"""Executable timed-release cooperative JAX MAPPO training loop."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp

from ant_byte_env import write_value_count
from ant_byte_env.runs import append_metrics, ensure_run_structure, write_json
from ant_byte_env.wandb_tracking import WandbTracker
from ant_byte_env.training.jax_mappo.checkpointing import checkpoint_args, save_checkpoint
from ant_byte_env.training.jax_mappo.curriculum import reset_batch
from ant_byte_env.training.jax_mappo.layout_audit import LayoutAuditTracker
from ant_byte_env.training.jax_mappo.models import init_agent_params
from ant_byte_env.training.jax_mappo.observations import (
    build_actor_observations,
    build_central_observations,
    food_observation_scale,
)
from ant_byte_env.training.jax_mappo.runner import (
    _autocurriculum_state_stats,
    _metric_is_better,
    _metrics_to_float,
    _rollout_stats,
    _should_report_update,
    _should_run_best_eval,
)
from ant_byte_env.training.jax_mappo.timed_release.cli import parse_args
from ant_byte_env.training.jax_mappo.timed_release.env import make_timed_release_env
from ant_byte_env.training.jax_mappo.timed_release.evaluation import (
    evaluate_params as evaluate_timed_release_params,
)
from ant_byte_env.training.jax_mappo.timed_release.rollout import collect_rollout
from ant_byte_env.training.jax_mappo.transfer import (
    load_checkpoint_for_training,
    warm_start_actor_params,
)
from ant_byte_env.training.jax_mappo.updates import init_adam_state, update_agent


def _timed_release_rollout_stats(rollout: Any) -> dict[str, float]:
    active_counts = jnp.sum(rollout.agent_masks.astype(jnp.float32), axis=-1)
    return {
        "mean_active_ants": float(jnp.mean(active_counts)),
        "final_mean_active_ants": float(jnp.mean(active_counts[-1])),
        "active_agent_fraction": float(jnp.mean(rollout.agent_masks.astype(jnp.float32))),
    }


def _best_eval_metrics(*, args: Any, params: Any) -> dict[str, float]:
    return evaluate_timed_release_params(
        params=params,
        args=args,
        num_episodes=int(args.best_eval_episodes),
        seed_offset=int(args.best_eval_seed_offset),
        action_mode=str(args.best_eval_action_mode),
        move_temperature=float(args.best_eval_move_temperature),
        write_temperature=float(args.best_eval_write_temperature),
        shuffle_positions=bool(args.best_eval_shuffle_positions),
    )


def main(
    argv: list[str] | None = None,
    *,
    progress_callback: Any | None = None,
    checkpoint_callback: Any | None = None,
) -> dict[str, float]:
    args = parse_args(argv)
    key = jax.random.PRNGKey(args.seed)
    run_name = f"{args.exp_name}__timed_release__seed_{args.seed}__{int(time.time())}"
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
                "workflow": "timed_release_roles",
                "run_name": run_name,
                "args": checkpoint_args(args),
                "release_steps": [
                    max(0, (rank - int(args.initial_active_ants) + 1) * int(args.release_interval))
                    for rank in range(int(args.num_ants))
                ],
            },
        )

    env = make_timed_release_env(args)
    reset_fn = jax.jit(lambda reset_key: reset_batch(args=args, env=env, key=reset_key))

    key, reset_key, init_key = jax.random.split(key, 3)
    states, obs = reset_fn(reset_key)
    layout_audit = LayoutAuditTracker.from_args(args, run_name=run_name)
    layout_audit.observe_observations(
        obs=obs,
        update=0,
        global_step=0,
        reason="initial_reset",
    )
    food_scale = food_observation_scale(
        food_count=args.food_count,
        food_sources=getattr(args, "food_sources", None),
    )
    central_obs = build_central_observations(
        obs,
        food_scale=food_scale,
        write_bits=args.write_bits,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
    )
    actor_obs = build_actor_observations(
        obs,
        food_scale=food_scale,
        actor_vision_radius=args.actor_vision_radius,
        write_bits=args.write_bits,
        agent_identity_types=getattr(args, "agent_identity_types", None),
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
        critic_architecture=getattr(args, "critic_architecture", "mlp"),
        critic_num_ants=args.num_ants,
        critic_obs_height=args.obs_height or args.height,
        critic_obs_width=args.obs_width or args.width,
    )
    opt_state = init_adam_state(params)
    if args.load_model is not None:
        if bool(getattr(args, "actor_only_warm_start", False)):
            params = warm_start_actor_params(
                params,
                args.load_model,
                actor_obs_dim=actor_obs_dim,
                target_write_bits=args.write_bits,
                actor_vision_radius=args.actor_vision_radius,
                target_num_ants=args.num_ants,
                target_agent_identity_types=getattr(args, "agent_identity_types", None),
                write_head_transfer=args.write_head_transfer,
            )
            opt_state = init_adam_state(params)
        else:
            checkpoint = load_checkpoint_for_training(
                args.load_model,
                central_obs_dim=central_obs_dim,
                actor_obs_dim=actor_obs_dim,
                target_write_bits=args.write_bits,
                actor_vision_radius=args.actor_vision_radius,
                target_num_ants=args.num_ants,
                target_agent_identity_types=getattr(args, "agent_identity_types", None),
                write_head_transfer=args.write_head_transfer,
                target_critic_architecture=getattr(args, "critic_architecture", "mlp"),
                reset_optimizer=bool(getattr(args, "reset_optimizer_on_load", False)),
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
    global_step = 0
    final_metrics: dict[str, float] = {
        "global_step": 0.0,
        "loss": 0.0,
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "episode_return": 0.0,
    }
    best_model_metric_value: float | None = None
    best_model_metrics: dict[str, float] | None = None

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
            layout_audit.observe_rollout_resets(
                rollout=rollout,
                update=update,
                global_step=global_step,
            )
            if _should_report_update(
                update=update,
                num_updates=num_updates,
                log_interval=args.log_interval,
            ):
                final_metrics = {
                    **_metrics_to_float(update_metrics),
                    **_autocurriculum_state_stats(states),
                    **_rollout_stats(rollout),
                    **_timed_release_rollout_stats(rollout),
                    **layout_audit.metrics(),
                    "global_step": float(global_step),
                    "learning_rate": float(learning_rate),
                }
                logged_metrics = {
                    "update": update,
                    "num_updates": num_updates,
                    **final_metrics,
                }
                reported_metrics = dict(final_metrics)
                if args.save_best_model is not None:
                    should_score_checkpoint = True
                    selection_metrics = dict(final_metrics)
                    if args.best_model_selection == "eval":
                        should_score_checkpoint = _should_run_best_eval(
                            args=args,
                            update=update,
                            num_updates=num_updates,
                        )
                        if should_score_checkpoint:
                            eval_metrics = _best_eval_metrics(args=args, params=params)
                            selection_metrics.update(eval_metrics)
                            reported_metrics.update(eval_metrics)
                            logged_metrics.update(eval_metrics)
                            logged_metrics.update(
                                {
                                    "best_eval_episodes": float(args.best_eval_episodes),
                                    "best_eval_seed_offset": float(args.best_eval_seed_offset),
                                }
                            )
                    if should_score_checkpoint:
                        if args.best_model_metric not in selection_metrics:
                            raise ValueError(
                                f"best model metric {args.best_model_metric!r} "
                                "was not reported by this training loop."
                            )
                        metric_value = float(selection_metrics[args.best_model_metric])
                        if _metric_is_better(
                            value=metric_value,
                            best_value=best_model_metric_value,
                            mode=args.best_model_mode,
                        ):
                            best_model_metric_value = metric_value
                            best_model_metrics = {
                                **selection_metrics,
                                "best_model_metric_value": metric_value,
                                "best_model_update": float(update),
                                "best_model_global_step": float(global_step),
                                "best_model_selection": str(args.best_model_selection),
                            }
                            save_checkpoint(
                                args.save_best_model,
                                params=params,
                                opt_state=opt_state,
                                args=args,
                                central_obs_dim=central_obs_dim,
                                actor_obs_dim=actor_obs_dim,
                                run_name=run_name,
                                metrics=best_model_metrics,
                            )
                    logged_metrics.update(
                        {
                            "best_model_metric_value": float(best_model_metric_value),
                        }
                    )
                if progress_callback is not None:
                    progress_callback(update, num_updates, reported_metrics)
                if checkpoint_callback is not None:
                    checkpoint_callback(
                        update=update,
                        num_updates=num_updates,
                        metrics=reported_metrics,
                        params=params,
                        opt_state=opt_state,
                        args=args,
                        central_obs_dim=central_obs_dim,
                        actor_obs_dim=actor_obs_dim,
                        run_name=run_name,
                        tracker=tracker,
                        global_step=global_step,
                    )
                tracker.log_metrics(logged_metrics, step=global_step)
                if not args.quiet:
                    print(
                        "update={update}/{num_updates} step={step} loss={loss:.4f} "
                        "return={episode_return:.3f} active={mean_active_ants:.2f}".format(
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
                "jax-mappo-timed-release-checkpoint",
                args.save_model,
                artifact_type="model",
                aliases=["latest"],
            )
        if args.save_best_model is not None and args.save_best_model.exists():
            tracker.log_artifact(
                "jax-mappo-timed-release-best-checkpoint",
                args.save_best_model,
                artifact_type="model",
                aliases=["best"],
            )
        if summary_path is not None:
            summary = {
                "backend": "jax",
                "workflow": "timed_release_roles",
                "run_name": run_name,
                "metrics": final_metrics,
                "checkpoint_path": args.save_model,
            }
            if args.save_best_model is not None:
                summary.update(
                    {
                        "best_checkpoint_path": args.save_best_model,
                        "best_checkpoint_metric": args.best_model_metric,
                        "best_checkpoint_selection": args.best_model_selection,
                        "best_checkpoint_metrics": best_model_metrics,
                    }
                )
            write_json(summary_path, summary)
        return final_metrics
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
