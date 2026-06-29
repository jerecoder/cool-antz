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
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams
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
        "behavior_anchor_kl": float(metrics.behavior_anchor_kl),
        "clipfrac": float(metrics.clipfrac),
        "grad_norm": float(metrics.grad_norm),
    }


def _should_report_update(*, update: int, num_updates: int, log_interval: int) -> bool:
    return update == 1 or update == num_updates or update % log_interval == 0


def _metric_is_better(
    *,
    value: float,
    best_value: float | None,
    mode: str,
) -> bool:
    if best_value is None:
        return True
    if mode == "min":
        return value < best_value
    return value > best_value


def _should_run_best_eval(*, args: Any, update: int, num_updates: int) -> bool:
    interval = int(args.best_eval_interval or args.log_interval)
    return update == 1 or update == num_updates or update % interval == 0


def _best_eval_metrics(
    *,
    args: Any,
    params: Any,
    opponent_params: Any,
    env: Any,
) -> dict[str, float]:
    eval_args = type(args)(
        **{
            **vars(args),
            "eval_episodes": int(args.best_eval_episodes),
        }
    )
    return evaluate_matrix(
        params=params,
        opponent_params=opponent_params,
        args=eval_args,
        env=env,
    )


def _best_model_metadata(best_model_metrics: dict[str, float] | None) -> dict[str, float]:
    if best_model_metrics is None:
        return {}
    keys = (
        "best_model_metric_value",
        "best_model_update",
        "best_model_global_step",
    )
    return {key: float(best_model_metrics[key]) for key in keys if key in best_model_metrics}


def _behavior_anchor_enabled(args: Any) -> bool:
    return float(getattr(args, "behavior_anchor_coef", 0.0)) > 0.0


def _copy_behavior_anchor_from_model(
    params: JaxMAPPOParams,
    checkpoint_path: Any,
    *,
    actor_obs_dim: int,
    target_write_bits: int,
) -> JaxMAPPOParams:
    return warm_start_actor_params(
        params,
        checkpoint_path,
        actor_obs_dim=actor_obs_dim,
        target_write_bits=target_write_bits,
    )


def _clone_params(params: JaxMAPPOParams) -> JaxMAPPOParams:
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), params)


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
    behavior_anchor_params: JaxMAPPOParams | None = None
    behavior_anchor_source: str | None = None
    if args.resume_model is not None:
        checkpoint = load_checkpoint(
            args.resume_model,
            central_obs_dim=central_obs_dim,
            actor_obs_dim=actor_obs_dim,
        )
        params = checkpoint["params"]
        opt_state = checkpoint["opt_state"]
        behavior_anchor_params = checkpoint.get("behavior_anchor_params")
        if behavior_anchor_params is not None:
            behavior_anchor_source = "resume_checkpoint"
    elif args.learner_load_model is not None:
        params = warm_start_actor_params(
            params,
            args.learner_load_model,
            actor_obs_dim=actor_obs_dim,
            target_write_bits=args.write_bits,
        )
        if _behavior_anchor_enabled(args):
            behavior_anchor_params = _clone_params(params)
            behavior_anchor_source = "post_transfer_learner_actor"
    if args.behavior_anchor_model is not None:
        behavior_anchor_params = _clone_params(
            _copy_behavior_anchor_from_model(
                params,
                args.behavior_anchor_model,
                actor_obs_dim=actor_obs_dim,
                target_write_bits=args.write_bits,
            )
        )
        behavior_anchor_source = str(args.behavior_anchor_model)
    elif behavior_anchor_params is None and _behavior_anchor_enabled(args):
        behavior_anchor_params = _clone_params(params)
        behavior_anchor_source = "current_learner_actor"
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
            behavior_anchor_params=behavior_anchor_params,
        ),
        donate_argnums=(0, 1),
    )

    steps_per_update = args.num_envs * args.num_steps
    num_updates = max(1, args.total_timesteps // steps_per_update)
    final_metrics: dict[str, float] = {"global_step": 0.0, "loss": 0.0, "episode_return": 0.0}
    global_step = 0
    best_model_metric_value: float | None = None
    best_model_metrics: dict[str, float] | None = None
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
        if _should_report_update(
            update=update,
            num_updates=num_updates,
            log_interval=args.log_interval,
        ):
            final_metrics = {
                **_metrics_to_float(update_metrics),
                **_rollout_stats(rollout, args=args),
                "global_step": float(global_step),
                "learning_rate": float(args.learning_rate),
            }
            logged_metrics = {"update": float(update), "num_updates": float(num_updates), **final_metrics}
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
                        eval_metrics = _best_eval_metrics(
                            args=args,
                            params=params,
                            opponent_params=opponent_params,
                            env=env,
                        )
                        selection_metrics.update(eval_metrics)
                        reported_metrics.update(eval_metrics)
                        logged_metrics.update(eval_metrics)
                        logged_metrics["best_eval_episodes"] = float(
                            args.best_eval_episodes
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
                            behavior_anchor_params=behavior_anchor_params,
                        )
                best_metadata = _best_model_metadata(best_model_metrics)
                logged_metrics.update(best_metadata)
                reported_metrics.update(best_metadata)
            if progress_callback is not None:
                progress_callback(update, num_updates, reported_metrics)
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
    final_metrics.update(_best_model_metadata(best_model_metrics))
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
            behavior_anchor_params=behavior_anchor_params,
        )
    if summary_path is not None:
        summary = {
            "backend": "jax",
            "workflow": "adversarial_frozen_opponent",
            "run_name": run_name,
            "metrics": final_metrics,
            "checkpoint_path": args.save_model,
            "behavior_anchor_source": behavior_anchor_source,
        }
        if args.save_best_model is not None:
            summary.update(
                {
                    "best_checkpoint_path": args.save_best_model,
                    "best_checkpoint_metric": args.best_model_metric,
                    "best_checkpoint_mode": args.best_model_mode,
                    "best_checkpoint_selection": args.best_model_selection,
                    "best_checkpoint_metrics": best_model_metrics,
                }
            )
        write_json(
            summary_path,
            summary,
        )
    return final_metrics


if __name__ == "__main__":
    main()
