"""Checkpoint helpers for adversarial evaluation and rendering."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jax

from ant_byte_env.training.jax_mappo.adversarial.cli import parse_args
from ant_byte_env.training.jax_mappo.adversarial.env import (
    JaxAdversarialAntByteEnv,
    reset_batch,
)
from ant_byte_env.training.jax_mappo.adversarial.evaluation import evaluate_matrix
from ant_byte_env.training.jax_mappo.adversarial.observations import (
    build_team_actor_observations,
    build_team_central_observations,
)
from ant_byte_env.training.jax_mappo.adversarial.setup import (
    init_adversarial_params,
    make_env,
)
from ant_byte_env.training.jax_mappo.adversarial.transfer import warm_start_actor_params
from ant_byte_env.training.jax_mappo.checkpointing import read_checkpoint
from ant_byte_env.training.jax_mappo.observations import food_observation_scale
from ant_byte_env.training.jax_mappo.types import JaxMAPPOParams

EvaluationProgressCallback = Callable[[str, int, int, dict[str, float]], None]

_PATH_ARG_NAMES = {
    "learner_load_model",
    "opponent_load_model",
    "resume_model",
    "behavior_anchor_model",
    "save_model",
    "save_best_model",
    "run_dir",
}
_DERIVED_ARG_NAMES = {"critic_architecture", "num_ants"}


@dataclass(frozen=True)
class AdversarialCheckpointBundle:
    args: argparse.Namespace
    env: JaxAdversarialAntByteEnv
    learner_params: JaxMAPPOParams
    opponent_params: JaxMAPPOParams
    checkpoint: dict[str, Any]


def load_checkpoint_for_evaluation(
    checkpoint_path: Path,
    *,
    argv: Sequence[str] | None = None,
    args: argparse.Namespace | None = None,
) -> AdversarialCheckpointBundle:
    """Load learner params plus the matching frozen-opponent params."""

    if argv is not None and args is not None:
        raise ValueError("pass either argv or args, not both.")

    checkpoint = read_checkpoint(Path(checkpoint_path))
    eval_args = _resolve_args(checkpoint, argv=argv, args=args)
    env = make_env(eval_args)

    key = jax.random.PRNGKey(int(eval_args.seed))
    _, reset_key, _, opponent_init_key = jax.random.split(key, 4)
    _, obs = reset_batch(args=eval_args, env=env, key=reset_key)
    food_scale = food_observation_scale(
        food_count=eval_args.food_count,
        food_sources=getattr(eval_args, "food_sources", None),
    )
    central_obs = build_team_central_observations(
        obs,
        team=eval_args.learner_team,
        num_ants_per_team=eval_args.num_ants_per_team,
        food_scale=food_scale,
        write_bits=eval_args.write_bits,
    )
    actor_obs = build_team_actor_observations(
        obs,
        team=eval_args.learner_team,
        num_ants_per_team=eval_args.num_ants_per_team,
        food_scale=food_scale,
        actor_vision_radius=eval_args.actor_vision_radius,
        write_bits=eval_args.write_bits,
    )
    _validate_checkpoint_dims(
        checkpoint,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
    )

    opponent_params = init_adversarial_params(
        opponent_init_key,
        args=eval_args,
        central_obs_dim=int(central_obs.shape[-1]),
        actor_obs_dim=int(actor_obs.shape[-1]),
    )
    opponent_load_model = eval_args.opponent_load_model or eval_args.learner_load_model
    if opponent_load_model is not None:
        opponent_params = warm_start_actor_params(
            opponent_params,
            opponent_load_model,
            actor_obs_dim=int(actor_obs.shape[-1]),
            target_write_bits=eval_args.write_bits,
        )

    return AdversarialCheckpointBundle(
        args=eval_args,
        env=env,
        learner_params=checkpoint["params"],
        opponent_params=opponent_params,
        checkpoint=checkpoint,
    )


def evaluate_checkpoint_matrix(
    checkpoint_path: Path,
    *,
    argv: Sequence[str] | None = None,
    args: argparse.Namespace | None = None,
    eval_episodes: int | None = None,
    eval_max_steps: int | None = None,
    progress_callback: EvaluationProgressCallback | None = None,
    progress_step_interval: int | None = None,
    fixed_hub_positions: Sequence[Sequence[int]] | None = None,
    fixed_food_positions: Sequence[Sequence[int]] | None = None,
) -> dict[str, float]:
    bundle = load_checkpoint_for_evaluation(checkpoint_path, argv=argv, args=args)
    eval_args = bundle.args
    eval_overrides = {}
    if eval_episodes is not None:
        eval_overrides["eval_episodes"] = int(eval_episodes)
    if eval_max_steps is not None:
        eval_overrides["max_steps"] = int(eval_max_steps)
    if eval_overrides:
        eval_args = argparse.Namespace(**{**vars(eval_args), **eval_overrides})
    return evaluate_matrix(
        params=bundle.learner_params,
        opponent_params=bundle.opponent_params,
        args=eval_args,
        env=bundle.env,
        progress_callback=progress_callback,
        progress_step_interval=progress_step_interval,
        fixed_hub_positions=fixed_hub_positions,
        fixed_food_positions=fixed_food_positions,
    )


def _resolve_args(
    checkpoint: dict[str, Any],
    *,
    argv: Sequence[str] | None,
    args: argparse.Namespace | None,
) -> argparse.Namespace:
    if args is not None:
        return args
    if argv is not None:
        return parse_args(list(argv))
    return _args_from_checkpoint_payload(checkpoint.get("args", {}))


def _args_from_checkpoint_payload(payload: dict[str, Any]) -> argparse.Namespace:
    defaults = vars(parse_args(["--allow-random-init"])).copy()
    values = {
        key: value
        for key, value in dict(payload).items()
        if key not in _DERIVED_ARG_NAMES
    }
    defaults.update(values)
    for name in _PATH_ARG_NAMES:
        if defaults.get(name) is not None:
            defaults[name] = Path(defaults[name])
    defaults["num_ants"] = int(defaults["num_ants_per_team"])
    defaults["critic_architecture"] = "mlp"
    return argparse.Namespace(**defaults)


def _validate_checkpoint_dims(
    checkpoint: dict[str, Any],
    *,
    central_obs_dim: int,
    actor_obs_dim: int,
) -> None:
    if int(checkpoint["central_obs_dim"]) != central_obs_dim:
        raise ValueError(
            "Checkpoint central observation dimension does not match adversarial args."
        )
    if int(checkpoint["actor_obs_dim"]) != actor_obs_dim:
        raise ValueError("Checkpoint actor observation dimension does not match adversarial args.")
