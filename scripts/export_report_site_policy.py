"""Export a local JAX MAPPO actor for the static report-site sandbox."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CHECKPOINT = Path(
    "runs/notebooks/fl50_60ants_half_food_sw_wc_8bits_stabilize_from_60best/checkpoints/"
    "best_full_layout_proximity_60ants_half_food_shared_writes_write_cost_8bits_stabilized.pkl"
)
DEFAULT_OUTPUT = Path(
    "docs/report-site/assets/data/policy-runner-50x50-frontier.js"
)


def _array(value: Any) -> list[Any]:
    return np.asarray(value, dtype=np.float32).tolist()


def _project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def _observation_layout(*, actor_obs_dim: int, args: dict[str, Any]) -> list[str]:
    radius = int(args.get("actor_vision_radius", 2))
    write_bits = int(args.get("write_bits", 1))
    num_ants = int(args.get("num_ants", 1))
    identity_types = args.get("agent_identity_types")
    identity_width = (
        0
        if num_ants <= 1
        else int(num_ants if identity_types is None else identity_types)
    )
    patch_size = (2 * radius + 1) ** 2
    current_dim = patch_size * (1 + 1 + write_bits + 1 + 1) + identity_width + 1 + 4
    legacy_dim = patch_size * (1 + write_bits + 1 + 1) + 1

    if int(actor_obs_dim) == current_dim:
        return [
            "local_food_patch",
            "local_ant_count_patch",
            "local_byte_bit_patches",
            "local_hub_patch",
            "local_border_or_obstacle_patch",
            "agent_identity_features",
            "own_carrying_flag",
            "own_facing_one_hot",
        ]
    if int(actor_obs_dim) == legacy_dim:
        return [
            "legacy_local_food_patch",
            "legacy_local_byte_bit_patches",
            "legacy_local_hub_patch",
            "legacy_local_border_patch",
            "own_carrying_flag",
        ]
    raise ValueError(
        f"Unsupported actor_obs_dim={actor_obs_dim}; expected {current_dim} current "
        f"or {legacy_dim} legacy features."
    )


def export_policy(checkpoint_path: Path, output_path: Path) -> None:
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)

    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = vars(args)

    params = checkpoint["params"]
    source_dir = checkpoint_path.parent.parent
    summary = _load_json(source_dir / "summary.json")
    actor_obs_dim = int(checkpoint["actor_obs_dim"])

    payload = {
        "schema": "cool_antz_report_site_policy_v1",
        "source": {
            "checkpoint": _project_path(checkpoint_path),
            "run_name": checkpoint.get("run_name"),
            "config": _project_path(source_dir / "config.json"),
            "summary": _project_path(source_dir / "summary.json"),
        },
        "env": {
            "width": int(args.get("width", 25)),
            "height": int(args.get("height", 25)),
            "num_ants": int(args.get("num_ants", 4)),
            "food_count": int(args.get("food_count", 23)),
            "food_sources": int(args.get("food_sources", 12)),
            "write_bits": int(args.get("write_bits", 3)),
            "actor_vision_radius": int(args.get("actor_vision_radius", 2)),
            "max_steps": int(args.get("max_steps", 2500)),
            "write_while_moving": bool(args.get("write_while_moving", False)),
            "per_ant_write_channels": bool(args.get("per_ant_write_channels", False)),
            "random_ant_spawn": bool(args.get("random_ant_spawn", False)),
            "agent_identity_types": args.get("agent_identity_types"),
            "food_scale": int(
                max(
                    1,
                    np.ceil(
                        float(args.get("food_count", 23))
                        / float(max(int(args.get("food_sources", 12)), 1))
                    ),
                )
            ),
        },
        "actor": {
            "actor_obs_dim": actor_obs_dim,
            "central_obs_dim": int(checkpoint["central_obs_dim"]),
            "observation_layout": _observation_layout(
                actor_obs_dim=actor_obs_dim,
                args=args,
            ),
            "actor_body": [
                {
                    "weight": _array(params.actor_body[0].weight),
                    "bias": _array(params.actor_body[0].bias),
                },
                {
                    "weight": _array(params.actor_body[1].weight),
                    "bias": _array(params.actor_body[1].bias),
                },
            ],
            "move_head": {
                "weight": _array(params.move_head.weight),
                "bias": _array(params.move_head.bias),
            },
            "write_head": {
                "weight": _array(params.write_head.weight),
                "bias": _array(params.write_head.bias),
            },
        },
        "training_metrics": checkpoint.get("metrics", {}),
        "summary_metrics": {} if summary is None else summary.get("metrics", {}),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, separators=(",", ":"))
    output_path.write_text(
        "window.CoolAntzPolicyRunnerData = "
        + serialized
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path} from {checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export_policy(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
