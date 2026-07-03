"""Export a local JAX MAPPO actor for the static report-site sandbox."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CHECKPOINT = Path(
    "runs/notebooks/ant_count_25x25_3_bits/4_ants/checkpoints/model.pkl"
)
DEFAULT_OUTPUT = Path(
    "docs/report-site/assets/data/policy-runner-25x25-4ants.js"
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


def export_policy(checkpoint_path: Path, output_path: Path) -> None:
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)

    args = checkpoint.get("args", {})
    if not isinstance(args, dict):
        args = vars(args)

    params = checkpoint["params"]
    source_dir = checkpoint_path.parent.parent
    summary = _load_json(source_dir / "summary.json")

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
            "actor_obs_dim": int(checkpoint["actor_obs_dim"]),
            "central_obs_dim": int(checkpoint["central_obs_dim"]),
            "observation_layout": [
                "legacy_local_food_patch",
                "legacy_local_byte_bit_patches",
                "legacy_local_hub_patch",
                "legacy_local_border_patch",
                "own_carrying_flag",
            ],
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
