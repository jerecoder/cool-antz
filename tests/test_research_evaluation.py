from __future__ import annotations

from pathlib import Path
from typing import Any

from ant_byte_env.research.evaluation import evaluate_research_plan_checkpoint


def test_evaluate_research_plan_checkpoint_returns_empty_when_disabled() -> None:
    assert (
        evaluate_research_plan_checkpoint(
            {"evaluation": {"deterministic_episodes": 0, "sampled_episodes": 0}},
            summary={},
            evaluate_checkpoint=lambda *args, **kwargs: {},
        )
        == {}
    )


def test_evaluate_research_plan_checkpoint_reports_missing_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing.pkl"

    result = evaluate_research_plan_checkpoint(
        {"evaluation": {"deterministic_episodes": 2, "sampled_episodes": 0}},
        summary={"final_checkpoint": str(checkpoint)},
        evaluate_checkpoint=lambda *args, **kwargs: {},
    )

    assert result == {
        "error": "missing_checkpoint_file",
        "checkpoint": str(checkpoint),
        "deterministic_episodes": 2,
        "sampled_episodes": 0,
        "action_modes": [],
    }


def test_evaluate_research_plan_checkpoint_calls_all_requested_modes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_evaluate(path: Path, **kwargs: Any) -> dict[str, float]:
        calls.append({"path": path, **kwargs})
        return {"episode_return": float(len(calls))}

    result = evaluate_research_plan_checkpoint(
        {
            "evaluation": {
                "deterministic_episodes": 2,
                "sampled_episodes": 3,
                "seed_offset": 7,
                "shuffle_positions": False,
                "action_modes": [
                    {"name": "greedy", "action_mode": "greedy", "episodes": 4},
                    {
                        "name": "hot",
                        "action_mode": "sampled",
                        "episodes": 5,
                        "seed_offset": 123,
                        "move_temperature": 0.8,
                    },
                ],
            }
        },
        summary={"final_checkpoint": str(checkpoint)},
        evaluate_checkpoint=fake_evaluate,
    )

    assert result["checkpoint"] == str(checkpoint)
    assert result["deterministic"] == {"episode_return": 1.0}
    assert result["sampled"] == {"episode_return": 2.0}
    assert result["greedy"] == {"episode_return": 3.0}
    assert result["hot"] == {"episode_return": 4.0}
    assert calls == [
        {
            "path": checkpoint,
            "num_episodes": 2,
            "seed_offset": 7,
            "deterministic": True,
            "shuffle_positions": False,
        },
        {
            "path": checkpoint,
            "num_episodes": 3,
            "seed_offset": 100_007,
            "deterministic": False,
            "shuffle_positions": False,
        },
        {
            "path": checkpoint,
            "num_episodes": 4,
            "seed_offset": 200_007,
            "action_mode": "greedy",
            "shuffle_positions": False,
        },
        {
            "path": checkpoint,
            "num_episodes": 5,
            "seed_offset": 123,
            "action_mode": "sampled",
            "shuffle_positions": False,
            "move_temperature": 0.8,
        },
    ]
