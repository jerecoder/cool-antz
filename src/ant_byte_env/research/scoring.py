"""Scoring helpers for comparing archived research-loop runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def target_stage_metrics(
    summary: Mapping[str, Any],
    *,
    target_stage: str,
) -> dict[str, Any]:
    rows = list(summary.get("curriculum", {}).get("stage_metrics", []))
    matches = [dict(row) for row in rows if str(row.get("stage_name")) == target_stage]
    return matches[-1] if matches else {}


def promotion_score(metrics: Mapping[str, Any]) -> float:
    episode_return = float(metrics.get("episode_return", 0.0))
    deliveries = float(metrics.get("delivery_events", 0.0))
    remaining = float(metrics.get("final_mean_remaining_food", 0.0))
    return episode_return + 0.02 * deliveries - 0.01 * remaining


def flatten_evaluation_metrics(evaluation: Mapping[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for mode, metrics in evaluation.items():
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                clean_key = str(key)
                if clean_key.startswith("eval_"):
                    clean_key = clean_key[len("eval_") :]
                flat[f"evaluation/{mode}/{clean_key}"] = float(value)
    if "score" in evaluation and isinstance(evaluation["score"], (int, float)):
        flat["evaluation/score"] = float(evaluation["score"])
    return flat


def summary_score(summary: Mapping[str, Any]) -> float:
    evaluation = summary.get("evaluation", {})
    if isinstance(evaluation, Mapping):
        score = evaluation_score(evaluation)
        if score is not None:
            return score

    target_stage = str(summary.get("target_stage", "25x25"))
    metrics = (
        target_stage_metrics(summary, target_stage=target_stage)
        if "curriculum" in summary
        else {}
    )
    if metrics:
        return promotion_score(metrics)
    return 0.0


def evaluation_score(evaluation: Mapping[str, Any]) -> float | None:
    rows = [
        metrics
        for metrics in evaluation.values()
        if isinstance(metrics, Mapping) and "eval_mean_episode_return" in metrics
    ]
    if not rows:
        return None
    mean_return = sum(
        float(row.get("eval_mean_episode_return", 0.0)) for row in rows
    ) / len(rows)
    mean_delivery = sum(
        float(row.get("eval_mean_delivered_food", 0.0)) for row in rows
    ) / len(rows)
    mean_fraction = sum(
        float(row.get("eval_mean_delivered_fraction", 0.0)) for row in rows
    ) / len(rows)
    return mean_return + 0.02 * mean_delivery + 0.5 * mean_fraction


def extra_evaluation_summary(evaluation: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    extras: dict[str, dict[str, float]] = {}
    for mode, metrics in evaluation.items():
        if mode in {"deterministic", "sampled"} or not isinstance(metrics, Mapping):
            continue
        if "eval_mean_episode_return" not in metrics:
            continue
        extras[str(mode)] = {
            "return": float(metrics.get("eval_mean_episode_return", 0.0)),
            "delivered": float(metrics.get("eval_mean_delivered_food", 0.0)),
            "fraction": float(metrics.get("eval_mean_delivered_fraction", 0.0)),
            "success_rate": float(metrics.get("eval_success_rate", 0.0)),
            "episode_length": float(metrics.get("eval_mean_episode_length", 0.0)),
        }
    return extras


__all__ = [
    "evaluation_score",
    "extra_evaluation_summary",
    "flatten_evaluation_metrics",
    "promotion_score",
    "summary_score",
    "target_stage_metrics",
]
