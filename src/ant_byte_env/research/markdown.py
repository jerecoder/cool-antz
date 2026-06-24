"""Markdown rendering for archived research plans."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def research_experiment_markdown(plan: Mapping[str, Any]) -> str:
    target = dict(plan.get("target", {}))
    lines = [
        f"# {plan['id']}: {plan.get('title', '')}",
        "",
        f"Family: {plan.get('family', '')}",
        f"Mode: {plan.get('mode', '')}",
        f"Run directory: `{plan.get('run_dir', '')}`",
        *(
            [f"Source checkpoint: `{plan['source_checkpoint']}`"]
            if plan.get("source_checkpoint")
            else []
        ),
        "",
        "## Hypothesis",
        str(plan.get("hypothesis", "")),
        "",
        "## Intervention",
        str(plan.get("intervention", "")),
        "",
        "## Baseline To Beat",
        str(target.get("baseline", "")),
        "",
        "## Success Signal",
        str(plan.get("success_signal", "")),
        "",
        "## Evaluation Gate",
        "```json",
        json.dumps(plan.get("evaluation", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Report Notes",
        str(plan.get("report_notes", "")),
        "",
        "## Key Settings",
        "```json",
        json.dumps(plan.get("resolved_args", {}), indent=2, sort_keys=True),
        "```",
    ]
    if plan.get("mode") == "forage_curriculum":
        lines.extend(
            [
                "",
                "## Stage Schedule",
                "```json",
                json.dumps(plan.get("stages", []), indent=2, sort_keys=True),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


__all__ = ["research_experiment_markdown"]
