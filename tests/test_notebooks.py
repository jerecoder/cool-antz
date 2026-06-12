from __future__ import annotations

import json
from pathlib import Path


def test_notebooks_are_clean_and_use_packaged_trainers() -> None:
    stale_imports = (
        "import train_mappo_jax",
        "from train_mappo_jax",
        "train_mappo_jax_core",
        "load_raw_checkpoint",
        "BASE_CHECKPOINT",
        "runs/legacy",
    )

    for path in sorted(Path("notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None, path
                assert cell.get("outputs", []) == [], path
            source = "".join(cell.get("source", []))
            for stale_import in stale_imports:
                assert stale_import not in source, (path, stale_import)


def test_communication_notebook_writes_distinct_vision_rollouts() -> None:
    notebook = json.loads(
        Path("notebooks/train_jax_communication_curriculum.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "checkpoint_path.stem}_rollout" not in source
    assert "stage_name = checkpoint_path.parent.parent.name" in source
    assert "jax_mappo_15x15_{stage_name}_vision_rollout.gif" in source
