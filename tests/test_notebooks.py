from __future__ import annotations

import json
from pathlib import Path


MAPPO_STAGE_SIZES = (4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25)
ANT_COUNT_STAGES = (2, 3, 4, 6, 8)


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


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


def test_mappo_curriculum_keeps_stage_count_and_reaches_25x25() -> None:
    source = notebook_source(Path("notebooks/train_mappo_curriculum.ipynb"))

    assert f"STAGE_SIZES = {MAPPO_STAGE_SIZES!r}" in source
    assert "for size in STAGE_SIZES" in source
    assert "from `4x4` through `25x25`" in source

    spec = json.loads(Path("experiments/forage_curriculum.json").read_text(encoding="utf-8"))
    assert spec["args"]["width"] == 25
    assert spec["args"]["height"] == 25
    assert spec["args"]["obs_width"] == 25
    assert spec["args"]["obs_height"] == 25
    assert spec["metadata"]["stage_sizes"] == list(MAPPO_STAGE_SIZES)
    assert spec["metadata"]["stage_count"] == len(MAPPO_STAGE_SIZES)


def test_communication_notebook_writes_distinct_vision_rollouts() -> None:
    source = notebook_source(Path("notebooks/train_jax_communication_curriculum.ipynb"))

    assert "checkpoint_path.stem}_rollout" not in source
    assert "stage_name = checkpoint_path.parent.parent.name" in source
    assert "jax_mappo_15x15_{stage_name}_vision_rollout.gif" in source


def test_ant_count_curriculum_starts_from_three_bit_25x25_checkpoint() -> None:
    source = notebook_source(Path("notebooks/train_jax_ant_count_curriculum.ipynb"))

    assert "COMMUNICATION_BITS = 3" in source
    assert "width\": 25" in source
    assert "height\": 25" in source
    assert "food_count\": 23" in source
    assert "food_sources\": 12" in source
    assert "max_steps\": 2500" in source
    assert "ANT_STAGES = [2, 3, 4, 6, 8]" in source
    assert f"ANT_STAGES = {list(ANT_COUNT_STAGES)!r}" in source
    assert "communication_bits_25x25/3_bits/checkpoints/model.pkl" in source
    assert "prepare_ant_count_checkpoint" in source
    assert "expand_critic_input_for_ant_count" in source


def test_direct_goal_notebook_uses_direct_goal_config_and_evaluation() -> None:
    source = notebook_source(Path("notebooks/train_jax_direct_goal_baseline.ipynb"))

    assert "experiments\" / \"direct_goal_baseline.json\"" in source
    assert "evaluate_checkpoint(CHECKPOINT_PATH, num_episodes=8)" in source
    assert "50x50" in source
    assert "3x3" in source
    assert "5` writable bits" in source
    assert "10` ants" in source
