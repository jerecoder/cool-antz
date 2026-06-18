from __future__ import annotations

import json
from pathlib import Path


MAPPO_STAGE_SIZES = tuple(range(4, 51))
ANT_COUNT_STAGES = (2, 3, 4, 6, 8)
MAX_NOTEBOOK_CODE_LINES = 40


def notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def code_cell_line_count(cell: dict[str, object]) -> int:
    source = "".join(cell.get("source", []))
    return len(source.splitlines())


def test_notebooks_are_clean_short_and_use_shared_workflows() -> None:
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
        notebook_text = notebook_source(path)
        assert "notebook_workflows" in notebook_text
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None, path
                assert cell.get("outputs", []) == [], path
                assert code_cell_line_count(cell) <= MAX_NOTEBOOK_CODE_LINES, path
            source = "".join(cell.get("source", []))
            for stale_import in stale_imports:
                assert stale_import not in source, (path, stale_import)


def test_mappo_curriculum_keeps_stage_count_and_reaches_50x50() -> None:
    source = notebook_source(Path("notebooks/train_mappo_curriculum.ipynb"))

    assert "STAGE_SIZES = range(4, 51)" in source
    assert "workflows.build_forage_curriculum_stages(STAGE_SIZES)" in source
    assert "workflows.run_forage_curriculum" in source
    assert "workflows.render_forage_rollouts" in source
    assert "from `4x4` through `50x50`" in source
    assert "WANDB_PROJECT = None" in source
    assert 'WANDB_GROUP = "forage_curriculum_50x50"' in source
    assert 'WANDB_MODE = "online"' in source
    assert "WANDB_VIDEO_MAX_FRAMES = 600" in source
    assert "wandb_project=WANDB_PROJECT" in source
    assert "wandb_video_max_frames=WANDB_VIDEO_MAX_FRAMES" in source

    spec = json.loads(Path("experiments/forage_curriculum.json").read_text(encoding="utf-8"))
    assert spec["args"]["width"] == 50
    assert spec["args"]["height"] == 50
    assert spec["args"]["obs_width"] == 50
    assert spec["args"]["obs_height"] == 50
    assert spec["args"]["write_while_moving"] is True
    assert spec["args"]["distance_bonus"] == 0.0
    assert spec["metadata"]["stage_sizes"] == list(MAPPO_STAGE_SIZES)
    assert spec["metadata"]["stage_count"] == len(MAPPO_STAGE_SIZES)


def test_autocurriculum_notebook_uses_single_env_curriculum_config() -> None:
    source = notebook_source(Path("notebooks/train_mappo_autocurriculum.ipynb"))

    assert "workflows.run_autocurriculum_training" in source
    assert "workflows.render_autocurriculum_rollout" in source
    assert 'WANDB_GROUP = "autocurriculum_50x50"' in source
    assert "AUTOCURRICULUM_CONFIG" in source
    assert "ADVANCE_COOKIES = 6" in source
    assert 'experiment_args["autocurriculum_success_cookies"] = ADVANCE_COOKIES' in source
    assert 'experiment_args["food_count"] = ADVANCE_COOKIES * FOOD_SOURCES' in source
    assert "workflows.config_common_args(\n    experiment_args," in source
    assert "from `4x4` through `50x50` inside each episode" in source

    spec = json.loads(Path("experiments/autocurriculum.json").read_text(encoding="utf-8"))
    assert spec["args"]["autocurriculum"] is True
    assert spec["args"]["width"] == 50
    assert spec["args"]["height"] == 50
    assert spec["args"]["obs_width"] == 50
    assert spec["args"]["obs_height"] == 50
    assert spec["args"]["food_count"] == 12
    assert spec["args"]["food_sources"] == 2
    assert spec["metadata"]["autocurriculum_start_size"] == 4
    assert spec["metadata"]["autocurriculum_success_cookies"] == 6
    assert spec["metadata"]["food_source_count"] == 2
    assert spec["metadata"]["cookies_per_source"] == 6


def test_notebook_render_cells_do_not_cap_rollout_frames() -> None:
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        source = notebook_source(path)

        assert "ROLLOUT_TILE_SIZE = workflows.NOTEBOOK_ROLLOUT_TILE_SIZE" in source
        assert "ROLLOUT_MAX_FRAMES" not in source
        assert "    max_frames=" not in source
        assert "tile_size=ROLLOUT_TILE_SIZE" in source


def test_jax_notebooks_configure_runtime_before_importing_jax() -> None:
    for path in sorted(Path("notebooks").glob("*.ipynb")):
        source = notebook_source(path)

        assert 'XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35"' in source
        assert "workflows.configure_jax_notebook_runtime()" in source
        assert "workflows.assert_notebook_resources_available(runtime_status)" in source
        assert source.index("configure_jax_notebook_runtime") < source.index("import jax")


def test_communication_notebook_writes_distinct_vision_rollouts() -> None:
    source = notebook_source(Path("notebooks/train_jax_communication_curriculum.ipynb"))
    helper_source = Path("src/ant_byte_env/notebook_workflows.py").read_text(encoding="utf-8")

    assert "checkpoint_path.stem}_rollout" not in source
    assert "workflows.run_communication_bit_curriculum" in source
    assert "workflows.run_communication_post_stage_sequence" in source
    assert "POLISH_CONFIG" in source
    assert "workflows.render_communication_rollouts" in source
    assert "extra_checkpoint_paths=CONSOLIDATED_CHECKPOINTS" in source
    assert "jax_mappo_25x25_{checkpoint.parent.parent.name}_vision_rollout.mp4" in helper_source
    assert "NOTEBOOK_ROLLOUT_POLICY_TEMPERATURE = 1.0" in helper_source

    spec = json.loads(Path("experiments/communication_bits.json").read_text(encoding="utf-8"))
    assert spec["args"]["width"] == 25
    assert spec["args"]["height"] == 25
    assert spec["args"]["obs_width"] == 50
    assert spec["args"]["obs_height"] == 50
    assert spec["args"]["write_while_moving"] is True
    assert spec["args"]["distance_bonus"] == 0.0
    assert spec["args"]["write_bit_entropy_bonus"] == 0.5
    assert spec["args"]["ent_coef"] == 0.02
    assert spec["metadata"]["consolidation"]["enabled"] is True
    assert spec["metadata"]["consolidation"]["global_update_cap"] == 5000
    assert spec["metadata"]["consolidation"]["args"]["write_bit_entropy_bonus"] == 0.05
    assert spec["metadata"]["polish"]["enabled"] is True
    assert spec["metadata"]["polish"]["global_update_cap"] == 2500
    assert spec["metadata"]["polish"]["args"]["write_bit_entropy_bonus"] == 0.0
    assert spec["metadata"]["polish"]["args"]["ent_coef"] == 0.0


def test_ant_count_curriculum_starts_from_three_bit_50_padded_checkpoint() -> None:
    source = notebook_source(Path("notebooks/train_jax_ant_count_curriculum.ipynb"))
    helper_source = Path("src/ant_byte_env/notebook_workflows.py").read_text(encoding="utf-8")

    assert "COMMUNICATION_BITS = 3" in source
    assert "workflows.ant_count_training_args" in source
    assert '"width": 25' in helper_source
    assert '"height": 25' in helper_source
    assert '"obs_width": 50' in helper_source
    assert '"obs_height": 50' in helper_source
    assert '"food_count": 23' in helper_source
    assert '"food_sources": 6' in helper_source
    assert '"max_steps": 2500' in helper_source
    assert '"write_while_moving": True' in helper_source
    assert "ANT_STAGES = [2, 3, 4, 6, 8]" in source
    assert f"ANT_STAGES = {list(ANT_COUNT_STAGES)!r}" in source
    assert "communication_bits_25x25/3_bits/checkpoints/model.pkl" in source
    assert "workflows.run_ant_count_curriculum" in source
    assert "workflows.render_ant_count_rollouts" in source
    assert "def prepare_ant_count_checkpoint" in helper_source
    assert "def expand_critic_input_for_ant_count" in helper_source
    assert "def prepare_ant_count_checkpoint" not in source
    assert "def expand_critic_input_for_ant_count" not in source


def test_direct_goal_notebook_uses_direct_goal_config_and_evaluation() -> None:
    source = notebook_source(Path("notebooks/train_jax_direct_goal_baseline.ipynb"))

    assert "experiments\" / \"direct_goal_baseline.json\"" in source
    assert "evaluate_checkpoint(CHECKPOINT_PATH, num_episodes=8)" in source
    assert "50x50" in source
    assert "3x3" in source
    assert "5` writable bits" in source
    assert "10` ants" in source
