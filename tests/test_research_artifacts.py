from pathlib import Path

from ant_byte_env.research import artifacts


def test_resolve_run_dir_preserves_matrix_relative_layout() -> None:
    assert artifacts.resolve_run_dir(
        Path("runs/autoresearch/forage_loop/A"),
        matrix_root=Path("runs/autoresearch"),
        override=Path("/tmp/research"),
    ) == Path("/tmp/research") / "forage_loop" / "A"


def test_resolve_run_dir_falls_back_to_name_for_external_paths() -> None:
    assert artifacts.resolve_run_dir(
        Path("/outside/A"),
        matrix_root=Path("runs/autoresearch"),
        override=Path("/tmp/research"),
    ) == Path("/tmp/research") / "A"


def test_forage_stage_checkpoint_path_respects_best_selection() -> None:
    checkpoint_dir = Path("runs/checkpoints")

    assert artifacts.forage_stage_checkpoint_path(
        checkpoint_dir,
        {"name": "25x25", "select_best_checkpoint": False},
        selected=True,
    ) == checkpoint_dir / "jax_mappo_forage_stage1_25x25.pkl"
    assert artifacts.forage_stage_checkpoint_path(
        checkpoint_dir,
        {"name": "25x25", "select_best_checkpoint": True},
        selected=True,
    ) == checkpoint_dir / "jax_mappo_forage_stage1_25x25_best.pkl"


def test_planned_stage_checkpoints_returns_selected_stage_paths() -> None:
    assert artifacts.planned_stage_checkpoints(
        {
            "checkpoint_dir": "runs/checkpoints",
            "stages": [
                {"name": "4x4"},
                {"name": "25x25", "select_best_checkpoint": True},
            ],
        }
    ) == [
        "runs/checkpoints/jax_mappo_forage_stage1_4x4.pkl",
        "runs/checkpoints/jax_mappo_forage_stage1_25x25_best.pkl",
    ]
