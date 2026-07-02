import json
from pathlib import Path


EXPECTED_NOTEBOOKS = {
    Path("baselines/direct_goal.ipynb"),
    Path("communication/bit_curriculum.ipynb"),
    Path("curriculum/autocurriculum.ipynb"),
    Path("curriculum/exploration.ipynb"),
    Path("curriculum/forage.ipynb"),
    Path("curriculum/maze_exploration.ipynb"),
    Path("exploration_to_forage/base_50x50.ipynb"),
    Path("historical/map_ant_gated_mlp_curriculum.ipynb"),
    Path("scaling/ant_count_curriculum.ipynb"),
    Path("scaling/full_layout_8ants_half_food_50x50.ipynb"),
    Path("scaling/full_layout_8ants_half_food_shared_writes_50x50.ipynb"),
    Path("source_layouts/padded_sources_50x50.ipynb"),
    Path("source_layouts/proximity_sources_50x50.ipynb"),
    Path("source_layouts/scratch_smooth_sources_50x50.ipynb"),
    Path("timed_release/roles.ipynb"),
}


def _notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return "".join(
        line
        for cell in notebook.get("cells", [])
        for line in cell.get("source", [])
        if isinstance(line, str)
    )


def test_notebooks_are_grouped_by_workflow() -> None:
    notebooks_root = Path("notebooks")
    notebook_paths = {
        path.relative_to(notebooks_root)
        for path in notebooks_root.glob("*/*.ipynb")
    }

    assert notebook_paths == EXPECTED_NOTEBOOKS
    assert not list(notebooks_root.glob("*.ipynb"))


def test_notebooks_resolve_project_root_from_nested_folders() -> None:
    for relative_path in EXPECTED_NOTEBOOKS:
        source = _notebook_source(Path("notebooks") / relative_path)

        assert 'PROJECT_ROOT = Path.cwd()' in source
        assert 'pyproject.toml' in source
        assert 'src" / "ant_byte_env' in source
        assert 'Path.cwd().name == "notebooks"' not in source
