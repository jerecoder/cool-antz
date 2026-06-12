from __future__ import annotations

import json
from pathlib import Path


def test_notebooks_are_clean_and_use_packaged_trainers() -> None:
    stale_imports = ("import train_mappo_jax", "from train_mappo_jax", "train_mappo_jax_core")

    for path in sorted(Path("notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None, path
                assert cell.get("outputs", []) == [], path
            source = "".join(cell.get("source", []))
            for stale_import in stale_imports:
                assert stale_import not in source, (path, stale_import)


def test_legacy_jax_checkpoint_shim_exists() -> None:
    shim = Path("train_mappo_jax_core.py")

    assert shim.is_file()
    assert "ant_byte_env.training.jax_mappo.core" in shim.read_text(encoding="utf-8")
