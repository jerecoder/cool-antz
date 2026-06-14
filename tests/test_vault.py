from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from ant_byte_env.vault import create_vault_entry


def test_create_vault_entry_copies_assets_and_writes_metadata(tmp_path) -> None:
    asset_path = tmp_path / "rollout.mp4"
    asset_path.write_bytes(b"video bytes")

    entry_dir = create_vault_entry(
        vault_dir=tmp_path / "vault",
        title="Policy rollout",
        description="A rollout video worth keeping.",
        assets=[asset_path],
        metadata={"stage": "4x4"},
        created_at=datetime(2026, 6, 11, 12, 30, 5, tzinfo=timezone.utc),
    )

    assert entry_dir.name == "20260611T123005Z"
    assert (entry_dir / "rollout.mp4").read_bytes() == b"video bytes"
    payload = json.loads((entry_dir / "metadata.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["created_at"] == "2026-06-11T12:30:05Z"
    assert payload["title"] == "Policy rollout"
    assert payload["description"] == "A rollout video worth keeping."
    assert payload["metadata"] == {"stage": "4x4"}
    assert payload["assets"] == [
        {
            "filename": "rollout.mp4",
            "source_path": str(asset_path),
            "size_bytes": len(b"video bytes"),
        }
    ]


def test_create_vault_entry_uses_unique_folder_and_asset_names(tmp_path) -> None:
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    first_asset = left_dir / "plot.png"
    second_asset = right_dir / "plot.png"
    first_asset.write_bytes(b"first")
    second_asset.write_bytes(b"second")

    fixed_time = datetime(2026, 6, 11, 12, 30, 5, tzinfo=timezone.utc)
    first_entry = create_vault_entry(
        vault_dir=tmp_path / "vault",
        title="First",
        description="First result.",
        assets=[first_asset, second_asset],
        created_at=fixed_time,
    )
    second_entry = create_vault_entry(
        vault_dir=tmp_path / "vault",
        title="Second",
        description="Second result.",
        assets=[first_asset],
        created_at=fixed_time,
    )

    assert first_entry.name == "20260611T123005Z"
    assert second_entry.name == "20260611T123005Z-02"
    assert (first_entry / "plot.png").read_bytes() == b"first"
    assert (first_entry / "plot_2.png").read_bytes() == b"second"


def test_create_vault_entry_rejects_missing_or_empty_inputs(tmp_path) -> None:
    asset_path = tmp_path / "plot.png"
    asset_path.write_bytes(b"plot")

    with pytest.raises(ValueError, match="title"):
        create_vault_entry(
            vault_dir=tmp_path / "vault",
            title=" ",
            description="Result.",
            assets=[asset_path],
        )
    with pytest.raises(ValueError, match="asset"):
        create_vault_entry(
            vault_dir=tmp_path / "vault",
            title="Result",
            description="Result.",
            assets=[],
        )
    with pytest.raises(FileNotFoundError):
        create_vault_entry(
            vault_dir=tmp_path / "vault",
            title="Result",
            description="Result.",
            assets=[tmp_path / "missing.png"],
        )
