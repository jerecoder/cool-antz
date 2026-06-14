from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def create_vault_entry(
    *,
    vault_dir: Path,
    title: str,
    description: str,
    assets: Sequence[Path],
    metadata: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Copy assets into a timestamped vault entry and write metadata.json."""

    if not title.strip():
        raise ValueError("title must not be empty.")
    if not description.strip():
        raise ValueError("description must not be empty.")
    if not assets:
        raise ValueError("at least one asset is required.")

    actual_created_at = _utc_datetime(created_at)
    entry_dir = _next_entry_dir(vault_dir, actual_created_at)
    entry_dir.mkdir(parents=True)

    asset_records: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for asset in assets:
        source_path = Path(asset)
        if not source_path.is_file():
            raise FileNotFoundError(f"vault asset does not exist: {source_path}")

        destination_name = _unique_asset_name(source_path.name, used_names)
        destination_path = entry_dir / destination_name
        shutil.copy2(source_path, destination_path)
        used_names.add(destination_name)
        asset_records.append(
            {
                "filename": destination_name,
                "source_path": str(source_path),
                "size_bytes": destination_path.stat().st_size,
            }
        )

    metadata_path = entry_dir / "metadata.json"
    metadata_payload = {
        "schema_version": 1,
        "created_at": actual_created_at.isoformat().replace("+00:00", "Z"),
        "title": title,
        "description": description,
        "assets": asset_records,
        "metadata": dict(metadata or {}),
    }
    metadata_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return entry_dir


def _utc_datetime(created_at: datetime | None) -> datetime:
    if created_at is None:
        return datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        return created_at.replace(tzinfo=timezone.utc)
    return created_at.astimezone(timezone.utc)


def _next_entry_dir(vault_dir: Path, created_at: datetime) -> Path:
    stem = created_at.strftime("%Y%m%dT%H%M%SZ")
    candidate = vault_dir / stem
    suffix = 2
    while candidate.exists():
        candidate = vault_dir / f"{stem}-{suffix:02d}"
        suffix += 1
    return candidate


def _unique_asset_name(filename: str, used_names: set[str]) -> str:
    candidate = Path(filename).name
    if candidate not in used_names:
        return candidate

    stem = Path(candidate).stem
    suffix = Path(candidate).suffix
    index = 2
    while True:
        numbered_candidate = f"{stem}_{index}{suffix}"
        if numbered_candidate not in used_names:
            return numbered_candidate
        index += 1
