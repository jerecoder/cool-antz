"""Curated result indexing for AntByte experiment artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def index_result_metadata(source_root: Path, output_path: Path) -> dict[str, Any]:
    source_root = Path(source_root)
    entries = [_entry_from_metadata(path, source_root) for path in sorted(source_root.rglob("metadata.json"))]
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_root": str(source_root),
        "entry_count": len(entries),
        "entries": entries,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _entry_from_metadata(metadata_path: Path, source_root: Path) -> dict[str, Any]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    entry_dir = metadata_path.parent
    assets = []
    for asset in payload.get("assets", []):
        filename = asset.get("filename")
        asset_path = entry_dir / filename if filename else None
        assets.append(
            {
                "filename": filename,
                "size_bytes": asset.get(
                    "size_bytes",
                    asset_path.stat().st_size if asset_path is not None and asset_path.exists() else 0,
                ),
            }
        )
    return {
        "id": entry_dir.name,
        "path": str(entry_dir.relative_to(source_root) if entry_dir.is_relative_to(source_root) else entry_dir),
        "created_at": payload.get("created_at"),
        "title": payload.get("title", ""),
        "description": payload.get("description", ""),
        "assets": assets,
        "metadata": payload.get("metadata", {}),
    }
