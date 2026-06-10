from __future__ import annotations

from pathlib import Path

import pygame

from ant_byte_env.sprites import ASSET_FILENAMES, load_sprites


def test_loads_checked_in_sprite_assets() -> None:
    sprites = load_sprites(tile_size=18)

    assert set(sprites) == set(ASSET_FILENAMES)
    for surface in sprites.values():
        assert isinstance(surface, pygame.Surface)
        assert surface.get_size() == (18, 18)


def test_missing_assets_use_generated_fallbacks(tmp_path: Path) -> None:
    sprites = load_sprites(tile_size=20, asset_dir=tmp_path)

    assert set(sprites) == set(ASSET_FILENAMES)
    for surface in sprites.values():
        assert isinstance(surface, pygame.Surface)
        assert surface.get_size() == (20, 20)
