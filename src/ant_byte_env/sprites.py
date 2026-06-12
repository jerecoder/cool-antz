"""Pygame sprite loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

SpriteName = Literal["ant", "food", "hub", "tile"]

ASSET_FILENAMES = {
    "ant": "ant.png",
    "food": "food.png",
    "hub": "hub.png",
    "tile": "tile.png",
}

ASSET_DIR = Path(__file__).with_name("assets")


def load_sprites(tile_size: int, asset_dir: Path | None = None) -> dict[SpriteName, pygame.Surface]:
    """Load sprite assets, falling back to generated placeholder surfaces."""

    source_dir = asset_dir or ASSET_DIR
    return {
        name: _load_sprite(name, source_dir / filename, tile_size)
        for name, filename in ASSET_FILENAMES.items()
    }


def _load_sprite(name: SpriteName, path: Path, tile_size: int) -> pygame.Surface:
    if path.exists():
        try:
            sprite = pygame.image.load(str(path))
            return pygame.transform.smoothscale(sprite, (tile_size, tile_size))
        except pygame.error:
            pass

    return _fallback_sprite(name, tile_size)


def _fallback_sprite(name: SpriteName, tile_size: int) -> pygame.Surface:
    surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
    if name == "tile":
        surface.fill((219, 208, 177))
        pygame.draw.rect(surface, (176, 162, 127), surface.get_rect(), max(1, tile_size // 16))
        return surface

    if name == "food":
        radius = max(3, tile_size // 4)
        center = (tile_size // 2, tile_size // 2)
        pygame.draw.circle(surface, (74, 163, 74), center, radius)
        pygame.draw.circle(
            surface,
            (136, 205, 98),
            (center[0] - radius // 3, center[1] - radius // 3),
            max(2, radius // 2),
        )
        pygame.draw.line(
            surface,
            (59, 119, 55),
            (center[0], center[1] - radius),
            (center[0] + radius // 2, center[1] - radius - max(2, tile_size // 10)),
            max(1, tile_size // 16),
        )
        return surface

    if name == "hub":
        margin = max(2, tile_size // 8)
        rect = pygame.Rect(margin, margin, tile_size - 2 * margin, tile_size - 2 * margin)
        pygame.draw.ellipse(surface, (129, 96, 70), rect)
        pygame.draw.ellipse(surface, (76, 54, 42), rect.inflate(-tile_size // 4, -tile_size // 4))
        pygame.draw.rect(surface, (87, 67, 51), rect, max(1, tile_size // 14), border_radius=2)
        return surface

    body_color = (83, 55, 43)
    accent_color = (42, 31, 28)
    center_y = tile_size // 2
    pygame.draw.ellipse(
        surface,
        body_color,
        pygame.Rect(tile_size // 3, center_y - tile_size // 6, tile_size // 3, tile_size // 3),
    )
    pygame.draw.circle(surface, body_color, (tile_size // 3, center_y), max(3, tile_size // 7))
    pygame.draw.circle(surface, body_color, (2 * tile_size // 3, center_y), max(3, tile_size // 8))
    for offset in (-tile_size // 5, 0, tile_size // 5):
        pygame.draw.line(
            surface,
            accent_color,
            (tile_size // 2, center_y),
            (tile_size // 5, center_y + offset),
            max(1, tile_size // 18),
        )
        pygame.draw.line(
            surface,
            accent_color,
            (tile_size // 2, center_y),
            (4 * tile_size // 5, center_y + offset),
            max(1, tile_size // 18),
        )
    return surface
