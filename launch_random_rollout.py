#!/usr/bin/env python3
"""Launch a Pygame random rollout for AntByteForagingEnv."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol

import numpy as np
import pygame

from ant_byte_env import AntByteForagingEnv


class VideoWriter(Protocol):
    def append_data(self, frame: np.ndarray) -> None:
        """Append one RGB frame to the output video."""

    def close(self) -> None:
        """Flush and close the output video."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Pygame random rollout.")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--num-ants", type=int, default=4)
    parser.add_argument("--food-count", type=int, default=8)
    parser.add_argument("--food-sources", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--tile-size", type=int, default=56)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video", type=Path, default=None, help="Export rollout video.")
    parser.add_argument(
        "--video-fps",
        type=int,
        default=AntByteForagingEnv.metadata["render_fps"],
        help="Frames per second for --video output.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without opening a Pygame window. Useful with --video.",
    )
    return parser.parse_args()


def show_frame(frame: np.ndarray, window: pygame.Surface) -> None:
    surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    window.blit(surface, (0, 0))
    pygame.display.flip()


def append_video_frame(writer: VideoWriter | None, frame: np.ndarray) -> None:
    if writer is not None:
        writer.append_data(frame)


def main() -> None:
    args = parse_args()
    show_window = not args.no_window
    env = AntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        food_source_count=args.food_sources,
        max_steps=args.max_steps,
        render_mode="rgb_array",
        tile_size=args.tile_size,
    )

    if args.seed is not None:
        env.action_space.seed(args.seed)

    writer: VideoWriter | None = None
    window = None
    clock = None

    try:
        if args.video is not None:
            import imageio.v2 as imageio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=args.video_fps)

        if show_window:
            pygame.display.init()
            window = pygame.display.set_mode(
                (args.width * args.tile_size, args.height * args.tile_size)
            )
            pygame.display.set_caption("AntByteForaging random rollout")
            clock = pygame.time.Clock()

        _, info = env.reset(seed=args.seed)
        frame = env.render()
        assert frame is not None
        append_video_frame(writer, frame)
        if window is not None:
            show_frame(frame, window)
        print(
            "random rollout started: delivered={delivered_food} remaining={remaining_food}".format(
                **info
            )
        )

        terminated = False
        truncated = False
        while not terminated and not truncated:
            if window is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            frame = env.render()
            assert frame is not None
            append_video_frame(writer, frame)
            if window is not None:
                show_frame(frame, window)
                assert clock is not None
                clock.tick(args.video_fps)
            print(
                "step={step_count} reward={reward:.2f} delivered={delivered_food} "
                "remaining={remaining_food}".format(reward=reward, **info)
            )
    finally:
        if writer is not None:
            writer.close()
            print(f"video saved to {args.video}")
        env.close()
        if pygame.display.get_init():
            pygame.display.quit()


if __name__ == "__main__":
    main()
