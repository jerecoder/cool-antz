#!/usr/bin/env python3
"""Launch a Pygame random rollout for AntByteForagingEnv."""

from __future__ import annotations

import argparse

import pygame

from ant_byte_env import AntByteForagingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Pygame random rollout.")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--num-ants", type=int, default=4)
    parser.add_argument("--food-count", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--tile-size", type=int, default=56)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = AntByteForagingEnv(
        width=args.width,
        height=args.height,
        num_ants=args.num_ants,
        food_count=args.food_count,
        max_steps=args.max_steps,
        render_mode="human",
        tile_size=args.tile_size,
    )

    if args.seed is not None:
        env.action_space.seed(args.seed)

    _, info = env.reset(seed=args.seed)
    print(
        "random rollout started: delivered={delivered_food} remaining={remaining_food}".format(
            **info
        )
    )

    try:
        terminated = False
        truncated = False
        while not terminated and not truncated:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            _, reward, terminated, truncated, info = env.step(env.action_space.sample())
            print(
                "step={step_count} reward={reward:.2f} delivered={delivered_food} "
                "remaining={remaining_food}".format(reward=reward, **info)
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
