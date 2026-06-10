from __future__ import annotations

from ant_byte_env import AntByteForagingEnv


def main() -> None:
    env = AntByteForagingEnv(render_mode="human")
    obs, info = env.reset()
    print(f"start delivered={info['delivered_food']} remaining={info['remaining_food']}")

    terminated = False
    truncated = False
    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(
            "step={step_count} reward={reward:.2f} delivered={delivered_food} "
            "remaining={remaining_food}".format(reward=reward, **info)
        )

    env.close()


if __name__ == "__main__":
    main()
