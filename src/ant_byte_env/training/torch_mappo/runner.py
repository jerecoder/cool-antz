"""Executable Torch MAPPO training loop."""

from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
import torch
import torch.optim as optim

from ant_byte_env import write_value_count
from ant_byte_env.runs import append_metrics, ensure_run_structure, write_json
from ant_byte_env.training.torch_mappo.checkpointing import checkpoint_args, load_agent_checkpoint
from ant_byte_env.training.torch_mappo.cli import parse_args
from ant_byte_env.training.torch_mappo.model import MAPPOAgent, make_mappo_loss
from ant_byte_env.training.torch_mappo.observations import (
    build_actor_observations,
    build_central_observations,
    obs_to_tensor,
)
from ant_byte_env.training.torch_mappo.rollout import (
    collect_rollout,
    make_envs,
    make_rollout_storage,
    reset_env,
    rollout_storage_to_tensordict,
    stack_obs,
    update_agent,
)


def main(argv: list[str] | None = None) -> dict[str, float]:
    args = parse_args(argv)
    run_name = f"{args.exp_name}__seed_{args.seed}__{int(time.time())}"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    metrics_path = None
    summary_path = None
    if args.run_dir is not None:
        ensure_run_structure(args.run_dir)
        if args.save_model is None:
            args.save_model = args.run_dir / "checkpoints" / "model.pt"
        metrics_path = args.run_dir / "metrics.jsonl"
        summary_path = args.run_dir / "summary.json"
        write_json(
            args.run_dir / "config.json",
            {
                "backend": "torch",
                "run_name": run_name,
                "args": checkpoint_args(args),
            },
        )

    envs = make_envs(args)
    for env_index, env in enumerate(envs):
        env.action_space.seed(args.seed + env_index)

    try:
        obs_items = [
            reset_env(env, seed=args.seed + env_index, args=args)[0]
            for env_index, env in enumerate(envs)
        ]
        next_obs = stack_obs(obs_items)
        obs_tensor = obs_to_tensor(next_obs, device)
        central_obs = build_central_observations(
            obs_tensor,
            food_scale=args.food_count,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        actor_obs = build_actor_observations(
            obs_tensor,
            central_obs,
            food_scale=args.food_count,
            actor_vision_radius=args.actor_vision_radius,
            write_bits=args.write_bits,
            obs_width=args.obs_width,
            obs_height=args.obs_height,
        )
        central_obs_dim = central_obs.shape[-1]
        actor_obs_dim = actor_obs.shape[-1]

        agent = MAPPOAgent(
            central_obs_dim=central_obs_dim,
            actor_obs_dim=actor_obs_dim,
            hidden_size=args.hidden_size,
            write_value_count=write_value_count(args.write_bits),
        ).to(device)
        loaded_checkpoint: dict[str, Any] | None = None
        if args.load_model is not None:
            loaded_checkpoint = load_agent_checkpoint(
                agent=agent,
                checkpoint_path=args.load_model,
                central_obs_dim=central_obs_dim,
                actor_obs_dim=actor_obs_dim,
                write_bits=args.write_bits,
                actor_vision_radius=args.actor_vision_radius,
                target_num_ants=args.num_ants,
                device=device,
            )
        loss_module = make_mappo_loss(args, agent).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
        if loaded_checkpoint is not None and "optimizer_state_dict" in loaded_checkpoint:
            optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
        storage = make_rollout_storage(
            args=args,
            actor_obs_dim=actor_obs_dim,
            central_obs_dim=central_obs_dim,
            device=device,
        )

        next_done = torch.zeros(args.num_envs, device=device)
        global_step = 0
        num_updates = max(1, args.total_timesteps // (args.num_envs * args.num_steps))
        final_metrics: dict[str, float] = {
            "global_step": 0.0,
            "loss": 0.0,
            "episode_return": 0.0,
            "episode_length": 0.0,
        }

        for update in range(1, num_updates + 1):
            if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * args.learning_rate

            next_obs, next_done, global_step, rollout_stats = collect_rollout(
                args=args,
                agent=agent,
                envs=envs,
                storage=storage,
                next_obs=next_obs,
                next_done=next_done,
                global_step=global_step,
                device=device,
            )

            update_metrics = update_agent(
                args=args,
                agent=agent,
                optimizer=optimizer,
                loss_module=loss_module,
                rollout=rollout_storage_to_tensordict(storage),
            )

            final_metrics = {
                **update_metrics,
                **rollout_stats,
                "global_step": float(global_step),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if not args.quiet:
                print(
                    "update={update}/{num_updates} step={step} loss={loss:.4f} "
                    "return={episode_return:.3f} len={episode_length:.1f} "
                    "entropy={entropy:.3f}".format(
                        update=update,
                        num_updates=num_updates,
                        step=global_step,
                        **final_metrics,
                    )
                )
            if metrics_path is not None:
                append_metrics(
                    metrics_path,
                    {
                        "update": update,
                        "num_updates": num_updates,
                        **final_metrics,
                    },
                )

        if args.save_model is not None:
            args.save_model.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "agent_state_dict": agent.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": checkpoint_args(args),
                    "central_obs_dim": central_obs_dim,
                    "actor_obs_dim": actor_obs_dim,
                    "run_name": run_name,
                },
                args.save_model,
            )
        if summary_path is not None:
            write_json(
                summary_path,
                {
                    "backend": "torch",
                    "run_name": run_name,
                    "metrics": final_metrics,
                    "checkpoint_path": args.save_model,
                },
            )
        return final_metrics
    finally:
        for env in envs:
            env.close()



if __name__ == "__main__":
    main()
