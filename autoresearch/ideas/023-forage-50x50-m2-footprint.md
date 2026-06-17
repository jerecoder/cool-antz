# 023 - Single-Ant 50x50 M2 Footprint Failure

## Run

- W&B: https://wandb.ai/jerefigueiredo-universidad-de-san-andr-s/cool-antz/runs/gwgbxmjc
- Local log: `runs/autoresearch/forage_50x50_memory_m2_live/logs/M2_live.log`
- Partial checkpoints: through `20x20`
- Final summary: not produced
- Videos: none produced

## Setup

- Phase/id: `memory/M2`
- Intended stages: `4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50`
- Updates per stage: `250`
- Env steps per update: `4 envs * 256 steps = 1024`
- Reward: `pickup_bonus=0.25`, `distance_bonus=0.02`, `write_bit_entropy_bonus=0.01`
- Algorithm: `gamma=0.995`, `gae_lambda=0.97`, `ent_coef=0.01`
- Network: `hidden_size=256`, `write_bits=3`
- No-cheat constraints: `num_ants=1`, `actor_vision_radius=1`, no food/hub coordinates or direction vectors in actor observations.

## Outcome

`M2` saved checkpoints through `20x20` and stopped while compiling or starting `25x25`. There was no Python traceback in the main log and no `sweep_summary.json`, so this is an incomplete run rather than a policy-quality result.

The most plausible cause is footprint pressure from combining `write_bits=3` with `hidden_size=256` on a machine that was already at `99%` disk use and had recently saturated swap. Because `M1` completed with `hidden_size=128`, the next useful test is to isolate the write channel from the wider network.

## Next Experiment

Run `memory/M3`: `write_bits=3`, `hidden_size=128`, same long-horizon setup, and the new write-memory diagnostics enabled.
