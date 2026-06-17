# 022 - Single-Ant 50x50 M1 Two-Bit Memory Screen

## Run

- W&B: https://wandb.ai/jerefigueiredo-universidad-de-san-andr-s/cool-antz/runs/ascbraz6
- Local summary: `runs/autoresearch/forage_50x50_memory_live/memory/M1/sweep_summary.json`
- Local log: `runs/autoresearch/forage_50x50_memory_live/logs/M1_live.log`
- Final checkpoint: `runs/autoresearch/forage_50x50_memory_live/memory/M1/checkpoints/jax_mappo_forage_stage1_50x50.pkl`
- Videos: `25x25`, `40x40`, `50x50`

## Setup

- Phase/id: `memory/M1`
- Stages: `4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50`
- Updates per stage: `250`
- Env steps per update: `4 envs * 256 steps = 1024`
- Total env steps: `2,816,000`
- Reward: `pickup_bonus=0.25`, `distance_bonus=0.02`, `write_bit_entropy_bonus=0.02`
- Algorithm: `gamma=0.995`, `gae_lambda=0.97`, `ent_coef=0.01`
- Network: `hidden_size=128`, `write_bits=2`
- No-cheat constraints: `num_ants=1`, `actor_vision_radius=1`, no food/hub coordinates or direction vectors in actor observations.

## Results

| Stage | Last episode return | Best episode return | Last env return | Last pickups | Last deliveries | Remaining food |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x4 | 4.4262 | 8.8460 | 3.00 | 22.0 | 12.0 | 1.25 |
| 8x8 | 5.1900 | 5.4855 | 4.00 | 18.0 | 16.0 | 3.75 |
| 12x12 | 2.1582 | 4.9000 | 1.50 | 10.0 | 6.0 | 9.00 |
| 16x16 | 3.0279 | 3.6480 | 2.50 | 8.0 | 10.0 | 9.25 |
| 20x20 | 1.1390 | 3.0906 | 1.00 | 2.0 | 4.0 | 11.75 |
| 25x25 | 0.6367 | 2.1488 | 0.50 | 2.0 | 2.0 | 18.50 |
| 30x30 | 0.9508 | 2.1485 | 0.75 | 3.0 | 3.0 | 22.50 |
| 35x35 | 1.3223 | 1.8269 | 1.00 | 5.0 | 4.0 | 31.25 |
| 40x40 | 1.0671 | 1.3183 | 0.75 | 5.0 | 3.0 | 31.00 |
| 45x45 | 0.0009 | 1.1968 | 0.00 | 0.0 | 0.0 | 34.75 |
| 50x50 | 0.2613 | 2.0198 | 0.25 | 0.0 | 1.0 | 42.50 |

## Interpretation

Two-bit self-memory helped some large-map exploration signals but did not solve the final curriculum. Compared with `H1`, `M1` improved `50x50` best return (`2.0198` versus `0.8799`) and final remaining food (`42.50` versus `45.25`), and it kept nonzero delivery behavior at `35x35` and `40x40`.

The failure remains unstable: `45x45` still ended with zero pickups and deliveries, and the final `50x50` rollout delivered only once. This suggests the extra bit channel is promising enough to test, but the current run does not prove that the policy learned a reliable route-marking strategy.

## Next Experiments

1. Run `memory/M2` with `write_bits=3` and the new write-memory diagnostics enabled.
2. Compare `write_action_nonzero_rate`, `final_mean_nonzero_byte_tiles`, pickups, deliveries, and remaining food on `35x35+`.
3. If `M2` writes heavily without improving deliveries, reduce `write_bit_entropy_bonus` or add a small write-value penalty instead of adding more shaping.
