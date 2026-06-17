# 024 - Single-Ant 50x50 M3 Write-Noise Screen

## Run

- W&B: https://wandb.ai/jerefigueiredo-universidad-de-san-andr-s/cool-antz/runs/sqsq2bdr
- Local summary: `runs/autoresearch/forage_50x50_memory_m3_live/memory/M3/sweep_summary.json`
- Local log: `runs/autoresearch/forage_50x50_memory_m3_live/logs/M3_live.log`
- Final checkpoint: `runs/autoresearch/forage_50x50_memory_m3_live/memory/M3/checkpoints/jax_mappo_forage_stage1_50x50.pkl`
- Videos: `40x40`, `50x50`

## Setup

- Phase/id: `memory/M3`
- Stages: `4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50`
- Updates per stage: `250`
- Env steps per update: `4 envs * 256 steps = 1024`
- Total env steps: `2,816,000`
- Reward: `pickup_bonus=0.25`, `distance_bonus=0.02`, `write_bit_entropy_bonus=0.01`
- Algorithm: `gamma=0.995`, `gae_lambda=0.97`, `ent_coef=0.01`
- Network: `hidden_size=128`, `write_bits=3`
- No-cheat constraints: `num_ants=1`, `actor_vision_radius=1`, no food/hub coordinates or direction vectors in actor observations.

## Results

| Stage | Last episode return | Best episode return | Pickups | Deliveries | Carrying rate | Remaining food | Write rate | Final marked tiles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25x25 | 0.8850 | 1.8293 | 2.0 | 3.0 | 0.850 | 21.00 | 0.042 | 15.8 |
| 30x30 | 0.3789 | 1.8276 | 2.0 | 1.0 | 0.470 | 22.00 | 0.005 | 3.8 |
| 35x35 | 0.3881 | 1.6371 | 2.0 | 1.0 | 0.570 | 32.00 | 0.088 | 18.2 |
| 40x40 | 0.0078 | 1.1433 | 0.0 | 0.0 | 1.000 | 34.75 | 0.392 | 354.0 |
| 45x45 | 0.0682 | 0.9437 | 1.0 | 0.0 | 0.836 | 37.75 | 0.153 | 131.8 |
| 50x50 | 0.0685 | 0.9463 | 1.0 | 0.0 | 0.838 | 43.75 | 0.134 | 122.5 |

## Interpretation

`M3` completed cleanly on the smaller network, so the `M2` failure was likely the wider 256-hidden footprint rather than a hard incompatibility with three write bits. The result is worse than `M1` and `H1` on the largest maps: `40x40+` ends with little or no delivery behavior, and the policy often carries food without getting it home.

The new diagnostics make the failure mode clearer. On `40x40`, the policy marks about `354` tiles while producing zero pickups and zero deliveries. On `45x45` and `50x50`, it still marks more than `100` tiles while delivering nothing. This looks like noisy overmarking from the extra write capacity and entropy pressure, not useful trail memory.

## Next Experiment

Run `memory/M4`: return to `write_bits=2`, keep `hidden_size=128`, and lower `write_bit_entropy_bonus` from `0.02` to `0.005`. This isolates whether `M1`'s best large-map behavior came from useful two-bit memory or from excessive write exploration that happened to help occasionally.
