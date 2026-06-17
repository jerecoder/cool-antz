# 021 - Single-Ant 50x50 H1 Horizon Screen

## Run

- W&B: https://wandb.ai/jerefigueiredo-universidad-de-san-andr-s/cool-antz/runs/lvmkhnmy
- Local summary: `runs/autoresearch/forage_50x50_horizon_live/algorithm/H1/sweep_summary.json`
- Local log: `runs/autoresearch/forage_50x50_horizon_live/logs/H1_live.log`
- Final checkpoint: `runs/autoresearch/forage_50x50_horizon_live/algorithm/H1/checkpoints/jax_mappo_forage_stage1_50x50.pkl`
- Videos: `25x25`, `40x40`, `50x50`

## Setup

- Phase/id: `algorithm/H1`
- Stages: `4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50`
- Updates per stage: `250`
- Env steps per update: `4 envs * 256 steps = 1024`
- Total env steps: `2,816,000`
- Reward: `pickup_bonus=0.25`, `distance_bonus=0.02`
- Algorithm: `gamma=0.995`, `gae_lambda=0.97`, `ent_coef=0.01`
- Network: `hidden_size=128`
- No-cheat constraints: `num_ants=1`, `actor_vision_radius=1`, no food/hub coordinates or direction vectors in actor observations.

## Results

| Stage | Last episode return | Best episode return | Last env return | Last pickups | Last deliveries | Remaining food |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4x4 | 8.8058 | 8.8058 | 6.50 | 36.0 | 26.0 | 0.75 |
| 8x8 | 2.2596 | 5.3386 | 1.75 | 8.0 | 7.0 | 5.25 |
| 12x12 | 2.5791 | 4.8509 | 2.00 | 9.0 | 8.0 | 8.75 |
| 16x16 | 2.5818 | 4.5848 | 2.00 | 9.0 | 8.0 | 10.00 |
| 20x20 | 2.2016 | 3.0178 | 1.75 | 7.0 | 7.0 | 8.75 |
| 25x25 | 0.3782 | 2.2650 | 0.25 | 2.0 | 1.0 | 17.00 |
| 30x30 | 0.6295 | 2.0751 | 0.50 | 2.0 | 2.0 | 23.25 |
| 35x35 | 1.0729 | 1.7576 | 0.75 | 5.0 | 3.0 | 31.75 |
| 40x40 | 0.8795 | 2.9455 | 0.75 | 2.0 | 3.0 | 29.00 |
| 45x45 | -0.0010 | 1.8816 | 0.00 | 0.0 | 0.0 | 34.50 |
| 50x50 | 0.2483 | 0.8799 | 0.25 | 0.0 | 1.0 | 45.25 |

## Interpretation

The longer rollout horizon is a real improvement over `R1`: it produces stable deliveries through `20x20` and nonzero delivery behavior through `40x40`. The remaining failure is now mostly large-map path memory and sparse rediscovery, not basic pickup mechanics.

The `45x45` and `50x50` diagnostics suggest the ant often spends long stretches carrying or failing to find new pickup opportunities. The next no-cheat test should therefore increase only the self-written local byte channel, keeping the actor's visual radius and observations unchanged.

## Next Experiments

1. Add and run `memory/M1`: H1 horizon with `write_bits=2` and a small `write_bit_entropy_bonus`, still local-only.
2. If `M1` increases pickups or deliveries on `35x35+`, compare `memory/M2` with `write_bits=3`.
3. If larger write channels produce high carrying but low delivery, add a write-use diagnostic/probe and consider a small write penalty decay rather than increasing reward shaping.
