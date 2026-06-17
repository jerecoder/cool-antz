# 020 - Single-Ant 50x50 R1 Live Curriculum

## Run

- W&B: https://wandb.ai/jerefigueiredo-universidad-de-san-andr-s/cool-antz/runs/xz8ipcbo
- Local summary: `runs/autoresearch/forage_50x50_live/reward/R1/sweep_summary.json`
- Local log: `runs/autoresearch/forage_50x50_live/logs/R1_live.log`
- Final checkpoint: `runs/autoresearch/forage_50x50_live/reward/R1/checkpoints/jax_mappo_forage_stage1_50x50.pkl`
- Videos: `25x25`, `40x40`, `50x50`

## Setup

- Phase/id: `reward/R1`
- Stages: `4, 8, 12, 16, 20, 25, 30, 35, 40, 45, 50`
- Updates per stage: `500`
- Env steps per update: `4 envs * 80 steps = 320`
- Total env steps: `1,760,000`
- Reward: `pickup_bonus=0.25`, `distance_bonus=0.02`
- Algorithm: `gamma=0.99`, `gae_lambda=0.95`, `ent_coef=0.01`
- Network: `hidden_size=128`
- No-cheat constraints: `num_ants=1`, `actor_vision_radius=1`, no food/hub coordinates or direction vectors in actor observations.

## Results

| Stage | Last episode return | Best episode return | Last env return |
| --- | ---: | ---: | ---: |
| 4x4 | 1.5183 | 4.3450 | 1.00 |
| 8x8 | 1.5743 | 2.7011 | 1.25 |
| 12x12 | 0.3766 | 1.7577 | 0.25 |
| 16x16 | 0.4443 | 1.3835 | 0.25 |
| 20x20 | 0.0637 | 1.2507 | 0.00 |
| 25x25 | 0.0025 | 0.9424 | 0.00 |
| 30x30 | 0.0647 | 0.9403 | 0.00 |
| 35x35 | 0.0005 | 1.2474 | 0.00 |
| 40x40 | 0.0008 | 1.2536 | 0.00 |
| 45x45 | 0.0011 | 0.3754 | 0.00 |
| 50x50 | 0.0008 | 0.3140 | 0.00 |

## Interpretation

The curriculum transfers movement/pickup behavior through small maps, but the policy does not maintain reliable delivery once maps reach roughly `20x20`. The nonzero best returns on large maps suggest occasional useful trajectories, but the last returns near zero indicate the learned behavior is not stable.

Current evidence points more toward long-horizon credit assignment and self-memory/path marking than toward more pickup shaping alone.

## Next Experiments

1. Run the long-horizon algorithm candidate `H1`: `num_steps=256`, `num_envs=8`, `gamma=0.995`, `gae_lambda=0.97`, `distance_bonus=0.02`.
2. Run a memory-marker candidate with `write_bits=2` or `write_bits=3`, still with `actor_vision_radius=1`, to test whether self-laid trail state helps return-to-hub without exposing hub coordinates.
3. Inspect rollout videos at `25x25`, `40x40`, and `50x50` for failure mode: no pickup, pickup without return, orbiting, or unstable exploration.
4. If video shows pickup-without-return, prioritize marker/write-bit and longer GAE experiments. If video shows no pickup, prioritize exploration/entropy and reward scaling.
