# 000: Communication Bits Collapse To `1`

## Problem

The communication curriculum can train through wider write alphabets, but the
learned policy mostly writes `1`. A wider action space is not enough; the ants
need an incentive and a training setup that makes extra bits useful.

## Current Evidence

- The communication curriculum uses staged bit growth: `2 -> 3 -> 5 -> 8`.
- The local branch has a terminal entropy bonus over final nonzero byte values.
- Saved checkpoint probes still show the write policy collapsing to `1`.
- `num_steps = 80` means each PPO update sees 80 sequential steps per env.
- On the 25x25 setup, `max_steps = 2500`, so a full time-limit episode spans
  roughly 31 PPO updates.

## Working Hypothesis

Terminal-only entropy is too sparse. Most PPO updates do not contain an episode
end, so the write head gets many gradients from ordinary forage behavior and
old warm-start bias before it receives any entropy-shaped communication signal.

## Candidate Interventions

1. Increase rollout horizon.

   Try `num_steps = 256` first. If memory is tight, reduce `num_envs` from `16`
   to `8` so the batch is still manageable.

2. Reward chunk-local write entropy.

   Add a small reward based on the entropy of write actions inside each rollout
   chunk, excluding `0`. This gives PPO a communication signal every update
   instead of only at episode end.

3. Reward bit-level usage, not byte-level alphabet usage.

   For larger alphabets, entropy over 255 possible nonzero numbers may be too
   diffuse. Per-bit activation balance is closer to the actual objective:
   make the policy use more bits.

4. De-bias expanded write heads.

   Current staged transfer copies lower-bit write logits into the larger
   alphabet. That preserves behavior, but it can also preserve collapse.
   Consider adding small noise or neutral initialization for newly available
   values.

5. Probe sampled and deterministic behavior separately.

   Temperature-zero rollout can hide near-ties by always picking the lowest
   argmax value. But current probes suggest this is a real collapse, not only a
   rendering artifact.

## First Experiment

Change only the communication rollout horizon:

```json
{
  "num_steps": 256,
  "num_envs": 8
}
```

Keep the total update count comparable enough to inspect quickly, then compare:

- forage return
- write-action histogram
- final-grid byte histogram
- deterministic rollout
- sampled rollout

## Stop Rule

If `num_steps = 256` still writes almost only `1`, move the signal from terminal
grid entropy to chunk-local write entropy. Do not keep increasing horizon
blindly.
