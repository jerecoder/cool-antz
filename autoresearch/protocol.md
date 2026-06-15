# Autoresearch Protocol

Use this when a training run surprises us or when a reward/config change is
tempting but not yet grounded.

## 1. Observe

Record the concrete failure. Include paths, not vibes.

Good:

```text
runs/notebooks/communication_bits_25x25/8_bits/checkpoints/model.pkl writes 1
with probability about 0.977 on reset-state probes.
```

Weak:

```text
The ants do not communicate.
```

## 2. Hypothesize

Write one causal claim.

```text
Terminal-only entropy is too sparse because 25x25 episodes span many PPO
updates, so most 80-step rollout chunks get no communication-diversity credit.
```

## 3. Mutate

Change one dominant variable first. Examples:

- Increase `num_steps` from `80` to `256`.
- Reward nonterminal write-action bit entropy inside the rollout chunk.
- De-bias expanded write heads when growing the alphabet.
- Evaluate sampled and deterministic rollouts separately.

## 4. Measure

Every run should capture:

- exact git branch and commit
- dirty diff summary, if any
- training command or notebook cell
- config path
- checkpoint path
- summary metrics
- write-action histogram
- final-grid byte histogram
- deterministic rollout artifact
- sampled rollout artifact, when relevant

## 5. Decide

Use one of these verdicts:

- `keep`: evidence improved and no obvious regression appeared
- `mutate`: evidence moved, but the intervention is incomplete
- `drop`: evidence did not support the hypothesis
- `blocked`: run could not answer the question

Do not stack three clever changes before one boring measurement. That is how
we lose the thread.
