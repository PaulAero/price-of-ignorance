# The price of ignorance

Your machine wears out invisibly. **Looking costs $5. Fixing costs $50. A
breakdown costs $500** (plus the forced $50 replacement). When is knowing worth
its price?

```bash
python ignorance.py   # no dependencies, ~170 lines, runs in seconds
```

```
policy                              total    doing  knowing  failing  fails/ep
head_in_sand                       1078.0     98.0      0.0    980.0      1.96
clockwork(k=8)                      422.9    311.2      0.0    111.8      0.22
check_then_fix(k=3, 0.55)           253.4    139.2     80.0     34.2      0.07
tracker(check@0.35, fix@0.60)       354.5    151.6     59.6    143.2      0.29
adaptive(4->2@0.25, fix@0.50)       245.8    154.3     81.5     10.0      0.02
```

Same machines, same 2000 seeds — and paying to look **never** changes the
physics (two RNG streams; there's a test for it). The scoreboard's three cost
columns are the point: **doing** (repairs), **knowing** (paid readings),
**failing** (breakdowns). The whole game is the split between them.

## The puzzle

The `tracker` maintains a wear estimate and only pays to check when the
estimate says so. It **loses to the dumb periodic checker by ~100 points** —
its mean-rate drift model gets burned by bad-batch machines that age twice as
fast, so it looks exactly when its model says "fine".

(Confession, kept because it's instructive: our first tracker lost by 150. An
adversarial review of this repo found that a third of that was a state bug —
it often paid for two repairs in a row. Fixed; the honest gap is ~100, and it
is entirely a modelling failure, not a bug.)

`adaptive` — check slowly, check fast after a worrying reading, fix early —
was contributed by that same review and is the **current bar: 245.8**.

## Beat 245.8

- Your policy is a function `(obs, state) -> "run" | "check" | "fix"`.
  `obs` gives you `t`, `failed`, and a `reading` only on the step you paid.
  No touching `machine.wear` (that's the point; there's a test for that too).
- Dev on the public seeds (`evaluate(policy)`, seeds 0-1999). A submission
  counts if it beats the bar **by more than statistical noise** (paired
  difference on the same seeds, > 2 SE) and holds on the held-out range
  (seeds 10000-11999, checked on the PR).
- Known headroom: an *illegal* oracle that reads true wear scores **~120**.
  The legal optimum is unknown — somewhere between 120 and 245.8. That gap is
  the leaderboard.

## Why this exists

Every RL environment you know gives observations away for free. Real
engineering doesn't: an inspection immobilizes the machine, a lab test burns a
sample, a high-fidelity simulation burns a day of compute. Costly observations
are an old idea in the literature (active sensing, ACNO-MDPs); what has no
standard home is the **cost of information as a first-class channel of the
environment API** — the way safe RL standardized the cost of constraint
violation. Here, every action is either **physical** or **epistemic**, every
action has a price, and `info["costs"]` always splits into
`physical / information / failure`. That's the whole contract, in miniature.

## The serious version

The full contract (typed priced actions, step semantics, an AgentView boundary)
plus two engineering-grade environments — fatigue-crack inspection planning and
turbofan degradation with biased free sensors — with tuned baselines, paired
confidence intervals, raw per-episode data, and a public multi-round
adversarial audit trail:

**→ [coin-envs](https://github.com/PI-Project-AI/coin-envs)** · essay: coming soon

Built with heavy AI assistance (Claude), adversarially reviewed by another
model (which found two real bugs in these 170 lines — see the confession
above — and whose fixes and tests are in the history). MIT.

Questions, submissions that beat the bar, or anything else:
**paul.provost@pi-project.ai** (or open an issue).
