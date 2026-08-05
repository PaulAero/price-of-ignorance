# The price of ignorance

Your machine wears out invisibly. **Looking costs $5. Fixing costs $50. A
breakdown costs $500.** When is knowing worth its price?

```bash
python ignorance.py   # no dependencies, ~150 lines, runs in seconds
```

```
policy                              total    doing  knowing  failing  fails/ep
head_in_sand                       1064.5     96.8      0.0    967.8      1.94
clockwork(k=8)                      421.6    311.1      0.0    110.5      0.22
check_then_fix(k=3, 0.55)           255.1    139.1     80.0     36.0      0.07
tracker(check@0.35, fix@0.60)       408.6    211.4     58.7    138.5      0.28
```

Same machine, same 2000 seeds. Note the scoreboard's three cost columns:
**doing** (repairs), **knowing** (paid sensor readings), **failing**
(breakdowns). The whole game is the split between them.

And a puzzle to start you off: our "smart" tracker — it maintains a wear
estimate and only pays to check when the estimate says so — **loses to the
dumb periodic checker**. Its mean-rate drift model gets burned by bad-batch
machines that age twice as fast. Fixing that is exactly the interesting
problem. Beat 255.1, open a PR, the scoreboard is the leaderboard.

## Why this exists

Every RL environment you know gives you observations **for free**. Real
engineering doesn't: an inspection immobilizes the machine, a lab test burns a
sample, a high-fidelity simulation burns a day of compute. The economics of
*when to pay to know* is the daily job of every maintenance planner — and it
has no standard home in our APIs. Safe RL made the cost of *violating a
constraint* a first-class channel of the environment interface; nothing ever
did that for the cost of *information*.

This file is the smallest possible version of that idea: every action is
either **physical** (changes the machine) or **epistemic** (changes what you
know about it), every action has a price, and `info["costs"]` always splits
into `physical / information / failure`.

## The serious version

The full contract (typed priced actions, the information-cost channel, step
semantics) plus two engineering-grade environments — fatigue-crack inspection
planning and turbofan degradation with biased free sensors — with tuned
baselines, paired confidence intervals, raw per-episode data, and a public
three-round adversarial audit trail, lives here:

**→ [coin-envs](https://github.com/PaulAero/coin-envs)** · essay: coming soon

## Rules of the leaderboard

- Your policy is a function `(obs, state) -> "run" | "check" | "fix"`. No
  peeking at `machine.wear` (that's the point).
- Evaluation: `evaluate(policy)` as shipped — 2000 episodes, seeds 0-1999.
- PRs add a row to the scoreboard with your numbers, reproduced by CI.

Built with heavy AI assistance (Claude); the design and its mistakes are
documented in the serious repo's audit trail. MIT.
