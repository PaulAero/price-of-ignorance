"""The price of ignorance — a 150-line decision environment.

Your machine wears out invisibly. Looking costs $5. Fixing costs $50.
A breakdown costs $500. When is knowing worth its price?

    python ignorance.py        # runs every baseline, prints the scoreboard

No dependencies. Beat the best baseline, open a PR.

The idea in one line: costly observations are an old idea (active sensing,
ACNO-MDPs); what has no standard home is the COST OF INFORMATION as a
first-class channel of the environment API, the way safe RL standardized
the cost of constraint violation. This file is the smallest possible demo
of that contract. The
serious, audited version (fatigue cracks, turbofans, tuned baselines,
adversarial audit trail) lives at: https://github.com/PI-Project-AI/coin-envs
"""

import math
import random

PRICE_CHECK = 5.0     # a sensor reading (information, not action)
PRICE_FIX = 50.0      # preventive replacement
PRICE_FAILURE = 500.0  # breakdown; a forced $50 replacement is added on top
HORIZON = 50
WEAR_RATE = 0.030     # median wear per step; every machine draws its own
RATE_SPREAD = 0.5     # lognormal sigma of that draw (the "bad batch" risk)
NOISE = 0.08          # sensor noise on a paid check


class Machine:
    """step(action) -> (obs, reward, done, info).

    Actions: "run" ($0), "check" ($5, returns a noisy wear reading),
    "fix" ($50, fresh part). Wear is hidden; only breakdowns are public.
    info["costs"] always splits into physical / information / failure —
    the point of the whole exercise.
    """

    def __init__(self, seed=None):
        if seed is None:
            seed = random.getrandbits(64)
        # two independent streams: paying to LOOK must never touch the
        # physics (an epistemic action changes what you know, not the world)
        self.world = random.Random(f"{seed}-world")
        self.sensor = random.Random(f"{seed}-sensor")

    def reset(self):
        self.wear = 0.0
        self.rate = self.world.lognormvariate(math.log(WEAR_RATE), RATE_SPREAD)
        self.t = 0
        return {"t": 0, "reading": None, "failed": False}

    def step(self, action):
        costs = {"physical": 0.0, "information": 0.0, "failure": 0.0}
        if action == "fix":
            costs["physical"] += PRICE_FIX
            self._renew()
        elif action == "check":
            costs["information"] += PRICE_CHECK
        elif action != "run":
            raise ValueError(f"unknown action {action!r}")

        # the world turns whether you look or not (wear accelerates);
        # order each step: action effect -> wear -> failure/renewal -> observation
        self.wear += self.rate * (1 + 2 * self.wear) * self.world.uniform(0.5, 1.5)
        failed = self.wear >= 1.0
        if failed:
            costs["failure"] += PRICE_FAILURE
            costs["physical"] += PRICE_FIX  # forced replacement
            self._renew()

        reading = None
        if action == "check":  # you paid: you see the END-of-step state
            # (if the machine broke this step, you read the fresh part)
            reading = max(0.0, self.wear + self.sensor.gauss(0.0, NOISE))

        self.t += 1
        obs = {"t": self.t, "reading": reading, "failed": failed}
        return obs, -sum(costs.values()), self.t >= HORIZON, {"costs": costs}

    def _renew(self):
        self.wear = 0.0
        self.rate = self.world.lognormvariate(math.log(WEAR_RATE), RATE_SPREAD)


# ----------------------------- baselines ------------------------------------
# Deliberately simple. The interesting question is not "what's optimal" but
# "how much of the gap can YOUR policy close, and how much does it spend on
# knowing vs doing".

def head_in_sand(obs, state):
    """Never look, never fix. The lower anchor."""
    return "run"


def clockwork(k):
    """Fix every k steps, never look. Maintenance by calendar."""
    def policy(obs, state):
        return "fix" if (obs["t"] > 0 and obs["t"] % k == 0) else "run"
    return policy


def check_then_fix(k, threshold):
    """Check every k steps; fix when the reading crosses the threshold."""
    def policy(obs, state):
        if obs["reading"] is not None and obs["reading"] >= threshold:
            return "fix"
        return "check" if obs["t"] % k == k - 1 else "run"
    return policy


def tracker(check_at, fix_at):
    """Track expected wear with a mean-rate model; check only when the
    ESTIMATE gets uncertain enough to matter, fix on a bad reading.
    ~12 lines of belief; no learning."""
    def policy(obs, state):
        if obs["failed"] or "wear" not in state:
            state.update(wear=0.0, drift=WEAR_RATE * math.exp(RATE_SPREAD ** 2 / 2))
        if obs["reading"] is not None:
            state["wear"] = obs["reading"]           # trust the paid reading
            if obs["reading"] >= fix_at:
                state["wear"] = 0.0                   # we fix: fresh part
                return "fix"
        state["wear"] += state["drift"] * (1 + 2 * state["wear"])
        if state["wear"] >= fix_at:
            state["wear"] = 0.0                       # we are about to fix
            return "fix"
        return "check" if state["wear"] >= check_at else "run"
    return policy


def adaptive(slow=4, fast=2, alert=0.25, fix_at=0.50):
    """Check slowly; after a worrying reading, check fast; fix early.
    Found by the adversarial reviewer of this repo — the bar to beat."""
    def policy(obs, state):
        k = state.get("k", slow)
        if obs["failed"]:
            state["k"] = k = slow
        r = obs["reading"]
        if r is not None:
            if r >= fix_at:
                state["k"] = slow
                return "fix"
            state["k"] = k = fast if r >= alert else slow
        return "check" if obs["t"] % k == k - 1 else "run"
    return policy


# ----------------------------- harness --------------------------------------

def evaluate(policy, episodes=2000, seed0=0):
    tot = {"physical": 0.0, "information": 0.0, "failure": 0.0}
    failures = 0
    for ep in range(episodes):
        m = Machine(seed=seed0 + ep)
        obs, done, state = m.reset(), False, {}
        while not done:
            obs, r, done, info = m.step(policy(obs, state))
            for key in tot:
                tot[key] += info["costs"][key]
            failures += int(obs["failed"])
    n = episodes
    return {k: v / n for k, v in tot.items()} | {
        "total": sum(tot.values()) / n, "failures_per_ep": failures / n}


if __name__ == "__main__":
    baselines = [
        ("head_in_sand", head_in_sand),
        ("clockwork(k=8)", clockwork(8)),
        ("check_then_fix(k=3, 0.55)", check_then_fix(3, 0.55)),
        ("tracker(check@0.35, fix@0.60)", tracker(0.35, 0.60)),
        ("adaptive(4->2@0.25, fix@0.50)", adaptive()),
    ]
    print(f"{'policy':32s} {'total':>8s} {'doing':>8s} {'knowing':>8s} "
          f"{'failing':>8s} {'fails/ep':>9s}")
    for name, pol in baselines:
        r = evaluate(pol)
        print(f"{name:32s} {r['total']:8.1f} {r['physical']:8.1f} "
              f"{r['information']:8.1f} {r['failure']:8.1f} {r['failures_per_ep']:9.2f}")
    print("\nSame machine, same 2000 seeds. The whole game is the split between")
    print("the three columns. Can you do better? -> README.md")
