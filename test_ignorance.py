"""Tests: contract shape, causal invariants, and scoreboard regression.
Run with `python test_ignorance.py` (no dependencies) or pytest."""

from ignorance import (
    Machine,
    adaptive,
    check_then_fix,
    evaluate,
    head_in_sand,
    tracker,
)


def test_cost_channels_are_typed():
    m = Machine(seed=0)
    m.reset()
    _, _, _, info = m.step("check")
    assert info["costs"]["information"] == 5.0 and info["costs"]["physical"] == 0.0
    _, _, _, info = m.step("fix")
    assert info["costs"]["physical"] == 50.0 and info["costs"]["information"] == 0.0


def test_reward_equals_minus_costs():
    m = Machine(seed=3)
    m.reset()
    for action in ("run", "check", "fix", "run"):
        _, r, _, info = m.step(action)
        assert r == -sum(info["costs"].values())


def test_reading_only_when_paid():
    m = Machine(seed=1)
    m.reset()
    obs, *_ = m.step("run")
    assert obs["reading"] is None
    obs, *_ = m.step("check")
    assert obs["reading"] is not None


def test_checking_never_changes_the_physics():
    """THE causal invariant (found broken by the adversarial review, fixed):
    an epistemic action must not touch the world. Two policies that differ
    only by ignored checks must produce identical physical trajectories."""
    for seed in range(30):
        phys = {}
        for label, checks in (("without", False), ("with", True)):
            m = Machine(seed=seed)
            obs, done = m.reset(), False
            tot_phys = tot_fail = 0.0
            t = 0
            while not done:
                action = "check" if (checks and t % 2 == 0) else "run"
                obs, _r, done, info = m.step(action)
                tot_phys += info["costs"]["physical"]
                tot_fail += info["costs"]["failure"]
                t += 1
            phys[label] = (tot_phys, tot_fail)
        assert phys["without"] == phys["with"], f"seed {seed}: {phys}"


def test_tracker_never_double_fixes():
    """Regression for the state bug the review found (~48 pts of its loss)."""
    pol = tracker(0.35, 0.60)
    for seed in range(200):
        m = Machine(seed=seed)
        obs, done, state = m.reset(), False, {}
        prev = None
        while not done:
            action = pol(obs, state)
            assert not (action == "fix" and prev == "fix" and not obs["failed"]), \
                f"double fix at seed {seed}, t={obs['t']}"
            obs, _r, done, _info = m.step(action)
            prev = action


def test_deterministic_given_seed():
    r1 = evaluate(head_in_sand, episodes=50)
    r2 = evaluate(head_in_sand, episodes=50)
    assert r1 == r2


def test_scoreboard_regression():
    """The published numbers must be reproducible (tolerance for platform
    float noise only)."""
    expected = {
        "check_then_fix": (check_then_fix(3, 0.55), 253.4),
        "adaptive": (adaptive(), 245.8),
    }
    for name, (pol, target) in expected.items():
        got = evaluate(pol)["total"]
        assert abs(got - target) < 0.5, f"{name}: {got} vs published {target}"


def test_checking_beats_ignorance():
    blind = evaluate(head_in_sand, episodes=300)
    informed = evaluate(check_then_fix(3, 0.55), episodes=300)
    assert informed["total"] < blind["total"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests pass")
