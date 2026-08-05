"""Smoke tests: contract shape, determinism, scoreboard reproducibility."""

from ignorance import Machine, check_then_fix, evaluate, head_in_sand


def test_cost_channels_are_typed():
    m = Machine(seed=0)
    m.reset()
    _, _, _, info = m.step("check")
    assert info["costs"]["information"] == 5.0 and info["costs"]["physical"] == 0.0
    _, _, _, info = m.step("fix")
    assert info["costs"]["physical"] == 50.0 and info["costs"]["information"] == 0.0


def test_reading_only_when_paid():
    m = Machine(seed=1)
    m.reset()
    obs, *_ = m.step("run")
    assert obs["reading"] is None
    obs, *_ = m.step("check")
    assert obs["reading"] is not None


def test_deterministic_given_seed():
    r1 = evaluate(head_in_sand, episodes=50)
    r2 = evaluate(head_in_sand, episodes=50)
    assert r1 == r2


def test_checking_beats_ignorance():
    blind = evaluate(head_in_sand, episodes=300)
    informed = evaluate(check_then_fix(3, 0.55), episodes=300)
    assert informed["total"] < blind["total"]


if __name__ == "__main__":
    test_cost_channels_are_typed()
    test_reading_only_when_paid()
    test_deterministic_given_seed()
    test_checking_beats_ignorance()
    print("all tests pass")
