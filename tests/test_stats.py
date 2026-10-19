from pytest_agentcheck.stats import (
    compare_two_proportions,
    flakiness_score,
    is_flaky,
    normal_quantile,
    sequential_verdict,
    wilson_score_interval,
)


def test_normal_quantile() -> None:
    assert abs(normal_quantile(0.5)) < 1e-7
    assert abs(normal_quantile(0.975) - 1.95996) < 1e-4
    assert abs(normal_quantile(0.95) - 1.64485) < 1e-4
    assert abs(normal_quantile(0.995) - 2.57583) < 1e-4


def test_wilson_score_interval_known_values() -> None:
    low_0, high_0 = wilson_score_interval(0, 0)
    assert low_0 == 0.0
    assert high_0 == 1.0

    low_10, high_10 = wilson_score_interval(10, 10, confidence=0.95)
    assert abs(low_10 - 0.7225) < 1e-3
    assert high_10 == 1.0

    low_zero, high_zero = wilson_score_interval(0, 10, confidence=0.95)
    assert low_zero == 0.0
    assert abs(high_zero - 0.2775) < 1e-3

    low_100, high_100 = wilson_score_interval(90, 100, confidence=0.95)
    assert abs(low_100 - 0.8256) < 1e-3
    assert abs(high_100 - 0.9448) < 1e-3


def test_compare_two_proportions_equal() -> None:
    res = compare_two_proportions(90, 100, 90, 100)
    assert res.difference == 0.0
    assert res.is_significant is False
    assert res.is_degradation is False


def test_compare_two_proportions_degradation() -> None:
    res = compare_two_proportions(60, 100, 95, 100, alpha=0.05)
    assert res.is_degradation is True
    assert res.is_significant is True
    assert res.p_value < 0.001


def test_compare_two_proportions_improvement() -> None:
    res = compare_two_proportions(95, 100, 70, 100, alpha=0.05)
    assert res.is_degradation is False
    assert res.is_significant is False


def test_sequential_verdict_impossible() -> None:
    decision = sequential_verdict(
        successes=0,
        completed=3,
        total_runs=20,
        min_pass_rate=0.9,
    )
    assert decision.should_stop is True
    assert decision.verdict == "FAIL"
    assert decision.reason == "impossible_to_reach_threshold"


def test_sequential_verdict_guaranteed() -> None:
    decision = sequential_verdict(
        successes=18,
        completed=18,
        total_runs=20,
        min_pass_rate=0.9,
    )
    assert decision.should_stop is True
    assert decision.verdict == "PASS"
    assert decision.reason == "guaranteed_threshold_met"


def test_sequential_verdict_in_progress() -> None:
    decision = sequential_verdict(
        successes=5,
        completed=6,
        total_runs=20,
        min_pass_rate=0.8,
    )
    assert decision.should_stop is False
    assert decision.verdict == "INCONCLUSIVE"


def test_sequential_verdict_all_completed() -> None:
    decision = sequential_verdict(
        successes=19,
        completed=20,
        total_runs=20,
        min_pass_rate=0.8,
    )
    assert decision.should_stop is True
    assert decision.verdict == "PASS"


def test_is_flaky() -> None:
    assert is_flaky(0.5) is True
    assert is_flaky(0.1) is True
    assert is_flaky(0.9) is True
    assert is_flaky(0.04) is False
    assert is_flaky(0.96) is False
    assert is_flaky(50.0) is True
    assert is_flaky(98.0) is False


def test_flakiness_score() -> None:
    assert flakiness_score(0.5) == 1.0
    assert flakiness_score(0.0) == 0.0
    assert flakiness_score(1.0) == 0.0
    assert abs(flakiness_score(0.7) - 0.6) < 1e-6
