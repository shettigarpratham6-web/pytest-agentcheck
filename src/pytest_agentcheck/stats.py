from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TwoProportionResult:
    p1: float
    p2: float
    difference: float
    z_score: float
    p_value: float
    is_significant: bool
    is_degradation: bool


@dataclass(frozen=True)
class EarlyStoppingDecision:
    should_stop: bool
    verdict: str
    reason: str
    pass_rate: float
    interval: tuple[float, float]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def normal_quantile(p: float) -> float:
    if p <= 0.0:
        return -float("inf")
    if p >= 1.0:
        return float("inf")
    if p == 0.5:
        return 0.0

    t = math.sqrt(-2.0 * math.log(min(p, 1.0 - p)))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    x = t - num / den
    if p < 0.5:
        x = -x

    for _ in range(3):
        pdf = normal_pdf(x)
        if pdf > 0.0:
            x -= (normal_cdf(x) - p) / pdf

    return x


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)

    p_hat = successes / total
    alpha = 1.0 - confidence
    z = normal_quantile(1.0 - alpha / 2.0)
    z2 = z * z

    denom = 1.0 + z2 / total
    center = p_hat + z2 / (2.0 * total)
    margin = z * math.sqrt(
        (p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total))
    )

    lower = max(0.0, (center - margin) / denom)
    upper = min(1.0, (center + margin) / denom)
    if successes == total:
        upper = 1.0
    if successes == 0:
        lower = 0.0
    return (lower, upper)


def compare_two_proportions(
    k1: int,
    n1: int,
    k2: int,
    n2: int,
    alpha: float = 0.05,
) -> TwoProportionResult:
    if n1 <= 0 or n2 <= 0:
        return TwoProportionResult(
            p1=0.0,
            p2=0.0,
            difference=0.0,
            z_score=0.0,
            p_value=1.0,
            is_significant=False,
            is_degradation=False,
        )

    p1 = k1 / n1
    p2 = k2 / n2
    diff = p1 - p2

    pooled = (k1 + k2) / (n1 + n2)
    if pooled <= 0.0 or pooled >= 1.0:
        return TwoProportionResult(
            p1=p1,
            p2=p2,
            difference=diff,
            z_score=0.0,
            p_value=1.0,
            is_significant=False,
            is_degradation=False,
        )

    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        z = 0.0
    else:
        z = diff / se

    p_val = normal_cdf(z)
    is_deg = diff < 0.0
    is_sig = is_deg and (p_val < alpha)

    return TwoProportionResult(
        p1=p1,
        p2=p2,
        difference=diff,
        z_score=z,
        p_value=p_val,
        is_significant=is_sig,
        is_degradation=is_deg,
    )


def sequential_verdict(
    successes: int,
    completed: int,
    total_runs: int,
    min_pass_rate: float,
    confidence: float = 0.95,
) -> EarlyStoppingDecision:
    if min_pass_rate > 1.0:
        min_pass_rate = min_pass_rate / 100.0

    current_pass_rate = successes / completed if completed > 0 else 0.0
    interval = wilson_score_interval(successes, completed, confidence)

    remaining = total_runs - completed
    max_possible_successes = successes + remaining
    min_possible_successes = successes

    if (max_possible_successes / total_runs) < min_pass_rate:
        return EarlyStoppingDecision(
            should_stop=True,
            verdict="FAIL",
            reason="impossible_to_reach_threshold",
            pass_rate=current_pass_rate,
            interval=interval,
        )

    if (min_possible_successes / total_runs) >= min_pass_rate:
        return EarlyStoppingDecision(
            should_stop=True,
            verdict="PASS",
            reason="guaranteed_threshold_met",
            pass_rate=current_pass_rate,
            interval=interval,
        )

    if completed >= 5:
        if interval[1] < min_pass_rate:
            return EarlyStoppingDecision(
                should_stop=True,
                verdict="FAIL",
                reason="wilson_upper_below_threshold",
                pass_rate=current_pass_rate,
                interval=interval,
            )
        if interval[0] >= min_pass_rate and completed >= max(
            10, total_runs // 2
        ):
            return EarlyStoppingDecision(
                should_stop=True,
                verdict="PASS",
                reason="wilson_lower_above_threshold",
                pass_rate=current_pass_rate,
                interval=interval,
            )

    if completed >= total_runs:
        if interval[0] >= min_pass_rate:
            verdict = "PASS"
        elif interval[1] < min_pass_rate:
            verdict = "FAIL"
        else:
            verdict = "INCONCLUSIVE"
        return EarlyStoppingDecision(
            should_stop=True,
            verdict=verdict,
            reason="all_runs_completed",
            pass_rate=current_pass_rate,
            interval=interval,
        )

    return EarlyStoppingDecision(
        should_stop=False,
        verdict="INCONCLUSIVE",
        reason="in_progress",
        pass_rate=current_pass_rate,
        interval=interval,
    )


def is_flaky(
    pass_rate: float,
    lower: float = 0.05,
    upper: float = 0.95,
) -> bool:
    if pass_rate > 1.0:
        pass_rate = pass_rate / 100.0
    return lower < pass_rate < upper


def flakiness_score(pass_rate: float) -> float:
    if pass_rate > 1.0:
        pass_rate = pass_rate / 100.0
    clamped = max(0.0, min(1.0, pass_rate))
    return max(0.0, 1.0 - 2.0 * abs(clamped - 0.5))
