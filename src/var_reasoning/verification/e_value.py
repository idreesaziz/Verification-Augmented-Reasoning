"""E-value based statistical testing for claim verification.

Implements sequential hypothesis testing using e-values, which provide
anytime-valid inference: you can stop sampling at any time and the
false-positive guarantee still holds.  E-values compose multiplicatively
across steps—no Bonferroni correction needed.

References:
    Grünwald, de Heide & Koolen (2024). "Safe Testing."
    Vovk & Wang (2021). "E-values: Calibration, combination and applications."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Verdict(str, Enum):
    """Outcome of an e-value hypothesis test."""

    REJECT = "reject"  # Strong evidence claim is wrong
    ACCEPT = "accept"  # No evidence against claim after max samples
    SUSPECT = "suspect"  # Moderate evidence—flag for extra scrutiny


@dataclass(frozen=True)
class EValueResult:
    """Result of comparing an analytical claim against empirical evidence."""

    verdict: Verdict
    e_value: float  # Evidence against H₀ (higher = more evidence)
    claimed_value: float
    empirical_value: float
    sample_size: int
    confidence_interval: tuple[float, float]  # 99% CI of empirical estimate
    detail: str = ""


# ── Thresholds ───────────────────────────────────────────────────────

E_REJECT = 1000.0  # e > 1000  →  reject H₀   (α ≈ 0.001)
E_SUSPECT = 20.0  # e > 20    →  suspect       (α ≈ 0.05)


# ── Proportion test (Bernoulli claims) ───────────────────────────────

def proportion_e_value(successes: int, trials: int, p0: float) -> float:
    """Likelihood-ratio e-value for H₀: p = p₀.

    Uses the MLE (p̂ = k/n) as the alternative:
        E = (p̂/p₀)^k  ·  ((1−p̂)/(1−p₀))^(n−k)

    Valid as an e-value when n ≥ 1 and p₀ ∈ (0, 1).
    """
    if trials <= 0:
        return 1.0  # no evidence
    p0 = max(1e-12, min(1 - 1e-12, p0))
    p_hat = successes / trials
    p_hat = max(1e-12, min(1 - 1e-12, p_hat))

    log_e = (
        successes * math.log(p_hat / p0)
        + (trials - successes) * math.log((1 - p_hat) / (1 - p0))
    )
    # Clamp to avoid overflow
    return math.exp(min(log_e, 700))


def _wilson_ci(successes: int, trials: int, z: float = 2.576) -> tuple[float, float]:
    """Wilson score 99% confidence interval for a proportion."""
    if trials == 0:
        return (0.0, 1.0)
    n = trials
    p_hat = successes / n
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ── Mean test (continuous expected-value claims) ─────────────────────

def mean_e_value(
    sample_mean: float,
    sample_std: float,
    sample_size: int,
    mu0: float,
) -> float:
    """Gaussian likelihood-ratio e-value for H₀: μ = μ₀.

    Uses the CLT approximation for large n:
        z = (x̄ − μ₀) / (s / √n)
        E = exp(z² / 2)

    Conservative: assumes Gaussian tails, which underestimates evidence
    for heavy-tailed data.
    """
    if sample_size <= 1 or sample_std <= 0:
        return 1.0
    se = sample_std / math.sqrt(sample_size)
    z = (sample_mean - mu0) / se
    return math.exp(min(z * z / 2, 700))


def _normal_ci(
    sample_mean: float, sample_std: float, sample_size: int, z: float = 2.576
) -> tuple[float, float]:
    """99% confidence interval for a mean (normal approximation)."""
    if sample_size <= 1 or sample_std <= 0:
        return (sample_mean, sample_mean)
    se = sample_std / math.sqrt(sample_size)
    return (sample_mean - z * se, sample_mean + z * se)


# ── Exact test (deterministic count claims) ──────────────────────────

def exact_e_value(claimed: float, observed: float) -> float:
    """For deterministic claims: either exact match or infinite evidence."""
    if math.isclose(claimed, observed, rel_tol=1e-9, abs_tol=1e-9):
        return 1.0  # no evidence against
    return float("inf")  # definitive rejection


# ── Unified test interface ───────────────────────────────────────────

@dataclass(frozen=True)
class SimulationStats:
    """Summary statistics extracted from a simulation run."""

    sample_mean: float
    sample_size: int
    sample_std: float = 0.0
    # For Bernoulli: successes out of sample_size trials
    successes: int | None = None

    @property
    def is_proportion(self) -> bool:
        return self.successes is not None


def test_claim(claimed_value: float, stats: SimulationStats) -> EValueResult:
    """Test whether a claimed value is consistent with simulation evidence.

    Automatically selects the right test:
      - Proportion test for Bernoulli data (successes/trials)
      - Mean test for continuous data (sample mean ± std)
      - Exact test for deterministic results (std == 0)
    """
    # Deterministic case: simulation always produces the same value
    if stats.sample_std == 0 and stats.successes is None:
        e = exact_e_value(claimed_value, stats.sample_mean)
        ci = (stats.sample_mean, stats.sample_mean)
        verdict = Verdict.REJECT if e > E_REJECT else Verdict.ACCEPT
        return EValueResult(
            verdict=verdict,
            e_value=e,
            claimed_value=claimed_value,
            empirical_value=stats.sample_mean,
            sample_size=stats.sample_size,
            confidence_interval=ci,
            detail="exact_match" if e == 1.0 else "exact_mismatch",
        )

    # Proportion test
    if stats.is_proportion:
        e = proportion_e_value(stats.successes, stats.sample_size, claimed_value)
        ci = _wilson_ci(stats.successes, stats.sample_size)
        empirical = stats.successes / max(1, stats.sample_size)
    else:
        # Continuous mean test
        e = mean_e_value(
            stats.sample_mean, stats.sample_std, stats.sample_size, claimed_value
        )
        ci = _normal_ci(stats.sample_mean, stats.sample_std, stats.sample_size)
        empirical = stats.sample_mean

    if e > E_REJECT:
        verdict = Verdict.REJECT
    elif e > E_SUSPECT:
        verdict = Verdict.SUSPECT
    else:
        verdict = Verdict.ACCEPT

    return EValueResult(
        verdict=verdict,
        e_value=e,
        claimed_value=claimed_value,
        empirical_value=empirical,
        sample_size=stats.sample_size,
        confidence_interval=ci,
        detail=f"e={e:.2f}, CI=({ci[0]:.6f}, {ci[1]:.6f})",
    )
