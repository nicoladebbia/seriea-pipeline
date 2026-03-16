"""Statistical validation utilities for football prediction claims.

Provides rigorous statistical tools to prevent cherry-picked accuracy claims:
- Binomial confidence intervals (Clopper-Pearson exact)
- Minimum sample size enforcement
- Significance tests vs base rate (binomial test)
- Benjamini-Hochberg FDR correction for multiple comparisons
- Subgroup analysis with proper error bars

All accuracy claims in the system should pass through these functions
before being reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum sample sizes for reliable claims at different tiers
MIN_SAMPLES_REPORT = 30       # Minimum to report any metric
MIN_SAMPLES_RELIABLE = 100    # Minimum for "reliable" claims
MIN_SAMPLES_ROBUST = 300      # Minimum for "robust" claims

# Serie A base rates (from full dataset)
BASE_RATE_HOME_WIN = 0.444
BASE_RATE_DRAW = 0.265
BASE_RATE_AWAY_WIN = 0.291
BASE_RATE_RANDOM = 1 / 3


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def binomial_ci(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Clopper-Pearson binomial confidence interval.

    Returns (ci_lo, ci_hi) for the true proportion.
    Uses beta distribution quantiles for exact coverage.
    """
    if trials == 0:
        return (0.0, 1.0)

    from scipy import stats

    ci_lo = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
    ci_hi = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)

    # Edge cases
    if successes == 0:
        ci_lo = 0.0
    if successes == trials:
        ci_hi = 1.0

    return (round(float(ci_lo), 4), round(float(ci_hi), 4))


def accuracy_with_ci(
    correct: int,
    total: int,
    alpha: float = 0.05,
) -> dict:
    """Compute accuracy with exact binomial confidence interval.

    Returns dict with point estimate and CI bounds.
    Includes reliability tier based on sample size.
    """
    if total == 0:
        return {
            "accuracy": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 1.0,
            "n": 0,
            "reliability": "no_data",
        }

    acc = round(correct / total, 4)
    ci_lo, ci_hi = binomial_ci(correct, total, alpha)

    if total >= MIN_SAMPLES_ROBUST:
        reliability = "robust"
    elif total >= MIN_SAMPLES_RELIABLE:
        reliability = "reliable"
    elif total >= MIN_SAMPLES_REPORT:
        reliability = "preliminary"
    else:
        reliability = "insufficient"

    return {
        "accuracy": acc,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_width": round(ci_hi - ci_lo, 4),
        "n": total,
        "reliability": reliability,
    }


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------

def binomial_test(successes: int, trials: int, base_rate: float) -> dict:
    """Two-sided binomial test: is observed rate significantly different from base_rate?

    Returns dict with p-value, effect size, and significance verdict.
    """
    if trials == 0:
        return {
            "observed_rate": 0.0,
            "base_rate": base_rate,
            "effect_size": 0.0,
            "p_value": 1.0,
            "significant_005": False,
            "significant_001": False,
            "n": 0,
        }

    from scipy import stats

    result = stats.binomtest(successes, trials, base_rate, alternative="two-sided")
    p_value = float(result.pvalue)
    observed_rate = successes / trials
    effect_size = round(observed_rate - base_rate, 4)

    return {
        "observed_rate": round(observed_rate, 4),
        "base_rate": base_rate,
        "effect_size": effect_size,
        "p_value": round(p_value, 6),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "n": trials,
    }


def lift_with_significance(
    successes: int,
    trials: int,
    base_rate: float,
    alpha: float = 0.05,
) -> dict:
    """Compute lift over base rate with confidence interval and significance test.

    This is the proper way to report factor effectiveness:
    "Factor X: 62.3% accuracy (95% CI: 55.1%-69.0%, p=0.003 vs 44.4% base)"
    instead of just "Factor X: 62.3% (+17.9% lift)".
    """
    acc_info = accuracy_with_ci(successes, trials, alpha)
    sig_info = binomial_test(successes, trials, base_rate)

    lift = round(acc_info["accuracy"] - base_rate, 4) if trials > 0 else 0.0
    lift_ci_lo = round(acc_info["ci_lo"] - base_rate, 4) if trials > 0 else 0.0
    lift_ci_hi = round(acc_info["ci_hi"] - base_rate, 4) if trials > 0 else 0.0

    return {
        **acc_info,
        "base_rate": base_rate,
        "lift": lift,
        "lift_ci_lo": lift_ci_lo,
        "lift_ci_hi": lift_ci_hi,
        "p_value": sig_info["p_value"],
        "significant": sig_info["significant_005"],
    }


# ---------------------------------------------------------------------------
# Multiple comparisons correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction for multiple comparisons.

    Given a list of p-values from multiple tests, returns a list of bools
    indicating which tests remain significant after FDR correction.

    With 200 tests at alpha=0.05, we expect ~10 false positives by chance.
    BH controls the false discovery rate to alpha.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values with original indices
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])

    # Step-up: find largest k where p_(k) <= alpha * k / m, then reject all i <= k
    max_significant_rank = 0
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        if p <= alpha * rank / m:
            max_significant_rank = rank

    significant = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed, 1):
        if rank <= max_significant_rank:
            significant[orig_idx] = True

    return significant


def validate_multiple_claims(
    claims: list[dict],
    alpha: float = 0.05,
) -> list[dict]:
    """Validate a batch of accuracy claims with FDR correction.

    Each claim dict must have: successes, trials, base_rate, label.
    Returns enriched claims with CIs, p-values, and FDR-corrected significance.
    """
    if not claims:
        return []

    # Compute individual tests
    results = []
    p_values = []
    for claim in claims:
        info = lift_with_significance(
            claim["successes"],
            claim["trials"],
            claim["base_rate"],
            alpha,
        )
        info["label"] = claim.get("label", "")
        results.append(info)
        p_values.append(info["p_value"])

    # Apply FDR correction
    fdr_significant = benjamini_hochberg(p_values, alpha)
    for result, fdr_sig in zip(results, fdr_significant):
        result["fdr_significant"] = fdr_sig

    return results


# ---------------------------------------------------------------------------
# Subgroup analysis
# ---------------------------------------------------------------------------

@dataclass
class SubgroupResult:
    """Result of a subgroup accuracy analysis."""
    group_name: str
    correct: int
    total: int
    accuracy: float
    ci_lo: float
    ci_hi: float
    reliability: str
    p_value: float
    significant: bool


def subgroup_analysis(
    predictions: list[dict],
    group_fn,
    base_rate: float = BASE_RATE_HOME_WIN,
    min_samples: int = MIN_SAMPLES_REPORT,
) -> list[SubgroupResult]:
    """Analyze prediction accuracy across subgroups with proper statistics.

    Args:
        predictions: list of dicts with at least 'correct' (bool) and data for grouping
        group_fn: callable(pred) -> group_name (str or None to exclude)
        base_rate: null hypothesis rate
        min_samples: minimum group size to include

    Returns:
        List of SubgroupResult, sorted by accuracy descending.
        Groups with fewer than min_samples are excluded.
    """
    groups: dict[str, dict] = {}

    for pred in predictions:
        group = group_fn(pred)
        if group is None:
            continue
        if group not in groups:
            groups[group] = {"correct": 0, "total": 0}
        groups[group]["total"] += 1
        if pred.get("correct", False):
            groups[group]["correct"] += 1

    results = []
    for name, data in groups.items():
        if data["total"] < min_samples:
            continue

        info = lift_with_significance(data["correct"], data["total"], base_rate)
        results.append(SubgroupResult(
            group_name=name,
            correct=data["correct"],
            total=data["total"],
            accuracy=info["accuracy"],
            ci_lo=info["ci_lo"],
            ci_hi=info["ci_hi"],
            reliability=info["reliability"],
            p_value=info["p_value"],
            significant=info["significant"],
        ))

    results.sort(key=lambda x: x.accuracy, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Reproducibility metadata
# ---------------------------------------------------------------------------

def get_reproducibility_metadata(seed: int = 42) -> dict:
    """Generate reproducibility metadata for result files.

    Includes git hash, timestamp, python version, key package versions,
    and random seed used.
    """
    import sys
    from datetime import datetime

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "random_seed": seed,
        "python_version": sys.version.split()[0],
    }

    # Git commit hash
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            metadata["git_hash"] = result.stdout.strip()[:12]
    except Exception as e:
        log.debug(f"Failed to retrieve git hash for metadata: {e}")

    # Key package versions
    for pkg in ["numpy", "pandas", "scikit-learn", "scipy"]:
        try:
            import importlib
            mod = importlib.import_module(pkg.replace("-", "_").replace("scikit_learn", "sklearn"))
            metadata[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass

    return metadata


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_accuracy_claim(
    label: str,
    correct: int,
    total: int,
    base_rate: float,
    alpha: float = 0.05,
) -> str:
    """Format a single accuracy claim with proper statistical context.

    Returns a string like:
    "Factor X: 62.3% (95% CI: 55.1%-69.0%, n=150, p=0.003 vs 44.4% base) [reliable]"
    or:
    "Factor Y: 85.7% (95% CI: 42.1%-99.6%, n=7) [insufficient — below minimum sample size]"
    """
    if total == 0:
        return f"{label}: no data"

    info = lift_with_significance(correct, total, base_rate, alpha)

    if total < MIN_SAMPLES_REPORT:
        return (
            f"{label}: {info['accuracy']:.1%} "
            f"(95% CI: {info['ci_lo']:.1%}-{info['ci_hi']:.1%}, n={total}) "
            f"[insufficient - below min sample size of {MIN_SAMPLES_REPORT}]"
        )

    sig_str = f"p={info['p_value']:.4f}" if info['p_value'] >= 0.0001 else "p<0.0001"
    sig_marker = "*" if info["significant"] else ""

    return (
        f"{label}: {info['accuracy']:.1%} "
        f"(95% CI: {info['ci_lo']:.1%}-{info['ci_hi']:.1%}, "
        f"n={total}, {sig_str} vs {base_rate:.1%} base){sig_marker} "
        f"[{info['reliability']}]"
    )
