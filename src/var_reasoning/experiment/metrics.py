"""Metrics computation from experiment results."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_results(filepath: str | Path) -> list[dict]:
    results = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def compute_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get("is_correct", False))
    return correct / len(results)


def compute_accuracy_ci(
    results: list[dict], confidence: float = 0.95
) -> tuple[float, float, float]:
    """Compute accuracy with Wilson score confidence interval."""
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0
    p = compute_accuracy(results)
    z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return p, max(0.0, centre - spread), min(1.0, centre + spread)


def compute_accuracy_by_difficulty(
    results: list[dict],
) -> dict[str, float]:
    by_difficulty: dict[str, list[dict]] = {}
    for r in results:
        diff = r.get("difficulty") or "unknown"
        by_difficulty.setdefault(diff, []).append(r)
    return {k: compute_accuracy(v) for k, v in sorted(by_difficulty.items())}


def compute_aggregate_metrics(results: list[dict]) -> dict:
    """Compute all aggregate metrics for a set of results."""
    accuracy, ci_low, ci_high = compute_accuracy_ci(results)
    total_tokens = sum(
        r.get("total_input_tokens", 0) + r.get("total_output_tokens", 0)
        for r in results
    )
    total_cost = sum(r.get("cost_usd", 0.0) for r in results)
    n = len(results) or 1

    total_backtracks = sum(r.get("total_backtracks", 0) for r in results)
    problems_with_backtracks = sum(
        1 for r in results if r.get("total_backtracks", 0) > 0
    )
    backtrack_successes = sum(
        1
        for r in results
        if r.get("total_backtracks", 0) > 0 and r.get("is_correct", False)
    )

    # Verification metrics
    vtype_counts: dict[str, int] = {}
    total_informal = 0
    total_verifications = 0
    for r in results:
        for vtype, count in r.get("verification_type_counts", {}).items():
            vtype_counts[vtype] = vtype_counts.get(vtype, 0) + count
            total_verifications += count
        total_informal += r.get("informal_skip_count", 0)

    unsolvable = sum(
        1 for r in results if r.get("predicted_answer") == "UNSOLVABLE"
    )

    return {
        "num_problems": len(results),
        "accuracy": round(accuracy, 4),
        "accuracy_ci_low": round(ci_low, 4),
        "accuracy_ci_high": round(ci_high, 4),
        "accuracy_by_difficulty": compute_accuracy_by_difficulty(results),
        "avg_backtracks_per_problem": round(total_backtracks / n, 2),
        "backtrack_success_rate": (
            round(backtrack_successes / problems_with_backtracks, 4)
            if problems_with_backtracks > 0
            else 0.0
        ),
        "cascade_rate": round(
            sum(r.get("total_backtracks", 0) > 3 for r in results) / n, 4
        ),
        "unsolvable_rate": round(unsolvable / n, 4),
        "verification_type_distribution": {
            k: round(v / total_verifications, 4) if total_verifications > 0 else 0
            for k, v in vtype_counts.items()
        },
        "informal_skip_rate": (
            round(total_informal / total_verifications, 4)
            if total_verifications > 0
            else 0.0
        ),
        "formalization_rate": (
            round(
                (total_verifications - total_informal) / total_verifications, 4
            )
            if total_verifications > 0
            else 0.0
        ),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "cost_per_problem": round(total_cost / n, 6),
        "cost_normalized_accuracy": (
            round(accuracy / (total_cost / n), 4)
            if total_cost > 0
            else 0.0
        ),
    }


def find_result_files(
    results_dir: str | Path, benchmark: str | None = None
) -> list[Path]:
    """Find all JSONL result files, optionally filtered by benchmark."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    files = sorted(results_path.glob("*.jsonl"))
    if benchmark:
        files = [f for f in files if benchmark in f.stem]
    return files
