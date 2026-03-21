"""Compare results across experimental conditions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from var_reasoning.experiment.conditions import Condition, CONDITION_DESCRIPTIONS
from var_reasoning.experiment.metrics import (
    compute_accuracy_ci,
    compute_aggregate_metrics,
    find_result_files,
    load_results,
)


def _paired_bootstrap_test(
    results_a: list[dict],
    results_b: list[dict],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> float:
    """Paired bootstrap test for accuracy difference. Returns p-value."""
    import random

    rng = random.Random(seed)

    # Build aligned pair by problem_id
    b_by_id = {r["problem_id"]: r for r in results_b}
    pairs = []
    for r in results_a:
        pid = r["problem_id"]
        if pid in b_by_id:
            pairs.append(
                (
                    1 if r.get("is_correct") else 0,
                    1 if b_by_id[pid].get("is_correct") else 0,
                )
            )

    if not pairs:
        return 1.0

    observed_diff = sum(b - a for a, b in pairs) / len(pairs)
    count = 0
    for _ in range(n_bootstrap):
        sample = rng.choices(pairs, k=len(pairs))
        boot_diff = sum(b - a for a, b in sample) / len(sample)
        if boot_diff <= 0:
            count += 1
    return count / n_bootstrap


def compare(benchmark: str, results_dir: str = "results") -> None:
    """Print comparison tables for all conditions on a benchmark."""
    files = find_result_files(results_dir, benchmark)
    if not files:
        print(f"No result files found for benchmark '{benchmark}' in {results_dir}/")
        return

    # Group by condition
    condition_results: dict[str, list[dict]] = {}
    for f in files:
        stem = f.stem
        # Expected format: {condition}_{benchmark}_{timestamp}
        cond_letter = stem.split("_")[0]
        data = load_results(f)
        if cond_letter in condition_results:
            condition_results[cond_letter].extend(data)
        else:
            condition_results[cond_letter] = data

    # Accuracy table
    print(f"\n{'='*70}")
    print(f"  Results for benchmark: {benchmark}")
    print(f"{'='*70}\n")
    print(f"{'Condition':<12} {'N':>5} {'Accuracy':>10} {'95% CI':>18} {'Cost/Problem':>14}")
    print("-" * 65)

    for cond in ["A", "B", "C", "D"]:
        if cond not in condition_results:
            continue
        results = condition_results[cond]
        acc, ci_low, ci_high = compute_accuracy_ci(results)
        metrics = compute_aggregate_metrics(results)
        desc = CONDITION_DESCRIPTIONS.get(Condition(cond), cond)
        print(
            f"{cond:<12} {len(results):>5} {acc:>10.1%} "
            f"[{ci_low:.1%}, {ci_high:.1%}]{metrics['cost_per_problem']:>12.4f} $"
        )

    # Statistical significance
    if "A" in condition_results and "C" in condition_results:
        p_val = _paired_bootstrap_test(
            condition_results["A"], condition_results["C"]
        )
        print(f"\nPaired bootstrap: C vs A  p = {p_val:.4f}")

    if "B" in condition_results and "C" in condition_results:
        p_val = _paired_bootstrap_test(
            condition_results["B"], condition_results["C"]
        )
        print(f"Paired bootstrap: C vs B  p = {p_val:.4f}")

    # Detailed metrics for condition C
    if "C" in condition_results:
        metrics = compute_aggregate_metrics(condition_results["C"])
        print(f"\n--- Condition C Details ---")
        print(f"  Avg backtracks/problem:  {metrics['avg_backtracks_per_problem']}")
        print(f"  Backtrack success rate:  {metrics['backtrack_success_rate']:.1%}")
        print(f"  Cascade rate:            {metrics['cascade_rate']:.1%}")
        print(f"  Unsolvable rate:         {metrics['unsolvable_rate']:.1%}")
        print(f"  Formalization rate:      {metrics['formalization_rate']:.1%}")
        print(f"  Informal skip rate:      {metrics['informal_skip_rate']:.1%}")
        print(f"  Verification distribution: {metrics['verification_type_distribution']}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experimental conditions")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    compare(args.benchmark, args.results_dir)


if __name__ == "__main__":
    main()
