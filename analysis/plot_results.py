"""Generate plots from experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from var_reasoning.experiment.metrics import (
    compute_accuracy,
    compute_aggregate_metrics,
    find_result_files,
    load_results,
)


def _load_all_conditions(
    benchmark: str, results_dir: str
) -> dict[str, list[dict]]:
    files = find_result_files(results_dir, benchmark)
    condition_results: dict[str, list[dict]] = {}
    for f in files:
        cond_letter = f.stem.split("_")[0]
        data = load_results(f)
        if cond_letter in condition_results:
            condition_results[cond_letter].extend(data)
        else:
            condition_results[cond_letter] = data
    return condition_results


def plot_accuracy_by_condition(
    benchmark: str, results_dir: str = "results", output_dir: str = "figures"
) -> None:
    """Bar chart: accuracy by condition."""
    cond_results = _load_all_conditions(benchmark, results_dir)
    if not cond_results:
        print(f"No results found for {benchmark}")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    labels = []
    accuracies = []
    for cond in ["A", "B", "C", "D"]:
        if cond in cond_results:
            labels.append(f"Condition {cond}")
            accuracies.append(compute_accuracy(cond_results[cond]))

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, accuracies, color=sns.color_palette("muted", len(labels)))
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by Condition — {benchmark}")
    ax.set_ylim(0, 1.0)
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(f"{output_dir}/accuracy_{benchmark}.png", dpi=150)
    plt.close(fig)
    print(f"Saved accuracy_{benchmark}.png")


def plot_cost_per_problem(
    benchmark: str, results_dir: str = "results", output_dir: str = "figures"
) -> None:
    """Bar chart: cost per problem by condition."""
    cond_results = _load_all_conditions(benchmark, results_dir)
    if not cond_results:
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    labels = []
    costs = []
    for cond in ["A", "B", "C", "D"]:
        if cond in cond_results:
            metrics = compute_aggregate_metrics(cond_results[cond])
            labels.append(f"Condition {cond}")
            costs.append(metrics["cost_per_problem"])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, costs, color=sns.color_palette("Set2", len(labels)))
    ax.set_ylabel("Cost per Problem (USD)")
    ax.set_title(f"Cost per Problem — {benchmark}")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/cost_{benchmark}.png", dpi=150)
    plt.close(fig)
    print(f"Saved cost_{benchmark}.png")


def plot_accuracy_vs_cost(
    results_dir: str = "results", output_dir: str = "figures"
) -> None:
    """Scatter plot: accuracy vs cost per problem across all benchmarks."""
    all_files = find_result_files(results_dir)
    if not all_files:
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    points: list[tuple[float, float, str]] = []
    seen: set[str] = set()
    for f in all_files:
        parts = f.stem.split("_")
        cond = parts[0]
        bench = "_".join(parts[1:-2]) if len(parts) > 3 else parts[1]
        key = f"{cond}_{bench}"
        if key in seen:
            continue
        seen.add(key)
        data = load_results(f)
        metrics = compute_aggregate_metrics(data)
        points.append((metrics["cost_per_problem"], metrics["accuracy"], key))

    if not points:
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    xs, ys, labs = zip(*points)
    ax.scatter(xs, ys, s=80)
    for x, y, lab in zip(xs, ys, labs):
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Cost per Problem (USD)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Cost")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/accuracy_vs_cost.png", dpi=150)
    plt.close(fig)
    print("Saved accuracy_vs_cost.png")


def plot_verification_distribution(
    benchmark: str, results_dir: str = "results", output_dir: str = "figures"
) -> None:
    """Stacked bar chart: verification type distribution for Condition C."""
    cond_results = _load_all_conditions(benchmark, results_dir)
    if "C" not in cond_results:
        print("No Condition C results found")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics = compute_aggregate_metrics(cond_results["C"])
    dist = metrics["verification_type_distribution"]
    if not dist:
        return

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    types = list(dist.keys())
    values = list(dist.values())
    ax.bar(types, values, color=sns.color_palette("pastel", len(types)))
    ax.set_ylabel("Proportion")
    ax.set_title(f"Verification Type Distribution — {benchmark} (Condition C)")
    fig.tight_layout()
    fig.savefig(f"{output_dir}/verification_dist_{benchmark}.png", dpi=150)
    plt.close(fig)
    print(f"Saved verification_dist_{benchmark}.png")


def plot_steps_histogram(
    benchmark: str, results_dir: str = "results", output_dir: str = "figures"
) -> None:
    """Histogram: number of steps per problem for Conditions B and C."""
    cond_results = _load_all_conditions(benchmark, results_dir)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    for cond, color in [("B", "steelblue"), ("C", "coral")]:
        if cond not in cond_results:
            continue
        steps = [r.get("num_steps", 0) for r in cond_results[cond]]
        ax.hist(steps, bins=range(0, 27), alpha=0.6, label=f"Condition {cond}", color=color)

    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Count")
    ax.set_title(f"Steps per Problem — {benchmark}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_dir}/steps_hist_{benchmark}.png", dpi=150)
    plt.close(fig)
    print(f"Saved steps_hist_{benchmark}.png")


def plot_all(benchmark: str, results_dir: str = "results", output_dir: str = "figures") -> None:
    plot_accuracy_by_condition(benchmark, results_dir, output_dir)
    plot_cost_per_problem(benchmark, results_dir, output_dir)
    plot_accuracy_vs_cost(results_dir, output_dir)
    plot_verification_distribution(benchmark, results_dir, output_dir)
    plot_steps_histogram(benchmark, results_dir, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate experiment plots")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="figures")
    args = parser.parse_args()
    plot_all(args.benchmark, args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
