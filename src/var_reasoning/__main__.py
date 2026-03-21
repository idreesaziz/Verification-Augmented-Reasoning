"""CLI entry point for the VAR system."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv


def _get_loader(benchmark: str):
    """Get the appropriate benchmark loader."""
    if benchmark == "gsm8k":
        from var_reasoning.benchmarks.gsm8k import GSM8KLoader
        return GSM8KLoader()
    elif benchmark == "math":
        from var_reasoning.benchmarks.math_bench import MATHLoader
        return MATHLoader()
    elif benchmark == "folio":
        from var_reasoning.benchmarks.folio import FOLIOLoader
        return FOLIOLoader()
    elif benchmark == "humaneval":
        from var_reasoning.benchmarks.humaneval import HumanEvalLoader
        from var_reasoning.sandbox.executor import CodeExecutor
        return HumanEvalLoader(executor=CodeExecutor())
    else:
        print(f"Unknown benchmark: {benchmark}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Run an experimental condition on a benchmark."""
    from var_reasoning.experiment.conditions import Condition
    from var_reasoning.experiment.runner import ExperimentRunner

    condition = Condition(args.condition)
    loader = _get_loader(args.benchmark)

    runner = ExperimentRunner(
        condition=condition,
        benchmark_loader=loader,
        num_problems=args.num_problems,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    resume_path = Path(args.resume) if args.resume else None
    output_path = runner.run(args.benchmark, resume_path=resume_path)
    print(f"Results saved to: {output_path}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze and compare results across conditions."""
    from analysis.compare_conditions import compare
    compare(args.benchmark, args.results_dir)


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots from results."""
    from analysis.plot_results import plot_all
    plot_all(args.benchmark, args.results_dir, args.output_dir)


def main() -> None:
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        prog="var_reasoning",
        description="Verification-Augmented Reasoning System",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run an experiment")
    run_parser.add_argument(
        "--condition",
        required=True,
        choices=["A", "B", "C", "D"],
        help="Experimental condition",
    )
    run_parser.add_argument(
        "--benchmark",
        required=True,
        choices=["gsm8k", "math", "folio", "humaneval"],
        help="Benchmark dataset",
    )
    run_parser.add_argument(
        "--num-problems",
        type=int,
        default=200,
        help="Number of problems to run (default: 200)",
    )
    run_parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    run_parser.add_argument(
        "--resume",
        default=None,
        help="Path to existing result file to resume from",
    )
    run_parser.set_defaults(func=cmd_run)

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Compare results across conditions"
    )
    analyze_parser.add_argument("--benchmark", required=True)
    analyze_parser.add_argument("--results-dir", default="results")
    analyze_parser.set_defaults(func=cmd_analyze)

    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Generate plots")
    plot_parser.add_argument("--benchmark", required=True)
    plot_parser.add_argument("--results-dir", default="results")
    plot_parser.add_argument("--output-dir", default="figures")
    plot_parser.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
