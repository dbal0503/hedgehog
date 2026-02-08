#!/usr/bin/env python3
"""
Main entry point for landscape probe experiments.

Usage:
    # Run sanity check (quick test)
    python run_experiment.py sanity

    # Run full sweep on SST-2
    python run_experiment.py sweep --task sst2

    # Run sweep on multiple tasks
    python run_experiment.py sweep --task sst2 cola mrpc

    # Analyze results
    python run_experiment.py analyze

    # Full pipeline (sanity + sweep + analyze)
    python run_experiment.py full
"""

import os
import sys
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run_sweep import run_sweep, run_sanity_check
from analysis.correlation_analysis import generate_full_report


def check_environment():
    """Check that the environment is properly configured."""
    print("=" * 50)
    print("ENVIRONMENT CHECK")
    print("=" * 50)

    # PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.2f} GB")

        # Recommend batch size based on VRAM
        if vram < 4:
            print("⚠️  Low VRAM detected. Recommended: bert-tiny with batch_size=16")
        elif vram < 6:
            print("✓ VRAM OK for bert-mini with batch_size=16")
        else:
            print("✓ VRAM OK for bert-small or DistilBERT")
    else:
        print("⚠️  CUDA not available. Training will be slow on CPU.")

    # Check optional dependencies
    print("\nOptional dependencies:")
    try:
        import wandb
        print(f"  ✓ wandb {wandb.__version__}")
    except ImportError:
        print("  ✗ wandb (experiment tracking disabled)")

    try:
        import matplotlib
        import seaborn
        print(f"  ✓ matplotlib {matplotlib.__version__}")
        print(f"  ✓ seaborn {seaborn.__version__}")
    except ImportError:
        print("  ✗ matplotlib/seaborn (plotting disabled)")

    try:
        import sklearn
        print(f"  ✓ scikit-learn {sklearn.__version__}")
    except ImportError:
        print("  ✗ scikit-learn (partial correlations disabled)")

    print("=" * 50)
    return torch.cuda.is_available()


def run_sanity(args):
    """Run sanity check to verify probes work."""
    print("\n" + "=" * 50)
    print("SANITY CHECK")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_sanity_check(
        model_name=args.model,
        task_name="sst2",
        batch_size=args.batch_size,
        n_seeds=5,
        device=device,
        output_dir=args.output_dir
    )


def run_sweeps(args):
    """Run hyperparameter sweeps on specified tasks."""
    print("\n" + "=" * 50)
    print("HYPERPARAMETER SWEEP")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tasks = args.task if args.task else ["sst2"]

    for task in tasks:
        print(f"\n--- Running sweep for {task} ---")

        resume_path = None
        if args.resume:
            resume_path = os.path.join(args.output_dir, f"{task}_results.json")
            if not os.path.exists(resume_path):
                resume_path = None

        run_sweep(
            task_name=task,
            model_name=args.model,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            output_dir=args.output_dir,
            device=device,
            use_wandb=args.wandb,
            seed=args.seed,
            resume_from=resume_path
        )


def run_analysis(args):
    """Run correlation analysis on results."""
    print("\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)

    generate_full_report(
        results_dir=args.output_dir,
        output_dir=args.figures_dir,
        tasks=args.task
    )


def run_full_pipeline(args):
    """Run complete pipeline: sanity check → sweeps → analysis."""
    print("\n" + "=" * 50)
    print("FULL PIPELINE")
    print("=" * 50)

    # 1. Environment check
    check_environment()

    # 2. Sanity check
    run_sanity(args)

    # 3. Sweeps on all specified tasks
    run_sweeps(args)

    # 4. Analysis
    run_analysis(args)

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Landscape Probe Correlation Study",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=str, default="prajjwal1/bert-mini",
                        help="HuggingFace model ID")
    common.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (reduce if OOM)")
    common.add_argument("--output-dir", type=str, default="./results",
                        help="Directory for results")
    common.add_argument("--figures-dir", type=str, default="./figures",
                        help="Directory for plots")

    # Sanity check
    sanity_parser = subparsers.add_parser("sanity", parents=[common],
                                          help="Run sanity check")

    # Sweep
    sweep_parser = subparsers.add_parser("sweep", parents=[common],
                                         help="Run hyperparameter sweep")
    sweep_parser.add_argument("--task", type=str, nargs="+", default=["sst2"],
                              help="GLUE tasks to run (sst2, cola, mrpc, qnli)")
    sweep_parser.add_argument("--epochs", type=int, default=3,
                              help="Training epochs per run")
    sweep_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed")
    sweep_parser.add_argument("--wandb", action="store_true",
                              help="Enable wandb logging")
    sweep_parser.add_argument("--resume", action="store_true",
                              help="Resume from existing results")

    # Analysis
    analyze_parser = subparsers.add_parser("analyze", parents=[common],
                                           help="Analyze results")
    analyze_parser.add_argument("--task", type=str, nargs="+", default=None,
                                help="Tasks to analyze (default: all)")

    # Full pipeline
    full_parser = subparsers.add_parser("full", parents=[common],
                                        help="Run full pipeline")
    full_parser.add_argument("--task", type=str, nargs="+", default=["sst2"],
                             help="GLUE tasks to run")
    full_parser.add_argument("--epochs", type=int, default=3,
                             help="Training epochs per run")
    full_parser.add_argument("--seed", type=int, default=42,
                             help="Random seed")
    full_parser.add_argument("--wandb", action="store_true",
                             help="Enable wandb logging")
    full_parser.add_argument("--resume", action="store_true",
                             help="Resume from existing results")

    # Environment check
    env_parser = subparsers.add_parser("env", help="Check environment")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "env":
        check_environment()
    elif args.command == "sanity":
        check_environment()
        run_sanity(args)
    elif args.command == "sweep":
        check_environment()
        run_sweeps(args)
    elif args.command == "analyze":
        run_analysis(args)
    elif args.command == "full":
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
