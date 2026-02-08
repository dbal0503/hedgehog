"""
Run hyperparameter sweep with probe measurements.
Designed for local GPU (RTX 3050 4GB) with checkpointing.
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Dict, Any
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probes.landscape_probes import LandscapeProbes


def setup_wandb(project_name: str = "landscape-probes", run_name: Optional[str] = None):
    """Initialize wandb for experiment tracking."""
    try:
        import wandb
        wandb.init(project=project_name, name=run_name)
        return True
    except ImportError:
        print("wandb not installed, skipping experiment tracking")
        return False


def load_glue_task(task_name: str, tokenizer, max_length: int = 128):
    """Load and tokenize a GLUE task."""
    dataset = load_dataset("glue", task_name)

    # Task-specific preprocessing
    task_to_keys = {
        "sst2": ("sentence", None),
        "cola": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    if task_name not in task_to_keys:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(task_to_keys.keys())}")

    sentence1_key, sentence2_key = task_to_keys[task_name]

    def tokenize(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized


def train_one_config(
    model,
    train_loader,
    val_loader,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    num_epochs: int = 3,
    device: str = "cuda",
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Train model with given hyperparameters.

    Returns dict with:
        - val_accuracy: final validation accuracy
        - final_train_loss: training loss at end
    """
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    final_train_loss = 0.0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        final_train_loss = epoch_loss / num_batches

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        'val_accuracy': correct / total,
        'final_train_loss': final_train_loss
    }


def run_sweep(
    task_name: str = "sst2",
    model_name: str = "prajjwal1/bert-mini",
    batch_size: int = 16,
    num_epochs: int = 3,
    output_dir: str = "./results",
    device: str = "cuda",
    learning_rates: Optional[List[float]] = None,
    warmup_ratios: Optional[List[float]] = None,
    weight_decays: Optional[List[float]] = None,
    use_wandb: bool = False,
    seed: int = 42,
    resume_from: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Run full hyperparameter sweep with probe measurements.

    Args:
        task_name: GLUE task name (sst2, cola, mrpc, qnli)
        model_name: HuggingFace model ID
        batch_size: Training batch size (reduce if OOM)
        num_epochs: Number of training epochs
        output_dir: Directory to save results
        device: Device to use (cuda/cpu)
        learning_rates: List of learning rates to try
        warmup_ratios: List of warmup ratios to try
        weight_decays: List of weight decay values to try
        use_wandb: Whether to log to wandb
        seed: Random seed for reproducibility
        resume_from: Path to existing results file to resume from

    Returns:
        List of result dictionaries
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Default hyperparameter grid
    if learning_rates is None:
        learning_rates = [1e-5, 2e-5, 5e-5, 1e-4]
    if warmup_ratios is None:
        warmup_ratios = [0.0, 0.1]
    if weight_decays is None:
        weight_decays = [0.0, 0.01]

    # Setup wandb if requested
    if use_wandb:
        setup_wandb(project_name="landscape-probes", run_name=f"{task_name}-sweep")

    # Load tokenizer and data
    print(f"Loading tokenizer and dataset for {task_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_glue_task(task_name, tokenizer)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if device == "cuda" else False
    )
    val_loader = DataLoader(
        dataset["validation"],
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )

    # Determine number of labels
    num_labels = len(set(dataset["train"]["labels"]))
    print(f"Task {task_name}: {num_labels} labels, {len(dataset['train'])} train examples")

    # Load existing results if resuming
    results = []
    completed_configs = set()

    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from {resume_from}")
        with open(resume_from) as f:
            results = json.load(f)
        for r in results:
            config_key = (r['lr'], r['warmup_ratio'], r['weight_decay'])
            completed_configs.add(config_key)
        print(f"Found {len(completed_configs)} completed configurations")

    # Calculate total runs
    total_runs = len(learning_rates) * len(warmup_ratios) * len(weight_decays)
    current_run = 0

    for lr in learning_rates:
        for warmup in warmup_ratios:
            for wd in weight_decays:
                current_run += 1
                config_key = (lr, warmup, wd)

                if config_key in completed_configs:
                    print(f"[{current_run}/{total_runs}] Skipping completed config: lr={lr}, warmup={warmup}, wd={wd}")
                    continue

                print(f"\n[{current_run}/{total_runs}] Config: lr={lr}, warmup={warmup}, wd={wd}")

                # Fresh model for each config
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=num_labels, use_safetensors=True
                ).to(device)

                criterion = torch.nn.CrossEntropyLoss()

                # Measure probes at initialization
                print("  Computing probes at initialization...")
                prober = LandscapeProbes(model, criterion, device)
                try:
                    probes = prober.compute_all_probes(train_loader, n_batches=5, n_hutchinson_samples=50)
                    print(f"  Probes: grad_norm={probes['gradient_norm']:.4f}, "
                          f"sam_sharpness={probes['sam_sharpness']:.4f}, "
                          f"trace={probes['hutchinson_trace']:.4f}")
                except Exception as e:
                    print(f"  Warning: Probe computation failed: {e}")
                    probes = {
                        'gradient_norm': np.nan,
                        'gradient_variance': np.nan,
                        'sam_sharpness': np.nan,
                        'hutchinson_trace': np.nan,
                        'top_eigenvalue': np.nan,
                        'loss_at_init': np.nan
                    }

                # Train
                print("  Training...")
                try:
                    train_results = train_one_config(
                        model, train_loader, val_loader,
                        lr=lr, warmup_ratio=warmup, weight_decay=wd,
                        num_epochs=num_epochs, device=device
                    )
                    val_acc = train_results['val_accuracy']
                    final_loss = train_results['final_train_loss']
                    print(f"  Validation accuracy: {val_acc:.4f}")
                except Exception as e:
                    print(f"  Training failed: {e}")
                    val_acc = np.nan
                    final_loss = np.nan

                # Clear GPU memory
                del model
                torch.cuda.empty_cache()

                # Log result
                result = {
                    "task": task_name,
                    "model": model_name,
                    "lr": lr,
                    "warmup_ratio": warmup,
                    "weight_decay": wd,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "val_accuracy": val_acc,
                    "final_train_loss": final_loss,
                    "seed": seed,
                    **probes
                }
                results.append(result)

                # Save incrementally (for crash recovery)
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/{task_name}_seed{seed}_results.json"
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Results saved to {output_path}")

                # Log to wandb
                if use_wandb:
                    try:
                        import wandb
                        wandb.log(result)
                    except:
                        pass

    print(f"\nCompleted {len(results)} runs for {task_name}")
    return results


def run_sanity_check(
    model_name: str = "prajjwal1/bert-tiny",
    task_name: str = "sst2",
    batch_size: int = 32,
    n_seeds: int = 5,
    device: str = "cuda",
    output_dir: str = "./results"
) -> Dict[str, Any]:
    """
    Run sanity check: verify probes vary across random seeds.

    This is a quick test to ensure probes are capturing meaningful
    variation before running the full sweep.
    """
    print(f"Running sanity check with {n_seeds} seeds...")
    print(f"Model: {model_name}, Task: {task_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_glue_task(task_name, tokenizer)

    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    num_labels = len(set(dataset["train"]["labels"]))

    all_probes = []

    for seed in range(n_seeds):
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, use_safetensors=True
        ).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        prober = LandscapeProbes(model, criterion, device)

        probes = prober.compute_all_probes(train_loader, n_batches=5)
        all_probes.append(probes)

        print(f"    grad_norm={probes['gradient_norm']:.4f}, "
              f"sam_sharpness={probes['sam_sharpness']:.4f}, "
              f"trace={probes['hutchinson_trace']:.4f}")

        del model
        torch.cuda.empty_cache()

    # Compute coefficient of variation for each probe
    print("\n=== Sanity Check Results ===")
    results = {}

    for probe_name in all_probes[0].keys():
        values = [p[probe_name] for p in all_probes]
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = (std_val / abs(mean_val) * 100) if abs(mean_val) > 1e-10 else 0

        results[probe_name] = {
            'mean': mean_val,
            'std': std_val,
            'cv_percent': cv,
            'values': values
        }

        print(f"{probe_name}:")
        print(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}, CV: {cv:.2f}%")

    # Save sanity check results
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/sanity_check_{task_name}.json"

    # Convert numpy types for JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {
            'mean': float(v['mean']),
            'std': float(v['std']),
            'cv_percent': float(v['cv_percent']),
            'values': [float(x) for x in v['values']]
        }

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSanity check results saved to {output_path}")

    # Check if any probe has CV < 5%
    low_variance_probes = [k for k, v in results.items() if v['cv_percent'] < 5]
    if low_variance_probes:
        print(f"\n⚠️  Warning: Low variance probes (CV < 5%): {low_variance_probes}")
        print("   These probes may not capture meaningful differences.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run landscape probe experiments")
    parser.add_argument("--task", type=str, default="sst2",
                        help="GLUE task name (sst2, cola, mrpc, qnli)")
    parser.add_argument("--model", type=str, default="prajjwal1/bert-mini",
                        help="HuggingFace model ID")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (reduce if OOM)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use-wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--sanity-check", action="store_true",
                        help="Run sanity check instead of full sweep")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to results file to resume from")

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    if args.sanity_check:
        run_sanity_check(
            model_name=args.model,
            task_name=args.task,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir
        )
    else:
        run_sweep(
            task_name=args.task,
            model_name=args.model,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            output_dir=args.output_dir,
            device=args.device,
            use_wandb=args.use_wandb,
            seed=args.seed,
            resume_from=args.resume
        )


if __name__ == "__main__":
    main()
