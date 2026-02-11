"""
Pretrain Checkpoint Landscape Probe Experiment
===============================================

Tests: Do landscape probes measured during pretraining (under MLM loss)
predict which fine-tuning hyperparameters work best?

Uses MultiBERTs intermediate checkpoints (google/multiberts-seed_X-step_Nk)
as a source of pretrained models at different stages of pretraining.

Design:
- 4 pretraining stages x 5 seeds = 20 pretrained models
- Each model: measure 6 probes under MLM loss
- Each model: fine-tune on SST-2 with 16 HP configs (or custom grid)
- Correlate pretrain probes with fine-tuning outcomes

Usage:
    # Run experiment
    python pretrain_probe_sweep.py run --device cuda

    # Run with fewer checkpoints (faster)
    python pretrain_probe_sweep.py run --steps step_600k step_2000k --seeds 0 1 2

    # Resume interrupted experiment
    python pretrain_probe_sweep.py run --resume ./results_pretrain/pretrain_probe_results.json

    # Analyze results
    python pretrain_probe_sweep.py analyze
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
)
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional, List, Dict, Any
import argparse
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probes.landscape_probes import LandscapeProbes


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_STEPS = ["step_200k", "step_600k", "step_1000k", "step_2000k"]
SEEDS = [0, 1, 2, 3, 4]

DEFAULT_LRS = [1e-5, 2e-5, 5e-5, 1e-4]
DEFAULT_WARMUPS = [0.0, 0.1]
DEFAULT_WEIGHT_DECAYS = [0.0, 0.01]


# ============================================================
# MLM Criterion and Data
# ============================================================

class MLMCriterion(nn.Module):
    """CrossEntropyLoss wrapper that reshapes MLM logits/labels.

    AutoModelForMaskedLM outputs logits of shape (B, L, V).
    Labels have -100 for non-masked tokens (ignored by CrossEntropyLoss).
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()  # ignore_index=-100 by default

    def forward(self, logits, labels):
        return self.ce(logits.view(-1, logits.size(-1)), labels.view(-1))


def load_mlm_data(
    tokenizer,
    max_length: int = 128,
    n_samples: int = 1000,
    mlm_probability: float = 0.15,
    batch_size: int = 16,
):
    """Load WikiText-2 and prepare MLM DataLoader.

    Uses HuggingFace's DataCollatorForLanguageModeling to handle
    random token masking at collation time.
    """
    print("  Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Filter empty lines
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_special_tokens_mask=True,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Use a subset for probe computation (probes only need a few batches)
    if len(tokenized) > n_samples:
        tokenized = tokenized.select(range(n_samples))

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,
    )

    print(f"  MLM data ready: {len(tokenized)} samples, {len(dataloader)} batches")
    return dataloader


def load_finetune_data(tokenizer, task_name: str = "sst2", max_length: int = 128, batch_size: int = 16):
    """Load and tokenize a GLUE task for fine-tuning."""
    dataset = load_dataset("glue", task_name)

    task_to_keys = {
        "sst2": ("sentence", None),
        "cola": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
    }

    if task_name not in task_to_keys:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(task_to_keys.keys())}")

    s1_key, s2_key = task_to_keys[task_name]

    def tokenize_fn(examples):
        if s2_key is None:
            return tokenizer(
                examples[s1_key], truncation=True,
                max_length=max_length, padding="max_length",
            )
        return tokenizer(
            examples[s1_key], examples[s2_key],
            truncation=True, max_length=max_length, padding="max_length",
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    num_labels = len(set(tokenized["train"]["labels"]))

    train_loader = DataLoader(
        tokenized["train"], batch_size=batch_size, shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        tokenized["validation"], batch_size=batch_size, num_workers=0,
    )

    print(f"  {task_name}: {num_labels} labels, {len(tokenized['train'])} train, "
          f"{len(tokenized['validation'])} val")

    return train_loader, val_loader, num_labels


# ============================================================
# Pretrain Probe Measurement
# ============================================================

def measure_pretrain_probes(
    checkpoint_name: str,
    mlm_dataloader: DataLoader,
    device: str = "cuda",
    n_batches: int = 5,
    n_hutchinson_samples: int = 50,
) -> Dict[str, float]:
    """Measure landscape probes under MLM loss for a pretrained checkpoint."""
    print(f"  Loading MLM model: {checkpoint_name}")
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_name).to(device)
    model.eval()

    criterion = MLMCriterion()
    prober = LandscapeProbes(model, criterion, device)

    print(f"  Computing probes (n_hutchinson={n_hutchinson_samples})...")
    t0 = time.time()
    probes = prober.compute_all_probes(
        mlm_dataloader,
        n_batches=n_batches,
        n_hutchinson_samples=n_hutchinson_samples,
    )
    elapsed = time.time() - t0
    print(f"  Probes computed in {elapsed:.1f}s")

    del model
    torch.cuda.empty_cache()

    return probes


# ============================================================
# Fine-tuning
# ============================================================

def finetune_checkpoint(
    checkpoint_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_labels: int,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    num_epochs: int = 1,
    device: str = "cuda",
    max_grad_norm: float = 1.0,
) -> Dict[str, float]:
    """Fine-tune a single checkpoint with given HP config."""
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_name, num_labels=num_labels,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_train_loss = 0.0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{num_epochs}", leave=False)
        for batch in pbar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        final_train_loss = epoch_loss / n_batches

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

    del model
    torch.cuda.empty_cache()

    return {
        'val_accuracy': correct / total,
        'final_train_loss': final_train_loss,
    }


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(
    checkpoint_steps: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    task_name: str = "sst2",
    batch_size: int = 16,
    num_epochs: int = 1,
    learning_rates: Optional[List[float]] = None,
    warmup_ratios: Optional[List[float]] = None,
    weight_decays: Optional[List[float]] = None,
    device: str = "cuda",
    output_dir: str = "./results_pretrain",
    resume_from: Optional[str] = None,
    n_hutchinson_samples: int = 50,
):
    """Run the full pretrain probe -> fine-tuning experiment."""
    if checkpoint_steps is None:
        checkpoint_steps = CHECKPOINT_STEPS
    if seeds is None:
        seeds = SEEDS
    if learning_rates is None:
        learning_rates = DEFAULT_LRS
    if warmup_ratios is None:
        warmup_ratios = DEFAULT_WARMUPS
    if weight_decays is None:
        weight_decays = DEFAULT_WEIGHT_DECAYS

    os.makedirs(output_dir, exist_ok=True)

    n_hp_configs = len(learning_rates) * len(warmup_ratios) * len(weight_decays)
    n_checkpoints = len(seeds) * len(checkpoint_steps)
    total_runs = n_checkpoints * n_hp_configs

    print("=" * 60)
    print("PRETRAIN CHECKPOINT LANDSCAPE PROBE EXPERIMENT")
    print("=" * 60)
    print(f"Checkpoints: {checkpoint_steps}")
    print(f"Seeds: {seeds}")
    print(f"HP configs: {n_hp_configs} ({len(learning_rates)} LR x {len(warmup_ratios)} warmup x {len(weight_decays)} WD)")
    print(f"Total fine-tuning runs: {total_runs}")
    print(f"Task: {task_name}, Epochs: {num_epochs}, Batch size: {batch_size}")
    print()

    # Load tokenizer (shared across all MultiBERTs checkpoints)
    print("Loading tokenizer (bert-base-uncased)...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load MLM data for pretrain probe measurement
    print("Loading MLM data...")
    mlm_loader = load_mlm_data(tokenizer, batch_size=batch_size)

    # Load fine-tuning data
    print(f"Loading {task_name} data...")
    train_loader, val_loader, num_labels = load_finetune_data(
        tokenizer, task_name, batch_size=batch_size,
    )

    # Load existing results if resuming
    results = []
    completed = set()
    output_path = os.path.join(output_dir, "pretrain_probe_results.json")

    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from {resume_from}")
        with open(resume_from) as f:
            results = json.load(f)
        for r in results:
            completed.add((r['seed'], r['checkpoint_step'], r['lr'],
                          r['warmup_ratio'], r['weight_decay']))
        print(f"Found {len(completed)} completed runs, {total_runs - len(completed)} remaining")

    # Cache pretrain probes (one measurement per checkpoint)
    pretrain_probes_cache = {}

    run_count = 0
    experiment_start = time.time()

    for seed in seeds:
        for step in checkpoint_steps:
            checkpoint_name = f"google/multiberts-seed_{seed}-{step}"
            cache_key = (seed, step)

            # Check if all HP configs for this checkpoint are done
            all_done = all(
                (seed, step, lr, warmup, wd) in completed
                for lr in learning_rates
                for warmup in warmup_ratios
                for wd in weight_decays
            )
            if all_done:
                run_count += n_hp_configs
                print(f"\nSkipping {checkpoint_name} (all {n_hp_configs} configs completed)")
                continue

            print(f"\n{'=' * 60}")
            print(f"Checkpoint: {checkpoint_name}")
            print(f"{'=' * 60}")

            # Measure pretrain probes (once per checkpoint)
            if cache_key not in pretrain_probes_cache:
                try:
                    probes = measure_pretrain_probes(
                        checkpoint_name, mlm_loader, device,
                        n_hutchinson_samples=n_hutchinson_samples,
                    )
                    pretrain_probes_cache[cache_key] = probes
                    print(f"  grad_norm={probes['gradient_norm']:.4f}, "
                          f"trace={probes['hutchinson_trace']:.4f}, "
                          f"top_eig={probes['top_eigenvalue']:.4f}, "
                          f"sam={probes['sam_sharpness']:.4f}")
                except Exception as e:
                    print(f"  ERROR measuring probes: {e}")
                    pretrain_probes_cache[cache_key] = {
                        k: float('nan') for k in [
                            'gradient_norm', 'gradient_variance', 'sam_sharpness',
                            'hutchinson_trace', 'top_eigenvalue', 'loss_at_init',
                        ]
                    }

            # Fine-tune with each HP config
            for lr in learning_rates:
                for warmup in warmup_ratios:
                    for wd in weight_decays:
                        run_count += 1
                        config_key = (seed, step, lr, warmup, wd)

                        if config_key in completed:
                            continue

                        print(f"\n  [{run_count}/{total_runs}] lr={lr}, warmup={warmup}, wd={wd}")

                        try:
                            ft_result = finetune_checkpoint(
                                checkpoint_name, train_loader, val_loader,
                                num_labels, lr=lr, warmup_ratio=warmup,
                                weight_decay=wd, num_epochs=num_epochs,
                                device=device,
                            )
                            print(f"  -> val_accuracy={ft_result['val_accuracy']:.4f}")
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            ft_result = {
                                'val_accuracy': float('nan'),
                                'final_train_loss': float('nan'),
                            }

                        # Build result record
                        pretrain_probes = pretrain_probes_cache[cache_key]
                        result = {
                            'seed': seed,
                            'checkpoint_step': step,
                            'checkpoint_name': checkpoint_name,
                            'task': task_name,
                            'lr': lr,
                            'warmup_ratio': warmup,
                            'weight_decay': wd,
                            'batch_size': batch_size,
                            'num_epochs': num_epochs,
                            'val_accuracy': ft_result['val_accuracy'],
                            'final_train_loss': ft_result['final_train_loss'],
                            # Pretrain probes
                            'pretrain_gradient_norm': pretrain_probes['gradient_norm'],
                            'pretrain_gradient_variance': pretrain_probes['gradient_variance'],
                            'pretrain_sam_sharpness': pretrain_probes['sam_sharpness'],
                            'pretrain_hutchinson_trace': pretrain_probes['hutchinson_trace'],
                            'pretrain_top_eigenvalue': pretrain_probes['top_eigenvalue'],
                            'pretrain_loss': pretrain_probes['loss_at_init'],
                        }

                        results.append(result)

                        # Save incrementally (crash recovery)
                        with open(output_path, 'w') as f:
                            json.dump(results, f, indent=2)

    elapsed = time.time() - experiment_start
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT COMPLETE")
    print(f"  {len(results)} runs saved to {output_path}")
    print(f"  Total time: {elapsed / 3600:.1f} hours")
    print(f"{'=' * 60}")

    return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pretrain checkpoint landscape probe experiment",
    )
    subparsers = parser.add_subparsers(dest='command')

    # --- run ---
    run_parser = subparsers.add_parser('run', help='Run the experiment')
    run_parser.add_argument(
        '--steps', nargs='+', default=CHECKPOINT_STEPS,
        help='Pretraining checkpoint steps (default: step_200k step_600k step_1000k step_2000k)',
    )
    run_parser.add_argument(
        '--seeds', nargs='+', type=int, default=SEEDS,
        help='MultiBERTs seeds (default: 0 1 2 3 4)',
    )
    run_parser.add_argument('--task', default='sst2', help='GLUE task name')
    run_parser.add_argument('--batch-size', type=int, default=16)
    run_parser.add_argument('--epochs', type=int, default=1,
                           help='Fine-tuning epochs (default: 1)')
    run_parser.add_argument('--device', default='cuda')
    run_parser.add_argument('--output-dir', default='./results_pretrain')
    run_parser.add_argument('--resume', default=None,
                           help='Path to results JSON to resume from')
    run_parser.add_argument('--n-hutchinson', type=int, default=50,
                           help='Hutchinson samples for trace estimation')
    run_parser.add_argument('--lrs', nargs='+', type=float, default=None,
                           help='Custom learning rates')
    run_parser.add_argument('--warmups', nargs='+', type=float, default=None,
                           help='Custom warmup ratios')
    run_parser.add_argument('--weight-decays', nargs='+', type=float, default=None,
                           help='Custom weight decays')

    # --- analyze ---
    analyze_parser = subparsers.add_parser('analyze', help='Analyze results')
    analyze_parser.add_argument(
        '--results', default='./results_pretrain/pretrain_probe_results.json',
    )
    analyze_parser.add_argument('--output-dir', default='./figures_pretrain')

    args = parser.parse_args()

    if args.command == 'run':
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            args.device = 'cpu'

        if args.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"VRAM: {vram:.1f} GB")

        run_experiment(
            checkpoint_steps=args.steps,
            seeds=args.seeds,
            task_name=args.task,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rates=args.lrs,
            warmup_ratios=args.warmups,
            weight_decays=args.weight_decays,
            device=args.device,
            output_dir=args.output_dir,
            resume_from=args.resume,
            n_hutchinson_samples=args.n_hutchinson,
        )

    elif args.command == 'analyze':
        # Import here to avoid dependency issues if only running experiment
        from analysis.pretrain_analysis import analyze_pretrain_results
        analyze_pretrain_results(args.results, args.output_dir)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
