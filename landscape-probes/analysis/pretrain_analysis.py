"""
Analysis for pretrain checkpoint landscape probe experiment.

Correlates landscape probes measured during pretraining (under MLM loss)
with fine-tuning outcomes (accuracy, optimal hyperparameters).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

PRETRAIN_PROBES = [
    'pretrain_gradient_norm',
    'pretrain_gradient_variance',
    'pretrain_sam_sharpness',
    'pretrain_hutchinson_trace',
    'pretrain_top_eigenvalue',
    'pretrain_loss',
]

PROBE_SHORT_NAMES = {
    'pretrain_gradient_norm': 'grad_norm',
    'pretrain_gradient_variance': 'grad_var',
    'pretrain_sam_sharpness': 'sam_sharp',
    'pretrain_hutchinson_trace': 'hutch_trace',
    'pretrain_top_eigenvalue': 'top_eig',
    'pretrain_loss': 'mlm_loss',
}


def load_pretrain_results(filepath: str) -> pd.DataFrame:
    """Load pretrain probe experiment results."""
    with open(filepath) as f:
        results = json.load(f)
    return pd.DataFrame(results)


def spearman_with_sig(x, y):
    """Compute Spearman correlation with significance markers."""
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 3:
        return np.nan, np.nan, ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, p = stats.spearmanr(x[valid], y[valid])
    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
    return r, p, sig


def analyze_pretrain_results(results_path: str, output_dir: str = "./figures_pretrain"):
    """Full analysis of pretrain probe -> fine-tuning outcome correlations."""
    df = load_pretrain_results(results_path)
    os.makedirs(output_dir, exist_ok=True)

    available_probes = [p for p in PRETRAIN_PROBES if p in df.columns]

    print("=" * 70)
    print("PRETRAIN PROBE -> FINE-TUNING OUTCOME ANALYSIS")
    print("=" * 70)
    print(f"\nRuns loaded: {len(df)}")
    print(f"Checkpoint steps: {sorted(df['checkpoint_step'].unique())}")
    print(f"Seeds: {sorted(df['seed'].unique())}")

    n_checkpoints = df[['seed', 'checkpoint_step']].drop_duplicates().shape[0]
    n_hp = len(df) // n_checkpoints if n_checkpoints > 0 else 0
    print(f"Checkpoints: {n_checkpoints}, HP configs per checkpoint: {n_hp}")

    # ================================================================
    # 1. Pretrain probes vs BEST achievable accuracy (per checkpoint)
    # ================================================================
    print(f"\n{'=' * 70}")
    print("1. PRETRAIN PROBES vs BEST ACHIEVABLE ACCURACY")
    print(f"   (Spearman r, N = {n_checkpoints} checkpoint x seed combos)")
    print(f"{'=' * 70}")

    best_acc = df.groupby(['seed', 'checkpoint_step']).agg({
        'val_accuracy': 'max',
        **{p: 'first' for p in available_probes},
    }).reset_index()

    for probe in available_probes:
        r, p, sig = spearman_with_sig(
            best_acc[probe].values, best_acc['val_accuracy'].values,
        )
        short = PROBE_SHORT_NAMES.get(probe, probe)
        print(f"  {short:<20} r = {r:>7.3f}   p = {p:.4f} {sig}")

    # ================================================================
    # 2. Pretrain probes vs OPTIMAL learning rate
    # ================================================================
    print(f"\n{'=' * 70}")
    print("2. PRETRAIN PROBES vs OPTIMAL LEARNING RATE")
    print(f"   (Does pretraining geometry predict which LR works best?)")
    print(f"{'=' * 70}")

    optimal_lr_df = df.loc[df.groupby(['seed', 'checkpoint_step'])['val_accuracy'].idxmax()]

    for probe in available_probes:
        r, p, sig = spearman_with_sig(
            optimal_lr_df[probe].values, optimal_lr_df['lr'].values,
        )
        short = PROBE_SHORT_NAMES.get(probe, probe)
        print(f"  {short:<20} r = {r:>7.3f}   p = {p:.4f} {sig}")

    # Show which LR was optimal for each checkpoint
    print(f"\n  Optimal LR by checkpoint:")
    for step in sorted(df['checkpoint_step'].unique()):
        step_opt = optimal_lr_df[optimal_lr_df['checkpoint_step'] == step]
        lr_counts = step_opt['lr'].value_counts().to_dict()
        lr_str = ", ".join(f"{lr:.0e}({n})" for lr, n in sorted(lr_counts.items()))
        print(f"    {step}: {lr_str}")

    # ================================================================
    # 3. Overall correlation (all runs)
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"3. PRETRAIN PROBES vs VAL ACCURACY (all {len(df)} runs)")
    print(f"{'=' * 70}")

    for probe in available_probes:
        r, p, sig = spearman_with_sig(
            df[probe].values, df['val_accuracy'].values,
        )
        short = PROBE_SHORT_NAMES.get(probe, probe)
        print(f"  {short:<20} r = {r:>7.3f}   p = {p:.4f} {sig}")

    # ================================================================
    # 4. Per-checkpoint performance summary
    # ================================================================
    print(f"\n{'=' * 70}")
    print("4. PER-CHECKPOINT PERFORMANCE SUMMARY")
    print(f"{'=' * 70}")

    for step in sorted(df['checkpoint_step'].unique()):
        step_df = df[df['checkpoint_step'] == step]
        best = step_df.loc[step_df['val_accuracy'].idxmax()]
        mean_acc = step_df['val_accuracy'].mean()
        std_acc = step_df['val_accuracy'].std()

        print(f"\n  {step}:")
        print(f"    Accuracy: {mean_acc:.4f} +/- {std_acc:.4f} (best: {best['val_accuracy']:.4f})")
        print(f"    Best HP:  lr={best['lr']:.0e}, warmup={best['warmup_ratio']}, wd={best['weight_decay']}")

        # Pretrain probes (same for all HP configs at this checkpoint/seed)
        probe_vals = step_df.groupby('seed')[available_probes].first()
        for probe in available_probes:
            short = PROBE_SHORT_NAMES.get(probe, probe)
            vals = probe_vals[probe]
            print(f"    {short}: {vals.mean():.4f} +/- {vals.std():.4f}")

    # ================================================================
    # 5. Does the optimal HP change across pretraining stages?
    # ================================================================
    print(f"\n{'=' * 70}")
    print("5. OPTIMAL HP STABILITY ACROSS PRETRAINING STAGES")
    print(f"{'=' * 70}")

    for seed in sorted(df['seed'].unique()):
        seed_df = df[df['seed'] == seed]
        print(f"\n  Seed {seed}:")
        for step in sorted(seed_df['checkpoint_step'].unique()):
            step_df = seed_df[seed_df['checkpoint_step'] == step]
            best = step_df.loc[step_df['val_accuracy'].idxmax()]
            print(f"    {step}: lr={best['lr']:.0e}, warmup={best['warmup_ratio']}, "
                  f"wd={best['weight_decay']}, acc={best['val_accuracy']:.4f}")

    # Check if optimal HP is consistent
    hp_changes = 0
    hp_total = 0
    for seed in df['seed'].unique():
        seed_opt = optimal_lr_df[optimal_lr_df['seed'] == seed].sort_values('checkpoint_step')
        lrs = seed_opt['lr'].values
        for i in range(1, len(lrs)):
            hp_total += 1
            if lrs[i] != lrs[i - 1]:
                hp_changes += 1

    if hp_total > 0:
        pct = hp_changes / hp_total * 100
        print(f"\n  Optimal LR changed across consecutive checkpoints: "
              f"{hp_changes}/{hp_total} ({pct:.0f}%)")
        if pct < 20:
            print("  -> HP choice is STABLE across pretraining stages (limited predictive value)")
        elif pct > 60:
            print("  -> HP choice VARIES across pretraining stages (probes could be useful)")
        else:
            print("  -> HP choice shows MODERATE variation across stages")

    # ================================================================
    # 6. Probe-accuracy correlation by checkpoint step
    # ================================================================
    print(f"\n{'=' * 70}")
    print("6. CORRELATION STRENGTH BY PRETRAINING STAGE")
    print(f"   (Does correlation get stronger for later checkpoints?)")
    print(f"{'=' * 70}")

    steps = sorted(df['checkpoint_step'].unique())

    # Header
    print(f"\n  {'probe':<20}", end="")
    for step in steps:
        print(f"  {step:>12}", end="")
    print()
    print("  " + "-" * (20 + 14 * len(steps)))

    for probe in available_probes:
        short = PROBE_SHORT_NAMES.get(probe, probe)
        print(f"  {short:<20}", end="")
        for step in steps:
            step_df = df[df['checkpoint_step'] == step]
            r, p, sig = spearman_with_sig(
                step_df[probe].values, step_df['val_accuracy'].values,
            )
            if np.isnan(r):
                print(f"  {'--':>12}", end="")
            else:
                print(f"  {r:>9.3f}{sig:<3}", end="")
        print()

    # ================================================================
    # Plots
    # ================================================================
    if HAS_PLOTTING:
        _generate_plots(df, best_acc, optimal_lr_df, available_probes, output_dir)
    else:
        print("\nmatplotlib/seaborn not available, skipping plots")

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # Find strongest correlations
    all_corrs = []
    for probe in available_probes:
        # vs best accuracy
        r_acc, p_acc, _ = spearman_with_sig(
            best_acc[probe].values, best_acc['val_accuracy'].values,
        )
        all_corrs.append(('best_accuracy', probe, r_acc, p_acc))

        # vs optimal LR
        r_lr, p_lr, _ = spearman_with_sig(
            optimal_lr_df[probe].values, optimal_lr_df['lr'].values,
        )
        all_corrs.append(('optimal_lr', probe, r_lr, p_lr))

    all_corrs.sort(key=lambda x: abs(x[2]) if not np.isnan(x[2]) else 0, reverse=True)

    print("\nTop correlations:")
    for outcome, probe, r, p in all_corrs[:5]:
        short = PROBE_SHORT_NAMES.get(probe, probe)
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {short} <-> {outcome}: r={r:.3f} (p={p:.4f}) {sig}")

    n_sig = sum(1 for _, _, r, p in all_corrs if p < 0.05 and not np.isnan(p))
    max_r = max((abs(r) for _, _, r, _ in all_corrs if not np.isnan(r)), default=0)

    if max_r > 0.5 and n_sig >= 2:
        verdict = "STRONG signal — pretrain probes predict fine-tuning outcomes"
    elif max_r > 0.3 or n_sig >= 1:
        verdict = "MODERATE signal — some predictive power, investigate further"
    else:
        verdict = "WEAK/NO signal — pretrain probes do not predict fine-tuning outcomes in this setup"

    print(f"\nVerdict: {verdict}")
    print(f"Max |r|: {max_r:.3f}, Significant correlations: {n_sig}/{len(all_corrs)}")

    # Save report
    report = {
        'n_runs': len(df),
        'n_checkpoints': n_checkpoints,
        'checkpoints': sorted(df['checkpoint_step'].unique().tolist()),
        'seeds': sorted(df['seed'].unique().tolist()),
        'max_abs_correlation': float(max_r),
        'n_significant': n_sig,
        'verdict': verdict,
        'correlations': [
            {'outcome': o, 'probe': PROBE_SHORT_NAMES.get(p, p),
             'r': float(r) if not np.isnan(r) else None,
             'p': float(pv) if not np.isnan(pv) else None}
            for o, p, r, pv in all_corrs
        ],
    }

    report_path = os.path.join(output_dir, "pretrain_analysis_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    return report


def _generate_plots(df, best_acc, optimal_lr_df, available_probes, output_dir):
    """Generate all analysis plots."""
    # --- Plot 1: Pretrain probes vs best accuracy ---
    n_probes = len(available_probes)
    n_cols = min(3, n_probes)
    n_rows = (n_probes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_probes == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    steps = sorted(best_acc['checkpoint_step'].unique())
    step_to_num = {s: i for i, s in enumerate(steps)}

    for i, probe in enumerate(available_probes):
        ax = axes[i]
        valid = ~(best_acc[probe].isna() | best_acc['val_accuracy'].isna())
        plot_data = best_acc[valid]

        if len(plot_data) == 0:
            ax.set_title(f'{PROBE_SHORT_NAMES.get(probe, probe)}\n(no data)')
            continue

        colors = [step_to_num[s] for s in plot_data['checkpoint_step']]
        scatter = ax.scatter(
            plot_data[probe], plot_data['val_accuracy'],
            c=colors, cmap='viridis', alpha=0.7, s=60, edgecolors='gray',
        )

        r, p, sig = spearman_with_sig(
            plot_data[probe].values, plot_data['val_accuracy'].values,
        )
        short = PROBE_SHORT_NAMES.get(probe, probe)
        ax.set_title(f'{short}\nr={r:.3f}{sig} (p={p:.3f})')
        ax.set_xlabel(short)
        ax.set_ylabel('Best Val Accuracy')

    for i in range(len(available_probes), len(axes)):
        axes[i].axis('off')

    plt.suptitle(
        'Pretrain Probes vs Best Achievable Fine-tuning Accuracy',
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "pretrain_probes_vs_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Plot 2: Pretrain probes vs optimal LR ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_probes == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for i, probe in enumerate(available_probes):
        ax = axes[i]
        valid = ~(optimal_lr_df[probe].isna() | optimal_lr_df['lr'].isna())
        plot_data = optimal_lr_df[valid]

        if len(plot_data) == 0:
            continue

        ax.scatter(
            plot_data[probe], np.log10(plot_data['lr']),
            c=[step_to_num.get(s, 0) for s in plot_data['checkpoint_step']],
            cmap='viridis', alpha=0.7, s=60, edgecolors='gray',
        )

        r, p, sig = spearman_with_sig(
            plot_data[probe].values, plot_data['lr'].values,
        )
        short = PROBE_SHORT_NAMES.get(probe, probe)
        ax.set_title(f'{short}\nr={r:.3f}{sig} (p={p:.3f})')
        ax.set_xlabel(short)
        ax.set_ylabel('log10(Optimal LR)')

    for i in range(len(available_probes), len(axes)):
        axes[i].axis('off')

    plt.suptitle(
        'Pretrain Probes vs Optimal Fine-tuning Learning Rate',
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "pretrain_probes_vs_optimal_lr.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Plot 3: Accuracy by pretraining stage ---
    fig, ax = plt.subplots(figsize=(10, 6))
    steps_sorted = sorted(df['checkpoint_step'].unique())
    positions = range(len(steps_sorted))

    box_data = [df[df['checkpoint_step'] == step]['val_accuracy'].dropna().values
                for step in steps_sorted]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)

    for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0.2, 0.8, len(steps_sorted)))):
        patch.set_facecolor(color)

    ax.set_xticks(positions)
    ax.set_xticklabels(steps_sorted, rotation=45, ha='right')
    ax.set_xlabel('Pretraining Checkpoint')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Fine-tuning Accuracy by Pretraining Stage')
    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_by_checkpoint.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # --- Plot 4: Correlation heatmap ---
    fig, ax = plt.subplots(figsize=(8, 6))
    outcomes = ['val_accuracy', 'lr']
    probe_labels = [PROBE_SHORT_NAMES.get(p, p) for p in available_probes]

    corr_matrix = np.zeros((len(available_probes), len(outcomes)))
    annot = np.empty_like(corr_matrix, dtype=object)

    for i, probe in enumerate(available_probes):
        for j, outcome in enumerate(outcomes):
            r, p, sig = spearman_with_sig(
                df[probe].values, df[outcome].values,
            )
            corr_matrix[i, j] = r if not np.isnan(r) else 0
            annot[i, j] = f"{r:.2f}{sig}" if not np.isnan(r) else "--"

    sns.heatmap(
        corr_matrix, annot=annot, fmt='', cmap='RdBu_r', center=0,
        xticklabels=outcomes, yticklabels=probe_labels, ax=ax,
        vmin=-1, vmax=1,
    )
    ax.set_title('Pretrain Probe Correlations with Fine-tuning Outcomes\n(* p<0.05, ** p<0.01)')
    plt.tight_layout()
    path = os.path.join(output_dir, "pretrain_correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze pretrain probe experiment results")
    parser.add_argument(
        '--results', default='./results_pretrain/pretrain_probe_results.json',
        help='Path to results JSON',
    )
    parser.add_argument(
        '--output-dir', default='./figures_pretrain',
        help='Output directory for plots and reports',
    )
    args = parser.parse_args()
    analyze_pretrain_results(args.results, args.output_dir)


if __name__ == "__main__":
    main()
