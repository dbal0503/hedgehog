"""
Analyze correlations between probes and optimal hyperparameters.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Any
import warnings

# Try to import visualization libraries (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("matplotlib/seaborn not installed, plotting disabled")


def load_results(filepath: str) -> pd.DataFrame:
    """Load experiment results into DataFrame."""
    with open(filepath) as f:
        results = json.load(f)
    return pd.DataFrame(results)


def load_all_results(results_dir: str, tasks: Optional[List[str]] = None) -> pd.DataFrame:
    """Load results from multiple tasks into a single DataFrame."""
    all_results = []

    if tasks is None:
        # Find all result files
        for f in os.listdir(results_dir):
            if f.endswith('_results.json') and not f.startswith('sanity'):
                filepath = os.path.join(results_dir, f)
                df = load_results(filepath)
                all_results.append(df)
    else:
        for task in tasks:
            filepath = os.path.join(results_dir, f"{task}_results.json")
            if os.path.exists(filepath):
                df = load_results(filepath)
                all_results.append(df)

    if not all_results:
        raise ValueError(f"No results found in {results_dir}")

    return pd.concat(all_results, ignore_index=True)


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman correlations between probes and outcomes.

    Returns DataFrame with columns:
        - probe: probe name
        - outcome: outcome variable name
        - correlation: Spearman r
        - p_value: p-value
        - significant: boolean (p < 0.05)
    """
    probes = ['gradient_norm', 'gradient_variance', 'sam_sharpness',
              'hutchinson_trace', 'top_eigenvalue', 'loss_at_init']
    outcomes = ['val_accuracy', 'lr']

    correlations = []

    for probe in probes:
        for outcome in outcomes:
            if probe not in df.columns or outcome not in df.columns:
                continue

            # Remove NaN values
            valid_mask = ~(df[probe].isna() | df[outcome].isna())
            if valid_mask.sum() < 3:
                continue

            probe_vals = df.loc[valid_mask, probe]
            outcome_vals = df.loc[valid_mask, outcome]

            # Spearman correlation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, p = stats.spearmanr(probe_vals, outcome_vals)

            correlations.append({
                'probe': probe,
                'outcome': outcome,
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05,
                'n_samples': valid_mask.sum()
            })

    return pd.DataFrame(correlations)


def compute_partial_correlations(df: pd.DataFrame, target: str = 'val_accuracy') -> pd.DataFrame:
    """
    Compute partial correlations of each probe with target,
    controlling for other probes.

    This helps identify which probes add unique information.
    """
    probes = ['gradient_norm', 'gradient_variance', 'sam_sharpness',
              'hutchinson_trace', 'top_eigenvalue']

    # Filter to available probes
    available_probes = [p for p in probes if p in df.columns]

    if target not in df.columns:
        raise ValueError(f"Target {target} not in DataFrame")

    results = []

    for probe in available_probes:
        # Control variables: all other probes
        control_vars = [p for p in available_probes if p != probe]

        if len(control_vars) == 0:
            # No control variables, just compute regular correlation
            r, p = stats.spearmanr(df[probe], df[target])
            results.append({
                'probe': probe,
                'target': target,
                'partial_correlation': r,
                'p_value': p,
                'controlled_for': []
            })
            continue

        # Compute partial correlation using regression residuals
        # This is a simplified approach; for full partial correlation,
        # consider using pingouin library
        from sklearn.linear_model import LinearRegression

        # Regress probe on controls
        X_controls = df[control_vars].fillna(0).values
        probe_vals = df[probe].fillna(0).values
        target_vals = df[target].fillna(0).values

        reg_probe = LinearRegression().fit(X_controls, probe_vals)
        probe_residuals = probe_vals - reg_probe.predict(X_controls)

        reg_target = LinearRegression().fit(X_controls, target_vals)
        target_residuals = target_vals - reg_target.predict(X_controls)

        # Correlation of residuals
        r, p = stats.spearmanr(probe_residuals, target_residuals)

        results.append({
            'probe': probe,
            'target': target,
            'partial_correlation': r,
            'p_value': p,
            'controlled_for': control_vars
        })

    return pd.DataFrame(results)


def find_optimal_lr_per_probe_bin(
    df: pd.DataFrame,
    probe: str,
    n_bins: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Bin by probe value, find optimal LR in each bin.
    Tests if probe value predicts what LR works best.
    """
    if probe not in df.columns:
        raise ValueError(f"Probe {probe} not in DataFrame")

    # Remove NaN values
    valid_df = df.dropna(subset=[probe, 'val_accuracy', 'lr'])

    if len(valid_df) < n_bins:
        raise ValueError(f"Not enough data points ({len(valid_df)}) for {n_bins} bins")

    # Create bins
    try:
        valid_df = valid_df.copy()
        valid_df['probe_bin'] = pd.qcut(
            valid_df[probe],
            n_bins,
            labels=['low', 'medium', 'high'][:n_bins],
            duplicates='drop'
        )
    except ValueError:
        # If qcut fails due to duplicate edges, use cut instead
        valid_df['probe_bin'] = pd.cut(
            valid_df[probe],
            n_bins,
            labels=['low', 'medium', 'high'][:n_bins]
        )

    optimal_lrs = {}
    for bin_label in valid_df['probe_bin'].unique():
        if pd.isna(bin_label):
            continue

        bin_df = valid_df[valid_df['probe_bin'] == bin_label]
        if len(bin_df) == 0:
            continue

        best_idx = bin_df['val_accuracy'].idxmax()
        optimal_lrs[str(bin_label)] = {
            'optimal_lr': bin_df.loc[best_idx, 'lr'],
            'best_accuracy': bin_df.loc[best_idx, 'val_accuracy'],
            'mean_probe_value': bin_df[probe].mean(),
            'n_samples': len(bin_df)
        }

    return optimal_lrs


def analyze_probe_predictive_power(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze how well probes predict optimal hyperparameters.

    For each configuration, determine if it achieved near-optimal performance
    and check if probes could have predicted this.
    """
    probes = ['gradient_norm', 'gradient_variance', 'sam_sharpness',
              'hutchinson_trace', 'top_eigenvalue']
    available_probes = [p for p in probes if p in df.columns]

    # Find optimal accuracy for reference
    max_acc = df['val_accuracy'].max()
    threshold = max_acc * 0.95  # Within 5% of best

    df = df.copy()
    df['is_good_config'] = df['val_accuracy'] >= threshold

    results = []
    for probe in available_probes:
        # Check if probe distinguishes good from bad configs
        good_probes = df.loc[df['is_good_config'], probe].dropna()
        bad_probes = df.loc[~df['is_good_config'], probe].dropna()

        if len(good_probes) < 2 or len(bad_probes) < 2:
            continue

        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(good_probes, bad_probes, alternative='two-sided')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(good_probes), len(bad_probes)
        effect_size = 1 - (2 * stat) / (n1 * n2)

        results.append({
            'probe': probe,
            'good_config_mean': good_probes.mean(),
            'bad_config_mean': bad_probes.mean(),
            'p_value': p,
            'effect_size': effect_size,
            'significant': p < 0.05
        })

    return pd.DataFrame(results)


def plot_probe_vs_accuracy(df: pd.DataFrame, output_dir: str = "./figures"):
    """Generate scatter plots of each probe vs validation accuracy."""
    if not HAS_PLOTTING:
        print("Plotting libraries not available")
        return

    probes = ['gradient_norm', 'gradient_variance', 'sam_sharpness',
              'hutchinson_trace', 'top_eigenvalue']
    available_probes = [p for p in probes if p in df.columns]

    n_probes = len(available_probes)
    if n_probes == 0:
        print("No probes found in data")
        return

    n_cols = min(3, n_probes)
    n_rows = (n_probes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_probes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, probe in enumerate(available_probes):
        ax = axes[i]

        # Remove NaN values
        valid_mask = ~(df[probe].isna() | df['val_accuracy'].isna())
        plot_df = df.loc[valid_mask]

        if len(plot_df) == 0:
            ax.set_title(f'{probe}\n(no valid data)')
            continue

        # Color by learning rate
        scatter = ax.scatter(
            plot_df[probe],
            plot_df['val_accuracy'],
            c=np.log10(plot_df['lr']),
            cmap='viridis',
            alpha=0.7,
            s=50
        )

        # Fit line
        try:
            z = np.polyfit(plot_df[probe], plot_df['val_accuracy'], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(plot_df[probe].min(), plot_df[probe].max(), 100)
            ax.plot(x_line, p_line(x_line), "r--", alpha=0.8, linewidth=2)
        except:
            pass

        # Correlation
        r, pval = stats.spearmanr(plot_df[probe], plot_df['val_accuracy'])
        sig_marker = "*" if pval < 0.05 else ""
        ax.set_title(f'{probe}\nr={r:.3f}{sig_marker} (p={pval:.3f})')
        ax.set_xlabel(probe)
        ax.set_ylabel('Validation Accuracy')

    # Hide unused subplots
    for i in range(len(available_probes), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "probe_correlations.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str = "./figures"):
    """Generate heatmap of all probe-outcome correlations."""
    if not HAS_PLOTTING:
        print("Plotting libraries not available")
        return

    probes = ['gradient_norm', 'gradient_variance', 'sam_sharpness',
              'hutchinson_trace', 'top_eigenvalue', 'loss_at_init']
    outcomes = ['val_accuracy', 'lr', 'final_train_loss']

    available_probes = [p for p in probes if p in df.columns]
    available_outcomes = [o for o in outcomes if o in df.columns]

    # Compute correlation matrix
    corr_matrix = np.zeros((len(available_probes), len(available_outcomes)))
    p_matrix = np.zeros((len(available_probes), len(available_outcomes)))

    for i, probe in enumerate(available_probes):
        for j, outcome in enumerate(available_outcomes):
            valid_mask = ~(df[probe].isna() | df[outcome].isna())
            if valid_mask.sum() >= 3:
                r, p = stats.spearmanr(df.loc[valid_mask, probe], df.loc[valid_mask, outcome])
                corr_matrix[i, j] = r
                p_matrix[i, j] = p

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create annotation strings with significance markers
    annot = np.empty_like(corr_matrix, dtype=object)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            sig = "**" if p_matrix[i, j] < 0.01 else "*" if p_matrix[i, j] < 0.05 else ""
            annot[i, j] = f"{corr_matrix[i, j]:.2f}{sig}"

    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt='',
        cmap='RdBu_r',
        center=0,
        xticklabels=available_outcomes,
        yticklabels=available_probes,
        ax=ax,
        vmin=-1,
        vmax=1
    )

    ax.set_title('Probe-Outcome Spearman Correlations\n(* p<0.05, ** p<0.01)')

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")
    plt.close()


def generate_full_report(
    results_dir: str = "./results",
    output_dir: str = "./figures",
    tasks: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive analysis report.

    Returns dict with all analysis results and generates plots.
    """
    print("=" * 60)
    print("LANDSCAPE PROBE CORRELATION ANALYSIS REPORT")
    print("=" * 60)

    # Load all results
    df = load_all_results(results_dir, tasks)
    print(f"\nLoaded {len(df)} experiment runs")

    if 'task' in df.columns:
        print(f"Tasks: {df['task'].unique().tolist()}")

    report = {
        'n_experiments': len(df),
        'tasks': df['task'].unique().tolist() if 'task' in df.columns else [],
    }

    # 1. Basic correlations
    print("\n" + "-" * 40)
    print("1. PROBE-OUTCOME CORRELATIONS (Spearman)")
    print("-" * 40)

    corr_df = compute_correlations(df)
    report['correlations'] = corr_df.to_dict('records')

    print(corr_df.to_string(index=False))

    # Highlight significant findings
    sig_corrs = corr_df[corr_df['significant']]
    if len(sig_corrs) > 0:
        print("\n✓ Significant correlations found:")
        for _, row in sig_corrs.iterrows():
            print(f"  {row['probe']} ↔ {row['outcome']}: r={row['correlation']:.3f} (p={row['p_value']:.4f})")
    else:
        print("\n⚠ No significant correlations found")

    # 2. Partial correlations (if sklearn available)
    print("\n" + "-" * 40)
    print("2. PARTIAL CORRELATIONS (controlling for other probes)")
    print("-" * 40)

    try:
        partial_df = compute_partial_correlations(df, target='val_accuracy')
        report['partial_correlations'] = partial_df.to_dict('records')
        print(partial_df.to_string(index=False))
    except ImportError:
        print("sklearn not available, skipping partial correlations")

    # 3. Optimal LR by probe bins
    print("\n" + "-" * 40)
    print("3. OPTIMAL LEARNING RATE BY PROBE BINS")
    print("-" * 40)

    probe_lr_analysis = {}
    for probe in ['hutchinson_trace', 'sam_sharpness', 'gradient_norm']:
        if probe not in df.columns:
            continue

        print(f"\n{probe}:")
        try:
            optimal = find_optimal_lr_per_probe_bin(df, probe)
            probe_lr_analysis[probe] = optimal

            for bin_label, info in optimal.items():
                print(f"  {bin_label}: optimal_lr={info['optimal_lr']:.0e}, "
                      f"acc={info['best_accuracy']:.4f}, n={info['n_samples']}")

            # Check if there's a trend
            if 'low' in optimal and 'high' in optimal:
                lr_low = optimal['low']['optimal_lr']
                lr_high = optimal['high']['optimal_lr']
                if lr_low != lr_high:
                    direction = "higher" if lr_high > lr_low else "lower"
                    print(f"  → Higher {probe} suggests {direction} optimal LR")

        except Exception as e:
            print(f"  Could not analyze: {e}")

    report['optimal_lr_by_probe'] = probe_lr_analysis

    # 4. Predictive power analysis
    print("\n" + "-" * 40)
    print("4. PROBE PREDICTIVE POWER (good vs bad configs)")
    print("-" * 40)

    pred_df = analyze_probe_predictive_power(df)
    report['predictive_power'] = pred_df.to_dict('records')

    if len(pred_df) > 0:
        print(pred_df.to_string(index=False))

        sig_preds = pred_df[pred_df['significant']]
        if len(sig_preds) > 0:
            print("\n✓ Probes that distinguish good from bad configs:")
            for _, row in sig_preds.iterrows():
                print(f"  {row['probe']}: effect_size={row['effect_size']:.3f}")
    else:
        print("Not enough data for predictive power analysis")

    # 5. Generate plots
    print("\n" + "-" * 40)
    print("5. GENERATING PLOTS")
    print("-" * 40)

    if HAS_PLOTTING:
        plot_probe_vs_accuracy(df, output_dir)
        plot_correlation_heatmap(df, output_dir)
    else:
        print("Plotting libraries not available, skipping plots")

    # 6. Summary and recommendations
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Determine signal strength
    max_corr = corr_df['correlation'].abs().max() if len(corr_df) > 0 else 0
    n_significant = corr_df['significant'].sum() if len(corr_df) > 0 else 0

    if max_corr > 0.5 and n_significant >= 2:
        signal = "STRONG"
        recommendation = "Proceed to Phase 2"
    elif max_corr > 0.3 or n_significant >= 1:
        signal = "MODERATE"
        recommendation = "Investigate further, consider different probes or timing"
    else:
        signal = "WEAK"
        recommendation = "Consider pivoting or documenting negative result"

    print(f"\nSignal strength: {signal}")
    print(f"Max |correlation|: {max_corr:.3f}")
    print(f"Significant correlations: {n_significant}")
    print(f"\nRecommendation: {recommendation}")

    report['summary'] = {
        'signal_strength': signal,
        'max_correlation': max_corr,
        'n_significant': n_significant,
        'recommendation': recommendation
    }

    # Save report
    report_path = os.path.join(output_dir, "analysis_report.json")
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    report = convert_numpy(report)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")

    return report


def load_multistep_results(results_dir: str) -> pd.DataFrame:
    """Load only multistep result files."""
    all_results = []
    for f in sorted(os.listdir(results_dir)):
        if f.endswith('_multistep_results.json'):
            filepath = os.path.join(results_dir, f)
            df = load_results(filepath)
            all_results.append(df)

    if not all_results:
        raise ValueError(f"No multistep results found in {results_dir}")

    return pd.concat(all_results, ignore_index=True)


def detect_probe_steps(df: pd.DataFrame) -> List[str]:
    """Detect step prefixes from column names (e.g. 'init', 'step10', 'step100')."""
    probe_names = ['gradient_norm', 'gradient_variance', 'sam_sharpness',
                   'hutchinson_trace', 'top_eigenvalue', 'loss_at_init']
    prefixes = set()
    for col in df.columns:
        for probe in probe_names:
            if col.endswith(f'_{probe}'):
                prefix = col[:col.rfind(f'_{probe}')]
                if prefix:
                    prefixes.add(prefix)
    # Sort: init first, then by step number
    def sort_key(p):
        if p == 'init':
            return 0
        return int(p.replace('step', ''))
    return sorted(prefixes, key=sort_key)


def compute_correlations_by_step(df: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    """Compute Spearman correlations between probes at each step and val_accuracy."""
    probe_names = ['gradient_norm', 'gradient_variance', 'sam_sharpness',
                   'hutchinson_trace', 'top_eigenvalue', 'loss_at_init']
    rows = []
    for prefix in prefixes:
        step_label = prefix if prefix == 'init' else prefix.replace('step', '')
        for probe in probe_names:
            col = f'{prefix}_{probe}'
            if col not in df.columns or 'val_accuracy' not in df.columns:
                continue
            valid = ~(df[col].isna() | df['val_accuracy'].isna())
            if valid.sum() < 3:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, p = stats.spearmanr(df.loc[valid, col], df.loc[valid, 'val_accuracy'])
            rows.append({
                'step': step_label,
                'probe': probe,
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05,
                'n_samples': int(valid.sum()),
            })
    return pd.DataFrame(rows)


def plot_correlation_evolution(corr_df: pd.DataFrame, output_dir: str = "./figures"):
    """Plot how probe-accuracy correlation evolves across training steps."""
    if not HAS_PLOTTING:
        print("Plotting libraries not available")
        return

    probe_names = corr_df['probe'].unique()
    steps = corr_df['step'].unique()

    # Convert step labels to numeric for x-axis
    step_nums = []
    for s in steps:
        step_nums.append(0 if s == 'init' else int(s))

    fig, ax = plt.subplots(figsize=(10, 6))

    for probe in probe_names:
        pdf = corr_df[corr_df['probe'] == probe].set_index('step')
        y = [pdf.loc[s, 'correlation'] if s in pdf.index else np.nan for s in steps]
        sig = [pdf.loc[s, 'significant'] if s in pdf.index else False for s in steps]
        ax.plot(step_nums, y, 'o-', label=probe, markersize=6)
        # Mark significant points
        for xi, yi, si in zip(step_nums, y, sig):
            if si:
                ax.plot(xi, yi, 'o', color='red', markersize=10, zorder=5, fillstyle='none', linewidth=2)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Spearman Correlation with Val Accuracy')
    ax.set_title('Probe-Accuracy Correlation vs Training Step\n(red circles = p < 0.05)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_xticks(step_nums)
    ax.set_xticklabels([str(s) for s in steps], rotation=45)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "correlation_evolution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved correlation evolution plot to {output_path}")
    plt.close()


def generate_multistep_report(
    results_dir: str = "./results",
    output_dir: str = "./figures",
) -> Dict[str, Any]:
    """Generate analysis report for multistep probe results."""
    print("=" * 60)
    print("MULTISTEP LANDSCAPE PROBE ANALYSIS REPORT")
    print("=" * 60)

    df = load_multistep_results(results_dir)
    print(f"\nLoaded {len(df)} multistep experiment runs")

    if 'task' in df.columns:
        print(f"Tasks: {df['task'].unique().tolist()}")
    if 'probe_steps' in df.columns:
        print(f"Probe steps: {df['probe_steps'].iloc[0]}")

    prefixes = detect_probe_steps(df)
    print(f"Detected step prefixes: {prefixes}")

    report = {'n_experiments': len(df)}

    # 1. Correlations by step
    print("\n" + "-" * 60)
    print("1. PROBE-ACCURACY CORRELATIONS BY TRAINING STEP (Spearman)")
    print("-" * 60)

    corr_df = compute_correlations_by_step(df, prefixes)
    report['correlations_by_step'] = corr_df.to_dict('records')

    # Pivot table for display
    probe_names = ['gradient_norm', 'gradient_variance', 'sam_sharpness',
                   'hutchinson_trace', 'top_eigenvalue', 'loss_at_init']
    steps = [p if p == 'init' else p.replace('step', '') for p in prefixes]

    print(f"\n{'probe':<22}", end="")
    for s in steps:
        print(f"{'step ' + s:>12}" if s != 'init' else f"{'init':>12}", end="")
    print()
    print("-" * (22 + 12 * len(steps)))

    for probe in probe_names:
        pdf = corr_df[corr_df['probe'] == probe]
        print(f"{probe:<22}", end="")
        for s in steps:
            row = pdf[pdf['step'] == s]
            if len(row) == 0:
                print(f"{'--':>12}", end="")
            else:
                r = row.iloc[0]['correlation']
                sig = row.iloc[0]['significant']
                marker = "**" if row.iloc[0]['p_value'] < 0.01 else "*" if sig else ""
                print(f"{r:>9.3f}{marker:<3}", end="")
        print()

    # 2. Highlight best step per probe
    print("\n" + "-" * 60)
    print("2. STRONGEST CORRELATION PER PROBE")
    print("-" * 60)

    for probe in probe_names:
        pdf = corr_df[corr_df['probe'] == probe]
        if len(pdf) == 0:
            continue
        best_idx = pdf['correlation'].abs().idxmax()
        best = pdf.loc[best_idx]
        sig = "*" if best['significant'] else ""
        print(f"  {probe:<22} best at step {best['step']:>4}: r={best['correlation']:.3f}{sig} (p={best['p_value']:.4f})")

    # 3. Predictive power at each step
    print("\n" + "-" * 60)
    print("3. MANN-WHITNEY PREDICTIVE POWER BY STEP (good vs bad configs)")
    print("-" * 60)

    max_acc = df['val_accuracy'].max()
    threshold = max_acc * 0.95
    df_copy = df.copy()
    df_copy['is_good'] = df_copy['val_accuracy'] >= threshold
    n_good = df_copy['is_good'].sum()
    n_bad = (~df_copy['is_good']).sum()
    print(f"  Good configs (>= {threshold:.4f}): {n_good}, Bad: {n_bad}")

    for prefix in prefixes:
        step_label = prefix if prefix == 'init' else prefix.replace('step', '')
        sig_probes = []
        for probe in ['gradient_norm', 'gradient_variance', 'sam_sharpness',
                       'hutchinson_trace', 'top_eigenvalue']:
            col = f'{prefix}_{probe}'
            if col not in df_copy.columns:
                continue
            good_vals = df_copy.loc[df_copy['is_good'], col].dropna()
            bad_vals = df_copy.loc[~df_copy['is_good'], col].dropna()
            if len(good_vals) < 2 or len(bad_vals) < 2:
                continue
            stat, p = stats.mannwhitneyu(good_vals, bad_vals, alternative='two-sided')
            if p < 0.05:
                n1, n2 = len(good_vals), len(bad_vals)
                effect = 1 - (2 * stat) / (n1 * n2)
                sig_probes.append(f"{probe}(p={p:.3f}, d={effect:.2f})")

        step_display = f"step {step_label}" if step_label != 'init' else 'init'
        if sig_probes:
            print(f"  {step_display:>10}: {', '.join(sig_probes)}")
        else:
            print(f"  {step_display:>10}: no significant probes")

    # 4. Plot
    print("\n" + "-" * 60)
    print("4. GENERATING PLOTS")
    print("-" * 60)

    if HAS_PLOTTING:
        plot_correlation_evolution(corr_df, output_dir)
    else:
        print("Plotting libraries not available")

    # 5. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    max_corr = corr_df['correlation'].abs().max() if len(corr_df) > 0 else 0
    n_sig = corr_df['significant'].sum() if len(corr_df) > 0 else 0
    best_row = corr_df.loc[corr_df['correlation'].abs().idxmax()] if len(corr_df) > 0 else None

    print(f"\nMax |correlation|: {max_corr:.3f}")
    print(f"Significant correlations: {n_sig} / {len(corr_df)}")
    if best_row is not None:
        print(f"Best: {best_row['probe']} at step {best_row['step']} "
              f"(r={best_row['correlation']:.3f}, p={best_row['p_value']:.4f})")

    # Save report
    report_path = os.path.join(output_dir, "multistep_analysis_report.json")
    os.makedirs(output_dir, exist_ok=True)

    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    report = convert_numpy(report)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze landscape probe experiments")
    parser.add_argument("--results-dir", type=str, default="./results",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", type=str, default="./figures",
                        help="Directory for output plots and reports")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="Specific tasks to analyze (default: all)")
    parser.add_argument("--multistep", action="store_true",
                        help="Analyze multistep probe results (step{N}_ prefixed fields)")

    args = parser.parse_args()

    if args.multistep:
        generate_multistep_report(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
        )
    else:
        generate_full_report(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            tasks=args.tasks
        )


if __name__ == "__main__":
    main()
