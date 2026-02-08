"""Correlation analysis for landscape probe experiments."""

from .correlation_analysis import (
    load_results,
    compute_correlations,
    find_optimal_lr_per_probe_bin,
    plot_probe_vs_accuracy,
    generate_full_report
)

__all__ = [
    'load_results',
    'compute_correlations',
    'find_optimal_lr_per_probe_bin',
    'plot_probe_vs_accuracy',
    'generate_full_report'
]
