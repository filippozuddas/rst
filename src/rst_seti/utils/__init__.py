# -*- coding: utf-8 -*-
"""RST — Shared utilities."""

from rst_seti.utils.visualization import (
    # Style
    apply_light_style,
    # Spectrogram / cadence plots
    plot_spectrogram,
    plot_cadence_panels,
    plot_candidate,
    plot_attention_map,
    # Evaluation plots
    plot_threshold_sweep,
    # Distribution plots
    plot_prob_distribution,
    plot_prob_split,
    plot_prob_ccdf,
    # Attention extraction
    AttentionExtractor,
)

__all__ = [
    'apply_light_style',
    'plot_spectrogram',
    'plot_cadence_panels',
    'plot_candidate',
    'plot_attention_map',
    'plot_threshold_sweep',
    'plot_prob_distribution',
    'plot_prob_split',
    'plot_prob_ccdf',
    'AttentionExtractor',
]
