# -*- coding: utf-8 -*-
"""
rst-seti — Radio Spectrogram Transformer

Transformer-based technosignature detection in radio observations (SETI).
Based on DeiT / AST architecture adapted for 1-channel radio spectrograms.
"""

__version__ = "0.1.0"
__author__ = "Filippo Zuddas"

from rst_seti.inference.engine import InferenceEngine

__all__ = ["InferenceEngine", "__version__"]
