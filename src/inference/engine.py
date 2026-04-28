# -*- coding: utf-8 -*-
"""
RST — Inference Engine

End-to-end inference pipeline for raw HDF5 cadences.
Extracts overlapping snippets via sliding window, runs batch inference
through the Transformer model, and returns per-snippet probabilities.
"""

import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional

from src.models.rst_model import RSTModel
from src.data.preprocessing import preprocess_cadence


class InferenceEngine:
    """
    End-to-end inference engine for RST.

    Loads a trained model and runs inference on raw cadence arrays,
    using a sliding window to extract overlapping snippets.

    Args:
        config_path: Path to YAML config file.
        checkpoint_path: Path to model checkpoint (.pth).
        device: Device string ('cuda' or 'cpu'). Auto-detects if None.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: str = None,
    ):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        m_cfg = self.config['model']
        d_cfg = self.config['data']
        i_cfg = self.config.get('inference', {})

        # Store inference parameters
        self.snippet_width = d_cfg.get('snippet_width', 1024)
        self.sliding_step = i_cfg.get('sliding_window_step',
                                       d_cfg.get('sliding_window_step', 512))
        self.batch_size = i_cfg.get('batch_size', 128)
        self.threshold = i_cfg.get('threshold', 0.5)
        self.attn_threshold = i_cfg.get('attn_threshold', 0.9)


        # Device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Build model
        self.model = RSTModel(
            label_dim=m_cfg['label_dim'],
            fstride=m_cfg['stride'],
            tstride=m_cfg['stride'],
            input_fdim=m_cfg['input_fdim'],
            input_tdim=m_cfg['input_tdim'],
            imagenet_pretrain=False,  # Weights from checkpoint
            model_size=m_cfg['model_size'],
            verbose=False,
        )

        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ Model loaded on {self.device}")
        print(f"   Snippet width: {self.snippet_width}, "
              f"Sliding step: {self.sliding_step}")



    def _extract_snippets(
        self,
        cadence: np.ndarray,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Extract overlapping snippets from a cadence using sliding window.
        Each snippet is normalised independently (sample-wise z-score).

        Args:
            cadence: Array of shape (6, 16, n_freq), raw values.

        Returns:
            List of (center_channel, spectrogram) tuples.
            Each spectrogram has shape (96, 1024), normalized.
        """
        n_freq = cadence.shape[2]
        half = self.snippet_width // 2
        snippets = []

        start_center = half
        end_center = n_freq - half

        center = start_center
        while center <= end_center:
            spec = preprocess_cadence(
                cadence=cadence,
                center_channel=center,
                snippet_width=self.snippet_width,
            )
            snippets.append((center, spec))
            center += self.sliding_step

        # Ensure the very last window is included (edge case)
        if snippets and snippets[-1][0] < end_center:
            spec = preprocess_cadence(
                cadence=cadence,
                center_channel=end_center,
                snippet_width=self.snippet_width,
            )
            snippets.append((end_center, spec))

        return snippets

    @torch.no_grad()
    def _batch_inference(
        self,
        spectrograms: np.ndarray,
    ) -> np.ndarray:
        """
        Run batch inference on an array of spectrograms.

        Args:
            spectrograms: Array of shape (N, 96, 1024).

        Returns:
            Probabilities array of shape (N,).
        """
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(spectrograms)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=(self.device.type == 'cuda'),
        )

        all_probs = []
        for (batch,) in loader:
            # Check for NaNs/Infs in the input (critical for debugging real HDF5 data)
            if not torch.isfinite(batch).all():
                n_nan = torch.isnan(batch).sum().item()
                n_inf = torch.isinf(batch).sum().item()
                print(f"  ⚠️  WARNING: Input batch contains {n_nan} NaNs and {n_inf} Infs! "
                      f"Replacing with 0 for this batch.")
                batch = torch.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)

            batch = batch.to(self.device)
            # Mixed precision removed to avoid NaNs on real SRT data overflow
            logits = self.model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0).flatten()


    def cluster_detections(
        self,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Group adjacent ETI snippet detections into distinct signal clusters.

        Two snippets belong to the same cluster if their center_channels
        differ by at most `sliding_step` (i.e. they are adjacent in the
        sliding window grid).  A snippet below threshold breaks the chain.

        Args:
            results: Raw per-snippet DataFrame from run_cadence.

        Returns:
            DataFrame with one row per cluster, sorted by peak_probability
            descending.  Columns:
            - center_channel: channel of the peak-probability snippet
            - peak_probability: max P(ETI) in the cluster
            - mean_probability: mean P(ETI) across cluster snippets
            - cluster_width: (last - first center_channel) + snippet_width
            - n_snippets: number of snippets in the cluster
            - freq_mhz: frequency at the peak channel (0 if unavailable)
        """
        # Keep only above-threshold snippets, sorted by position
        eti = results[results['classification'] == 'ETI'].copy()
        if eti.empty:
            return pd.DataFrame(columns=[
                'center_channel', 'peak_probability', 'mean_probability',
                'cluster_width', 'n_snippets', 'freq_mhz',
            ])

        eti = eti.sort_values('center_channel').reset_index(drop=True)

        # Connected-component grouping on the 1D grid
        clusters = []
        cluster_start = 0

        for i in range(1, len(eti)):
            gap = eti.loc[i, 'center_channel'] - eti.loc[i - 1, 'center_channel']
            if gap > self.sliding_step:
                clusters.append((cluster_start, i - 1))
                cluster_start = i
        clusters.append((cluster_start, len(eti) - 1))  # last cluster

        # Aggregate per cluster
        rows = []
        for start_idx, end_idx in clusters:
            chunk = eti.iloc[start_idx:end_idx + 1]
            peak_idx = chunk['probability'].idxmax()
            rows.append({
                'center_channel': int(chunk.loc[peak_idx, 'center_channel']),
                'peak_probability': float(chunk['probability'].max()),
                'mean_probability': float(chunk['probability'].mean()),
                'cluster_width': int(
                    chunk['center_channel'].iloc[-1]
                    - chunk['center_channel'].iloc[0]
                    + self.snippet_width
                ),
                'n_snippets': len(chunk),
                'freq_mhz': float(chunk.loc[peak_idx, 'freq_mhz']),
            })

        cluster_df = pd.DataFrame(rows)
        cluster_df = cluster_df.sort_values(
            'peak_probability', ascending=False,
        ).reset_index(drop=True)

        return cluster_df

    def run_cadence(
        self,
        cadence: np.ndarray,
        freq_start_mhz: float = 0.0,
        freq_resolution_mhz: float = 0.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run inference on a single cadence.

        Args:
            cadence: Array of shape (6, 16, n_freq), raw values.
            freq_start_mhz: Starting frequency in MHz (from HDF5 header).
            freq_resolution_mhz: Channel width in MHz (from HDF5 header).

        Returns:
            Tuple of (raw_results, clustered_results).
            raw_results: per-snippet DataFrame (center_channel, probability,
                         classification, freq_mhz).
            clustered_results: per-cluster DataFrame (center_channel,
                               peak_probability, mean_probability,
                               cluster_width, n_snippets, freq_mhz).
        """
        # 1. Extract overlapping snippets (sample-wise normalization)
        snippets = self._extract_snippets(cadence)
        if not snippets:
            print("  ⚠️  No snippets extracted (input too narrow?)")
            return pd.DataFrame(), pd.DataFrame()

        centers = np.array([s[0] for s in snippets])
        
        print(f"  Extracted {len(snippets)} overlapping snippets "
              f"(step={self.sliding_step})")

        # 2. Batch inference (streaming to save RAM)
        all_probs = []
        chunk_size = 1024 
        for i in tqdm(range(0, len(snippets), chunk_size), desc="    Batch Inference", leave=False):
            chunk_snippets = snippets[i:i+chunk_size]
            chunk_specs = np.array([s[1] for s in chunk_snippets])
            probs_chunk = self._batch_inference(chunk_specs)
            all_probs.append(probs_chunk)
        
        probs = np.concatenate(all_probs)

        # 3. Build raw results DataFrame
        results = pd.DataFrame({
            'center_channel': centers,
            'probability': probs,
            'classification': ['ETI' if p >= self.threshold else 'RFI'
                               for p in probs],
        })

        # Add frequency info if available
        if freq_start_mhz != 0.0 and freq_resolution_mhz != 0.0:
            results['freq_mhz'] = (
                freq_start_mhz + centers * freq_resolution_mhz
            )
        else:
            results['freq_mhz'] = 0.0

        # Sort by probability descending
        results = results.sort_values('probability', ascending=False)
        results = results.reset_index(drop=True)

        # 4. Cluster adjacent detections
        clusters = self.cluster_detections(results)

        # 5. Summary
        n_eti = (results['classification'] == 'ETI').sum()
        n_clusters = len(clusters)
        n_high = (clusters['peak_probability'] >= self.attn_threshold).sum() if not clusters.empty else 0

        print(f"\n  Raw detections: {n_eti} ETI snippets (threshold={self.threshold})")
        print(f"  Clustered:      {n_clusters} distinct signals "
              f"({n_high} high-confidence, p≥{self.attn_threshold})")

        if not clusters.empty:
            print("\n  Top clusters:")
            print(clusters[['center_channel', 'peak_probability',
                            'n_snippets', 'cluster_width']].head(5).to_string(index=False))

        return results, clusters

    def get_snippet_spectrogram(
        self,
        cadence: np.ndarray,
        center_channel: int,
    ) -> np.ndarray:
        """
        Extract and preprocess a single snippet for visualization.

        Args:
            cadence: Array of shape (6, 16, n_freq).
            center_channel: Center channel for the snippet.

        Returns:
            Array of shape (96, 1024), normalized (sample-wise).
        """
        spec = preprocess_cadence(
            cadence=cadence,
            center_channel=center_channel,
            snippet_width=self.snippet_width,
        )
        return spec
