#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Dataset Builder

Extracts RAW background snippets from HDF5 observation files
for use with CadenceGenerator in training.

Telescope-agnostic: supports SRT, GBT, and any GUPPI/filterbank-format data
with ON/OFF cadence structure. Handles multiple frequency bands
and train/inference splitting.
"""

import os
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import json
import h5py
from tqdm import tqdm
import warnings


# ---- Default frequency band configuration ----
# Users can extend this by passing custom band_config to DatasetBuilder.
DEFAULT_BAND_CONFIG = {
    '6GHz': {
        'name': 'C-band (4-8 GHz)',
        'f_min': 4000,
        'f_max': 8000,
    },
    '18GHz': {
        'name': 'K-band (~18 GHz)',
        'f_min': 17000,
        'f_max': 19000,
    },
    '1.4GHz': {
        'name': 'L-band (~1.4 GHz)',
        'f_min': 1000,
        'f_max': 2000,
    },
}

# ---- Filename parsing patterns ----
# Each pattern extracts (target_name, obs_type) from different telescope formats.
FILENAME_PATTERNS = [
    # SRT format: ...TIC82452140_ON_0001.0000.h5
    re.compile(r'(TIC\d+)_(ON|OFF)'),
    # GBT format: ...HIP12345_ON_0001.h5 or ...GJ411_ON_0001.h5
    re.compile(r'(HIP\d+|GJ\d+[A-Za-z]?)_(ON|OFF)'),
    # Generic: any alphanumeric target followed by _ON or _OFF
    re.compile(r'([A-Za-z0-9_]+?)_(ON|OFF)(?:_|\.)'),
]


@dataclass
class CadenceInfo:
    """Information about a cadence (6 ON/OFF files)."""
    target_name: str
    date: str
    files: List[Path] = field(default_factory=list)
    n_channels: int = 0
    freq_start: float = 0.0
    freq_end: float = 0.0
    freq_band: str = 'unknown'

    @property
    def is_complete(self) -> bool:
        """Check if cadence has all 6 files."""
        return len(self.files) == 6

    @property
    def n_snippets(self) -> int:
        """Estimated number of snippets at given width."""
        return self.n_channels // DatasetBuilder.SNIPPET_WIDTH if self.n_channels > 0 else 0


class DatasetBuilder:
    """
    Build training datasets from radio telescope observation files.

    Telescope-agnostic: works with any GUPPI/filterbank HDF5 data
    organized in ON/OFF cadences.

    Extracts RAW 1024-channel backgrounds for use with CadenceGenerator.
    Signal injection and preprocessing happen in the training pipeline.
    """

    SNIPPET_WIDTH = 1024  # Frequency channels per snippet (native resolution)

    def __init__(self,
                 output_dir: str = "data/training",
                 band_config: Optional[Dict] = None,
                 snippet_width: int = 1024):
        """
        Args:
            output_dir: Directory to save extracted backgrounds.
            band_config: Custom frequency band configuration.
                         If None, uses DEFAULT_BAND_CONFIG.
            snippet_width: Frequency channels per snippet (default 1024).
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cadences: Dict[str, CadenceInfo] = {}
        self.band_config = band_config or DEFAULT_BAND_CONFIG
        DatasetBuilder.SNIPPET_WIDTH = snippet_width

    def scan_directory(self, directory: str, recursive: bool = True) -> List[Path]:
        """Scan directory for HDF5 files (.h5 and .hdf5)."""
        directory = Path(directory)
        files = []
        for ext in ['h5', 'hdf5']:
            pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
            files.extend(directory.glob(pattern))
        print(f"Found {len(files)} HDF5 files in {directory}")
        return files

    def parse_filename(self, filepath: Path) -> Optional[Dict]:
        """
        Parse observation filename to extract metadata.

        Supports multiple telescope formats (SRT, GBT, generic).
        Returns None if the filename doesn't match any known pattern.
        """
        name = filepath.stem

        # Try each pattern until one matches
        target = None
        obs_type = None
        for pattern in FILENAME_PATTERNS:
            match = pattern.search(name)
            if match:
                target = match.group(1)
                obs_type = match.group(2)
                break

        if target is None:
            return None

        # Extract timestamp from GUPPI header in filename
        time_match = re.search(r'_(\d{5})_(\d+)_', name)
        mjd = time_match.group(1) if time_match else "unknown"

        # Extract date from parent directory if available
        parent = filepath.parent.name
        date_match = re.search(r'(\d{8})', parent)
        date = date_match.group(1) if date_match else mjd

        return {
            'target': target,
            'obs_type': obs_type,
            'mjd': mjd,
            'date': date,
            'filepath': filepath
        }

    def group_into_cadences(self, files: List[Path]) -> Dict[str, CadenceInfo]:
        """Group files into cadences by target name with timestamp-based validation."""

        def extract_timestamp(filepath: Path) -> int:
            """Extract timestamp from GUPPI filename for sorting."""
            match = re.search(r'guppi_(\d+)_(\d+)_', filepath.name)
            if match:
                return int(match.group(1)) * 1000000 + int(match.group(2))
            return 0

        groups = defaultdict(list)

        for f in files:
            info = self.parse_filename(f)
            if info:
                # Key by target + date + directory (separate different frequency observations)
                key = f"{info['target']}_{info['date']}_{f.parent}"
                info['timestamp'] = extract_timestamp(f)
                groups[key].append(info)

        cadences = {}
        invalid_count = 0

        for key, file_infos in groups.items():
            file_infos.sort(key=lambda x: x['timestamp'])

            if len(file_infos) != 6:
                invalid_count += 1
                continue

            # Validate ON/OFF pattern
            expected_pattern = ['ON', 'OFF', 'ON', 'OFF', 'ON', 'OFF']
            actual_pattern = [f['obs_type'] for f in file_infos]

            if actual_pattern != expected_pattern:
                invalid_count += 1
                continue

            cadence_files = [f['filepath'] for f in file_infos]

            cadence = CadenceInfo(
                target_name=file_infos[0]['target'],
                date=file_infos[0]['date'],
                files=cadence_files
            )

            if cadence.is_complete:
                try:
                    with h5py.File(cadence_files[0], 'r') as f:
                        header = dict(f['data'].attrs)
                        cadence.n_channels = header.get('nchans', 0)
                        cadence.freq_start = header.get('fch1', 0.0)

                        for band_name, config in self.band_config.items():
                            if config['f_min'] <= cadence.freq_start <= config['f_max']:
                                cadence.freq_band = band_name
                                break
                except Exception:
                    continue

            cadences[key] = cadence

        if invalid_count > 0:
            print(f"  (Skipped {invalid_count} incomplete/invalid cadences)")

        self.cadences = cadences
        return cadences

    def get_cadences_by_band(self, band: str = None) -> Dict[str, List[CadenceInfo]]:
        """Get cadences grouped by frequency band."""
        complete = [c for c in self.cadences.values() if c.is_complete]

        by_band = defaultdict(list)
        for c in complete:
            by_band[c.freq_band].append(c)

        if band:
            return {band: by_band.get(band, [])}
        return dict(by_band)

    def print_cadence_summary(self):
        """Print summary of found cadences with band and telescope breakdown."""
        complete = [c for c in self.cadences.values() if c.is_complete]

        print(f"\n{'='*60}")
        print("CADENCE SUMMARY")
        print(f"{'='*60}")
        print(f"  Complete cadences: {len(complete)}")

        # Band breakdown
        by_band = self.get_cadences_by_band()
        print(f"\n  By frequency band:")
        for band_name, config in self.band_config.items():
            band_cadences = by_band.get(band_name, [])
            if band_cadences:
                print(f"    {config['name']}: {len(band_cadences)} cadences")

        unknown = by_band.get('unknown', [])
        if unknown:
            print(f"    Unknown band: {len(unknown)} cadences")

        if complete:
            total_snippets = sum(c.n_snippets for c in complete)
            print(f"\n  Total potential snippets: {total_snippets:,}")
            print(f"  Snippet width: {self.SNIPPET_WIDTH} channels")

            print(f"\n  Sample cadences:")
            for c in complete[:5]:
                print(f"    - {c.target_name} ({c.freq_band}): {c.n_snippets:,} snippets")
            if len(complete) > 5:
                print(f"    ... and {len(complete) - 5} more")

    def extract_backgrounds(self,
                           cadence: CadenceInfo,
                           n_snippets: int = None,
                           random_sample: bool = True) -> np.ndarray:
        """
        Extract RAW background snippets from a cadence.
        Returns array of shape (n_snippets, 6, 16, SNIPPET_WIDTH) — RAW, not normalized.
        """
        from blimpy import Waterfall

        if not cadence.is_complete:
            raise ValueError(f"Cadence {cadence.target_name} is not complete")

        cadence_data = []
        for i, filepath in enumerate(cadence.files):
            print(f"\n    Loading file {i+1}/6: {filepath.name}...", end=" ", flush=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wf = Waterfall(str(filepath))
                data = wf.data.squeeze()
                cadence_data.append(data)
                print(f"✓ ({data.shape})")
            except OSError as e:
                print(f"❌ FAILED")
                if "truncated file" in str(e).lower():
                    raise OSError(f"File is truncated or corrupt: {filepath}") from e
                raise OSError(f"Could not open HDF5 file {filepath}: {e}") from e
            except Exception as e:
                print(f"❌ ERROR")
                raise RuntimeError(f"Unexpected error loading {filepath}: {e}") from e

        cadence_array = np.stack(cadence_data, axis=0)
        n_freq = cadence_array.shape[2]
        total_snippets = n_freq // self.SNIPPET_WIDTH

        if n_snippets is None:
            n_snippets = total_snippets

        if random_sample and n_snippets < total_snippets:
            indices = np.random.choice(total_snippets, n_snippets, replace=False)
            indices.sort()
        else:
            indices = range(min(n_snippets, total_snippets))

        snippets = []
        for idx in indices:
            start = idx * self.SNIPPET_WIDTH
            end = (idx + 1) * self.SNIPPET_WIDTH
            snippet = cadence_array[:, :, start:end]
            snippets.append(snippet)

        return np.array(snippets, dtype=np.float32)

    def build_training_dataset(self,
                               cadences: List[CadenceInfo] = None,
                               snippets_per_cadence: int = 500,
                               max_total_snippets: int = 20000,
                               output_name: str = "backgrounds") -> str:
        """
        Build a training dataset from multiple cadences.
        Saves RAW backgrounds for use with CadenceGenerator.
        Returns path to saved dataset.
        """
        if cadences is None:
            cadences = [c for c in self.cadences.values() if c.is_complete]

        if not cadences:
            raise ValueError("No complete cadences found")

        print(f"\n{'='*60}")
        print("BUILDING RAW TRAINING DATASET")
        print(f"{'='*60}")
        print(f"  Cadences: {len(cadences)}")
        print(f"  Snippets per cadence: {snippets_per_cadence}")
        print(f"  Max total: {max_total_snippets}")
        print(f"  Output shape: (N, 6, 16, {self.SNIPPET_WIDTH})")

        all_snippets = []
        metadata = []

        for cadence in tqdm(cadences, desc="Processing"):
            try:
                n_to_extract = min(snippets_per_cadence, cadence.n_snippets)
                if n_to_extract == 0:
                    continue

                snippets = self.extract_backgrounds(
                    cadence,
                    n_snippets=n_to_extract,
                    random_sample=True
                )

                all_snippets.append(snippets)
                metadata.extend([{
                    'target': cadence.target_name,
                    'date': cadence.date
                }] * len(snippets))

                if sum(len(s) for s in all_snippets) >= max_total_snippets:
                    break

            except Exception as e:
                print(f"  Error: {cadence.target_name}: {e}")
                continue

        if not all_snippets:
            raise ValueError("No snippets extracted")

        dataset = np.concatenate(all_snippets, axis=0)

        if len(dataset) > max_total_snippets:
            indices = np.random.choice(len(dataset), max_total_snippets, replace=False)
            dataset = dataset[indices]

        # Save
        output_path = self.output_dir / f"{output_name}.npz"
        np.savez_compressed(output_path, backgrounds=dataset, n_samples=len(dataset))

        meta_path = self.output_dir / f"{output_name}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'n_samples': len(dataset),
                'n_cadences': len(cadences),
                'shape': list(dataset.shape),
                'fchans': self.SNIPPET_WIDTH,
                'targets': list(set(m['target'] for m in metadata))
            }, f, indent=2)

        print(f"\n✅ Dataset saved:")
        print(f"   {output_path} ({dataset.shape})")
        print(f"   {meta_path}")

        return str(output_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="RST — Extract backgrounds for training (multi-telescope, multi-band)"
    )
    parser.add_argument('--scan', '-s', nargs='+', required=True,
                        help='Directories to scan for HDF5 files')
    parser.add_argument('--output', '-o', default='data/training',
                        help='Output directory')
    parser.add_argument('--snippet-width', type=int, default=1024,
                        help='Frequency channels per snippet (default: 1024)')
    parser.add_argument('--snippets-per-cadence', '-n', type=int, default=500,
                        help='Max snippets per cadence')
    parser.add_argument('--max-snippets', '-m', type=int, default=15000,
                        help='Max total snippets per band')
    parser.add_argument('--name', default='backgrounds',
                        help='Output dataset name prefix')
    parser.add_argument('--band', '-b', choices=['6GHz', '18GHz', '1.4GHz', 'all'],
                        default='all', help='Frequency band to process')
    parser.add_argument('--training-cadences', '-t', type=int, default=None,
                        help='Number of cadences for training (rest for inference)')
    parser.add_argument('--list-only', action='store_true',
                        help='Only list cadences, do not extract')

    args = parser.parse_args()

    builder = DatasetBuilder(
        output_dir=args.output,
        snippet_width=args.snippet_width
    )

    all_files = []
    for directory in args.scan:
        all_files.extend(builder.scan_directory(directory))

    print(f"\nTotal files: {len(all_files)}")

    builder.group_into_cadences(all_files)
    builder.print_cadence_summary()

    if args.list_only:
        return

    bands_to_process = [args.band] if args.band != 'all' else list(builder.band_config.keys())

    for band_name in bands_to_process:
        by_band = builder.get_cadences_by_band(band_name)
        band_cadences = by_band.get(band_name, [])

        if not band_cadences:
            print(f"\n⚠️  No cadences found for {band_name}")
            continue

        print(f"\n{'='*60}")
        print(f"PROCESSING: {builder.band_config[band_name]['name']}")
        print(f"{'='*60}")
        print(f"  Cadences: {len(band_cadences)}")

        # Split train/inference if requested
        if args.training_cadences and args.training_cadences < len(band_cadences):
            training_cadences = band_cadences[:args.training_cadences]
            inference_cadences = band_cadences[args.training_cadences:]

            print(f"  Training: {len(training_cadences)} cadences")
            print(f"  Inference: {len(inference_cadences)} cadences")

            inference_path = builder.output_dir / f"inference_cadences_{band_name}.txt"
            with open(inference_path, 'w') as f:
                for c in inference_cadences:
                    files_str = ','.join(str(fp) for fp in c.files)
                    f.write(f"{c.target_name}|{files_str}\n")
            print(f"  Saved: {inference_path}")
        else:
            training_cadences = band_cadences

        output_name = f"{args.name}_{band_name}"
        builder.build_training_dataset(
            cadences=training_cadences,
            snippets_per_cadence=args.snippets_per_cadence,
            max_total_snippets=args.max_snippets,
            output_name=output_name
        )

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
