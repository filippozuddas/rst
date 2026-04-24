#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — SRT Cluster Band Cadence Scanner

Recursively scans a directory (the SRT cluster storage) to find all
complete cadences (6 ON/OFF observations) in a selected frequency band.
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.background_extractor import DatasetBuilder

BAND_MAPPING = {
    'C': '6GHz',
    'K': '18GHz'
}

BAND_DISPLAY = {
    'C': 'C-Band (4-8 GHz)',
    'K': 'K-Band (~18 GHz)'
}

def main():
    parser = argparse.ArgumentParser(
        description="Recursively scan SRT cluster for complete cadences in a specific band"
    )
    parser.add_argument('--scan-dir', '-s', required=True,
                        help='Root directory of the SRT cluster to scan')
    parser.add_argument('--band', '-b', choices=['C', 'K'], default='C',
                        help='Frequency band to search for (default: C)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file to save the list of found cadences (default: <band>_band_cadences_list.txt)')

    args = parser.parse_args()
    
    output_file = args.output or f"{args.band.lower()}_band_cadences_list.txt"
    target_band_key = BAND_MAPPING[args.band]
    display_name = BAND_DISPLAY[args.band]
    
    print(f"\n{'='*60}")
    print(f"SRT CLUSTER SCANNER: {display_name}")
    print(f"{'='*60}")
    print(f"Scanning directory: {args.scan_dir}")
    print("This may take a while depending on the cluster size...")

    # We only need the DatasetBuilder for its scanning and parsing capabilities
    builder = DatasetBuilder()
    
    # 1. Scan for all HDF5 files recursively
    files = builder.scan_directory(args.scan_dir, recursive=True)
    if not files:
        print("No HDF5 files found. Exiting.")
        return

    # 2. Group into cadences and parse headers
    print("Grouping files into cadences and checking frequency headers...")
    builder.group_into_cadences(files)
    
    # 3. Filter for the requested band
    by_band = builder.get_cadences_by_band(target_band_key)
    found_cadences = by_band.get(target_band_key, [])
    
    if not found_cadences:
        print(f"\nNo complete {display_name} cadences found.")
        return
        
    print(f"\n✅ Found {len(found_cadences)} complete {display_name} cadences!")
    
    # 4. Output the results
    with open(output_file, 'w') as f:
        f.write(f"# {display_name} Cadences Found in {args.scan_dir}\n")
        f.write(f"# Total: {len(found_cadences)}\n\n")
        for i, cadence in enumerate(found_cadences, 1):
            target = cadence.target_name
            date = cadence.date
            f.write(f"[{i}] Target: {target} | Date: {date} | Snippets: {cadence.n_snippets}\n")
            for file in cadence.files:
                f.write(f"    {file}\n")
            f.write("\n")
            
    print(f"List saved to: {output_file}")

    # Optionally print a small sample
    print("\nSample cadences:")
    for c in found_cadences[:5]:
        print(f"  - {c.target_name} ({c.date}): {c.n_snippets} potential snippets")
    if len(found_cadences) > 5:
        print(f"  ... and {len(found_cadences) - 5} more.")

if __name__ == '__main__':
    main()
