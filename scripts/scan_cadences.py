#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RST — Full Cadence Scanner

Recursively scans a directory to find all complete cadences
and reports their exact frequency ranges and statistics.
Useful for analyzing the data distribution before dataset creation.
"""

import argparse
from pathlib import Path
import sys
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.background_extractor import DatasetBuilder

def format_freq(freq_mhz):
    """Format frequency in MHz to a readable string."""
    if freq_mhz >= 1000:
        return f"{freq_mhz / 1000:.2f} GHz"
    return f"{freq_mhz:.1f} MHz"

def main():
    parser = argparse.ArgumentParser(
        description="Recursively scan directory for all complete cadences and their exact frequencies"
    )
    parser.add_argument('--scan-dir', '-s', required=True, nargs='+',
                        help='Root directory/directories to scan')
    parser.add_argument('--output', '-o', default='cadence_scan_report.txt',
                        help='Output file to save the report (default: cadence_scan_report.txt)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("RST - FULL CADENCE FREQUENCY SCANNER")
    print(f"{'='*60}")
    print(f"Scanning directories: {args.scan_dir}")
    print("This may take a while depending on the cluster size...")

    builder = DatasetBuilder()
    
    # Scan all directories
    all_files = []
    for d in args.scan_dir:
        all_files.extend(builder.scan_directory(d, recursive=True))
        
    if not all_files:
        print("No HDF5 files found. Exiting.")
        return

    print("Grouping files into cadences and parsing HDF5 headers...")
    builder.group_into_cadences(all_files)
    
    complete_cadences = [c for c in builder.cadences.values() if c.is_complete]
    
    if not complete_cadences:
        print("\nNo complete cadences (6 ON/OFF files) found.")
        return
        
    print(f"\n✅ Found {len(complete_cadences)} complete cadences!")
    
    # Group by Frequency Range (rounded to nearest 100 MHz for grouping)
    by_freq = defaultdict(list)
    for c in complete_cadences:
        # Round to nearest 0.1 GHz (100 MHz) for logical grouping
        freq_bin = round(c.freq_start / 100) * 100 
        by_freq[freq_bin].append(c)

    # Sort frequency bins
    sorted_bins = sorted(by_freq.keys())

    # Write report
    with open(args.output, 'w') as f:
        f.write(f"RST CADENCE SCAN REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total Complete Cadenze: {len(complete_cadences)}\n\n")
        
        f.write("DISTRIBUZIONE FREQUENZE:\n")
        for freq_bin in sorted_bins:
            cads = by_freq[freq_bin]
            f.write(f"  - ~{format_freq(freq_bin)}: {len(cads)} cadenze\n")
        f.write(f"\n{'='*60}\n\n")
        
        for freq_bin in sorted_bins:
            cads = by_freq[freq_bin]
            f.write(f"=== Range Frequenza: ~{format_freq(freq_bin)} ({len(cads)} cadenze) ===\n")
            for c in cads:
                f.write(f"Target: {c.target_name} | Date: {c.date} | Freq: {format_freq(c.freq_start)} | Snippets: {c.get_n_snippets(builder.snippet_width)}\n")
                f.write(f"Path: {c.files[0].parent}\n\n")
            
    print(f"Report completo salvato in: {args.output}")
    print("\nSommario Frequenze:")
    for freq_bin in sorted_bins:
        print(f"  ~{format_freq(freq_bin)}: {len(by_freq[freq_bin])} cadenze")

if __name__ == '__main__':
    main()
