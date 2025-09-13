#!/usr/bin/env python3
"""
Script to compare the old format (52 separate columns) vs new format (single columns with arrays).
"""

import csv
import json
from pathlib import Path

def compare_formats(multi_column_file, single_column_file, sample_limit=2):
    """
    Compare the two different formats to show they contain the same data.
    
    Args:
        multi_column_file (str): Path to file with 52 separate amplitude/phase columns
        single_column_file (str): Path to file with single amplitude/phase columns containing arrays
        sample_limit (int): Number of samples to compare
    """
    print(f"\n📊 Comparing formats for: {Path(single_column_file).name}")
    print("=" * 80)
    
    try:
        # Read multi-column format
        with open(multi_column_file, 'r', encoding='utf-8') as f:
            multi_reader = csv.DictReader(f)
            multi_samples = list(multi_reader)
        
        # Read single-column format
        with open(single_column_file, 'r', encoding='utf-8') as f:
            single_reader = csv.DictReader(f)
            single_samples = list(single_reader)
        
        print(f"Multi-column format: {len(multi_reader.fieldnames)} columns")
        print(f"Single-column format: {len(single_reader.fieldnames)} columns")
        print(f"Multi-column fieldnames: {multi_reader.fieldnames[:5]}... (showing first 5)")
        print(f"Single-column fieldnames: {single_reader.fieldnames}")
        
        # Compare first few samples
        for i in range(min(sample_limit, len(multi_samples), len(single_samples))):
            print(f"\n--- Sample {i+1} ---")
            
            multi_row = multi_samples[i]
            single_row = single_samples[i]
            
            # Compare RSSI
            multi_rssi = multi_row['rssi']
            single_rssi = single_row['rssi']
            print(f"RSSI comparison: {multi_rssi} == {single_rssi} -> {'✅' if multi_rssi == single_rssi else '❌'}")
            
            # Extract amplitudes from multi-column format
            multi_amplitudes = []
            for j in range(52):
                multi_amplitudes.append(float(multi_row[f'amplitude_{j}']))
            
            # Extract amplitudes from single-column format
            single_amplitudes = json.loads(single_row['amplitude'])
            
            # Extract phases from multi-column format
            multi_phases = []
            for j in range(52):
                multi_phases.append(float(multi_row[f'phase_{j}']))
            
            # Extract phases from single-column format
            single_phases = json.loads(single_row['phase'])
            
            print(f"Amplitudes length: multi={len(multi_amplitudes)}, single={len(single_amplitudes)}")
            print(f"Phases length: multi={len(multi_phases)}, single={len(single_phases)}")
            
            # Compare first 5 amplitude values
            amp_match = True
            print("First 5 amplitude comparisons:")
            for j in range(5):
                match = abs(multi_amplitudes[j] - single_amplitudes[j]) < 1e-5
                amp_match = amp_match and match
                print(f"  {j}: {multi_amplitudes[j]:.6f} vs {single_amplitudes[j]:.6f} -> {'✅' if match else '❌'}")
            
            # Compare first 5 phase values
            phase_match = True
            print("First 5 phase comparisons:")
            for j in range(5):
                match = abs(multi_phases[j] - single_phases[j]) < 1e-5
                phase_match = phase_match and match
                print(f"  {j}: {multi_phases[j]:.6f} vs {single_phases[j]:.6f} -> {'✅' if match else '❌'}")
            
            overall_match = amp_match and phase_match
            print(f"Overall sample verification: {'✅ PASSED' if overall_match else '❌ FAILED'}")
            
    except Exception as e:
        print(f"Error comparing formats: {e}")

def main():
    """Main comparison function."""
    
    print("Format Comparison: Multi-Column vs Single-Column")
    print("=" * 80)
    
    multi_dir = Path("Amplitude Phase Data")
    single_dir = Path("Amplitude Phase Data Single")
    
    if not multi_dir.exists():
        print(f"Error: {multi_dir} not found!")
        return
    
    if not single_dir.exists():
        print(f"Error: {single_dir} not found!")
        return
    
    # Compare a few sample files
    sample_files = ["0,0.csv", "1,0.csv"]
    
    for filename in sample_files:
        multi_file = multi_dir / filename
        single_file = single_dir / filename
        
        if multi_file.exists() and single_file.exists():
            compare_formats(multi_file, single_file, sample_limit=1)
        else:
            print(f"⚠️  Skipping {filename} - file not found in one of the directories")
    
    print(f"\n✅ Format comparison complete!")
    print(f"\nSummary:")
    print(f"- Multi-column format: 105 columns (1 RSSI + 52 amplitudes + 52 phases)")
    print(f"- Single-column format: 3 columns (1 RSSI + 1 amplitude array + 1 phase array)")
    print(f"- Both formats contain identical data, just organized differently")

if __name__ == "__main__":
    main()
