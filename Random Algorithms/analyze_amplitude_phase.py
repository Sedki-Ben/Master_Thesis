#!/usr/bin/env python3
"""
Script to analyze the extracted amplitude and phase data.
"""

import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_file(file_path, max_samples=100):
    """Analyze amplitude and phase data from a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            rssi_values = []
            amplitude_data = []
            phase_data = []
            
            sample_count = 0
            for row in reader:
                if sample_count >= max_samples:
                    break
                
                rssi_values.append(float(row['rssi']))
                
                # Extract amplitudes (52 values)
                amps = [float(row[f'amplitude_{i}']) for i in range(52)]
                amplitude_data.append(amps)
                
                # Extract phases (52 values)
                phases = [float(row[f'phase_{i}']) for i in range(52)]
                phase_data.append(phases)
                
                sample_count += 1
            
            amplitude_data = np.array(amplitude_data)
            phase_data = np.array(phase_data)
            
            print(f"File: {file_path.name}")
            print(f"Samples analyzed: {len(rssi_values)}")
            print(f"RSSI range: {min(rssi_values):.1f} to {max(rssi_values):.1f} dBm")
            print(f"Amplitude range: {amplitude_data.min():.3f} to {amplitude_data.max():.3f}")
            print(f"Phase range: {phase_data.min():.3f} to {phase_data.max():.3f} radians")
            print(f"Average amplitude per subcarrier: {amplitude_data.mean(axis=0).mean():.3f}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

def main():
    output_path = Path("Amplitude Phase Data")
    csv_files = list(output_path.glob("*.csv"))[:5]  # Analyze first 5 files
    
    print("Amplitude & Phase Data Analysis")
    print("=" * 60)
    
    for file_path in csv_files:
        analyze_file(file_path)

if __name__ == "__main__":
    main()
