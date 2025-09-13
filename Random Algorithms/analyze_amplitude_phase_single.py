#!/usr/bin/env python3
"""
Script to analyze the extracted amplitude and phase data (single column format).
"""

import csv
import json
import numpy as np
from pathlib import Path

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
                
                # Parse amplitude and phase arrays
                amplitudes = json.loads(row['amplitude'])
                phases = json.loads(row['phase'])
                
                amplitude_data.append(amplitudes)
                phase_data.append(phases)
                
                sample_count += 1
            
            amplitude_data = np.array(amplitude_data)
            phase_data = np.array(phase_data)
            
            print(f"File: {file_path.name}")
            print(f"Samples analyzed: {len(rssi_values)}")
            print(f"RSSI range: {min(rssi_values):.1f} to {max(rssi_values):.1f} dBm")
            print(f"Amplitude array shape: {amplitude_data.shape} (samples x subcarriers)")
            print(f"Phase array shape: {phase_data.shape} (samples x subcarriers)")
            print(f"Amplitude range: {amplitude_data.min():.3f} to {amplitude_data.max():.3f}")
            print(f"Phase range: {phase_data.min():.3f} to {phase_data.max():.3f} radians")
            print(f"Average amplitude per subcarrier: {amplitude_data.mean():.3f}")
            print(f"First sample amplitudes (first 10): {amplitude_data[0][:10].tolist()}")
            print(f"First sample phases (first 10): {phase_data[0][:10].tolist()}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

def main():
    output_path = Path("Amplitude Phase Data Single")
    csv_files = list(output_path.glob("*.csv"))[:3]  # Analyze first 3 files
    
    print("Amplitude & Phase Data Analysis (Single Column Format)")
    print("=" * 60)
    
    for file_path in csv_files:
        analyze_file(file_path)

if __name__ == "__main__":
    main()
