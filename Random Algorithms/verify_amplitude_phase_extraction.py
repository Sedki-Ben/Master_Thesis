#!/usr/bin/env python3
"""
Script to verify the amplitude and phase extraction by comparing with manual calculations.
"""

import csv
import json
import math
from pathlib import Path

def manual_calculation(csi_data):
    """
    Manually calculate amplitude and phase to verify the extraction.
    
    Args:
        csi_data (list): CSI data with 104 values (imag, real, imag, real...)
    
    Returns:
        tuple: (amplitudes, phases)
    """
    amplitudes = []
    phases = []
    
    print("Manual calculation for first 5 subcarriers:")
    print("Subcarrier | Imag | Real | Amplitude | Phase")
    print("-" * 50)
    
    for i in range(0, min(10, len(csi_data)), 2):  # First 5 pairs
        imag = csi_data[i]
        real = csi_data[i + 1]
        
        amplitude = math.sqrt(real * real + imag * imag)
        phase = math.atan2(imag, real)
        
        amplitudes.append(amplitude)
        phases.append(phase)
        
        subcarrier = i // 2
        print(f"{subcarrier:10d} | {imag:4d} | {real:4d} | {amplitude:9.6f} | {phase:9.6f}")
    
    return amplitudes, phases

def verify_file(processed_file, amplitude_phase_file, sample_limit=2):
    """
    Verify the extraction by comparing processed and amplitude/phase files.
    
    Args:
        processed_file (str): Path to processed file
        amplitude_phase_file (str): Path to amplitude/phase file
        sample_limit (int): Number of samples to verify
    """
    print(f"\n🔍 Verifying: {Path(processed_file).name}")
    print("=" * 70)
    
    try:
        # Read processed data
        with open(processed_file, 'r', encoding='utf-8') as f:
            processed_reader = csv.DictReader(f)
            processed_samples = list(processed_reader)
        
        # Read amplitude/phase data
        with open(amplitude_phase_file, 'r', encoding='utf-8') as f:
            ap_reader = csv.DictReader(f)
            ap_samples = list(ap_reader)
        
        for i in range(min(sample_limit, len(processed_samples), len(ap_samples))):
            print(f"\n--- Sample {i+1} ---")
            
            # Get data from processed file
            processed_row = processed_samples[i]
            csi_data = json.loads(processed_row['data'].strip('"'))
            rssi = processed_row['rssi']
            
            # Get data from amplitude/phase file
            ap_row = ap_samples[i]
            extracted_rssi = ap_row['rssi']
            
            print(f"RSSI verification: {rssi} == {extracted_rssi} -> {'✅' if rssi == extracted_rssi else '❌'}")
            print(f"CSI data length: {len(csi_data)} (expected: 104)")
            
            if len(csi_data) == 104:
                # Manual calculation for verification
                manual_amps, manual_phases = manual_calculation(csi_data)
                
                print("\\nVerifying first 5 extracted values:")
                print("Subcarrier | Manual Amp | Extracted Amp | Manual Phase | Extracted Phase | Match")
                print("-" * 80)
                
                all_match = True
                for j in range(5):
                    extracted_amp = float(ap_row[f'amplitude_{j}'])
                    extracted_phase = float(ap_row[f'phase_{j}'])
                    
                    amp_match = abs(manual_amps[j] - extracted_amp) < 1e-5
                    phase_match = abs(manual_phases[j] - extracted_phase) < 1e-5
                    match = amp_match and phase_match
                    all_match = all_match and match
                    
                    print(f"{j:10d} | {manual_amps[j]:10.6f} | {extracted_amp:13.6f} | {manual_phases[j]:12.6f} | {extracted_phase:15.6f} | {'✅' if match else '❌'}")
                
                print(f"\\nOverall verification: {'✅ PASSED' if all_match else '❌ FAILED'}")
            
    except Exception as e:
        print(f"Error during verification: {e}")

def main():
    """Main verification function."""
    
    print("Amplitude & Phase Extraction Verification")
    print("=" * 70)
    
    processed_dir = Path("Processed Labor Data")
    ap_dir = Path("Amplitude Phase Data")
    
    if not processed_dir.exists():
        print(f"Error: {processed_dir} not found!")
        return
    
    if not ap_dir.exists():
        print(f"Error: {ap_dir} not found!")
        return
    
    # Verify a few sample files
    sample_files = ["0,0.csv", "1,0.csv"]
    
    for filename in sample_files:
        processed_file = processed_dir / filename
        ap_file = ap_dir / filename
        
        if processed_file.exists() and ap_file.exists():
            verify_file(processed_file, ap_file, sample_limit=1)
        else:
            print(f"⚠️  Skipping {filename} - file not found")
    
    print(f"\\n✅ Verification complete!")

if __name__ == "__main__":
    main()
