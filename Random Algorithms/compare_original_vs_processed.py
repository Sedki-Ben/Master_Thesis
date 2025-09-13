#!/usr/bin/env python3
"""
Script to compare original vs processed CSI data to show the transformation.
"""

import csv
import json
from pathlib import Path

def compare_files(original_file, processed_file, sample_limit=3):
    """
    Compare original and processed files to show the transformation.
    
    Args:
        original_file (str): Path to original file
        processed_file (str): Path to processed file
        sample_limit (int): Number of samples to show
    """
    print(f"\n📊 Comparing: {Path(original_file).name}")
    print("=" * 80)
    
    try:
        # Read original data
        with open(original_file, 'r', encoding='utf-8') as f:
            original_reader = csv.DictReader(f)
            original_samples = list(original_reader)
        
        # Read processed data
        with open(processed_file, 'r', encoding='utf-8') as f:
            processed_reader = csv.DictReader(f)
            processed_samples = list(processed_reader)
        
        print(f"Original samples: {len(original_samples)}")
        print(f"Processed samples: {len(processed_samples)}")
        
        # Compare first few samples
        for i in range(min(sample_limit, len(original_samples), len(processed_samples))):
            print(f"\n--- Sample {i+1} (ID: {original_samples[i].get('id', 'N/A')}) ---")
            
            # Parse original CSI data
            orig_data_str = original_samples[i].get('data', '')
            orig_csi = json.loads(orig_data_str.strip('"'))
            
            # Parse processed CSI data
            proc_data_str = processed_samples[i].get('data', '')
            proc_csi = json.loads(proc_data_str.strip('"'))
            
            # Get count from processed file
            elements_count = processed_samples[i].get('csi_elements_count', 'N/A')
            
            print(f"Original CSI length: {len(orig_csi)} elements")
            print(f"Processed CSI length: {len(proc_csi)} elements")
            print(f"Elements count column: {elements_count}")
            
            print(f"Original first 15: {orig_csi[:15]}")
            print(f"Original last 15:  {orig_csi[-15:]}")
            print(f"Processed first 15: {proc_csi[:15]}")
            print(f"Processed last 15:  {proc_csi[-15:]}")
            
            # Verify the transformation
            expected_processed = orig_csi[12:-12]
            if proc_csi == expected_processed:
                print("✅ Transformation verified correctly!")
            else:
                print("❌ Transformation mismatch!")
    
    except Exception as e:
        print(f"Error comparing files: {e}")

def main():
    """Main function to run the comparison."""
    
    print("CSI Data Transformation Comparison")
    print("Original vs Processed Data")
    print("=" * 80)
    
    original_dir = Path("Labor Data CSV")
    processed_dir = Path("Processed Labor Data")
    
    if not original_dir.exists():
        print(f"Error: {original_dir} not found!")
        return
    
    if not processed_dir.exists():
        print(f"Error: {processed_dir} not found!")
        return
    
    # Compare a few sample files
    sample_files = ["0,0.csv", "1,0.csv", "3,0.csv"]
    
    for filename in sample_files:
        original_file = original_dir / filename
        processed_file = processed_dir / filename
        
        if original_file.exists() and processed_file.exists():
            compare_files(original_file, processed_file, sample_limit=2)
        else:
            print(f"⚠️  Skipping {filename} - file not found")
    
    print(f"\n✅ Comparison complete!")
    print(f"All processed files are in: {processed_dir}")

if __name__ == "__main__":
    main()
