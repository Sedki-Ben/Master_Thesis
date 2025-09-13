#!/usr/bin/env python3
"""
Create 250 sample dataset by subsampling from 750 sample dataset
"""

import csv
import json
import random
from pathlib import Path

def create_250_sample_dataset():
    """Create 250 sample dataset from 750 sample dataset"""
    
    print("🔄 Creating 250 sample dataset from 750 sample dataset...")
    
    source_dir = Path("CSI Dataset 750 Samples")
    target_dir = Path("CSI Dataset 250 Samples")
    target_dir.mkdir(exist_ok=True)
    
    test_source_dir = Path("Testing Points Dataset 750 Samples")
    test_target_dir = Path("Testing Points Dataset 250 Samples")
    test_target_dir.mkdir(exist_ok=True)
    
    # Process training data
    for csv_file in source_dir.glob("*.csv"):
        if csv_file.name == "dataset_info.md":
            continue
            
        print(f"   📁 Processing: {csv_file.name}")
        
        # Read all samples
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_samples = list(reader)
        
        # Randomly select 250 samples
        if len(all_samples) >= 250:
            selected_samples = random.sample(all_samples, 250)
        else:
            selected_samples = all_samples
            print(f"      ⚠️ Only {len(all_samples)} samples available")
        
        # Write to target file
        target_file = target_dir / csv_file.name
        with open(target_file, 'w', newline='', encoding='utf-8') as f:
            if selected_samples:
                writer = csv.DictWriter(f, fieldnames=selected_samples[0].keys())
                writer.writeheader()
                writer.writerows(selected_samples)
        
        print(f"      ✅ Created {len(selected_samples)} samples")
    
    # Process test data
    for csv_file in test_source_dir.glob("*.csv"):
        print(f"   📁 Processing test: {csv_file.name}")
        
        # Read all samples
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_samples = list(reader)
        
        # Randomly select 250 samples
        if len(all_samples) >= 250:
            selected_samples = random.sample(all_samples, 250)
        else:
            selected_samples = all_samples
            print(f"      ⚠️ Only {len(all_samples)} samples available")
        
        # Write to target file
        target_file = test_target_dir / csv_file.name
        with open(target_file, 'w', newline='', encoding='utf-8') as f:
            if selected_samples:
                writer = csv.DictWriter(f, fieldnames=selected_samples[0].keys())
                writer.writeheader()
                writer.writerows(selected_samples)
        
        print(f"      ✅ Created {len(selected_samples)} test samples")
    
    print("✅ 250 sample dataset created successfully!")

if __name__ == "__main__":
    random.seed(42)  # For reproducible results
    create_250_sample_dataset()


