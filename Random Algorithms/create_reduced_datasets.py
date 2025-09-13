#!/usr/bin/env python3
"""
Create reduced datasets from Amplitude Phase Data Single files.
This script creates two new datasets with 750 and 500 samples per location
for training efficiency and comparison studies.
"""

import csv
import json
import random
from pathlib import Path
import shutil

def create_reduced_dataset(input_dir, output_dir, samples_per_file, description):
    """
    Create a reduced dataset with specified number of samples per file.
    
    Args:
        input_dir (str): Source directory with original files
        output_dir (str): Destination directory for reduced files
        samples_per_file (int): Maximum samples to keep per file
        description (str): Description for logging
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    print(f"\n📁 Creating {description}")
    print(f"Source: {input_path}")
    print(f"Target: {output_path}")
    print(f"Max samples per file: {samples_per_file}")
    print("-" * 60)
    
    # Get all CSV files
    csv_files = sorted(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {input_dir}")
        return
    
    total_original_samples = 0
    total_reduced_samples = 0
    files_processed = 0
    
    for input_file in csv_files:
        try:
            # Read all samples from the input file
            samples = []
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                
                for row in reader:
                    samples.append(row)
            
            original_count = len(samples)
            total_original_samples += original_count
            
            # Randomly sample the specified number of samples
            if len(samples) > samples_per_file:
                # Set random seed for reproducible sampling
                random.seed(42 + files_processed)  # Different seed per file but reproducible
                selected_samples = random.sample(samples, samples_per_file)
                reduced_count = samples_per_file
            else:
                selected_samples = samples
                reduced_count = len(samples)
            
            total_reduced_samples += reduced_count
            
            # Write reduced dataset to output file
            output_file = output_path / input_file.name
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(selected_samples)
            
            print(f"📄 {input_file.name}: {original_count} → {reduced_count} samples")
            files_processed += 1
            
        except Exception as e:
            print(f"❌ Error processing {input_file.name}: {e}")
            continue
    
    print("-" * 60)
    print(f"✅ {description} Complete!")
    print(f"Files processed: {files_processed}")
    print(f"Total samples: {total_original_samples:,} → {total_reduced_samples:,}")
    print(f"Reduction: {((total_original_samples - total_reduced_samples) / total_original_samples * 100):.1f}%")
    print(f"Average samples per file: {total_reduced_samples / files_processed:.0f}")
    
    return {
        'files_processed': files_processed,
        'original_samples': total_original_samples,
        'reduced_samples': total_reduced_samples,
        'reduction_percentage': (total_original_samples - total_reduced_samples) / total_original_samples * 100
    }

def create_dataset_info_file(output_dir, stats, samples_per_file, description):
    """
    Create an information file documenting the dataset characteristics.
    
    Args:
        output_dir (str): Directory containing the dataset
        stats (dict): Statistics from the reduction process
        samples_per_file (int): Target samples per file
        description (str): Dataset description
    """
    info_content = f"""# {description}

## Dataset Information

**Creation Date**: {Path(__file__).stat().st_mtime}
**Source Dataset**: Amplitude Phase Data Single
**Target Samples per File**: {samples_per_file}
**Sampling Method**: Random sampling with fixed seed (reproducible)

## Statistics

- **Files Processed**: {stats['files_processed']}
- **Original Total Samples**: {stats['original_samples']:,}
- **Reduced Total Samples**: {stats['reduced_samples']:,}
- **Reduction Percentage**: {stats['reduction_percentage']:.1f}%
- **Average Samples per File**: {stats['reduced_samples'] / stats['files_processed']:.0f}

## Purpose

This reduced dataset is designed for:
- Faster training and experimentation
- Memory-constrained environments
- Quick prototyping and parameter tuning
- Computational efficiency studies

## Data Integrity

- Random sampling ensures representative data
- Fixed random seed (42 + file_index) ensures reproducibility
- All original features preserved (RSSI, amplitude array, phase array)
- Spatial coordinates maintained from original filenames

## Usage Notes

- Suitable for CNN model development and testing
- May require adjustments for final production models
- Consider data augmentation to compensate for reduced sample size
- Monitor for potential underfitting due to reduced data volume
"""
    
    info_file = Path(output_dir) / "dataset_info.md"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(info_content)
    
    print(f"📄 Dataset info saved: {info_file}")

def main():
    """
    Main function to create both reduced datasets.
    """
    print("🔄 CSI Dataset Reduction Tool")
    print("Creating reduced datasets for training efficiency")
    print("=" * 70)
    
    input_directory = "Amplitude Phase Data Single"
    
    # Check if input directory exists
    if not Path(input_directory).exists():
        print(f"❌ Input directory '{input_directory}' not found!")
        return
    
    # Create dataset with 750 samples per file
    stats_750 = create_reduced_dataset(
        input_dir=input_directory,
        output_dir="CSI Dataset 750 Samples",
        samples_per_file=750,
        description="CSI Dataset - 750 Samples per Location"
    )
    
    if stats_750:
        create_dataset_info_file(
            output_dir="CSI Dataset 750 Samples",
            stats=stats_750,
            samples_per_file=750,
            description="CSI Dataset - 750 Samples per Location"
        )
    
    # Create dataset with 500 samples per file
    stats_500 = create_reduced_dataset(
        input_dir=input_directory,
        output_dir="CSI Dataset 500 Samples",
        samples_per_file=500,
        description="CSI Dataset - 500 Samples per Location"
    )
    
    if stats_500:
        create_dataset_info_file(
            output_dir="CSI Dataset 500 Samples",
            stats=stats_500,
            samples_per_file=500,
            description="CSI Dataset - 500 Samples per Location"
        )
    
    print("\n" + "=" * 70)
    print("🎯 DATASET CREATION SUMMARY")
    print("=" * 70)
    
    if stats_750 and stats_500:
        print("✅ Both reduced datasets created successfully!")
        print(f"\n📊 Dataset Comparison:")
        print(f"Original Dataset: {stats_750['original_samples']:,} total samples")
        print(f"750-Sample Dataset: {stats_750['reduced_samples']:,} total samples ({100-stats_750['reduction_percentage']:.1f}% retained)")
        print(f"500-Sample Dataset: {stats_500['reduced_samples']:,} total samples ({100-stats_500['reduction_percentage']:.1f}% retained)")
        
        print(f"\n🎯 Use Cases:")
        print(f"- Original: Full-scale training and final model evaluation")
        print(f"- 750-Sample: Balanced training with good statistical power")
        print(f"- 500-Sample: Fast prototyping and hyperparameter tuning")
        
        print(f"\n📁 Generated Directories:")
        print(f"- CSI Dataset 750 Samples/ (with dataset_info.md)")
        print(f"- CSI Dataset 500 Samples/ (with dataset_info.md)")
    else:
        print("❌ Dataset creation encountered errors!")

if __name__ == "__main__":
    main()
