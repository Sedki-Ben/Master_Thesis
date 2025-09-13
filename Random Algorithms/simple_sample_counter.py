#!/usr/bin/env python3
"""
Simple script to count and display the number of samples in each labor data file.
"""

from pathlib import Path

def count_file_samples(file_path):
    """Count samples in a single file (total lines - 1 for header)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for line in f)
        return max(0, line_count - 1)  # Exclude header row
    except:
        return 0

def main():
    """Count samples in all labor data files."""
    
    # Try CSV files first, then TXT files
    csv_dir = Path("Labor Data CSV")
    txt_dir = Path("Labor Data")
    
    if csv_dir.exists():
        data_dir = csv_dir
        file_ext = "*.csv"
        print("Reading CSV files from 'Labor Data CSV'")
    elif txt_dir.exists():
        data_dir = txt_dir
        file_ext = "*.txt"
        print("Reading TXT files from 'Labor Data'")
    else:
        print("Error: No data directory found!")
        return
    
    # Get all files and sort them
    files = sorted(data_dir.glob(file_ext))
    
    if not files:
        print(f"No {file_ext} files found!")
        return
    
    print(f"\nSample counts for {len(files)} files:")
    print("-" * 40)
    
    total_samples = 0
    
    for file_path in files:
        sample_count = count_file_samples(file_path)
        total_samples += sample_count
        print(f"{file_path.name:<15}: {sample_count:>6} samples")
    
    print("-" * 40)
    print(f"{'TOTAL':<15}: {total_samples:>6} samples")
    print(f"{'FILES':<15}: {len(files):>6} files")
    print(f"{'AVERAGE':<15}: {total_samples/len(files):>6.1f} samples/file")

if __name__ == "__main__":
    main()
