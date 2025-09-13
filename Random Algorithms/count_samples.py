#!/usr/bin/env python3
"""
Script to count the number of samples (data rows) in each labor data file.
A sample is represented by a row containing CSI_DATA (excluding the header row).
"""

import csv
from pathlib import Path
import pandas as pd

def count_samples_in_file(file_path):
    """
    Count the number of samples (data rows) in a single file.
    
    Args:
        file_path (Path): Path to the file
    
    Returns:
        tuple: (filename, sample_count, total_lines)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Total lines including header
        total_lines = len(lines)
        
        # Sample count = total lines - 1 (excluding header)
        sample_count = max(0, total_lines - 1)
        
        return file_path.name, sample_count, total_lines
        
    except Exception as e:
        return file_path.name, 0, 0

def count_samples_csv_method(file_path):
    """
    Alternative method using CSV reader for more robust counting.
    
    Args:
        file_path (Path): Path to the file
    
    Returns:
        tuple: (filename, sample_count)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            sample_count = sum(1 for row in reader)
        
        return file_path.name, sample_count
        
    except Exception as e:
        return file_path.name, 0

def analyze_labor_data(data_dir="Labor Data", use_csv_files=True):
    """
    Analyze all labor data files and count samples in each.
    
    Args:
        data_dir (str): Directory containing the data files
        use_csv_files (bool): Whether to use CSV files or TXT files
    
    Returns:
        list: List of tuples with (filename, sample_count, total_lines)
    """
    
    if use_csv_files:
        data_path = Path("Labor Data CSV")
        file_pattern = "*.csv"
    else:
        data_path = Path(data_dir)
        file_pattern = "*.txt"
    
    if not data_path.exists():
        print(f"Error: Directory '{data_path}' does not exist!")
        return []
    
    # Get all data files
    data_files = sorted(data_path.glob(file_pattern))
    
    if not data_files:
        print(f"No {file_pattern} files found in '{data_path}'")
        return []
    
    print(f"Analyzing {len(data_files)} files in '{data_path}'...")
    print("=" * 60)
    
    results = []
    total_samples = 0
    
    for file_path in data_files:
        filename, sample_count, total_lines = count_samples_in_file(file_path)
        results.append((filename, sample_count, total_lines))
        total_samples += sample_count
        
        print(f"{filename:<15} | {sample_count:>6} samples | {total_lines:>6} total lines")
    
    print("=" * 60)
    print(f"{'TOTAL':<15} | {total_samples:>6} samples | {len(data_files)} files")
    
    return results

def generate_detailed_report(results):
    """
    Generate a detailed statistical report.
    
    Args:
        results (list): List of tuples with file analysis results
    """
    if not results:
        return
    
    sample_counts = [count for _, count, _ in results]
    
    print("\n" + "=" * 60)
    print("DETAILED STATISTICS")
    print("=" * 60)
    print(f"Total files analyzed: {len(results)}")
    print(f"Total samples across all files: {sum(sample_counts):,}")
    print(f"Average samples per file: {sum(sample_counts) / len(results):.1f}")
    print(f"Minimum samples in a file: {min(sample_counts):,}")
    print(f"Maximum samples in a file: {max(sample_counts):,}")
    
    # Find files with min and max samples
    min_count = min(sample_counts)
    max_count = max(sample_counts)
    
    min_files = [filename for filename, count, _ in results if count == min_count]
    max_files = [filename for filename, count, _ in results if count == max_count]
    
    print(f"File(s) with minimum samples ({min_count}): {', '.join(min_files)}")
    print(f"File(s) with maximum samples ({max_count}): {', '.join(max_files)}")

def save_results_to_csv(results, output_file="sample_counts.csv"):
    """
    Save the results to a CSV file for further analysis.
    
    Args:
        results (list): List of tuples with file analysis results
        output_file (str): Output CSV filename
    """
    if not results:
        return
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Sample_Count', 'Total_Lines'])
            writer.writerows(results)
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    """Main function to run the sample counting analysis."""
    
    print("Labor Data Sample Counter")
    print("=" * 60)
    
    # Check which files are available
    csv_dir = Path("Labor Data CSV")
    txt_dir = Path("Labor Data")
    
    if csv_dir.exists():
        print("Using CSV files from 'Labor Data CSV' directory")
        results = analyze_labor_data(use_csv_files=True)
    elif txt_dir.exists():
        print("Using TXT files from 'Labor Data' directory")
        results = analyze_labor_data(use_csv_files=False)
    else:
        print("Error: Neither 'Labor Data CSV' nor 'Labor Data' directory found!")
        return
    
    if results:
        generate_detailed_report(results)
        save_results_to_csv(results)

if __name__ == "__main__":
    main()
