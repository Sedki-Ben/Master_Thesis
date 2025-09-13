#!/usr/bin/env python3
"""
Display a summary of sample counts from the CSV results file.
"""

import csv

def main():
    try:
        with open('sample_counts.csv', 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        if not data:
            print("No data found in sample_counts.csv")
            return
        
        # Extract sample counts
        sample_counts = [int(row['Sample_Count']) for row in data]
        
        print("=" * 60)
        print("LABOR DATA SAMPLE COUNT SUMMARY")
        print("=" * 60)
        
        print(f"Total files analyzed: {len(data)}")
        print(f"Total samples: {sum(sample_counts):,}")
        print(f"Average samples per file: {sum(sample_counts) / len(data):.1f}")
        print(f"Minimum samples: {min(sample_counts):,}")
        print(f"Maximum samples: {max(sample_counts):,}")
        
        # Find files with min and max
        min_count = min(sample_counts)
        max_count = max(sample_counts)
        
        min_files = [row['Filename'] for row in data if int(row['Sample_Count']) == min_count]
        max_files = [row['Filename'] for row in data if int(row['Sample_Count']) == max_count]
        
        print(f"\nFiles with minimum samples ({min_count}): {', '.join(min_files)}")
        print(f"Files with maximum samples ({max_count}): {', '.join(max_files)}")
        
        print("\n" + "=" * 60)
        print("DETAILED BREAKDOWN BY FILE:")
        print("=" * 60)
        
        # Sort by filename for better readability
        sorted_data = sorted(data, key=lambda x: x['Filename'])
        
        for row in sorted_data:
            filename = row['Filename']
            sample_count = int(row['Sample_Count'])
            print(f"{filename:<15}: {sample_count:>6,} samples")
        
        print("=" * 60)
        
    except FileNotFoundError:
        print("Error: sample_counts.csv not found. Please run count_samples.py first.")
    except Exception as e:
        print(f"Error reading results: {e}")

if __name__ == "__main__":
    main()
