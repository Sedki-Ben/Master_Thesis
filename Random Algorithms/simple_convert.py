#!/usr/bin/env python3
"""
Simple script to convert all .txt files in Labor Data directory to .csv files.
Creates copies in a new directory while preserving originals.
"""

import shutil
from pathlib import Path

def main():
    """Convert all labor data .txt files to .csv files."""
    
    input_dir = Path("Labor Data")
    output_dir = Path("Labor Data CSV")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: '{input_dir}' directory not found!")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get all .txt files
    txt_files = list(input_dir.glob("*.txt"))
    
    if not txt_files:
        print("No .txt files found in Labor Data directory")
        return
    
    print(f"Converting {len(txt_files)} files...")
    
    # Convert each file
    for txt_file in txt_files:
        csv_file = output_dir / (txt_file.stem + ".csv")
        shutil.copy2(txt_file, csv_file)
        print(f"✓ {txt_file.name} -> {csv_file.name}")
    
    print(f"\nDone! Converted {len(txt_files)} files to CSV format.")

if __name__ == "__main__":
    main()
