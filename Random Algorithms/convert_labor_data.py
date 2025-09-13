#!/usr/bin/env python3
"""
Script to convert labor data from .txt files to .csv files.
All files in the "Labor Data" directory will be processed.
"""

import os
import shutil
from pathlib import Path

def convert_txt_to_csv(input_dir="Labor Data", output_dir="Labor Data CSV"):
    """
    Convert all .txt files in the input directory to .csv files in the output directory.
    
    Args:
        input_dir (str): Directory containing the .txt files
        output_dir (str): Directory where .csv files will be saved
    """
    
    # Create input and output paths
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    print(f"Created output directory: {output_path}")
    
    # Find all .txt files in the input directory
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in '{input_dir}'")
        return
    
    print(f"Found {len(txt_files)} .txt files to convert")
    
    # Convert each file
    converted_count = 0
    for txt_file in txt_files:
        try:
            # Create output filename with .csv extension
            csv_filename = txt_file.stem + ".csv"
            csv_file_path = output_path / csv_filename
            
            # Copy the file content (since it's already in CSV format)
            shutil.copy2(txt_file, csv_file_path)
            
            print(f"Converted: {txt_file.name} -> {csv_filename}")
            converted_count += 1
            
        except Exception as e:
            print(f"Error converting {txt_file.name}: {e}")
    
    print(f"\nConversion completed!")
    print(f"Successfully converted {converted_count} out of {len(txt_files)} files")
    print(f"CSV files saved in: {output_path}")

def convert_in_place(input_dir="Labor Data"):
    """
    Convert .txt files to .csv files in the same directory (rename files).
    
    Args:
        input_dir (str): Directory containing the .txt files
    """
    
    # Create input path
    input_path = Path(input_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Find all .txt files in the input directory
    txt_files = list(input_path.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in '{input_dir}'")
        return
    
    print(f"Found {len(txt_files)} .txt files to convert")
    
    # Convert each file by renaming
    converted_count = 0
    for txt_file in txt_files:
        try:
            # Create new filename with .csv extension
            csv_file_path = txt_file.with_suffix('.csv')
            
            # Rename the file
            txt_file.rename(csv_file_path)
            
            print(f"Converted: {txt_file.name} -> {csv_file_path.name}")
            converted_count += 1
            
        except Exception as e:
            print(f"Error converting {txt_file.name}: {e}")
    
    print(f"\nConversion completed!")
    print(f"Successfully converted {converted_count} out of {len(txt_files)} files")

if __name__ == "__main__":
    print("Labor Data File Converter")
    print("=" * 40)
    
    # Ask user for conversion method
    print("\nChoose conversion method:")
    print("1. Create new CSV files in separate directory (keeps original .txt files)")
    print("2. Rename .txt files to .csv files in-place (replaces original files)")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice == "1":
            print("\nConverting files to new directory...")
            convert_txt_to_csv()
            break
        elif choice == "2":
            print("\nConverting files in-place...")
            convert_in_place()
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
