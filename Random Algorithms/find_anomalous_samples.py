#!/usr/bin/env python3
"""
Script to find samples in labor data files that don't start with the expected sequence: 83, -80, 4
CSI data is organized as imaginary, then real parts.
"""

import csv
import json
from pathlib import Path

def parse_csi_data(data_string):
    """
    Parse the CSI data string into a list of numbers.
    
    Args:
        data_string (str): The data string from the CSV file
    
    Returns:
        list: List of integers, or None if parsing fails
    """
    try:
        # Remove quotes if present and parse as JSON array
        data_string = data_string.strip('"')
        return json.loads(data_string)
    except (json.JSONDecodeError, ValueError):
        return None

def check_sample_header(csi_data, expected_sequence=[83, -80, 4]):
    """
    Check if the CSI data starts with the expected sequence.
    
    Args:
        csi_data (list): List of CSI values
        expected_sequence (list): Expected starting sequence
    
    Returns:
        bool: True if matches expected sequence, False otherwise
    """
    if not csi_data or len(csi_data) < len(expected_sequence):
        return False
    
    return csi_data[:len(expected_sequence)] == expected_sequence

def analyze_file(file_path, expected_sequence=[83, -80, 4]):
    """
    Analyze a single file for anomalous samples.
    
    Args:
        file_path (Path): Path to the CSV file
        expected_sequence (list): Expected starting sequence
    
    Returns:
        list: List of tuples (row_number, sample_id, actual_sequence, csi_data)
    """
    anomalies = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 because of header
                # Extract the data column
                data_string = row.get('data', '')
                sample_id = row.get('id', 'unknown')
                
                # Parse CSI data
                csi_data = parse_csi_data(data_string)
                
                if csi_data is None:
                    anomalies.append((row_num, sample_id, "PARSE_ERROR", None))
                    continue
                
                # Check if it starts with expected sequence
                if not check_sample_header(csi_data, expected_sequence):
                    # Get the actual first few values
                    actual_start = csi_data[:len(expected_sequence)] if len(csi_data) >= len(expected_sequence) else csi_data
                    anomalies.append((row_num, sample_id, actual_start, len(csi_data)))
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [("FILE_ERROR", str(e), [], 0)]
    
    return anomalies

def scan_all_files(data_dir="Labor Data CSV", expected_sequence=[83, -80, 4]):
    """
    Scan all CSV files for anomalous samples.
    
    Args:
        data_dir (str): Directory containing CSV files
        expected_sequence (list): Expected starting sequence
    
    Returns:
        dict: Dictionary with filename as key and list of anomalies as value
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory '{data_dir}' does not exist!")
        return {}
    
    csv_files = sorted(data_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{data_dir}'")
        return {}
    
    print(f"Scanning {len(csv_files)} files for samples not starting with {expected_sequence}...")
    print("=" * 80)
    
    all_anomalies = {}
    total_samples = 0
    total_anomalies = 0
    
    for file_path in csv_files:
        print(f"Analyzing {file_path.name}...", end=" ")
        
        anomalies = analyze_file(file_path, expected_sequence)
        
        # Count total samples in this file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_samples = sum(1 for line in f) - 1  # Subtract header
        except:
            file_samples = 0
        
        total_samples += file_samples
        
        if anomalies:
            all_anomalies[file_path.name] = anomalies
            total_anomalies += len(anomalies)
            print(f"Found {len(anomalies)} anomalies out of {file_samples} samples")
        else:
            print(f"No anomalies found in {file_samples} samples")
    
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"Total files scanned: {len(csv_files)}")
    print(f"Total samples analyzed: {total_samples:,}")
    print(f"Total anomalies found: {total_anomalies}")
    print(f"Files with anomalies: {len(all_anomalies)}")
    
    return all_anomalies

def display_anomalies(anomalies, max_display=50):
    """
    Display the found anomalies in detail.
    
    Args:
        anomalies (dict): Dictionary of anomalies by filename
        max_display (int): Maximum number of anomalies to display in detail
    """
    if not anomalies:
        print("\n✅ No anomalies found! All samples start with the expected sequence [83, -80, 4]")
        return
    
    print("\n" + "=" * 80)
    print("DETAILED ANOMALY REPORT")
    print("=" * 80)
    
    displayed_count = 0
    
    for filename, file_anomalies in anomalies.items():
        print(f"\n📁 FILE: {filename}")
        print(f"   Anomalies found: {len(file_anomalies)}")
        print("   " + "-" * 60)
        
        for row_num, sample_id, actual_start, data_length in file_anomalies:
            if displayed_count >= max_display:
                print(f"   ... and {sum(len(anoms) for anoms in anomalies.values()) - displayed_count} more anomalies")
                return
            
            if actual_start == "PARSE_ERROR":
                print(f"   Row {row_num:4d} | Sample {sample_id:>6} | ❌ PARSE ERROR")
            elif actual_start == "FILE_ERROR":
                print(f"   FILE ERROR: {sample_id}")
            else:
                print(f"   Row {row_num:4d} | Sample {sample_id:>6} | Starts with: {actual_start} | Length: {data_length}")
            
            displayed_count += 1

def save_anomalies_report(anomalies, output_file="anomalous_samples.csv"):
    """
    Save anomalies to a CSV file for further analysis.
    
    Args:
        anomalies (dict): Dictionary of anomalies by filename
        output_file (str): Output CSV filename
    """
    if not anomalies:
        return
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Row_Number', 'Sample_ID', 'Actual_Start_Sequence', 'Data_Length'])
            
            for filename, file_anomalies in anomalies.items():
                for row_num, sample_id, actual_start, data_length in file_anomalies:
                    writer.writerow([filename, row_num, sample_id, str(actual_start), data_length])
        
        print(f"\n📝 Anomalies report saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving anomalies report: {e}")

def main():
    """Main function to run the anomaly detection."""
    
    print("CSI Data Anomaly Detector")
    print("Checking for samples that don't start with [83, -80, 4]")
    print("=" * 80)
    
    # Expected sequence
    expected_sequence = [83, -80, 4]
    
    # Scan all files
    anomalies = scan_all_files(expected_sequence=expected_sequence)
    
    # Display results
    display_anomalies(anomalies)
    
    # Save report if anomalies found
    if anomalies:
        save_anomalies_report(anomalies)

if __name__ == "__main__":
    main()
