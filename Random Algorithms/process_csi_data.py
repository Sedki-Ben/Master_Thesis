#!/usr/bin/env python3
"""
Script to process CSI data by removing the first 12 and last 12 elements from each sample,
and add a column showing the count of remaining elements.
"""

import csv
import json
import os
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

def process_csi_array(csi_data, remove_first=12, remove_last=12):
    """
    Remove first and last elements from CSI data array.
    
    Args:
        csi_data (list): Original CSI data array
        remove_first (int): Number of elements to remove from beginning
        remove_last (int): Number of elements to remove from end
    
    Returns:
        tuple: (processed_array, remaining_count)
    """
    if not csi_data or len(csi_data) <= (remove_first + remove_last):
        # If array is too short, return empty array
        return [], 0
    
    # Remove first and last elements
    processed = csi_data[remove_first:-remove_last] if remove_last > 0 else csi_data[remove_first:]
    
    return processed, len(processed)

def process_file(input_file, output_file, remove_first=12, remove_last=12):
    """
    Process a single CSV file by removing elements from CSI data.
    
    Args:
        input_file (Path): Input CSV file path
        output_file (Path): Output CSV file path
        remove_first (int): Number of elements to remove from beginning
        remove_last (int): Number of elements to remove from end
    
    Returns:
        dict: Processing statistics
    """
    stats = {
        'total_samples': 0,
        'processed_samples': 0,
        'parse_errors': 0,
        'original_length': 0,
        'new_length': 0
    }
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile)
            
            # Create new fieldnames with additional column
            fieldnames = reader.fieldnames + ['csi_elements_count']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                stats['total_samples'] += 1
                
                # Get original CSI data
                data_string = row.get('data', '')
                csi_data = parse_csi_data(data_string)
                
                if csi_data is None:
                    stats['parse_errors'] += 1
                    # Keep original data if parsing fails
                    row['csi_elements_count'] = 'PARSE_ERROR'
                    writer.writerow(row)
                    continue
                
                # Store original length for statistics
                if stats['total_samples'] == 1:
                    stats['original_length'] = len(csi_data)
                
                # Process CSI data
                processed_csi, remaining_count = process_csi_array(csi_data, remove_first, remove_last)
                
                # Update row with processed data
                row['data'] = json.dumps(processed_csi)
                row['csi_elements_count'] = remaining_count
                
                # Store new length for statistics
                if stats['processed_samples'] == 0:
                    stats['new_length'] = remaining_count
                
                stats['processed_samples'] += 1
                writer.writerow(row)
        
        return stats
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return stats

def process_all_files(input_dir="Labor Data CSV", output_dir="Processed Labor Data", 
                     remove_first=12, remove_last=12):
    """
    Process all CSV files in the input directory.
    
    Args:
        input_dir (str): Input directory containing CSV files
        output_dir (str): Output directory for processed files
        remove_first (int): Number of elements to remove from beginning
        remove_last (int): Number of elements to remove from end
    
    Returns:
        dict: Overall processing statistics
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Check if input directory exists
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return {}
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    print(f"Created output directory: {output_path}")
    
    # Get all CSV files
    csv_files = sorted(input_path.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{input_dir}'")
        return {}
    
    print(f"\nProcessing {len(csv_files)} files...")
    print(f"Removing first {remove_first} and last {remove_last} elements from CSI data")
    print("=" * 80)
    
    overall_stats = {
        'files_processed': 0,
        'total_samples': 0,
        'processed_samples': 0,
        'parse_errors': 0,
        'original_length': 0,
        'new_length': 0
    }
    
    for input_file in csv_files:
        output_file = output_path / input_file.name
        
        print(f"Processing {input_file.name}...", end=" ")
        
        file_stats = process_file(input_file, output_file, remove_first, remove_last)
        
        # Update overall statistics
        overall_stats['files_processed'] += 1
        overall_stats['total_samples'] += file_stats['total_samples']
        overall_stats['processed_samples'] += file_stats['processed_samples']
        overall_stats['parse_errors'] += file_stats['parse_errors']
        
        if file_stats['original_length'] > 0:
            overall_stats['original_length'] = file_stats['original_length']
        if file_stats['new_length'] > 0:
            overall_stats['new_length'] = file_stats['new_length']
        
        print(f"✓ {file_stats['processed_samples']} samples processed")
        
        if file_stats['parse_errors'] > 0:
            print(f"  ⚠️  {file_stats['parse_errors']} parse errors")
    
    return overall_stats

def display_summary(stats, remove_first=12, remove_last=12):
    """
    Display processing summary.
    
    Args:
        stats (dict): Processing statistics
        remove_first (int): Number of elements removed from beginning
        remove_last (int): Number of elements removed from end
    """
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    
    print(f"Files processed: {stats.get('files_processed', 0)}")
    print(f"Total samples: {stats.get('total_samples', 0):,}")
    print(f"Successfully processed: {stats.get('processed_samples', 0):,}")
    print(f"Parse errors: {stats.get('parse_errors', 0)}")
    
    print(f"\nCSI Data Transformation:")
    print(f"Original CSI array length: {stats.get('original_length', 0)} elements")
    print(f"Elements removed from start: {remove_first}")
    print(f"Elements removed from end: {remove_last}")
    print(f"Total elements removed: {remove_first + remove_last}")
    print(f"New CSI array length: {stats.get('new_length', 0)} elements")
    
    if stats.get('original_length', 0) > 0:
        reduction = ((remove_first + remove_last) / stats['original_length']) * 100
        print(f"Data reduction: {reduction:.1f}%")

def create_sample_verification(output_dir="Processed Labor Data"):
    """
    Create a script to verify the processed data.
    
    Args:
        output_dir (str): Directory containing processed files
    """
    verification_script = f"""#!/usr/bin/env python3
\"\"\"
Script to verify the processed CSI data.
\"\"\"

import csv
import json
from pathlib import Path

def verify_file(file_path):
    \"\"\"Verify a single processed file.\"\"\"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=2):
                data_string = row.get('data', '')
                count_string = row.get('csi_elements_count', '')
                
                if count_string == 'PARSE_ERROR':
                    continue
                
                # Parse data
                csi_data = json.loads(data_string.strip('"'))
                expected_count = int(count_string)
                actual_count = len(csi_data)
                
                if actual_count != expected_count:
                    print(f"MISMATCH in {{file_path.name}} row {{row_num}}: expected {{expected_count}}, got {{actual_count}}")
                    return False
        
        return True
    except Exception as e:
        print(f"Error verifying {{file_path}}: {{e}}")
        return False

def main():
    output_path = Path("{output_dir}")
    csv_files = list(output_path.glob("*.csv"))
    
    print(f"Verifying {{len(csv_files)}} processed files...")
    
    all_good = True
    for file_path in csv_files:
        if not verify_file(file_path):
            all_good = False
    
    if all_good:
        print("✅ All files verified successfully!")
    else:
        print("❌ Some files have verification errors!")

if __name__ == "__main__":
    main()
"""
    
    with open("verify_processed_data.py", 'w', encoding='utf-8') as f:
        f.write(verification_script)
    
    print(f"\n📝 Verification script created: verify_processed_data.py")

def main():
    """Main function to run the CSI data processing."""
    
    print("CSI Data Processor")
    print("Removes first 12 and last 12 elements from each CSI data array")
    print("=" * 80)
    
    # Processing parameters
    remove_first = 12
    remove_last = 12
    
    # Process all files
    stats = process_all_files(remove_first=remove_first, remove_last=remove_last)
    
    if stats:
        # Display summary
        display_summary(stats, remove_first, remove_last)
        
        # Create verification script
        create_sample_verification()
        
        print(f"\n✅ Processing complete! Check the 'Processed Labor Data' directory for results.")
    else:
        print("❌ Processing failed!")

if __name__ == "__main__":
    main()
