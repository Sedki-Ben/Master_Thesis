#!/usr/bin/env python3
"""
Script to verify the processed CSI data.
"""

import csv
import json
from pathlib import Path

def verify_file(file_path):
    """Verify a single processed file."""
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
                    print(f"MISMATCH in {file_path.name} row {row_num}: expected {expected_count}, got {actual_count}")
                    return False
        
        return True
    except Exception as e:
        print(f"Error verifying {file_path}: {e}")
        return False

def main():
    output_path = Path("Processed Labor Data")
    csv_files = list(output_path.glob("*.csv"))
    
    print(f"Verifying {len(csv_files)} processed files...")
    
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
