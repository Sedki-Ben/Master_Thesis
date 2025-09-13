#!/usr/bin/env python3
"""
Script to extract RSSI, amplitude, and phase from processed CSI data.
CSI data is ordered as imaginary, real, imaginary, real... (104 values = 52 complex pairs)
Creates files with only RSSI, amplitude (52 values), and phase (52 values) columns.
"""

import csv
import json
import math
from pathlib import Path

def extract_amplitude_phase(csi_data):
    """
    Extract amplitude and phase from CSI data.
    
    Args:
        csi_data (list): CSI data with 104 values (52 complex pairs: imag, real, imag, real...)
    
    Returns:
        tuple: (amplitudes, phases) - each list contains 52 values
    """
    if len(csi_data) != 104:
        raise ValueError(f"Expected 104 CSI values, got {len(csi_data)}")
    
    amplitudes = []
    phases = []
    
    # Process 52 complex pairs
    for i in range(0, 104, 2):
        imag = csi_data[i]      # Imaginary part
        real = csi_data[i + 1]  # Real part
        
        # Calculate amplitude: sqrt(real² + imag²)
        amplitude = math.sqrt(real * real + imag * imag)
        
        # Calculate phase: atan2(imag, real)
        phase = math.atan2(imag, real)
        
        amplitudes.append(amplitude)
        phases.append(phase)
    
    return amplitudes, phases

def process_file(input_file, output_file):
    """
    Process a single CSV file to extract RSSI, amplitude, and phase.
    
    Args:
        input_file (Path): Input processed CSV file
        output_file (Path): Output CSV file with extracted data
    
    Returns:
        dict: Processing statistics
    """
    stats = {
        'total_samples': 0,
        'processed_samples': 0,
        'parse_errors': 0,
        'length_errors': 0
    }
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            
            reader = csv.DictReader(infile)
            
            # Create fieldnames for output
            fieldnames = ['rssi']
            
            # Add amplitude columns (52 subcarriers)
            for i in range(52):
                fieldnames.append(f'amplitude_{i}')
            
            # Add phase columns (52 subcarriers)
            for i in range(52):
                fieldnames.append(f'phase_{i}')
            
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                stats['total_samples'] += 1
                
                # Extract RSSI
                rssi = row.get('rssi', '')
                
                # Extract and parse CSI data
                data_string = row.get('data', '')
                try:
                    csi_data = json.loads(data_string.strip('"'))
                except (json.JSONDecodeError, ValueError):
                    stats['parse_errors'] += 1
                    continue
                
                # Check if we have the expected 104 values
                if len(csi_data) != 104:
                    stats['length_errors'] += 1
                    continue
                
                try:
                    # Extract amplitude and phase
                    amplitudes, phases = extract_amplitude_phase(csi_data)
                    
                    # Create output row
                    output_row = {'rssi': rssi}
                    
                    # Add amplitude values
                    for i, amp in enumerate(amplitudes):
                        output_row[f'amplitude_{i}'] = round(amp, 6)
                    
                    # Add phase values
                    for i, phase in enumerate(phases):
                        output_row[f'phase_{i}'] = round(phase, 6)
                    
                    writer.writerow(output_row)
                    stats['processed_samples'] += 1
                    
                except Exception as e:
                    print(f"Error processing sample in {input_file.name}: {e}")
                    continue
        
        return stats
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return stats

def process_all_files(input_dir="Processed Labor Data", output_dir="Amplitude Phase Data"):
    """
    Process all processed CSV files to extract amplitude and phase data.
    
    Args:
        input_dir (str): Directory containing processed CSV files
        output_dir (str): Output directory for amplitude/phase files
    
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
    
    print(f"\nExtracting amplitude and phase from {len(csv_files)} files...")
    print("Each file will have: RSSI + 52 amplitudes + 52 phases = 105 columns")
    print("=" * 80)
    
    overall_stats = {
        'files_processed': 0,
        'total_samples': 0,
        'processed_samples': 0,
        'parse_errors': 0,
        'length_errors': 0
    }
    
    for input_file in csv_files:
        output_file = output_path / input_file.name
        
        print(f"Processing {input_file.name}...", end=" ")
        
        file_stats = process_file(input_file, output_file)
        
        # Update overall statistics
        overall_stats['files_processed'] += 1
        overall_stats['total_samples'] += file_stats['total_samples']
        overall_stats['processed_samples'] += file_stats['processed_samples']
        overall_stats['parse_errors'] += file_stats['parse_errors']
        overall_stats['length_errors'] += file_stats['length_errors']
        
        print(f"✓ {file_stats['processed_samples']} samples processed", end="")
        
        if file_stats['parse_errors'] > 0:
            print(f" ({file_stats['parse_errors']} parse errors)", end="")
        if file_stats['length_errors'] > 0:
            print(f" ({file_stats['length_errors']} length errors)", end="")
        
        print()  # New line
    
    return overall_stats

def display_summary(stats):
    """
    Display processing summary.
    
    Args:
        stats (dict): Processing statistics
    """
    print("\n" + "=" * 80)
    print("AMPLITUDE & PHASE EXTRACTION SUMMARY")
    print("=" * 80)
    
    print(f"Files processed: {stats.get('files_processed', 0)}")
    print(f"Total samples: {stats.get('total_samples', 0):,}")
    print(f"Successfully processed: {stats.get('processed_samples', 0):,}")
    print(f"Parse errors: {stats.get('parse_errors', 0)}")
    print(f"Length errors: {stats.get('length_errors', 0)}")
    
    if stats.get('total_samples', 0) > 0:
        success_rate = (stats['processed_samples'] / stats['total_samples']) * 100
        print(f"Success rate: {success_rate:.2f}%")
    
    print(f"\nOutput format:")
    print(f"- 1 RSSI column")
    print(f"- 52 amplitude columns (amplitude_0 to amplitude_51)")
    print(f"- 52 phase columns (phase_0 to phase_51)")
    print(f"- Total: 105 columns per file")

def create_sample_analysis(output_dir="Amplitude Phase Data"):
    """
    Create a script to analyze the extracted amplitude and phase data.
    
    Args:
        output_dir (str): Directory containing amplitude/phase files
    """
    analysis_script = f"""#!/usr/bin/env python3
\"\"\"
Script to analyze the extracted amplitude and phase data.
\"\"\"

import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def analyze_file(file_path, max_samples=100):
    \"\"\"Analyze amplitude and phase data from a single file.\"\"\"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            rssi_values = []
            amplitude_data = []
            phase_data = []
            
            sample_count = 0
            for row in reader:
                if sample_count >= max_samples:
                    break
                
                rssi_values.append(float(row['rssi']))
                
                # Extract amplitudes (52 values)
                amps = [float(row[f'amplitude_{{i}}']) for i in range(52)]
                amplitude_data.append(amps)
                
                # Extract phases (52 values)
                phases = [float(row[f'phase_{{i}}']) for i in range(52)]
                phase_data.append(phases)
                
                sample_count += 1
            
            amplitude_data = np.array(amplitude_data)
            phase_data = np.array(phase_data)
            
            print(f"File: {{file_path.name}}")
            print(f"Samples analyzed: {{len(rssi_values)}}")
            print(f"RSSI range: {{min(rssi_values):.1f}} to {{max(rssi_values):.1f}} dBm")
            print(f"Amplitude range: {{amplitude_data.min():.3f}} to {{amplitude_data.max():.3f}}")
            print(f"Phase range: {{phase_data.min():.3f}} to {{phase_data.max():.3f}} radians")
            print(f"Average amplitude per subcarrier: {{amplitude_data.mean(axis=0).mean():.3f}}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error analyzing {{file_path}}: {{e}}")

def main():
    output_path = Path("{output_dir}")
    csv_files = list(output_path.glob("*.csv"))[:5]  # Analyze first 5 files
    
    print("Amplitude & Phase Data Analysis")
    print("=" * 60)
    
    for file_path in csv_files:
        analyze_file(file_path)

if __name__ == "__main__":
    main()
"""
    
    with open("analyze_amplitude_phase.py", 'w', encoding='utf-8') as f:
        f.write(analysis_script)
    
    print(f"\n📊 Analysis script created: analyze_amplitude_phase.py")

def main():
    """Main function to run the amplitude and phase extraction."""
    
    print("CSI Amplitude & Phase Extractor")
    print("Extracts RSSI, amplitude, and phase from processed CSI data")
    print("CSI format: imaginary, real, imaginary, real... (52 complex pairs)")
    print("=" * 80)
    
    # Process all files
    stats = process_all_files()
    
    if stats:
        # Display summary
        display_summary(stats)
        
        # Create analysis script
        create_sample_analysis()
        
        print(f"\n✅ Extraction complete! Check the 'Amplitude Phase Data' directory for results.")
    else:
        print("❌ Extraction failed!")

if __name__ == "__main__":
    main()
