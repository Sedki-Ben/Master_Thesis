#!/usr/bin/env python3
"""
Plot amplitudes of 5 random samples from point (2,4) 
in relation to subcarrier indexes as line charts
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import random

def load_point_data(point_x, point_y, dataset_folder="CSI Dataset 250 Samples"):
    """Load all data for a specific point"""
    file_path = Path(dataset_folder) / f"{point_x},{point_y}.csv"
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return None
    
    amplitudes = []
    sample_info = []
    
    print(f"📂 Loading data from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            try:
                amps = json.loads(row['amplitude'])
                if len(amps) == 52:  # Ensure we have 52 subcarriers
                    amplitudes.append(amps)
                    sample_info.append({
                        'row_index': row_idx,
                        'rssi': float(row['rssi']),
                        'amplitudes': amps
                    })
            except Exception as e:
                print(f"⚠️ Error parsing row {row_idx}: {e}")
                continue
    
    print(f"✅ Loaded {len(amplitudes)} valid samples from point ({point_x}, {point_y})")
    return amplitudes, sample_info

def plot_amplitude_lines(point_x, point_y, num_samples=5, dataset_folder="CSI Dataset 250 Samples"):
    """Plot amplitude vs subcarrier index for random samples from a point"""
    
    # Load data
    amplitudes, sample_info = load_point_data(point_x, point_y, dataset_folder)
    
    if amplitudes is None or len(amplitudes) == 0:
        print(f"❌ No data available for point ({point_x}, {point_y})")
        return
    
    # Select random samples
    if len(amplitudes) < num_samples:
        print(f"⚠️ Only {len(amplitudes)} samples available, using all of them")
        selected_indices = list(range(len(amplitudes)))
    else:
        # Set seed for reproducibility
        random.seed(42)
        selected_indices = random.sample(range(len(amplitudes)), num_samples)
        selected_indices.sort()  # Sort for consistent ordering
    
    print(f"🎲 Selected sample indices: {selected_indices}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Color palette for different samples
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Subcarrier indices (0 to 51)
    subcarrier_indices = np.arange(52)
    
    # Plot each selected sample
    for i, sample_idx in enumerate(selected_indices):
        sample_amplitudes = amplitudes[sample_idx]
        sample_rssi = sample_info[sample_idx]['rssi']
        
        color = colors[i % len(colors)]
        
        plt.plot(subcarrier_indices, sample_amplitudes, 
                color=color, linewidth=2, alpha=0.8, marker='o', markersize=3,
                label=f'Sample {sample_idx+1}')
        
        print(f"📊 Sample {sample_idx+1}: Min={min(sample_amplitudes):.3f}, "
              f"Max={max(sample_amplitudes):.3f}, Mean={np.mean(sample_amplitudes):.3f}")
    
    # Customize the plot
    plt.title(f'CSI Amplitude vs Subcarrier Index\nPoint ({point_x}, {point_y}) - {len(selected_indices)} Random Samples', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Subcarrier Index', fontsize=12)
    plt.ylabel('CSI Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='upper right')
    
    # Remove the statistics box as requested
    
    # Set x-axis to show all subcarriers
    plt.xlim(-1, 52)
    plt.xticks(range(0, 52, 4))  # Show every 4th subcarrier for clarity
    
    # Save the plot
    output_filename = f"point_{point_x}_{point_y}_amplitude_analysis.png"
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved as: {output_filename}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 SUMMARY STATISTICS for Point ({point_x}, {point_y}):")
    print("="*60)
    for i, sample_idx in enumerate(selected_indices):
        sample_amps = amplitudes[sample_idx]
        sample_rssi = sample_info[sample_idx]['rssi']
        print(f"Sample {sample_idx+1:2d}: RSSI={sample_rssi:6.1f} dBm, "
              f"Min={min(sample_amps):6.3f}, Max={max(sample_amps):6.3f}, "
              f"Mean={np.mean(sample_amps):6.3f}, Std={np.std(sample_amps):6.3f}")
    
    # Get selected amplitudes for return
    all_selected_amps = [amplitudes[i] for i in selected_indices]
    return selected_indices, all_selected_amps

def main():
    """Main function"""
    print("🎯 CSI AMPLITUDE ANALYSIS FOR POINT (2,4)")
    print("="*60)
    
    # Plot amplitudes for point (2,4)
    point_x, point_y = 2, 4
    num_samples = 5
    
    # You can change the dataset here if needed
    dataset_options = [
        "CSI Dataset 250 Samples",
        "CSI Dataset 500 Samples", 
        "CSI Dataset 750 Samples"
    ]
    
    dataset = dataset_options[0]  # Using 250 samples by default
    
    print(f"📍 Target Point: ({point_x}, {point_y})")
    print(f"📊 Number of samples to plot: {num_samples}")
    print(f"📁 Dataset: {dataset}")
    print()
    
    # Create the plot
    selected_indices, amplitudes = plot_amplitude_lines(point_x, point_y, num_samples, dataset)
    
    if selected_indices:
        print(f"\n✅ Successfully plotted {len(selected_indices)} samples from point ({point_x}, {point_y})")
        print(f"🎲 Random samples used: {[i+1 for i in selected_indices]}")
    else:
        print(f"❌ Failed to create plot for point ({point_x}, {point_y})")

if __name__ == "__main__":
    main()
