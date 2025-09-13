#!/usr/bin/env python3
"""
Amplitude Analysis for 6 Specific Points from CSI Dataset 250 Samples
Creates 6 separate subplots, one for each point, showing 5 random samples each
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import random

def load_csi_samples(point_coords, dataset_folder="CSI Dataset 250 Samples", num_samples=5):
    """
    Load random samples from a specific coordinate point
    
    Args:
        point_coords: tuple (x, y) coordinates
        dataset_folder: folder containing the CSV files
        num_samples: number of random samples to extract
    
    Returns:
        list of dictionaries with amplitude, phase, rssi data
    """
    x, y = point_coords
    file_path = Path(dataset_folder) / f"{x},{y}.csv"
    
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return []
    
    # Load all samples from the file
    all_samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                amplitudes = json.loads(row['amplitude'])
                phases = json.loads(row['phase'])
                rssi = float(row['rssi'])
                
                if len(amplitudes) == 52 and len(phases) == 52:
                    all_samples.append({
                        'amplitude': amplitudes,
                        'phase': phases,
                        'rssi': rssi,
                        'coordinates': (x, y)
                    })
            except:
                continue
    
    # Randomly select samples
    if len(all_samples) >= num_samples:
        selected_samples = random.sample(all_samples, num_samples)
    else:
        selected_samples = all_samples
        print(f"Warning: Only {len(all_samples)} samples available for point ({x},{y})")
    
    return selected_samples

def plot_amplitude_subplots():
    """Create 6 subplots showing amplitude analysis for each point separately"""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define the 6 points to analyze
    points_to_analyze = [
        (1, 0), (2, 1), (1, 1), (2, 4), (4, 5), (5, 3)
    ]
    
    # Color palette for different samples within each point
    sample_colors = ['#FF6B35', '#1B998B', '#2E86AB', '#A23B72', '#F21905']
    
    # Create figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CSI Amplitude Analysis: 6 Points × 5 Random Samples Each\n'
                 'Dataset: CSI Dataset 250 Samples', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Subcarrier indices (0-51)
    subcarriers = list(range(52))
    
    all_samples_data = []  # Store for reproducibility
    
    # Process each point
    for point_idx, point_coords in enumerate(points_to_analyze):
        ax = axes[point_idx]
        x, y = point_coords
        
        print(f"Loading samples for point {point_coords}...")
        
        # Load 5 random samples for this point
        samples = load_csi_samples(point_coords, num_samples=5)
        
        # Plot each sample in this subplot
        for sample_idx, sample in enumerate(samples):
            amplitudes = sample['amplitude']
            
            # Create label with sample number
            label = f"Sample #{sample_idx+1}"
            
            # Plot amplitude across subcarriers
            ax.plot(subcarriers, amplitudes, 
                   color=sample_colors[sample_idx], 
                   alpha=0.8,
                   linewidth=2,
                   marker='.',
                   markersize=4,
                   label=label)
            
            # Store for reproducibility
            all_samples_data.append({
                'point': point_coords,
                'sample_number': sample_idx + 1,
                'amplitudes': amplitudes,
                'color': sample_colors[sample_idx],
                'label': label
            })
        
        # Customize each subplot
        ax.set_title(f'Point ({x}, {y})', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Subcarrier Index', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='upper right')
        
        # Set x-axis ticks
        ax.set_xticks(range(0, 52, 10))  # Show every 10th subcarrier
        ax.set_xlim(0, 51)
        
        # Calculate and display amplitude range for this point
        if samples:
            all_amps = []
            for sample in samples:
                all_amps.extend(sample['amplitude'])
            amp_min, amp_max = min(all_amps), max(all_amps)
            ax.text(0.02, 0.98, f'Range: {amp_min:.1f} - {amp_max:.1f}', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add overall information
    fig.text(0.02, 0.02, 
             f'Analysis Details: Random seed=42, Total samples={len(all_samples_data)}, '
             f'Subcarriers=52, Points analyzed: {len(points_to_analyze)}',
             fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)
    
    # Save the plot
    output_file = 'amplitude_analysis_6_subplots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f">>> Plot saved as '{output_file}'")
    
    plt.show()
    
    # Print detailed sample information for reproducibility
    print("\n" + "="*80)
    print("SAMPLE DETAILS FOR REPRODUCIBILITY")
    print("="*80)
    
    for point_coords in points_to_analyze:
        point_samples = [s for s in all_samples_data if s['point'] == point_coords]
        print(f"\nPoint {point_coords}:")
        for sample in point_samples:
            print(f"  Sample #{sample['sample_number']}: Color={sample['color']}")
            print(f"    Amplitude range: {min(sample['amplitudes']):.2f} - {max(sample['amplitudes']):.2f}")
            print(f"    Mean amplitude: {np.mean(sample['amplitudes']):.2f}")
            print(f"    First 5 amplitudes: {sample['amplitudes'][:5]}")
    
    return all_samples_data

def analyze_amplitude_patterns(all_samples_data):
    """Analyze patterns in the amplitude data"""
    print("\n" + "="*80)
    print("AMPLITUDE PATTERN ANALYSIS")
    print("="*80)
    
    points_to_analyze = [
        (1, 0), (2, 1), (1, 1), (2, 4), (4, 5), (5, 3)
    ]
    
    for point_coords in points_to_analyze:
        point_samples = [s for s in all_samples_data if s['point'] == point_coords]
        
        if point_samples:
            # Calculate statistics across all samples for this point
            all_point_amps = []
            for sample in point_samples:
                all_point_amps.extend(sample['amplitudes'])
            
            mean_amp = np.mean(all_point_amps)
            std_amp = np.std(all_point_amps)
            min_amp = min(all_point_amps)
            max_amp = max(all_point_amps)
            
            print(f"\nPoint {point_coords}:")
            print(f"  Mean amplitude: {mean_amp:.2f}")
            print(f"  Std deviation: {std_amp:.2f}")
            print(f"  Range: {min_amp:.2f} - {max_amp:.2f}")
            print(f"  Amplitude span: {max_amp - min_amp:.2f}")

def main():
    """Main execution function"""
    print("🔬 CSI AMPLITUDE ANALYSIS: 6 SUBPLOTS FOR 6 POINTS")
    print("="*60)
    
    # Create the subplot analysis
    all_samples_data = plot_amplitude_subplots()
    
    # Analyze patterns
    analyze_amplitude_patterns(all_samples_data)
    
    print(f"\n✅ Amplitude subplot analysis complete!")
    print(f"📊 Total samples plotted: {len(all_samples_data)}")
    print(f"📁 Plot saved with 6 separate subplots")
    print(f"🔄 Reproducible with random seed 42")
    
    return all_samples_data

if __name__ == "__main__":
    samples_data = main()

