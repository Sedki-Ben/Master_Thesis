#!/usr/bin/env python3
"""
Amplitude Analysis for 6 Specific Points from CSI Dataset 250 Samples
Plots 5 random samples from each point showing amplitude across 52 subcarriers
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

def plot_amplitude_analysis():
    """Create comprehensive amplitude analysis plot for 6 points"""
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define the 6 points to analyze
    points_to_analyze = [
        (1, 0), (2, 1), (1, 1), (2, 4), (4, 5), (5, 3)
    ]
    
    # Color palette for different points
    colors = ['#FF6B35', '#1B998B', '#2E86AB', '#A23B72', '#F21905', '#0D7377']
    
    # Set up the plot
    plt.figure(figsize=(16, 10))
    
    # Subcarrier indices (0-51)
    subcarriers = list(range(52))
    
    all_samples_data = []  # Store for reproducibility
    
    # Process each point
    for point_idx, point_coords in enumerate(points_to_analyze):
        print(f"Loading samples for point {point_coords}...")
        
        # Load 5 random samples for this point
        samples = load_csi_samples(point_coords, num_samples=5)
        
        # Plot each sample
        for sample_idx, sample in enumerate(samples):
            amplitudes = sample['amplitude']
            x, y = sample['coordinates']
            
            # Create label with point coordinates and sample number
            label = f"Point({x},{y}) Sample#{sample_idx+1}"
            
            # Plot amplitude across subcarriers
            plt.plot(subcarriers, amplitudes, 
                    color=colors[point_idx], 
                    alpha=0.7,
                    linewidth=1.5,
                    marker='.',
                    markersize=3,
                    label=label)
            
            # Store for reproducibility
            all_samples_data.append({
                'point': point_coords,
                'sample_number': sample_idx + 1,
                'amplitudes': amplitudes,
                'color': colors[point_idx],
                'label': label
            })
    
    # Customize the plot
    plt.xlabel('Subcarrier Index (0-51)', fontsize=14, fontweight='bold')
    plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
    plt.title('CSI Amplitude Analysis: 6 Points × 5 Random Samples Each\n'
              'Dataset: CSI Dataset 250 Samples', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis ticks to show all subcarriers
    plt.xticks(range(0, 52, 5))  # Show every 5th subcarrier for readability
    plt.grid(True, alpha=0.3)
    
    # Create legend with custom positioning
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
              fontsize=10, framealpha=0.9)
    
    # Add summary text box
    summary_text = f"""Analysis Summary:
Points: {len(points_to_analyze)} locations
Samples per point: 5 random samples
Total samples: {len(all_samples_data)}
Subcarriers: 52 (0-51)
Random seed: 42 (for reproducibility)"""
    
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'amplitude_analysis_6_points_5_samples.png'
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
            print(f"    First 10 amplitudes: {sample['amplitudes'][:10]}")
    
    return all_samples_data

def create_individual_plots(all_samples_data):
    """
    Create individual plots for each point (for future use)
    This function demonstrates how to create separate plots
    """
    print("\n>>> Creating individual plots for each point...")
    
    points = list(set([s['point'] for s in all_samples_data]))
    
    for point_coords in points:
        point_samples = [s for s in all_samples_data if s['point'] == point_coords]
        
        plt.figure(figsize=(12, 8))
        
        subcarriers = list(range(52))
        
        for sample in point_samples:
            plt.plot(subcarriers, sample['amplitudes'],
                    color=sample['color'],
                    alpha=0.8,
                    linewidth=2,
                    marker='.',
                    markersize=4,
                    label=f"Sample #{sample['sample_number']}")
        
        x, y = point_coords
        plt.xlabel('Subcarrier Index (0-51)', fontsize=14, fontweight='bold')
        plt.ylabel('Amplitude', fontsize=14, fontweight='bold')
        plt.title(f'CSI Amplitude Analysis: Point ({x},{y})\n'
                  f'5 Random Samples from CSI Dataset 250', 
                  fontsize=16, fontweight='bold')
        
        plt.xticks(range(0, 52, 5))
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save individual plot
        output_file = f'amplitude_point_{x}_{y}_individual.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        
        plt.close()  # Close to free memory

def main():
    """Main execution function"""
    print("🔬 CSI AMPLITUDE ANALYSIS: 6 POINTS × 5 SAMPLES")
    print("="*60)
    
    # Create the main comprehensive plot
    all_samples_data = plot_amplitude_analysis()
    
    # Optionally create individual plots (uncomment if needed)
    # create_individual_plots(all_samples_data)
    
    print(f"\n✅ Amplitude analysis complete!")
    print(f"📊 Total samples plotted: {len(all_samples_data)}")
    print(f"📁 Files saved with reproducible random seed (42)")
    
    return all_samples_data

if __name__ == "__main__":
    samples_data = main()

