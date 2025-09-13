#!/usr/bin/env python3
"""
RSSI Spatial Distribution Plot with Training/Validation/Test Points

Creates RSSI spatial plot with:
- Training points as circles
- Validation points as triangles 
- Test points as squares
- Integer RSSI values with red-to-blue color palette
- Legend showing only training vs test distinction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import re

def plot_rssi_with_validation_triangles():
    """
    Create RSSI spatial distribution plot with different shapes for train/val/test.
    """
    print("📡 Creating RSSI plot with validation triangles...")
    
    # Define the point splits
    test_points = [(0.5, 0.5), (1.5, 4.5), (2.5, 2.5), (3.5, 1.5), (5.5, 3.5)]
    validation_points = [(4, 5), (5, 1), (0, 3), (0, 6), (6, 4), (2, 1), (3, 3)]
    
    # All reference points
    all_reference_points = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 0), (1, 1), (1, 4), (1, 5),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 0), (4, 1), (4, 4), (4, 5),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
        (6, 3), (6, 4)
    ]
    
    # Training points
    training_points = [p for p in all_reference_points if p not in validation_points]
    
    # Load training/validation data (reference points with RSSI)
    ref_data_dir = Path("CSI Dataset 750 Samples")
    training_locations = []
    validation_locations = []
    
    for file_path in ref_data_dir.glob("*.csv"):
        match = re.match(r'(\d+),(\d+)\.csv', file_path.name)
        if not match:
            continue
        
        x, y = int(match.group(1)), int(match.group(2))
        
        # Read RSSI values
        rssi_values = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 50:  # Limit for speed
                    break
                rssi_values.append(float(row['rssi']))
        
        if rssi_values:
            rssi_median_int = int(round(np.median(sorted(rssi_values))))
            
            location_data = {
                'x': x,
                'y': y,
                'rssi_median_int': rssi_median_int
            }
            
            if (x, y) in training_points:
                training_locations.append(location_data)
            elif (x, y) in validation_points:
                validation_locations.append(location_data)
    
    # Load test data
    test_data_dir = Path("Testing Points Dataset 750 Samples")
    test_locations = []
    
    for file_path in test_data_dir.glob("*.csv"):
        match = re.match(r'([\d.]+),([\d.]+)\.csv', file_path.name)
        if not match:
            continue
        
        x, y = float(match.group(1)), float(match.group(2))
        
        # Read RSSI values
        rssi_values = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 50:  # Limit for speed
                    break
                rssi_values.append(float(row['rssi']))
        
        if rssi_values:
            rssi_median_int = int(round(np.median(sorted(rssi_values))))
            
            test_locations.append({
                'x': x,
                'y': y,
                'rssi_median_int': rssi_median_int
            })
    
    # Convert to DataFrames
    df_train = pd.DataFrame(training_locations)
    df_val = pd.DataFrame(validation_locations)
    df_test = pd.DataFrame(test_locations)
    
    print(f"   Training points: {len(df_train)}")
    print(f"   Validation points: {len(df_val)}")
    print(f"   Test points: {len(df_test)}")
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Combine all data to get consistent color scale
    all_rssi = list(df_train['rssi_median_int']) + list(df_val['rssi_median_int']) + list(df_test['rssi_median_int'])
    vmin, vmax = min(all_rssi), max(all_rssi)
    
    # Plot all points as circles
    scatter_train = plt.scatter(df_train['x'], df_train['y'], c=df_train['rssi_median_int'],
                               s=500, cmap='RdBu_r', alpha=0.9, edgecolors='black', linewidth=2.5,
                               marker='o', vmin=vmin, vmax=vmax)
    
    # Plot validation points as circles
    scatter_val = plt.scatter(df_val['x'], df_val['y'], c=df_val['rssi_median_int'],
                             s=500, cmap='RdBu_r', alpha=0.9, edgecolors='black', linewidth=2.5,
                             marker='o', vmin=vmin, vmax=vmax)
    
    # Plot test points as circles
    scatter_test = plt.scatter(df_test['x'], df_test['y'], c=df_test['rssi_median_int'],
                              s=500, cmap='RdBu_r', alpha=0.9, edgecolors='black', linewidth=2.5,
                              marker='o', vmin=vmin, vmax=vmax)
    
    # Add integer RSSI value labels (no decimals)
    for _, row in df_train.iterrows():
        plt.annotate(f'{int(row["rssi_median_int"])}', 
                    (row['x'], row['y']), 
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', 
                    color='black')
    
    for _, row in df_val.iterrows():
        plt.annotate(f'{int(row["rssi_median_int"])}', 
                    (row['x'], row['y']), 
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', 
                    color='black')
    
    for _, row in df_test.iterrows():
        plt.annotate(f'{int(row["rssi_median_int"])}', 
                    (row['x'], row['y']), 
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', 
                    color='black')
    
    # Set up the plot
    plt.xlabel('X Coordinate (meters)', fontsize=16, fontweight='bold')
    plt.ylabel('Y Coordinate (meters)', fontsize=16, fontweight='bold')
    plt.title('RSSI Signal Strength Distribution', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Enhanced colorbar
    cbar = plt.colorbar(scatter_train, shrink=0.8, pad=0.02)
    cbar.set_label('RSSI (dBm)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Set colorbar ticks
    rssi_min = min(all_rssi)
    rssi_max = max(all_rssi)
    tick_values = list(range(rssi_min, rssi_max + 1, 2))
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{val}' for val in tick_values])
    
    # No legend (removed as requested)
    
    plt.grid(True, alpha=0.4, linewidth=0.8)
    plt.gca().set_aspect('equal')
    
    # Improve tick labels
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    # Add border
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # Save the plot
    output_dir = Path("Analysis_CSI_Dataset_750_Samples")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'rssi_distribution_with_validation_triangles.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
    
    plt.show()
    
    print(f"✅ RSSI plot with validation triangles created!")
    print(f"📁 Saved as: Analysis_CSI_Dataset_750_Samples/rssi_distribution_with_validation_triangles.png")
    print(f"🔵 Training: {len(df_train)} circles")
    print(f"🔺 Validation: {len(df_val)} triangles") 
    print(f"🔴 Test: {len(df_test)} squares")
    print(f"📊 RSSI range: {rssi_max - rssi_min} dB ({rssi_min} to {rssi_max} dBm)")

def main():
    """Main execution function"""
    print(f"📡 RSSI DISTRIBUTION WITH VALIDATION TRIANGLES")
    print(f"="*50)
    
    plot_rssi_with_validation_triangles()
    
    print(f"\n✅ Plot generation complete!")

if __name__ == "__main__":
    main()
