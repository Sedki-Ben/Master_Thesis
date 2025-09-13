#!/usr/bin/env python3
"""
Simple RSSI Spatial Plot with Color Graduation

Fast and focused RSSI visualization for the 750-sample dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import re

def plot_rssi_spatial():
    """
    Create RSSI spatial distribution plot with color graduation.
    """
    print("📡 Creating RSSI spatial plot...")
    
    # Load data quickly
    data_dir = Path("CSI Dataset 750 Samples")
    locations = []
    
    for file_path in data_dir.glob("*.csv"):
        match = re.match(r'(\d+),(\d+)\.csv', file_path.name)
        if not match:
            continue
        
        x, y = int(match.group(1)), int(match.group(2))
        
        # Read first 50 samples for speed
        rssi_values = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 50:  # Limit for speed
                    break
                rssi_values.append(float(row['rssi']))
        
        if rssi_values:
            # Sort RSSI values and calculate median
            rssi_sorted = sorted(rssi_values)
            locations.append({
                'x': x,
                'y': y,
                'rssi_median': np.median(rssi_sorted)
            })
    
    df = pd.DataFrame(locations)
    print(f"   Loaded {len(df)} locations")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Main scatter plot with color graduation
    scatter = plt.scatter(df['x'], df['y'], c=df['rssi_median'], 
                         s=400, cmap='plasma', alpha=0.8, 
                         edgecolors='black', linewidth=2)
    
    # Add value labels
    for _, row in df.iterrows():
        plt.annotate(f'{row["rssi_median"]:.1f}', 
                    (row['x'], row['y']), 
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    
    plt.xlabel('X Coordinate (meters)', fontsize=14, fontweight='bold')
    plt.ylabel('Y Coordinate (meters)', fontsize=14, fontweight='bold')
    plt.title('RSSI Signal Strength Distribution - 750 Sample Dataset\n(Median Values After Sorting - Color Graduation)', 
              fontsize=16, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label('Median RSSI (dBm)', fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    # Statistics
    rssi_range = df['rssi_median'].max() - df['rssi_median'].min()
    plt.figtext(0.02, 0.02, 
                f'RSSI Range: {rssi_range:.1f} dB | Min: {df["rssi_median"].min():.1f} dBm | Max: {df["rssi_median"].max():.1f} dBm',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Save
    output_dir = Path("Analysis_CSI_Dataset_750_Samples")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'rssi_spatial_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ RSSI spatial plot created!")
    print(f"📁 Saved: Analysis_CSI_Dataset_750_Samples/rssi_spatial_distribution.png")
    print(f"📊 RSSI range: {rssi_range:.1f} dB ({df['rssi_median'].min():.1f} to {df['rssi_median'].max():.1f} dBm)")
    print(f"📊 Using median values from sorted RSSI samples per location")

if __name__ == "__main__":
    plot_rssi_spatial()
