#!/usr/bin/env python3
"""
Improved RSSI Spatial Distribution Plot

Creates RSSI spatial plot with:
- Integer RSSI values (no decimals)
- Red-to-blue color palette for better visualization
- Enhanced readability and aesthetics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import re

def plot_improved_rssi_spatial():
    """
    Create improved RSSI spatial distribution plot.
    """
    print("📡 Creating improved RSSI spatial plot...")
    
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
            # Sort RSSI values and calculate median, then round to integer
            rssi_sorted = sorted(rssi_values)
            rssi_median_int = int(round(np.median(rssi_sorted)))
            
            locations.append({
                'x': x,
                'y': y,
                'rssi_median_int': rssi_median_int
            })
    
    df = pd.DataFrame(locations)
    print(f"   Loaded {len(df)} locations")
    
    # Create the improved plot
    plt.figure(figsize=(14, 10))
    
    # Use red-to-blue colormap (RdBu_r for red-high, blue-low)
    scatter = plt.scatter(df['x'], df['y'], c=df['rssi_median_int'], 
                         s=500, cmap='RdBu_r', alpha=0.9, 
                         edgecolors='black', linewidth=2.5)
    
    # Add integer value labels that fit within the circles
    for _, row in df.iterrows():
        plt.annotate(f'{row["rssi_median_int"]}', 
                    (row['x'], row['y']), 
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', 
                    color='black')
    
    plt.xlabel('X Coordinate (meters)', fontsize=16, fontweight='bold')
    plt.ylabel('Y Coordinate (meters)', fontsize=16, fontweight='bold')
    plt.title('RSSI Signal Strength Distribution', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Enhanced colorbar
    cbar = plt.colorbar(scatter, shrink=0.8, pad=0.02)
    cbar.set_label('RSSI (dBm)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Ensure colorbar shows integer ticks
    rssi_min = int(df['rssi_median_int'].min())
    rssi_max = int(df['rssi_median_int'].max())
    tick_values = list(range(rssi_min, rssi_max + 1, 2))  # Every 2 dB
    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([f'{val}' for val in tick_values])
    
    plt.grid(True, alpha=0.4, linewidth=0.8)
    plt.gca().set_aspect('equal')
    
    # Enhanced statistics box - moved to top right of the plot area
    rssi_range = df['rssi_median_int'].max() - df['rssi_median_int'].min()
    
    # Get the current axes to place text within the plot
    ax = plt.gca()
    
    # Place text box in the top right corner of the plot
    ax.text(0.98, 0.98, 
            f'RSSI Range: {rssi_range} dB\nMin: {df["rssi_median_int"].min()} dBm\nMax: {df["rssi_median_int"].max()} dBm',
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Improve tick labels
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    
    # Add border around the plot
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # Save with high quality
    output_dir = Path("Analysis_CSI_Dataset_750_Samples")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'rssi_spatial_distribution_improved.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
    
    # Also overwrite the original file
    plt.savefig(output_dir / 'rssi_spatial_distribution.png', 
                dpi=300, bbox_inches='tight', facecolor='white', edgecolor='black')
    
    plt.show()
    
    print(f"✅ Improved RSSI spatial plot created!")
    print(f"📁 Saved as: Analysis_CSI_Dataset_750_Samples/rssi_spatial_distribution_improved.png")
    print(f"📁 Updated: Analysis_CSI_Dataset_750_Samples/rssi_spatial_distribution.png")
    print(f"📊 RSSI range: {rssi_range} dB ({df['rssi_median_int'].min()} to {df['rssi_median_int'].max()} dBm)")
    print(f"🎨 Color scheme: Red = Strong signal, Blue = Weak signal")
    print(f"🔢 Values displayed: Integer RSSI (no decimals)")
    
    return df

if __name__ == "__main__":
    plot_improved_rssi_spatial()
