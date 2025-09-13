#!/usr/bin/env python3
"""
RSSI Signal Strength Visualization for 750-Sample Dataset

Creates detailed RSSI spatial distribution plots with color graduation
to analyze signal attenuation patterns and spatial characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pathlib import Path
import re
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Set high-quality plotting parameters
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (16, 12),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 100
})

def load_rssi_data(data_dir):
    """
    Load RSSI data from the 750-sample dataset.
    """
    print("📡 Loading RSSI data from 750-sample dataset...")
    
    all_data = []
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    
    for file_path in csv_files:
        # Parse coordinates from filename
        match = re.match(r'(\d+),(\d+)\.csv', file_path.name)
        if not match:
            continue
        
        x, y = int(match.group(1)), int(match.group(2))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            rssi_values = []
            for row in reader:
                rssi = float(row['rssi'])
                rssi_values.append(rssi)
            
            # Calculate statistics for this location
            if rssi_values:
                location_data = {
                    'x': x,
                    'y': y,
                    'location': f"({x},{y})",
                    'rssi_mean': np.mean(rssi_values),
                    'rssi_std': np.std(rssi_values),
                    'rssi_min': np.min(rssi_values),
                    'rssi_max': np.max(rssi_values),
                    'rssi_median': np.median(rssi_values),
                    'sample_count': len(rssi_values),
                    'rssi_values': rssi_values
                }
                all_data.append(location_data)
    
    df = pd.DataFrame(all_data)
    print(f"   ✅ Loaded data from {len(df)} locations")
    print(f"   📊 Total samples: {df['sample_count'].sum():,}")
    print(f"   📡 RSSI range: {df['rssi_mean'].min():.1f} to {df['rssi_mean'].max():.1f} dBm")
    
    return df

def create_rssi_visualizations(df):
    """
    Create comprehensive RSSI visualizations with color gradation.
    """
    print("🎨 Creating RSSI visualizations...")
    
    # Create the main figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('RSSI Signal Strength Analysis - 750 Sample Dataset', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # 1. Main RSSI Spatial Distribution (Large plot)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Create scatter plot with color graduation
    scatter = ax1.scatter(df['x'], df['y'], c=df['rssi_mean'], 
                         s=300, cmap='plasma', alpha=0.8, 
                         edgecolors='black', linewidth=2)
    
    # Add location labels
    for _, row in df.iterrows():
        ax1.annotate(f'{row["rssi_mean"]:.1f}', 
                    (row['x'], row['y']), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax1.set_xlabel('X Coordinate (meters)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Y Coordinate (meters)', fontsize=13, fontweight='bold')
    ax1.set_title('Mean RSSI Distribution Across Room\n(Signal Attenuation Pattern)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar1.set_label('Mean RSSI (dBm)', fontsize=12, fontweight='bold')
    cbar1.ax.tick_params(labelsize=10)
    
    # 2. RSSI Standard Deviation (Variability)
    ax2 = fig.add_subplot(gs[0, 2])
    
    scatter2 = ax2.scatter(df['x'], df['y'], c=df['rssi_std'], 
                          s=200, cmap='viridis', alpha=0.8,
                          edgecolors='black', linewidth=1.5)
    
    ax2.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax2.set_ylabel('Y Coordinate (m)', fontweight='bold')
    ax2.set_title('RSSI Variability\n(Temporal Stability)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    cbar2.set_label('RSSI Std (dB)', fontsize=10, fontweight='bold')
    
    # 3. RSSI Distribution Histogram
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Plot histogram with color gradient
    n_bins = 25
    counts, bins, patches = ax3.hist(df['rssi_mean'], bins=n_bins, 
                                    alpha=0.7, edgecolor='black', linewidth=1)
    
    # Color gradient for histogram bars
    fracs = counts / counts.max()
    norm = plt.Normalize(fracs.min(), fracs.max())
    for frac, patch in zip(fracs, patches):
        color = plt.cm.plasma(norm(frac))
        patch.set_facecolor(color)
    
    ax3.set_xlabel('Mean RSSI (dBm)', fontweight='bold')
    ax3.set_ylabel('Number of Locations', fontweight='bold')
    ax3.set_title('RSSI Distribution\n(Signal Strength Spread)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics
    mean_rssi = df['rssi_mean'].mean()
    std_rssi = df['rssi_mean'].std()
    ax3.axvline(mean_rssi, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_rssi:.1f} dBm')
    ax3.axvline(mean_rssi - std_rssi, color='orange', linestyle=':', linewidth=2, 
                label=f'-1σ: {mean_rssi - std_rssi:.1f} dBm')
    ax3.axvline(mean_rssi + std_rssi, color='orange', linestyle=':', linewidth=2, 
                label=f'+1σ: {mean_rssi + std_rssi:.1f} dBm')
    ax3.legend()
    
    # 4. Distance vs RSSI Analysis
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Calculate distance from origin (0,0)
    df['distance'] = np.sqrt(df['x']**2 + df['y']**2)
    
    # Create scatter plot with color by location
    scatter4 = ax4.scatter(df['distance'], df['rssi_mean'], 
                          c=df['y'], s=150, cmap='coolwarm', 
                          alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add trend line
    z = np.polyfit(df['distance'], df['rssi_mean'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['distance'].min(), df['distance'].max(), 100)
    ax4.plot(x_trend, p(x_trend), "r--", linewidth=2, alpha=0.8, 
             label=f'Trend: {z[0]:.2f}x + {z[1]:.1f}')
    
    ax4.set_xlabel('Distance from Origin (m)', fontweight='bold')
    ax4.set_ylabel('Mean RSSI (dBm)', fontweight='bold')
    ax4.set_title('Distance-Attenuation\n(Path Loss Analysis)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    cbar4 = plt.colorbar(scatter4, ax=ax4, shrink=0.8)
    cbar4.set_label('Y Coordinate', fontsize=10, fontweight='bold')
    
    # 5. RSSI Range (Min-Max) Analysis
    ax5 = fig.add_subplot(gs[1, 2])
    
    df['rssi_range'] = df['rssi_max'] - df['rssi_min']
    
    scatter5 = ax5.scatter(df['x'], df['y'], c=df['rssi_range'], 
                          s=200, cmap='hot', alpha=0.8,
                          edgecolors='black', linewidth=1.5)
    
    ax5.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax5.set_ylabel('Y Coordinate (m)', fontweight='bold')
    ax5.set_title('RSSI Dynamic Range\n(Signal Variability)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    cbar5 = plt.colorbar(scatter5, ax=ax5, shrink=0.8)
    cbar5.set_label('RSSI Range (dB)', fontsize=10, fontweight='bold')
    
    # 6. 3D Surface Plot
    ax6 = fig.add_subplot(gs[2, :], projection='3d')
    
    # Create meshgrid for interpolation
    x_unique = sorted(df['x'].unique())
    y_unique = sorted(df['y'].unique())
    X, Y = np.meshgrid(x_unique, y_unique)
    
    # Create RSSI matrix
    Z = np.full(X.shape, np.nan)
    for i, y_val in enumerate(y_unique):
        for j, x_val in enumerate(x_unique):
            match = df[(df['x'] == x_val) & (df['y'] == y_val)]
            if not match.empty:
                Z[i, j] = match['rssi_mean'].iloc[0]
    
    # Create surface plot
    surf = ax6.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, 
                           linewidth=0, antialiased=True)
    
    # Add data points
    ax6.scatter(df['x'], df['y'], df['rssi_mean'], 
               c=df['rssi_mean'], cmap='plasma', s=50, alpha=1.0)
    
    ax6.set_xlabel('X Coordinate (m)', fontweight='bold')
    ax6.set_ylabel('Y Coordinate (m)', fontweight='bold')
    ax6.set_zlabel('RSSI (dBm)', fontweight='bold')
    ax6.set_title('3D RSSI Surface\n(Signal Propagation Landscape)', fontweight='bold')
    
    # Add colorbar for 3D plot
    cbar6 = plt.colorbar(surf, ax=ax6, shrink=0.6)
    cbar6.set_label('RSSI (dBm)', fontsize=12, fontweight='bold')
    
    # Save the plot
    output_dir = Path("Analysis_CSI_Dataset_750_Samples")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'rssi_detailed_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return df

def generate_rssi_summary_stats(df):
    """
    Generate detailed RSSI statistics and analysis.
    """
    print("\n📊 DETAILED RSSI ANALYSIS RESULTS:")
    print("=" * 70)
    
    # Basic statistics
    print(f"📡 SIGNAL STRENGTH CHARACTERISTICS:")
    print(f"   • Overall RSSI range: {df['rssi_mean'].min():.1f} to {df['rssi_mean'].max():.1f} dBm")
    print(f"   • Dynamic range: {df['rssi_mean'].max() - df['rssi_mean'].min():.1f} dB")
    print(f"   • Mean RSSI: {df['rssi_mean'].mean():.1f} ± {df['rssi_mean'].std():.1f} dBm")
    print(f"   • Median RSSI: {df['rssi_mean'].median():.1f} dBm")
    
    # Spatial gradient analysis
    print(f"\n🗺️  SPATIAL GRADIENT ANALYSIS:")
    corner_locations = [
        (df['x'].min(), df['y'].min()),  # Bottom-left
        (df['x'].max(), df['y'].min()),  # Bottom-right
        (df['x'].min(), df['y'].max()),  # Top-left
        (df['x'].max(), df['y'].max())   # Top-right
    ]
    
    corner_rssi = []
    for x, y in corner_locations:
        match = df[(df['x'] == x) & (df['y'] == y)]
        if not match.empty:
            rssi_val = match['rssi_mean'].iloc[0]
            corner_rssi.append(rssi_val)
            print(f"   • Corner ({x},{y}): {rssi_val:.1f} dBm")
    
    if len(corner_rssi) > 1:
        corner_range = max(corner_rssi) - min(corner_rssi)
        print(f"   • Corner-to-corner range: {corner_range:.1f} dB")
    
    # Distance correlation
    df['distance'] = np.sqrt(df['x']**2 + df['y']**2)
    correlation = np.corrcoef(df['distance'], df['rssi_mean'])[0, 1]
    print(f"\n📐 DISTANCE-ATTENUATION ANALYSIS:")
    print(f"   • Distance-RSSI correlation: {correlation:.3f}")
    print(f"   • Path loss pattern: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'}")
    
    # Variability analysis
    print(f"\n📊 SIGNAL VARIABILITY ANALYSIS:")
    print(f"   • Mean temporal variability: {df['rssi_std'].mean():.2f} dB")
    print(f"   • Most stable location: {df.loc[df['rssi_std'].idxmin(), 'location']} ({df['rssi_std'].min():.2f} dB std)")
    print(f"   • Most variable location: {df.loc[df['rssi_std'].idxmax(), 'location']} ({df['rssi_std'].max():.2f} dB std)")
    
    # Localization potential
    print(f"\n🎯 LOCALIZATION POTENTIAL ASSESSMENT:")
    unique_pairs = len(df)
    theoretical_max = unique_pairs * (unique_pairs - 1) / 2
    
    # Calculate distinguishable pairs (assuming 2 dB minimum difference for reliable distinction)
    distinguishable_count = 0
    rssi_values = df['rssi_mean'].values
    for i in range(len(rssi_values)):
        for j in range(i + 1, len(rssi_values)):
            if abs(rssi_values[i] - rssi_values[j]) >= 2.0:  # 2 dB threshold
                distinguishable_count += 1
    
    distinguishability = distinguishable_count / theoretical_max if theoretical_max > 0 else 0
    
    print(f"   • Total location pairs: {unique_pairs}")
    print(f"   • Distinguishable pairs (≥2dB): {distinguishable_count}")
    print(f"   • Distinguishability ratio: {distinguishability:.2%}")
    print(f"   • CNN classification potential: {'EXCELLENT' if distinguishability > 0.8 else 'GOOD' if distinguishability > 0.6 else 'MODERATE'}")
    
    # Expected CNN performance
    base_accuracy = 70
    rssi_contribution = min(25, (df['rssi_mean'].max() - df['rssi_mean'].min()) * 1.5)
    stability_contribution = min(5, (2.0 - df['rssi_std'].mean()) * 2.5)
    
    expected_accuracy = base_accuracy + rssi_contribution + stability_contribution
    
    print(f"\n🧠 CNN PERFORMANCE PREDICTION:")
    print(f"   • RSSI-only baseline accuracy: ~{expected_accuracy:.0f}%")
    print(f"   • With amplitude/phase features: ~{min(95, expected_accuracy + 15):.0f}%")
    print(f"   • Training difficulty: {'Easy' if expected_accuracy > 85 else 'Moderate' if expected_accuracy > 75 else 'Challenging'}")
    
    return df

def main():
    """
    Main function to create RSSI visualizations for 750-sample dataset.
    """
    print("🎯 RSSI SIGNAL STRENGTH VISUALIZATION - 750 SAMPLE DATASET")
    print("=" * 70)
    print("Creating detailed spatial distribution plots with color graduation...")
    
    # Load data
    data_dir = "CSI Dataset 750 Samples"
    df = load_rssi_data(data_dir)
    
    # Create visualizations
    df_with_analysis = create_rssi_visualizations(df)
    
    # Generate summary statistics
    generate_rssi_summary_stats(df_with_analysis)
    
    print(f"\n✅ RSSI ANALYSIS COMPLETE!")
    print(f"📁 Visualization saved: Analysis_CSI_Dataset_750_Samples/rssi_detailed_analysis.png")
    print(f"🎨 Generated comprehensive RSSI spatial distribution with color graduation")

if __name__ == "__main__":
    main()
