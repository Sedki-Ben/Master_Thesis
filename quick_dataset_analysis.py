#!/usr/bin/env python3
"""
Quick Analysis of Reduced CSI Datasets

Fast analysis focusing on key metrics for CNN development.
Generates essential visualizations with minimal processing time.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (12, 8),
    'axes.grid': True,
    'grid.alpha': 0.3
})

def analyze_dataset_quick(dataset_name, data_dir):
    """
    Quick analysis of a dataset with essential findings.
    """
    print(f"\n🔬 ANALYZING: {dataset_name}")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path(f"Analysis_{dataset_name.replace(' ', '_')}")
    output_dir.mkdir(exist_ok=True)
    
    # Load data quickly
    print("📊 Loading data...")
    all_data = []
    csv_files = list(Path(data_dir).glob("*.csv"))
    
    # Sample fewer files for speed if needed
    if len(csv_files) > 20:
        csv_files = csv_files[::2]  # Every other file
    
    for file_path in csv_files[:15]:  # Limit to 15 files for speed
        match = re.match(r'(\d+),(\d+)\.csv', file_path.name)
        if not match:
            continue
        
        x, y = int(match.group(1)), int(match.group(2))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                if i >= 50:  # Limit samples per file for speed
                    break
                    
                rssi = float(row['rssi'])
                amplitude = np.array(json.loads(row['amplitude']))
                phase = np.array(json.loads(row['phase']))
                
                all_data.append({
                    'x': x, 'y': y,
                    'rssi': rssi,
                    'amp_mean': np.mean(amplitude),
                    'amp_std': np.std(amplitude),
                    'phase_std': np.std(phase),
                    'amplitude_vector': amplitude,
                    'phase_vector': phase
                })
    
    df = pd.DataFrame(all_data)
    print(f"   Loaded {len(df)} samples from {len(df.groupby(['x', 'y']))} locations")
    
    # Quick Analysis 1: Spatial Distribution
    print("🗺️  Step 1: Spatial characteristics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Quick Analysis - {dataset_name}', fontsize=14, fontweight='bold')
    
    # RSSI spatial pattern
    scatter = axes[0, 0].scatter(df['x'], df['y'], c=df['rssi'], cmap='plasma', s=50, alpha=0.7)
    axes[0, 0].set_xlabel('X Coordinate (m)')
    axes[0, 0].set_ylabel('Y Coordinate (m)')
    axes[0, 0].set_title('RSSI Spatial Distribution')
    plt.colorbar(scatter, ax=axes[0, 0], label='RSSI (dBm)')
    
    # Amplitude distribution
    axes[0, 1].hist(df['amp_mean'], bins=25, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 1].set_xlabel('Mean Amplitude')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Amplitude Distribution')
    
    # Key findings
    rssi_range = df['rssi'].max() - df['rssi'].min()
    amp_range = df['amp_mean'].max() - df['amp_mean'].min()
    
    print(f"   RSSI range: {rssi_range:.1f} dB")
    print(f"   Amplitude range: {amp_range:.2f}")
    print(f"   Spatial gradient: {'Strong' if rssi_range > 15 else 'Moderate'}")
    
    # Quick Analysis 2: Frequency Characteristics
    print("📡 Step 2: Frequency domain...")
    
    # Sample amplitude vectors for analysis
    sample_amps = np.vstack([row for row in df['amplitude_vector'][:200]])  # Limit for speed
    mean_amp_per_sc = np.mean(sample_amps, axis=0)
    
    axes[1, 0].plot(range(52), mean_amp_per_sc, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Subcarrier Index')
    axes[1, 0].set_ylabel('Mean Amplitude')
    axes[1, 0].set_title('Frequency Response')
    
    freq_variation = np.std(mean_amp_per_sc) / np.mean(mean_amp_per_sc)
    print(f"   Frequency selectivity: {freq_variation:.3f}")
    print(f"   Channel quality: {'Frequency-selective' if freq_variation > 0.1 else 'Flat'}")
    
    # Quick Analysis 3: Feature Space
    print("📉 Step 3: Feature analysis...")
    
    # Quick PCA on subset
    if len(sample_amps) > 0:
        scaler = StandardScaler()
        amp_scaled = scaler.fit_transform(sample_amps)
        
        pca = PCA()
        pca.fit(amp_scaled)
        
        cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
        axes[1, 1].plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'g-', linewidth=2)
        axes[1, 1].axhline(y=0.95, color='red', linestyle='--', label='95% Variance')
        axes[1, 1].set_xlabel('Number of Components')
        axes[1, 1].set_ylabel('Cumulative Variance')
        axes[1, 1].set_title('PCA Analysis')
        axes[1, 1].legend()
        
        n_comp_95 = np.argmax(cumsum_ratio >= 0.95) + 1
        print(f"   Components for 95% variance: {n_comp_95}")
        print(f"   Feature efficiency: {'High' if n_comp_95 < 20 else 'Moderate'}")
    
    plt.tight_layout()
    plt.savefig(output_dir / 'quick_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    # CNN Recommendations
    print("🎯 CNN Development Recommendations:")
    
    # Calculate feasibility score
    spatial_score = min(5, rssi_range / 4)
    freq_score = min(5, freq_variation * 20)
    feature_score = min(5, (52 - n_comp_95) / 10) if 'n_comp_95' in locals() else 3
    
    overall_score = (spatial_score + freq_score + feature_score) / 3
    
    print(f"   Overall feasibility: {overall_score:.2f}/5.0")
    print(f"   Expected accuracy: {80 + min(15, overall_score * 3):.0f}%")
    
    if overall_score >= 3.5:
        print("   Status: EXCELLENT for CNN development")
        print("   Recommended: Primary training dataset")
    elif overall_score >= 2.5:
        print("   Status: GOOD for CNN development")
        print("   Recommended: Validation and prototyping")
    else:
        print("   Status: MODERATE - may need augmentation")
    
    # Architecture suggestions
    print(f"   Suggested CNN depth: {min(4, max(2, int(overall_score)))} layers")
    print(f"   Kernel size: 3-7 subcarriers")
    print(f"   Batch size: {min(128, max(32, len(df) // 100))}")
    
    return {
        'dataset_name': dataset_name,
        'sample_count': len(df),
        'overall_score': overall_score,
        'rssi_range': rssi_range,
        'freq_variation': freq_variation,
        'n_components_95': n_comp_95 if 'n_comp_95' in locals() else 25
    }

def main():
    """
    Quick analysis of both datasets.
    """
    print("🚀 QUICK DATASET ANALYSIS")
    print("Fast analysis for CNN development guidance")
    print("=" * 60)
    
    # Analyze both datasets
    results_750 = analyze_dataset_quick("CSI Dataset 750 Samples", "CSI Dataset 750 Samples")
    results_500 = analyze_dataset_quick("CSI Dataset 500 Samples", "CSI Dataset 500 Samples")
    
    # Quick comparison
    print(f"\n🏆 COMPARISON SUMMARY:")
    print(f"750-Sample Dataset: {results_750['overall_score']:.2f}/5.0 feasibility")
    print(f"500-Sample Dataset: {results_500['overall_score']:.2f}/5.0 feasibility")
    
    if results_750['overall_score'] > results_500['overall_score']:
        print("✅ Recommendation: Use 750-sample dataset for primary development")
    elif results_500['overall_score'] > results_750['overall_score']:
        print("✅ Recommendation: Use 500-sample dataset for primary development")
    else:
        print("✅ Recommendation: Both datasets equivalent - choose based on computational needs")
    
    print(f"\n📁 Generated visualizations:")
    print("   • Analysis_CSI_Dataset_750_Samples/quick_analysis.png")
    print("   • Analysis_CSI_Dataset_500_Samples/quick_analysis.png")
    
    print(f"\n✅ Quick analysis complete!")

if __name__ == "__main__":
    main()
