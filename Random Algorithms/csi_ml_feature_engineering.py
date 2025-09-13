#!/usr/bin/env python3
"""
Streamlined CSI Feature Engineering and ML Analysis

Quick and efficient analysis for CNN/DL model development preparation
focusing on the most critical aspects for indoor localization research.
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
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

def quick_data_load(data_dir="Amplitude Phase Data Single"):
    """Quick data loading and basic feature extraction."""
    print("🚀 Quick CSI Data Analysis for ML/DL")
    print("=" * 50)
    
    data_dir = Path(data_dir)
    all_data = []
    
    csv_files = sorted(data_dir.glob("*.csv"))
    print(f"📁 Loading {len(csv_files)} files...")
    
    for file_path in csv_files:
        # Parse coordinates
        match = re.match(r'(\d+),(\d+)\.csv', file_path.name)
        if not match:
            continue
        
        x, y = int(match.group(1)), int(match.group(2))
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            sample_count = 0
            for row in reader:
                if sample_count >= 50:  # Limit samples per location for speed
                    break
                
                rssi = float(row['rssi'])
                amplitude = np.array(json.loads(row['amplitude']))
                phase = np.array(json.loads(row['phase']))
                
                # Extract key features
                features = {
                    'x': x, 'y': y,
                    'rssi': rssi,
                    'amp_mean': np.mean(amplitude),
                    'amp_std': np.std(amplitude),
                    'amp_max': np.max(amplitude),
                    'amp_min': np.min(amplitude),
                    'phase_std': np.std(phase),
                    'freq_selectivity': np.std(amplitude) / (np.mean(amplitude) + 1e-10),
                    'amplitude_range': np.max(amplitude) - np.min(amplitude),
                    'spectral_centroid': np.sum(np.arange(52) * amplitude) / (np.sum(amplitude) + 1e-10)
                }
                
                # Add raw CSI features for CNN
                for i in range(52):
                    features[f'amp_{i}'] = amplitude[i]
                    features[f'phase_{i}'] = phase[i]
                
                all_data.append(features)
                sample_count += 1
    
    df = pd.DataFrame(all_data)
    print(f"✅ Loaded {len(df)} samples from {len(df.groupby(['x', 'y']))} locations")
    
    return df

def analyze_spatial_features(df):
    """Analyze spatial distribution and feature characteristics."""
    print("\n📍 Spatial Feature Analysis")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Location distribution
    locations = df.groupby(['x', 'y']).size().reset_index(name='count')
    scatter = axes[0, 0].scatter(locations['x'], locations['y'], 
                                c=locations['count'], s=100, cmap='viridis')
    axes[0, 0].set_xlabel('X Coordinate (m)')
    axes[0, 0].set_ylabel('Y Coordinate (m)')
    axes[0, 0].set_title('Sample Distribution by Location')
    plt.colorbar(scatter, ax=axes[0, 0], label='Sample Count')
    
    # 2. RSSI vs Position
    scatter = axes[0, 1].scatter(df['x'], df['y'], c=df['rssi'], s=20, cmap='plasma')
    axes[0, 1].set_xlabel('X Coordinate (m)')
    axes[0, 1].set_ylabel('Y Coordinate (m)')
    axes[0, 1].set_title('RSSI Distribution')
    plt.colorbar(scatter, ax=axes[0, 1], label='RSSI (dBm)')
    
    # 3. Amplitude characteristics
    scatter = axes[1, 0].scatter(df['x'], df['y'], c=df['amp_mean'], s=20, cmap='coolwarm')
    axes[1, 0].set_xlabel('X Coordinate (m)')
    axes[1, 0].set_ylabel('Y Coordinate (m)')
    axes[1, 0].set_title('Mean Amplitude Distribution')
    plt.colorbar(scatter, ax=axes[1, 0], label='Mean Amplitude')
    
    # 4. Frequency selectivity (multipath indicator)
    scatter = axes[1, 1].scatter(df['x'], df['y'], c=df['freq_selectivity'], s=20, cmap='hot')
    axes[1, 1].set_xlabel('X Coordinate (m)')
    axes[1, 1].set_ylabel('Y Coordinate (m)')
    axes[1, 1].set_title('Frequency Selectivity\n(Multipath Indicator)')
    plt.colorbar(scatter, ax=axes[1, 1], label='Selectivity')
    
    plt.tight_layout()
    plt.savefig('spatial_features_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    print(f"📊 Spatial Statistics:")
    print(f"   • X range: {df['x'].min()}-{df['x'].max()} meters")
    print(f"   • Y range: {df['y'].min()}-{df['y'].max()} meters")
    print(f"   • RSSI range: {df['rssi'].min():.1f} to {df['rssi'].max():.1f} dBm")
    print(f"   • Amplitude range: {df['amp_mean'].min():.2f} to {df['amp_mean'].max():.2f}")
    print(f"   • Multipath variation: {df['freq_selectivity'].min():.3f} to {df['freq_selectivity'].max():.3f}")

def dimensionality_analysis(df):
    """Quick dimensionality reduction analysis."""
    print("\n📉 Dimensionality Reduction Analysis")
    print("=" * 50)
    
    # Prepare raw CSI features
    amp_cols = [f'amp_{i}' for i in range(52)]
    phase_cols = [f'phase_{i}' for i in range(52)]
    raw_features = amp_cols + phase_cols
    
    X = df[raw_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA analysis
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Explained variance
    cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
    axes[0].plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'b-', linewidth=2)
    axes[0].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='95% Variance')
    axes[0].axhline(y=0.99, color='orange', linestyle='--', alpha=0.7, label='99% Variance')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Cumulative Explained Variance')
    axes[0].set_title('PCA: Explained Variance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. PCA visualization (first 2 components)
    unique_locations = df.groupby(['x', 'y']).first().reset_index()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_locations)))
    
    for i, (_, loc) in enumerate(unique_locations.iterrows()):
        if i >= 20:  # Limit for visibility
            break
        mask = (df['x'] == loc['x']) & (df['y'] == loc['y'])
        axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=f"({loc['x']},{loc['y']})", 
                       alpha=0.6, s=20)
    
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[1].set_title('PCA: Location Separability')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Feature importance (amplitude vs phase)
    amp_variance = np.sum(pca.explained_variance_ratio_[:52])
    phase_variance = np.sum(pca.explained_variance_ratio_[52:104])
    
    categories = ['Amplitude\nFeatures', 'Phase\nFeatures']
    variances = [amp_variance, phase_variance]
    
    bars = axes[2].bar(categories, variances, color=['blue', 'red'], alpha=0.7)
    axes[2].set_ylabel('Cumulative Explained Variance')
    axes[2].set_title('Feature Type Importance')
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, variance in zip(bars, variances):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{variance:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dimensionality_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find components for variance thresholds
    n_comp_95 = np.argmax(cumsum_ratio >= 0.95) + 1
    n_comp_99 = np.argmax(cumsum_ratio >= 0.99) + 1
    
    print(f"📊 PCA Results:")
    print(f"   • Original dimensions: {X_scaled.shape[1]}")
    print(f"   • Components for 95% variance: {n_comp_95}")
    print(f"   • Components for 99% variance: {n_comp_99}")
    print(f"   • Amplitude feature importance: {amp_variance:.3f}")
    print(f"   • Phase feature importance: {phase_variance:.3f}")
    
    return {
        'n_comp_95': n_comp_95,
        'n_comp_99': n_comp_99,
        'amp_importance': amp_variance,
        'phase_importance': phase_variance
    }

def cnn_input_formats(df):
    """Analyze optimal CNN input formats."""
    print("\n🧠 CNN Input Format Analysis")
    print("=" * 50)
    
    # Sample data from one location
    sample_data = df.iloc[0]
    
    # Extract amplitude and phase arrays
    amplitudes = [sample_data[f'amp_{i}'] for i in range(52)]
    phases = [sample_data[f'phase_{i}'] for i in range(52)]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. 1D CNN format (amplitude + phase as channels)
    axes[0, 0].plot(range(52), amplitudes, 'b-', linewidth=2, label='Amplitude')
    axes[0, 0].plot(range(52), phases, 'r-', linewidth=2, label='Phase')
    axes[0, 0].set_xlabel('Subcarrier Index')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('1D CNN Input Format\n(52 subcarriers × 2 channels)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 2D CNN format (location matrix)
    # Create amplitude matrix for multiple locations
    location_amps = []
    location_labels = []
    unique_locations = df.groupby(['x', 'y']).first().reset_index()[:15]  # First 15 locations
    
    for _, loc in unique_locations.iterrows():
        loc_data = df[(df['x'] == loc['x']) & (df['y'] == loc['y'])].iloc[0]
        loc_amps = [loc_data[f'amp_{i}'] for i in range(52)]
        location_amps.append(loc_amps)
        location_labels.append(f"({loc['x']},{loc['y']})")
    
    amp_matrix = np.array(location_amps)
    im = axes[0, 1].imshow(amp_matrix, cmap='viridis', aspect='auto')
    axes[0, 1].set_xlabel('Subcarrier Index')
    axes[0, 1].set_ylabel('Location')
    axes[0, 1].set_title('2D CNN Input Format\n(Location × Subcarrier)')
    axes[0, 1].set_yticks(range(len(location_labels)))
    axes[0, 1].set_yticklabels(location_labels, fontsize=8)
    plt.colorbar(im, ax=axes[0, 1])
    
    # 3. Feature scaling comparison
    amp_array = np.array(amplitudes)
    
    # Different scaling methods
    standardized = (amp_array - np.mean(amp_array)) / np.std(amp_array)
    normalized = (amp_array - np.min(amp_array)) / (np.max(amp_array) - np.min(amp_array))
    
    axes[1, 0].plot(range(52), amp_array, 'k-', linewidth=2, label='Original')
    axes[1, 0].plot(range(52), standardized, 'b-', linewidth=2, label='Standardized')
    axes[1, 0].plot(range(52), normalized, 'r-', linewidth=2, label='Min-Max Normalized')
    axes[1, 0].set_xlabel('Subcarrier Index')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].set_title('Feature Scaling for CNN')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Data augmentation example (noise addition)
    noise_levels = [0.01, 0.05, 0.1]
    
    axes[1, 1].plot(range(52), amplitudes, 'k-', linewidth=3, label='Original')
    
    for i, noise_level in enumerate(noise_levels):
        noise = np.random.normal(0, noise_level * np.std(amplitudes), 52)
        noisy_amps = np.array(amplitudes) + noise
        axes[1, 1].plot(range(52), noisy_amps, alpha=0.7, linewidth=2,
                       label=f'Noise σ={noise_level*100:.0f}%')
    
    axes[1, 1].set_xlabel('Subcarrier Index')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].set_title('Data Augmentation\n(Noise Addition)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_input_formats.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"🧠 CNN Recommendations:")
    print(f"   • 1D CNN input: (52, 2) for amplitude+phase channels")
    print(f"   • 2D CNN input: (n_locations, 52) or (time_steps, 52)")
    print(f"   • Preprocessing: StandardScaler recommended")
    print(f"   • Data augmentation: Gaussian noise (σ=1-5% of signal)")

def main():
    """Main analysis pipeline."""
    # Load data
    df = quick_data_load()
    
    if df.empty:
        print("❌ No data loaded. Exiting.")
        return
    
    # Spatial analysis
    analyze_spatial_features(df)
    
    # Dimensionality analysis
    dim_results = dimensionality_analysis(df)
    
    # CNN input analysis
    cnn_input_formats(df)
    
    print("\n🎯 ML/DL PREPARATION SUMMARY")
    print("=" * 50)
    print(f"📊 Dataset Overview:")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Unique locations: {len(df.groupby(['x', 'y']))}")
    print(f"   • Feature dimensions: 104 (52 amplitude + 52 phase)")
    print(f"   • Reduced dimensions (95% var): {dim_results['n_comp_95']}")
    
    print(f"\n🌊 Multipath Characteristics:")
    print(f"   • RSSI variation: {df['rssi'].max() - df['rssi'].min():.1f} dB")
    print(f"   • Frequency selectivity range: {df['freq_selectivity'].min():.3f} - {df['freq_selectivity'].max():.3f}")
    print(f"   • Amplitude dynamic range: {df['amp_mean'].max() - df['amp_mean'].min():.2f}")
    
    print(f"\n🤖 Deep Learning Readiness:")
    print(f"   • Amplitude features: {dim_results['amp_importance']:.3f} importance")
    print(f"   • Phase features: {dim_results['phase_importance']:.3f} importance")
    print(f"   • Recommended CNN input: (batch, 52, 2) for 1D or (batch, locations, 52) for 2D")
    print(f"   • Data augmentation: Ready for noise injection and temporal shifts")
    
    print(f"\n📊 Generated Visualizations:")
    print(f"   • spatial_features_analysis.png")
    print(f"   • dimensionality_analysis.png") 
    print(f"   • cnn_input_formats.png")
    
    return df, dim_results

if __name__ == "__main__":
    df, results = main()
