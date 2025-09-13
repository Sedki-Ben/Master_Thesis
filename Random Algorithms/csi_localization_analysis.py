#!/usr/bin/env python3
"""
Comprehensive CSI Data Analysis for Indoor Localization Research

This script analyzes Channel State Information (CSI) data collected in an obstacle-rich 
indoor environment for developing deep learning-based localization systems. The analysis
focuses on multipath effects, spatial diversity, and frequency-domain characteristics
that can be leveraged for accurate indoor positioning.

Research Context:
- Indoor localization using WiFi CSI measurements
- Multipath-rich environment with obstacles
- Preparation for CNN/Deep Learning model training
- Spatial coordinates encoded in filenames as (x,y) positions in meters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pathlib import Path
import re
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plotting style for scientific publications
plt.style.use('default')
sns.set_palette("husl")

class CSILocalizationAnalyzer:
    """
    Comprehensive analyzer for CSI-based indoor localization research.
    """
    
    def __init__(self, data_dir="Amplitude Phase Data Single"):
        self.data_dir = Path(data_dir)
        self.locations = {}  # Will store {(x,y): {'rssi': [], 'amplitude': [], 'phase': []}}
        self.all_data = []   # Will store all samples with location info
        self.coordinates = []  # List of (x,y) coordinates
        
    def load_and_parse_data(self):
        """
        Load CSI data and parse spatial coordinates from filenames.
        
        Returns:
            dict: Summary of loaded data
        """
        print("🔬 Loading and parsing CSI localization data...")
        print("=" * 70)
        
        csv_files = sorted(self.data_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {self.data_dir}")
            return {}
        
        total_samples = 0
        
        for file_path in csv_files:
            # Parse coordinates from filename (e.g., "3,4.csv" -> (3, 4))
            match = re.match(r'(\d+),(\d+)\.csv', file_path.name)
            if not match:
                print(f"⚠️  Skipping {file_path.name} - invalid coordinate format")
                continue
            
            x, y = int(match.group(1)), int(match.group(2))
            location = (x, y)
            
            # Read CSI data for this location
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    location_data = {
                        'rssi': [],
                        'amplitude': [],
                        'phase': [],
                        'samples': 0
                    }
                    
                    for row in reader:
                        rssi = float(row['rssi'])
                        amplitude = np.array(json.loads(row['amplitude']))
                        phase = np.array(json.loads(row['phase']))
                        
                        location_data['rssi'].append(rssi)
                        location_data['amplitude'].append(amplitude)
                        location_data['phase'].append(phase)
                        
                        # Store in all_data for comprehensive analysis
                        self.all_data.append({
                            'x': x, 'y': y, 'location': location,
                            'rssi': rssi, 'amplitude': amplitude, 'phase': phase
                        })
                    
                    location_data['samples'] = len(location_data['rssi'])
                    self.locations[location] = location_data
                    total_samples += location_data['samples']
                    
                    print(f"📍 Location ({x:2d},{y:2d}): {location_data['samples']:4d} samples")
            
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                continue
        
        self.coordinates = sorted(list(self.locations.keys()))
        
        summary = {
            'total_locations': len(self.locations),
            'total_samples': total_samples,
            'coordinate_range_x': (min(c[0] for c in self.coordinates), max(c[0] for c in self.coordinates)),
            'coordinate_range_y': (min(c[1] for c in self.coordinates), max(c[1] for c in self.coordinates)),
            'avg_samples_per_location': total_samples / len(self.locations) if self.locations else 0
        }
        
        print("\n📊 Data Loading Summary:")
        print(f"Total locations: {summary['total_locations']}")
        print(f"Total samples: {summary['total_samples']:,}")
        print(f"X coordinate range: {summary['coordinate_range_x']}")
        print(f"Y coordinate range: {summary['coordinate_range_y']}")
        print(f"Average samples per location: {summary['avg_samples_per_location']:.1f}")
        
        return summary
    
    def analyze_spatial_distribution(self):
        """
        Analyze the spatial distribution of measurement locations and signal characteristics.
        """
        print("\n🗺️  Spatial Distribution Analysis")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract coordinates and mean values
        x_coords = [coord[0] for coord in self.coordinates]
        y_coords = [coord[1] for coord in self.coordinates]
        mean_rssi = [np.mean(self.locations[coord]['rssi']) for coord in self.coordinates]
        mean_amplitude = [np.mean([np.mean(amp) for amp in self.locations[coord]['amplitude']]) 
                         for coord in self.coordinates]
        std_amplitude = [np.std([np.mean(amp) for amp in self.locations[coord]['amplitude']]) 
                        for coord in self.coordinates]
        
        # 1. Measurement locations layout
        axes[0, 0].scatter(x_coords, y_coords, c='red', s=100, alpha=0.7, edgecolors='black')
        for i, (x, y) in enumerate(self.coordinates):
            axes[0, 0].annotate(f'({x},{y})', (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        axes[0, 0].set_xlabel('X Coordinate (meters)')
        axes[0, 0].set_ylabel('Y Coordinate (meters)')
        axes[0, 0].set_title('Measurement Locations Layout')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_aspect('equal')
        
        # 2. RSSI spatial heatmap
        scatter = axes[0, 1].scatter(x_coords, y_coords, c=mean_rssi, s=200, 
                                   cmap='viridis', alpha=0.8, edgecolors='black')
        axes[0, 1].set_xlabel('X Coordinate (meters)')
        axes[0, 1].set_ylabel('Y Coordinate (meters)')
        axes[0, 1].set_title('Mean RSSI Distribution (dBm)')
        plt.colorbar(scatter, ax=axes[0, 1])
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Mean amplitude spatial distribution
        scatter = axes[1, 0].scatter(x_coords, y_coords, c=mean_amplitude, s=200, 
                                   cmap='plasma', alpha=0.8, edgecolors='black')
        axes[1, 0].set_xlabel('X Coordinate (meters)')
        axes[1, 0].set_ylabel('Y Coordinate (meters)')
        axes[1, 0].set_title('Mean CSI Amplitude Distribution')
        plt.colorbar(scatter, ax=axes[1, 0])
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Amplitude variability (proxy for multipath complexity)
        scatter = axes[1, 1].scatter(x_coords, y_coords, c=std_amplitude, s=200, 
                                   cmap='coolwarm', alpha=0.8, edgecolors='black')
        axes[1, 1].set_xlabel('X Coordinate (meters)')
        axes[1, 1].set_ylabel('Y Coordinate (meters)')
        axes[1, 1].set_title('CSI Amplitude Variability\n(Multipath Complexity Indicator)')
        plt.colorbar(scatter, ax=axes[1, 1])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('spatial_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical analysis
        print("📈 Spatial Statistics:")
        print(f"RSSI range: {min(mean_rssi):.1f} to {max(mean_rssi):.1f} dBm")
        print(f"Mean amplitude range: {min(mean_amplitude):.2f} to {max(mean_amplitude):.2f}")
        print(f"Amplitude variability range: {min(std_amplitude):.2f} to {max(std_amplitude):.2f}")
        
        return {
            'rssi_stats': {'min': min(mean_rssi), 'max': max(mean_rssi), 'mean': np.mean(mean_rssi)},
            'amplitude_stats': {'min': min(mean_amplitude), 'max': max(mean_amplitude), 'mean': np.mean(mean_amplitude)},
            'variability_stats': {'min': min(std_amplitude), 'max': max(std_amplitude), 'mean': np.mean(std_amplitude)}
        }
    
    def analyze_subcarrier_characteristics(self):
        """
        Analyze frequency-domain characteristics across subcarriers for multipath analysis.
        """
        print("\n📡 Subcarrier-Domain Analysis")
        print("=" * 70)
        
        # Aggregate all amplitude and phase data
        all_amplitudes = []
        all_phases = []
        
        for sample in self.all_data:
            all_amplitudes.append(sample['amplitude'])
            all_phases.append(sample['phase'])
        
        all_amplitudes = np.array(all_amplitudes)  # Shape: (samples, 52)
        all_phases = np.array(all_phases)          # Shape: (samples, 52)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Mean amplitude per subcarrier
        mean_amp_per_subcarrier = np.mean(all_amplitudes, axis=0)
        std_amp_per_subcarrier = np.std(all_amplitudes, axis=0)
        
        axes[0, 0].plot(range(52), mean_amp_per_subcarrier, 'b-', linewidth=2, label='Mean')
        axes[0, 0].fill_between(range(52), 
                               mean_amp_per_subcarrier - std_amp_per_subcarrier,
                               mean_amp_per_subcarrier + std_amp_per_subcarrier,
                               alpha=0.3, label='±1 STD')
        axes[0, 0].set_xlabel('Subcarrier Index')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Mean Amplitude per Subcarrier\n(Frequency Selectivity)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Phase characteristics per subcarrier
        mean_phase_per_subcarrier = np.angle(np.mean(np.exp(1j * all_phases), axis=0))
        phase_coherence = np.abs(np.mean(np.exp(1j * all_phases), axis=0))
        
        axes[0, 1].plot(range(52), mean_phase_per_subcarrier, 'r-', linewidth=2, label='Mean Phase')
        ax_twin = axes[0, 1].twinx()
        ax_twin.plot(range(52), phase_coherence, 'g--', linewidth=2, label='Phase Coherence')
        axes[0, 1].set_xlabel('Subcarrier Index')
        axes[0, 1].set_ylabel('Phase (radians)', color='r')
        ax_twin.set_ylabel('Phase Coherence', color='g')
        axes[0, 1].set_title('Phase Characteristics per Subcarrier')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Subcarrier correlation matrix (multipath fingerprint)
        corr_matrix = np.corrcoef(all_amplitudes.T)
        im = axes[0, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 2].set_xlabel('Subcarrier Index')
        axes[0, 2].set_ylabel('Subcarrier Index')
        axes[0, 2].set_title('Subcarrier Amplitude Correlation Matrix\n(Multipath Fingerprint)')
        plt.colorbar(im, ax=axes[0, 2])
        
        # 4. Amplitude distribution across all subcarriers
        all_amps_flat = all_amplitudes.flatten()
        axes[1, 0].hist(all_amps_flat, bins=50, alpha=0.7, density=True, edgecolor='black')
        axes[1, 0].set_xlabel('Amplitude')
        axes[1, 0].set_ylabel('Probability Density')
        axes[1, 0].set_title('Overall Amplitude Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add fitted normal distribution
        mu, sigma = stats.norm.fit(all_amps_flat)
        x = np.linspace(all_amps_flat.min(), all_amps_flat.max(), 100)
        axes[1, 0].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                       label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')
        axes[1, 0].legend()
        
        # 5. Phase unwrapping analysis (for delay spread estimation)
        # Compute phase difference between adjacent subcarriers
        phase_diff = np.diff(all_phases, axis=1)
        phase_diff_unwrapped = np.unwrap(phase_diff, axis=1)
        mean_phase_slope = np.mean(phase_diff_unwrapped, axis=0)
        
        axes[1, 1].plot(range(51), mean_phase_slope, 'purple', linewidth=2)
        axes[1, 1].set_xlabel('Subcarrier Index Difference')
        axes[1, 1].set_ylabel('Phase Slope (rad/subcarrier)')
        axes[1, 1].set_title('Mean Phase Slope\n(Related to Delay Spread)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Multipath richness indicator
        # Calculate amplitude variance per sample (across subcarriers)
        amp_variance_per_sample = np.var(all_amplitudes, axis=1)
        location_variance = []
        locations_for_variance = []
        
        for coord in self.coordinates:
            coord_indices = [i for i, sample in enumerate(self.all_data) if sample['location'] == coord]
            coord_variances = amp_variance_per_sample[coord_indices]
            location_variance.append(np.mean(coord_variances))
            locations_for_variance.append(f'({coord[0]},{coord[1]})')
        
        axes[1, 2].bar(range(len(location_variance)), location_variance, alpha=0.7, 
                      color='orange', edgecolor='black')
        axes[1, 2].set_xlabel('Location')
        axes[1, 2].set_ylabel('Mean Amplitude Variance')
        axes[1, 2].set_title('Multipath Richness per Location\n(Frequency Selectivity)')
        axes[1, 2].set_xticks(range(0, len(locations_for_variance), 2))
        axes[1, 2].set_xticklabels([locations_for_variance[i] for i in range(0, len(locations_for_variance), 2)], 
                                  rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('subcarrier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Subcarrier Analysis Results:")
        print(f"Amplitude dynamic range: {np.min(all_amplitudes):.2f} to {np.max(all_amplitudes):.2f}")
        print(f"Mean amplitude across all subcarriers: {np.mean(all_amplitudes):.2f}")
        print(f"Phase coherence range: {np.min(phase_coherence):.3f} to {np.max(phase_coherence):.3f}")
        print(f"Maximum subcarrier correlation: {np.max(corr_matrix[corr_matrix < 1]):.3f}")
        print(f"Minimum subcarrier correlation: {np.min(corr_matrix):.3f}")
        
        return {
            'amplitude_stats': {
                'min': np.min(all_amplitudes),
                'max': np.max(all_amplitudes),
                'mean': np.mean(all_amplitudes),
                'std': np.std(all_amplitudes)
            },
            'correlation_stats': {
                'max': np.max(corr_matrix[corr_matrix < 1]),
                'min': np.min(corr_matrix),
                'mean': np.mean(corr_matrix[corr_matrix < 1])
            }
        }
    
    def analyze_multipath_characteristics(self):
        """
        Detailed analysis of multipath propagation effects in the CSI data.
        """
        print("\n🌊 Multipath Propagation Analysis")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Select a few representative locations for detailed analysis
        representative_locations = self.coordinates[::len(self.coordinates)//6][:6]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(representative_locations)))
        
        # 1. Channel Transfer Function magnitude comparison
        for i, coord in enumerate(representative_locations):
            sample_amplitudes = self.locations[coord]['amplitude'][:10]  # First 10 samples
            for j, amp in enumerate(sample_amplitudes):
                alpha = 0.3 if j > 0 else 1.0  # Highlight first sample
                axes[0, 0].plot(range(52), amp, color=colors[i], alpha=alpha, 
                               linewidth=2 if j == 0 else 1)
            
            # Plot mean for this location
            mean_amp = np.mean(sample_amplitudes, axis=0)
            axes[0, 0].plot(range(52), mean_amp, color=colors[i], linewidth=3, 
                           label=f'Loc ({coord[0]},{coord[1]})')
        
        axes[0, 0].set_xlabel('Subcarrier Index')
        axes[0, 0].set_ylabel('Channel Amplitude')
        axes[0, 0].set_title('Channel Transfer Function Magnitude\n(Multipath Fading Patterns)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Phase response comparison
        for i, coord in enumerate(representative_locations):
            sample_phases = self.locations[coord]['phase'][:5]  # First 5 samples
            mean_phase = np.mean(sample_phases, axis=0)
            # Unwrap phase for better visualization
            mean_phase_unwrapped = np.unwrap(mean_phase)
            axes[0, 1].plot(range(52), mean_phase_unwrapped, color=colors[i], 
                           linewidth=2, marker='o', markersize=3, label=f'Loc ({coord[0]},{coord[1]})')
        
        axes[0, 1].set_xlabel('Subcarrier Index')
        axes[0, 1].set_ylabel('Unwrapped Phase (radians)')
        axes[0, 1].set_title('Channel Phase Response\n(Delay Characteristics)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Delay spread estimation via phase linearity
        delay_spreads = []
        location_labels = []
        
        for coord in self.coordinates:
            phases = np.array(self.locations[coord]['phase'])
            
            # Calculate delay spread for each sample
            sample_delays = []
            for phase_sample in phases[:20]:  # Use first 20 samples
                # Unwrap phase
                unwrapped = np.unwrap(phase_sample)
                # Linear fit to estimate group delay
                slope, intercept = np.polyfit(range(52), unwrapped, 1)
                # Phase deviation from linear trend (related to delay spread)
                linear_trend = slope * np.arange(52) + intercept
                phase_deviation = np.std(unwrapped - linear_trend)
                sample_delays.append(phase_deviation)
            
            delay_spreads.append(np.mean(sample_delays))
            location_labels.append(f'({coord[0]},{coord[1]})')
        
        bars = axes[0, 2].bar(range(len(delay_spreads)), delay_spreads, alpha=0.7, 
                             color='skyblue', edgecolor='navy')
        axes[0, 2].set_xlabel('Location')
        axes[0, 2].set_ylabel('Delay Spread Indicator')
        axes[0, 2].set_title('Relative Delay Spread per Location\n(Multipath Temporal Dispersion)')
        axes[0, 2].set_xticks(range(0, len(location_labels), 2))
        axes[0, 2].set_xticklabels([location_labels[i] for i in range(0, len(location_labels), 2)], 
                                  rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Amplitude vs Phase scatter plot (constellation analysis)
        sample_coord = representative_locations[0]  # Use first representative location
        sample_amps = np.array(self.locations[sample_coord]['amplitude'][:100])  # 100 samples
        sample_phases = np.array(self.locations[sample_coord]['phase'][:100])
        
        # Plot for multiple subcarriers
        subcarriers_to_plot = [0, 13, 26, 39, 51]  # Spread across frequency
        for sc in subcarriers_to_plot:
            axes[1, 0].scatter(sample_amps[:, sc], sample_phases[:, sc], 
                              alpha=0.6, label=f'Subcarrier {sc}', s=20)
        
        axes[1, 0].set_xlabel('Amplitude')
        axes[1, 0].set_ylabel('Phase (radians)')
        axes[1, 0].set_title(f'Amplitude-Phase Constellation\nLocation ({sample_coord[0]},{sample_coord[1]})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Temporal stability analysis
        # Calculate amplitude correlation over time for each location
        temporal_stability = []
        for coord in self.coordinates:
            amplitudes = np.array(self.locations[coord]['amplitude'])
            if len(amplitudes) > 1:
                # Calculate correlation between consecutive samples
                correlations = []
                for i in range(min(50, len(amplitudes)-1)):  # Use first 50 samples
                    corr = np.corrcoef(amplitudes[i], amplitudes[i+1])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                temporal_stability.append(np.mean(correlations) if correlations else 0)
            else:
                temporal_stability.append(0)
        
        axes[1, 1].bar(range(len(temporal_stability)), temporal_stability, alpha=0.7, 
                      color='lightcoral', edgecolor='darkred')
        axes[1, 1].set_xlabel('Location')
        axes[1, 1].set_ylabel('Temporal Correlation')
        axes[1, 1].set_title('Channel Temporal Stability\n(Consecutive Sample Correlation)')
        axes[1, 1].set_xticks(range(0, len(location_labels), 2))
        axes[1, 1].set_xticklabels([location_labels[i] for i in range(0, len(location_labels), 2)], 
                                  rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Frequency correlation analysis (coherence bandwidth estimation)
        all_amplitudes = np.vstack([np.array(self.locations[coord]['amplitude']) 
                                   for coord in self.coordinates])
        
        # Calculate correlation between subcarriers
        freq_separations = []
        correlations = []
        
        for sep in range(1, 26):  # Up to half the subcarriers
            corr_values = []
            for sc in range(52 - sep):
                corr = np.corrcoef(all_amplitudes[:, sc], all_amplitudes[:, sc + sep])[0, 1]
                if not np.isnan(corr):
                    corr_values.append(corr)
            
            if corr_values:
                freq_separations.append(sep)
                correlations.append(np.mean(corr_values))
        
        axes[1, 2].plot(freq_separations, correlations, 'o-', linewidth=2, markersize=6, 
                       color='darkgreen')
        axes[1, 2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                          label='50% Correlation Threshold')
        axes[1, 2].set_xlabel('Subcarrier Separation')
        axes[1, 2].set_ylabel('Amplitude Correlation')
        axes[1, 2].set_title('Frequency Correlation Function\n(Coherence Bandwidth Analysis)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multipath_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("🌊 Multipath Analysis Results:")
        print(f"Delay spread range: {min(delay_spreads):.3f} to {max(delay_spreads):.3f}")
        print(f"Temporal stability range: {min(temporal_stability):.3f} to {max(temporal_stability):.3f}")
        print(f"50% frequency correlation at separation: ~{freq_separations[np.argmin(np.abs(np.array(correlations) - 0.5))]} subcarriers")
        
        return {
            'delay_spread_stats': {'min': min(delay_spreads), 'max': max(delay_spreads), 'mean': np.mean(delay_spreads)},
            'temporal_stability_stats': {'min': min(temporal_stability), 'max': max(temporal_stability), 'mean': np.mean(temporal_stability)},
            'coherence_bandwidth': freq_separations[np.argmin(np.abs(np.array(correlations) - 0.5))] if correlations else None
        }

def main():
    """
    Main analysis pipeline for CSI-based indoor localization research.
    """
    print("🏢 CSI-Based Indoor Localization Analysis")
    print("Research Focus: Multipath-Rich Environment for Deep Learning")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = CSILocalizationAnalyzer()
    
    # Load and parse data
    data_summary = analyzer.load_and_parse_data()
    
    if not data_summary:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Perform comprehensive analysis
    print("\n🔬 Beginning comprehensive scientific analysis...")
    
    # 1. Spatial distribution analysis
    spatial_stats = analyzer.analyze_spatial_distribution()
    
    # 2. Subcarrier characteristics analysis
    subcarrier_stats = analyzer.analyze_subcarrier_characteristics()
    
    # 3. Multipath propagation analysis
    multipath_stats = analyzer.analyze_multipath_characteristics()
    
    # Summary for research conclusions
    print("\n📋 RESEARCH SUMMARY")
    print("=" * 70)
    print("🎯 Dataset Characteristics for Deep Learning:")
    print(f"   • Spatial diversity: {data_summary['total_locations']} locations")
    print(f"   • Temporal samples: {data_summary['total_samples']:,} total measurements")
    print(f"   • Frequency diversity: 52 subcarriers per measurement")
    print(f"   • Feature dimensionality: 104 features (52 amplitude + 52 phase)")
    
    print("\n🌊 Multipath Environment Analysis:")
    print(f"   • RSSI variation: {spatial_stats['rssi_stats']['max'] - spatial_stats['rssi_stats']['min']:.1f} dB")
    print(f"   • Amplitude dynamic range: {subcarrier_stats['amplitude_stats']['max'] - subcarrier_stats['amplitude_stats']['min']:.1f}")
    print(f"   • Delay spread variation: {multipath_stats['delay_spread_stats']['max'] - multipath_stats['delay_spread_stats']['min']:.3f}")
    print(f"   • Temporal stability range: {multipath_stats['temporal_stability_stats']['min']:.3f} - {multipath_stats['temporal_stability_stats']['max']:.3f}")
    
    print("\n🤖 Deep Learning Implications:")
    print("   • High spatial resolution features available for CNN training")
    print("   • Frequency-domain patterns suitable for 1D/2D convolutions")
    print("   • Multipath signatures provide location-specific fingerprints")
    print("   • Temporal variations require robust training strategies")
    
    print("\n📊 Generated Visualizations:")
    print("   • spatial_distribution_analysis.png - Spatial layout and signal distribution")
    print("   • subcarrier_analysis.png - Frequency-domain characteristics")
    print("   • multipath_analysis.png - Multipath propagation effects")
    
    print("\n✅ Analysis complete! Data ready for deep learning model development.")

if __name__ == "__main__":
    main()
