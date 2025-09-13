#!/usr/bin/env python3
"""
Comprehensive CSI Analysis Report for Indoor Localization Research

This report provides detailed analysis of Channel State Information (CSI) data
collected in an obstacle-rich indoor environment, with specific focus on
deep learning model development for accurate indoor positioning.

Author: AI Research Assistant
Research Context: Indoor Localization using WiFi CSI in Multipath-Rich Environment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pathlib import Path
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Set scientific plotting style
plt.rcParams.update({
    'font.size': 10,
    'figure.figsize': (12, 8),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9
})

class ComprehensiveCSIAnalyzer:
    """
    Comprehensive analyzer for CSI-based indoor localization research with detailed explanations.
    """
    
    def __init__(self, data_dir="Amplitude Phase Data Single"):
        self.data_dir = Path(data_dir)
        self.df = None
        self.analysis_results = {}
        
    def load_data_with_analysis(self):
        """
        Load CSI data with comprehensive metadata analysis.
        
        Goal: Understand the dataset structure, spatial coverage, and data quality
        for indoor localization model development.
        """
        print("🔬 COMPREHENSIVE CSI DATA ANALYSIS FOR INDOOR LOCALIZATION")
        print("=" * 80)
        print("Research Objective: Evaluate CSI data suitability for deep learning-based")
        print("indoor positioning in obstacle-rich environments")
        print("=" * 80)
        
        print("\n📊 PHASE 1: DATA LOADING AND STRUCTURE ANALYSIS")
        print("-" * 60)
        
        all_data = []
        location_stats = {}
        
        csv_files = sorted(self.data_dir.glob("*.csv"))
        print(f"Dataset Discovery: Found {len(csv_files)} measurement files")
        
        for file_path in csv_files:
            # Parse spatial coordinates from filename
            match = re.match(r'(\d+),(\d+)\.csv', file_path.name)
            if not match:
                continue
            
            x, y = int(match.group(1)), int(match.group(2))
            location_key = f"({x},{y})"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                location_samples = []
                for row in reader:
                    rssi = float(row['rssi'])
                    amplitude = np.array(json.loads(row['amplitude']))
                    phase = np.array(json.loads(row['phase']))
                    
                    # Comprehensive feature extraction
                    sample_features = {
                        'x': x, 'y': y, 'location': location_key,
                        'rssi': rssi,
                        # Raw CSI features (for CNN input)
                        'amplitude_vector': amplitude,
                        'phase_vector': phase,
                        # Statistical features
                        'amp_mean': np.mean(amplitude),
                        'amp_std': np.std(amplitude),
                        'amp_var': np.var(amplitude),
                        'amp_max': np.max(amplitude),
                        'amp_min': np.min(amplitude),
                        'amp_range': np.max(amplitude) - np.min(amplitude),
                        'amp_skewness': stats.skew(amplitude),
                        'amp_kurtosis': stats.kurtosis(amplitude),
                        'phase_mean': np.mean(phase),
                        'phase_std': np.std(phase),
                        'phase_var': np.var(phase),
                        # Frequency domain features
                        'spectral_centroid': np.sum(np.arange(52) * amplitude) / np.sum(amplitude),
                        'spectral_spread': np.sqrt(np.sum(((np.arange(52) - np.sum(np.arange(52) * amplitude) / np.sum(amplitude))**2) * amplitude) / np.sum(amplitude)),
                        # Multipath indicators
                        'frequency_selectivity': np.std(amplitude) / np.mean(amplitude),
                        'rms_delay_spread': self._calculate_rms_delay_spread(phase),
                        'channel_gain': np.mean(amplitude**2),
                        # Complex CSI characteristics
                        'phase_linearity': np.std(np.diff(np.unwrap(phase))),
                        'amplitude_envelope_variation': np.std(savgol_filter(amplitude, 5, 2)) / np.mean(amplitude)
                    }
                    
                    location_samples.append(sample_features)
                    all_data.append(sample_features)
                
                # Store location statistics
                location_stats[location_key] = {
                    'sample_count': len(location_samples),
                    'mean_rssi': np.mean([s['rssi'] for s in location_samples]),
                    'std_rssi': np.std([s['rssi'] for s in location_samples]),
                    'mean_amplitude': np.mean([s['amp_mean'] for s in location_samples]),
                    'multipath_complexity': np.mean([s['frequency_selectivity'] for s in location_samples])
                }
        
        self.df = pd.DataFrame(all_data)
        self.analysis_results['location_stats'] = location_stats
        
        print(f"✅ Data Loading Complete:")
        print(f"   • Total samples: {len(self.df):,}")
        print(f"   • Unique locations: {len(location_stats)}")
        print(f"   • Coordinate range: X[{self.df['x'].min()}-{self.df['x'].max()}], Y[{self.df['y'].min()}-{self.df['y'].max()}] meters")
        print(f"   • Samples per location: {self.df.groupby(['x', 'y']).size().describe()}")
        
        return self.df
    
    def _calculate_rms_delay_spread(self, phase):
        """Calculate RMS delay spread from phase response."""
        try:
            # Unwrap phase and calculate group delay
            unwrapped_phase = np.unwrap(phase)
            group_delay = -np.diff(unwrapped_phase)
            return np.sqrt(np.mean(group_delay**2)) if len(group_delay) > 0 else 0
        except:
            return 0
    
    def analyze_spatial_characteristics(self):
        """
        PHASE 2: SPATIAL CHARACTERISTICS ANALYSIS
        
        Goal: Evaluate spatial diversity and signal propagation patterns
        to determine localization feasibility and optimal model architecture.
        """
        print("\n📍 PHASE 2: SPATIAL CHARACTERISTICS ANALYSIS")
        print("-" * 60)
        print("Objective: Assess spatial signal diversity for location discrimination")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Analysis 1: Measurement Grid Layout and Coverage
        print("\n🗺️  Analysis 1: Spatial Coverage Assessment")
        print("Goal: Evaluate measurement grid completeness and spatial resolution")
        
        unique_locations = self.df.groupby(['x', 'y']).size().reset_index(name='sample_count')
        
        scatter = axes[0, 0].scatter(unique_locations['x'], unique_locations['y'], 
                                   c=unique_locations['sample_count'], 
                                   s=200, cmap='viridis', edgecolors='black', linewidth=1)
        
        # Add coordinate labels
        for _, row in unique_locations.iterrows():
            axes[0, 0].annotate(f'({int(row.x)},{int(row.y)})', 
                              (row.x, row.y), xytext=(3, 3), 
                              textcoords='offset points', fontsize=8, 
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        axes[0, 0].set_xlabel('X Coordinate (meters)')
        axes[0, 0].set_ylabel('Y Coordinate (meters)')
        axes[0, 0].set_title('Measurement Grid Layout\n(Sample Density Analysis)')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(scatter, ax=axes[0, 0], label='Samples per Location')
        
        # Calculate spatial resolution
        x_resolution = np.mean(np.diff(sorted(unique_locations['x'].unique())))
        y_resolution = np.mean(np.diff(sorted(unique_locations['y'].unique())))
        room_area = (unique_locations['x'].max() - unique_locations['x'].min()) * (unique_locations['y'].max() - unique_locations['y'].min())
        
        print(f"📏 Spatial Resolution Analysis:")
        print(f"   • Grid resolution: {x_resolution:.1f}m × {y_resolution:.1f}m")
        print(f"   • Room dimensions: {unique_locations['x'].max() - unique_locations['x'].min():.1f}m × {unique_locations['y'].max() - unique_locations['y'].min():.1f}m")
        print(f"   • Coverage area: {room_area:.1f} m²")
        print(f"   • Location density: {len(unique_locations) / room_area:.2f} points/m²")
        print(f"✅ Assessment: {'High' if len(unique_locations) >= 25 else 'Medium' if len(unique_locations) >= 15 else 'Low'} spatial resolution for localization")
        
        # Analysis 2: RSSI Spatial Distribution Pattern
        print("\n📡 Analysis 2: RSSI Spatial Distribution")
        print("Goal: Identify signal strength patterns and potential dead zones")
        
        scatter = axes[0, 1].scatter(self.df['x'], self.df['y'], c=self.df['rssi'], 
                                   s=30, cmap='plasma', alpha=0.7)
        axes[0, 1].set_xlabel('X Coordinate (meters)')
        axes[0, 1].set_ylabel('Y Coordinate (meters)')
        axes[0, 1].set_title('RSSI Spatial Distribution\n(Signal Strength Pattern)')
        cb = plt.colorbar(scatter, ax=axes[0, 1], label='RSSI (dBm)')
        
        # RSSI statistics
        rssi_range = self.df['rssi'].max() - self.df['rssi'].min()
        rssi_gradient = self._calculate_spatial_gradient('rssi')
        
        print(f"📊 RSSI Analysis Results:")
        print(f"   • RSSI range: {self.df['rssi'].min():.1f} to {self.df['rssi'].max():.1f} dBm ({rssi_range:.1f} dB span)")
        print(f"   • Spatial gradient: {rssi_gradient:.2f} dB/meter")
        print(f"   • Signal quality: {'Excellent' if rssi_range > 15 else 'Good' if rssi_range > 10 else 'Moderate'} for localization")
        print(f"✅ Localization potential: {'High' if rssi_range > 15 and rssi_gradient > 1 else 'Medium' if rssi_range > 10 else 'Low'}")
        
        # Analysis 3: Amplitude Spatial Characteristics
        print("\n🌊 Analysis 3: CSI Amplitude Spatial Analysis")
        print("Goal: Evaluate multipath-induced amplitude variations across space")
        
        scatter = axes[0, 2].scatter(self.df['x'], self.df['y'], c=self.df['amp_mean'], 
                                   s=30, cmap='coolwarm', alpha=0.7)
        axes[0, 2].set_xlabel('X Coordinate (meters)')
        axes[0, 2].set_ylabel('Y Coordinate (meters)')
        axes[0, 2].set_title('Mean CSI Amplitude Distribution\n(Channel Gain Pattern)')
        plt.colorbar(scatter, ax=axes[0, 2], label='Mean Amplitude')
        
        amp_range = self.df['amp_mean'].max() - self.df['amp_mean'].min()
        amp_gradient = self._calculate_spatial_gradient('amp_mean')
        
        print(f"📈 Amplitude Analysis Results:")
        print(f"   • Amplitude range: {self.df['amp_mean'].min():.2f} to {self.df['amp_mean'].max():.2f} ({amp_range:.2f} span)")
        print(f"   • Spatial gradient: {amp_gradient:.3f} units/meter")
        print(f"   • Multipath richness: {'High' if amp_range > 10 else 'Medium' if amp_range > 5 else 'Low'}")
        
        # Analysis 4: Frequency Selectivity (Multipath Complexity)
        print("\n🔄 Analysis 4: Multipath Complexity Assessment")
        print("Goal: Quantify frequency selectivity as multipath richness indicator")
        
        scatter = axes[1, 0].scatter(self.df['x'], self.df['y'], c=self.df['frequency_selectivity'], 
                                   s=30, cmap='hot', alpha=0.7)
        axes[1, 0].set_xlabel('X Coordinate (meters)')
        axes[1, 0].set_ylabel('Y Coordinate (meters)')
        axes[1, 0].set_title('Frequency Selectivity\n(Multipath Complexity Indicator)')
        plt.colorbar(scatter, ax=axes[1, 0], label='Frequency Selectivity')
        
        fs_stats = self.df['frequency_selectivity'].describe()
        multipath_zones = (self.df['frequency_selectivity'] > fs_stats['75%']).sum() / len(self.df) * 100
        
        print(f"🌊 Multipath Analysis Results:")
        print(f"   • Frequency selectivity range: {fs_stats['min']:.3f} to {fs_stats['max']:.3f}")
        print(f"   • Mean selectivity: {fs_stats['mean']:.3f} ± {fs_stats['std']:.3f}")
        print(f"   • High multipath zones: {multipath_zones:.1f}% of locations")
        print(f"✅ Multipath richness: {'Excellent' if fs_stats['mean'] > 0.3 else 'Good' if fs_stats['mean'] > 0.2 else 'Moderate'} for fingerprinting")
        
        # Analysis 5: Phase Characteristics and Delay Spread
        print("\n⏱️  Analysis 5: Phase Response and Delay Spread")
        print("Goal: Assess temporal dispersion characteristics for model design")
        
        scatter = axes[1, 1].scatter(self.df['x'], self.df['y'], c=self.df['rms_delay_spread'], 
                                   s=30, cmap='viridis', alpha=0.7)
        axes[1, 1].set_xlabel('X Coordinate (meters)')
        axes[1, 1].set_ylabel('Y Coordinate (meters)')
        axes[1, 1].set_title('RMS Delay Spread\n(Temporal Dispersion Pattern)')
        plt.colorbar(scatter, ax=axes[1, 1], label='RMS Delay Spread')
        
        delay_stats = self.df['rms_delay_spread'].describe()
        
        print(f"⏲️  Delay Spread Analysis:")
        print(f"   • RMS delay spread range: {delay_stats['min']:.3f} to {delay_stats['max']:.3f}")
        print(f"   • Mean delay spread: {delay_stats['mean']:.3f} ± {delay_stats['std']:.3f}")
        print(f"   • Temporal diversity: {'High' if delay_stats['std'] > 0.1 else 'Medium' if delay_stats['std'] > 0.05 else 'Low'}")
        
        # Analysis 6: Location Discriminability Score
        print("\n🎯 Analysis 6: Location Discriminability Assessment")
        print("Goal: Evaluate feature separability for classification accuracy")
        
        # Calculate inter-location feature distances
        location_centers = self.df.groupby(['x', 'y']).agg({
            'rssi': 'mean',
            'amp_mean': 'mean',
            'frequency_selectivity': 'mean',
            'rms_delay_spread': 'mean'
        }).reset_index()
        
        # Normalize features for distance calculation
        feature_cols = ['rssi', 'amp_mean', 'frequency_selectivity', 'rms_delay_spread']
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(location_centers[feature_cols])
        
        # Calculate minimum inter-location distances
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(normalized_features))
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        
        scatter = axes[1, 2].scatter(location_centers['x'], location_centers['y'], 
                                   c=min_distances, s=200, cmap='RdYlGn', 
                                   edgecolors='black', linewidth=1)
        axes[1, 2].set_xlabel('X Coordinate (meters)')
        axes[1, 2].set_ylabel('Y Coordinate (meters)')
        axes[1, 2].set_title('Location Discriminability Score\n(Feature Space Separation)')
        plt.colorbar(scatter, ax=axes[1, 2], label='Min Distance to Neighbor')
        
        discriminability_score = np.mean(min_distances)
        
        print(f"🔍 Discriminability Analysis:")
        print(f"   • Mean inter-location distance: {discriminability_score:.3f}")
        print(f"   • Minimum separation: {np.min(min_distances):.3f}")
        print(f"   • Maximum separation: {np.max(min_distances):.3f}")
        print(f"✅ Classification potential: {'Excellent' if discriminability_score > 2 else 'Good' if discriminability_score > 1 else 'Challenging'}")
        
        plt.tight_layout()
        plt.savefig('comprehensive_spatial_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store results for final assessment
        self.analysis_results['spatial'] = {
            'rssi_range': rssi_range,
            'amplitude_range': amp_range,
            'multipath_complexity': fs_stats['mean'],
            'discriminability_score': discriminability_score,
            'spatial_resolution': min(x_resolution, y_resolution),
            'room_coverage': len(unique_locations)
        }
        
        return self.analysis_results['spatial']
    
    def _calculate_spatial_gradient(self, feature):
        """Calculate spatial gradient of a feature across the measurement area."""
        locations = self.df.groupby(['x', 'y'])[feature].mean().reset_index()
        if len(locations) < 4:
            return 0
        
        x_range = locations['x'].max() - locations['x'].min()
        y_range = locations['y'].max() - locations['y'].min()
        feature_range = locations[feature].max() - locations[feature].min()
        
        if x_range == 0 and y_range == 0:
            return 0
        
        spatial_range = np.sqrt(x_range**2 + y_range**2)
        return feature_range / spatial_range if spatial_range > 0 else 0
    
    def analyze_frequency_domain_characteristics(self):
        """
        PHASE 3: FREQUENCY DOMAIN ANALYSIS
        
        Goal: Examine subcarrier-level characteristics for CNN architecture design
        and understand frequency-selective fading patterns.
        """
        print("\n📡 PHASE 3: FREQUENCY DOMAIN CHARACTERISTICS")
        print("-" * 60)
        print("Objective: Analyze subcarrier-level patterns for optimal CNN design")
        
        # Aggregate amplitude and phase data across all samples
        all_amplitudes = np.vstack([row for row in self.df['amplitude_vector']])
        all_phases = np.vstack([row for row in self.df['phase_vector']])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Analysis 1: Mean Amplitude Response Across Subcarriers
        print("\n📊 Analysis 1: Frequency Response Characterization")
        print("Goal: Identify frequency-selective fading patterns across subcarriers")
        
        mean_amplitude_per_subcarrier = np.mean(all_amplitudes, axis=0)
        std_amplitude_per_subcarrier = np.std(all_amplitudes, axis=0)
        
        axes[0, 0].plot(range(52), mean_amplitude_per_subcarrier, 'b-', linewidth=2, label='Mean Amplitude')
        axes[0, 0].fill_between(range(52), 
                               mean_amplitude_per_subcarrier - std_amplitude_per_subcarrier,
                               mean_amplitude_per_subcarrier + std_amplitude_per_subcarrier,
                               alpha=0.3, label='±1σ Variation')
        axes[0, 0].set_xlabel('Subcarrier Index')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Mean Frequency Response\n(Subcarrier Amplitude Profile)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Calculate frequency selectivity metrics
        amplitude_variation = np.std(mean_amplitude_per_subcarrier) / np.mean(mean_amplitude_per_subcarrier)
        coherence_bandwidth = self._estimate_coherence_bandwidth(all_amplitudes)
        
        print(f"📈 Frequency Response Analysis:")
        print(f"   • Amplitude variation across subcarriers: {amplitude_variation:.3f}")
        print(f"   • Peak amplitude subcarrier: {np.argmax(mean_amplitude_per_subcarrier)}")
        print(f"   • Minimum amplitude subcarrier: {np.argmin(mean_amplitude_per_subcarrier)}")
        print(f"   • Estimated coherence bandwidth: ~{coherence_bandwidth} subcarriers")
        print(f"✅ Frequency selectivity: {'High' if amplitude_variation > 0.3 else 'Medium' if amplitude_variation > 0.15 else 'Low'}")
        
        # Analysis 2: Subcarrier Correlation Matrix (Multipath Fingerprint)
        print("\n🔄 Analysis 2: Subcarrier Correlation Analysis")
        print("Goal: Understand frequency correlation structure for CNN kernel design")
        
        correlation_matrix = np.corrcoef(all_amplitudes.T)
        im = axes[0, 1].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 1].set_xlabel('Subcarrier Index')
        axes[0, 1].set_ylabel('Subcarrier Index')
        axes[0, 1].set_title('Subcarrier Correlation Matrix\n(Frequency Domain Coupling)')
        plt.colorbar(im, ax=axes[0, 1], label='Correlation Coefficient')
        
        # Calculate correlation statistics
        correlation_mean = np.mean(correlation_matrix[correlation_matrix != 1])
        correlation_std = np.std(correlation_matrix[correlation_matrix != 1])
        max_correlation = np.max(correlation_matrix[correlation_matrix != 1])
        
        print(f"🔗 Correlation Analysis Results:")
        print(f"   • Mean inter-subcarrier correlation: {correlation_mean:.3f}")
        print(f"   • Correlation standard deviation: {correlation_std:.3f}")
        print(f"   • Maximum correlation: {max_correlation:.3f}")
        print(f"✅ CNN implication: {'Strong' if correlation_mean > 0.3 else 'Moderate' if correlation_mean > 0.1 else 'Weak'} frequency coupling suggests {'larger' if correlation_mean > 0.3 else 'smaller'} kernel sizes")
        
        # Analysis 3: Phase Response and Group Delay
        print("\n⚡ Analysis 3: Phase Response Characteristics")
        print("Goal: Analyze phase linearity and group delay for temporal modeling")
        
        mean_phase_per_subcarrier = np.angle(np.mean(np.exp(1j * all_phases), axis=0))
        unwrapped_mean_phase = np.unwrap(mean_phase_per_subcarrier)
        
        axes[0, 2].plot(range(52), unwrapped_mean_phase, 'r-', linewidth=2, label='Unwrapped Phase')
        
        # Fit linear trend to estimate group delay
        slope, intercept = np.polyfit(range(52), unwrapped_mean_phase, 1)
        linear_trend = slope * np.arange(52) + intercept
        axes[0, 2].plot(range(52), linear_trend, 'k--', linewidth=2, label=f'Linear Fit (slope={slope:.3f})')
        
        axes[0, 2].set_xlabel('Subcarrier Index')
        axes[0, 2].set_ylabel('Unwrapped Phase (radians)')
        axes[0, 2].set_title('Phase Response Linearity\n(Group Delay Estimation)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Calculate phase linearity metrics
        phase_deviation = np.std(unwrapped_mean_phase - linear_trend)
        phase_coherence = np.abs(np.mean(np.exp(1j * all_phases), axis=0))
        
        print(f"📐 Phase Analysis Results:")
        print(f"   • Group delay slope: {slope:.3f} rad/subcarrier")
        print(f"   • Phase linearity deviation: {phase_deviation:.3f}")
        print(f"   • Mean phase coherence: {np.mean(phase_coherence):.3f}")
        print(f"✅ Phase stability: {'High' if phase_deviation < 0.5 else 'Medium' if phase_deviation < 1.0 else 'Low'}")
        
        # Analysis 4: Amplitude Distribution Analysis
        print("\n📊 Analysis 4: Amplitude Statistical Distribution")
        print("Goal: Characterize amplitude statistics for data normalization")
        
        all_amplitudes_flat = all_amplitudes.flatten()
        
        # Plot histogram with fitted distributions
        axes[1, 0].hist(all_amplitudes_flat, bins=50, density=True, alpha=0.7, 
                       color='skyblue', edgecolor='black', label='Empirical Distribution')
        
        # Fit normal and Rayleigh distributions
        mu_normal, sigma_normal = stats.norm.fit(all_amplitudes_flat)
        scale_rayleigh = stats.rayleigh.fit(all_amplitudes_flat)[1]
        
        x_range = np.linspace(all_amplitudes_flat.min(), all_amplitudes_flat.max(), 200)
        axes[1, 0].plot(x_range, stats.norm.pdf(x_range, mu_normal, sigma_normal), 
                       'r-', linewidth=2, label=f'Normal (μ={mu_normal:.2f}, σ={sigma_normal:.2f})')
        axes[1, 0].plot(x_range, stats.rayleigh.pdf(x_range, scale=scale_rayleigh), 
                       'g-', linewidth=2, label=f'Rayleigh (σ={scale_rayleigh:.2f})')
        
        axes[1, 0].set_xlabel('Amplitude')
        axes[1, 0].set_ylabel('Probability Density')
        axes[1, 0].set_title('Amplitude Distribution Analysis\n(Statistical Characterization)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistical tests
        ks_normal = stats.kstest(all_amplitudes_flat, lambda x: stats.norm.cdf(x, mu_normal, sigma_normal))
        ks_rayleigh = stats.kstest(all_amplitudes_flat, lambda x: stats.rayleigh.cdf(x, scale=scale_rayleigh))
        
        print(f"📈 Distribution Analysis:")
        print(f"   • Mean amplitude: {mu_normal:.3f}")
        print(f"   • Standard deviation: {sigma_normal:.3f}")
        print(f"   • Skewness: {stats.skew(all_amplitudes_flat):.3f}")
        print(f"   • Kurtosis: {stats.kurtosis(all_amplitudes_flat):.3f}")
        print(f"   • Normal fit quality (KS p-value): {ks_normal.pvalue:.4f}")
        print(f"   • Rayleigh fit quality (KS p-value): {ks_rayleigh.pvalue:.4f}")
        print(f"✅ Best fit: {'Rayleigh' if ks_rayleigh.pvalue > ks_normal.pvalue else 'Normal'} distribution")
        
        # Analysis 5: Dynamic Range and SNR Analysis
        print("\n🔊 Analysis 5: Dynamic Range and Signal Quality")
        print("Goal: Assess signal dynamic range for quantization and preprocessing")
        
        # Calculate dynamic range per sample
        sample_dynamic_ranges = []
        sample_snrs = []
        
        for i in range(min(1000, len(all_amplitudes))):  # Sample subset for efficiency
            sample_amp = all_amplitudes[i]
            dynamic_range = 20 * np.log10(np.max(sample_amp) / (np.min(sample_amp) + 1e-10))
            signal_power = np.mean(sample_amp**2)
            # Estimate noise as minimum amplitude (simplified)
            noise_power = np.min(sample_amp)**2
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            sample_dynamic_ranges.append(dynamic_range)
            sample_snrs.append(snr)
        
        axes[1, 1].hist(sample_dynamic_ranges, bins=30, alpha=0.7, color='orange', 
                       edgecolor='black', label=f'Mean: {np.mean(sample_dynamic_ranges):.1f} dB')
        axes[1, 1].set_xlabel('Dynamic Range (dB)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Dynamic Range Distribution\n(Signal Quantization Analysis)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        dynamic_range_stats = {
            'mean': np.mean(sample_dynamic_ranges),
            'std': np.std(sample_dynamic_ranges),
            'min': np.min(sample_dynamic_ranges),
            'max': np.max(sample_dynamic_ranges)
        }
        
        print(f"🔊 Dynamic Range Analysis:")
        print(f"   • Mean dynamic range: {dynamic_range_stats['mean']:.1f} ± {dynamic_range_stats['std']:.1f} dB")
        print(f"   • Range span: {dynamic_range_stats['min']:.1f} to {dynamic_range_stats['max']:.1f} dB")
        print(f"   • Effective bits required: {dynamic_range_stats['max']/6:.1f} bits")
        print(f"✅ Quantization recommendation: {'16-bit' if dynamic_range_stats['max'] > 60 else '12-bit' if dynamic_range_stats['max'] > 36 else '8-bit'}")
        
        # Analysis 6: Frequency Correlation vs Distance
        print("\n📏 Analysis 6: Frequency Correlation Function")
        print("Goal: Estimate coherence bandwidth for CNN receptive field design")
        
        # Calculate correlation vs frequency separation
        frequency_separations = range(1, 26)  # Up to half the subcarriers
        correlation_vs_separation = []
        
        for sep in frequency_separations:
            correlations = []
            for sc in range(52 - sep):
                corr = np.corrcoef(all_amplitudes[:, sc], all_amplitudes[:, sc + sep])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            if correlations:
                correlation_vs_separation.append(np.mean(correlations))
            else:
                correlation_vs_separation.append(0)
        
        axes[1, 2].plot(frequency_separations, correlation_vs_separation, 'go-', 
                       linewidth=2, markersize=6, label='Amplitude Correlation')
        axes[1, 2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                          label='50% Correlation Threshold')
        axes[1, 2].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, 
                          label='10% Correlation Threshold')
        axes[1, 2].set_xlabel('Subcarrier Separation')
        axes[1, 2].set_ylabel('Mean Correlation Magnitude')
        axes[1, 2].set_title('Frequency Correlation Function\n(Coherence Bandwidth Analysis)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # Find coherence bandwidth
        coherence_50 = next((i for i, corr in enumerate(correlation_vs_separation) if corr < 0.5), len(correlation_vs_separation))
        coherence_10 = next((i for i, corr in enumerate(correlation_vs_separation) if corr < 0.1), len(correlation_vs_separation))
        
        print(f"📡 Coherence Bandwidth Analysis:")
        print(f"   • 50% coherence bandwidth: {coherence_50 + 1} subcarriers")
        print(f"   • 10% coherence bandwidth: {coherence_10 + 1} subcarriers")
        print(f"   • Correlation decay rate: {-np.polyfit(frequency_separations[:10], correlation_vs_separation[:10], 1)[0]:.3f}/subcarrier")
        print(f"✅ CNN kernel recommendation: {coherence_50 + 1}-{coherence_10 + 1} subcarrier receptive field")
        
        plt.tight_layout()
        plt.savefig('frequency_domain_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Store frequency domain results
        self.analysis_results['frequency'] = {
            'amplitude_variation': amplitude_variation,
            'correlation_mean': correlation_mean,
            'phase_deviation': phase_deviation,
            'dynamic_range': dynamic_range_stats,
            'coherence_bandwidth_50': coherence_50 + 1,
            'coherence_bandwidth_10': coherence_10 + 1
        }
        
        return self.analysis_results['frequency']
    
    def _estimate_coherence_bandwidth(self, amplitudes):
        """Estimate coherence bandwidth from amplitude correlation."""
        if len(amplitudes) < 2:
            return 1
        
        correlations = []
        for sep in range(1, min(26, amplitudes.shape[1])):
            corr_values = []
            for sc in range(amplitudes.shape[1] - sep):
                corr = np.corrcoef(amplitudes[:, sc], amplitudes[:, sc + sep])[0, 1]
                if not np.isnan(corr):
                    corr_values.append(abs(corr))
            
            if corr_values and np.mean(corr_values) < 0.5:
                return sep
            
        return min(26, amplitudes.shape[1])

def main():
    """Execute comprehensive CSI analysis."""
    analyzer = ComprehensiveCSIAnalyzer()
    
    # Phase 1: Data loading and structure analysis
    df = analyzer.load_data_with_analysis()
    
    if df is None or df.empty:
        print("❌ Analysis failed: No data loaded")
        return
    
    # Phase 2: Spatial characteristics analysis
    spatial_results = analyzer.analyze_spatial_characteristics()
    
    # Phase 3: Frequency domain analysis
    frequency_results = analyzer.analyze_frequency_domain_characteristics()
    
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE LOCALIZATION FEASIBILITY ASSESSMENT")
    print("="*80)
    
    # Overall assessment scoring
    scores = {
        'spatial_diversity': min(5, spatial_results['discriminability_score'] * 2),
        'signal_strength': min(5, spatial_results['rssi_range'] / 4),
        'multipath_richness': min(5, spatial_results['multipath_complexity'] * 10),
        'frequency_selectivity': min(5, frequency_results['amplitude_variation'] * 10),
        'coverage_density': min(5, spatial_results['room_coverage'] / 10),
        'signal_quality': min(5, frequency_results['dynamic_range']['mean'] / 20)
    }
    
    overall_score = np.mean(list(scores.values()))
    
    print(f"\n📊 LOCALIZATION FEASIBILITY SCORES (0-5 scale):")
    for criterion, score in scores.items():
        status = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
        print(f"   {status} {criterion.replace('_', ' ').title()}: {score:.2f}/5")
    
    print(f"\n🎯 OVERALL FEASIBILITY SCORE: {overall_score:.2f}/5")
    
    if overall_score >= 4:
        feasibility = "🟢 EXCELLENT - Highly suitable for deep learning localization"
    elif overall_score >= 3:
        feasibility = "🟡 GOOD - Suitable with proper preprocessing and architecture"
    elif overall_score >= 2:
        feasibility = "🟠 MODERATE - Challenging but potentially feasible"
    else:
        feasibility = "🔴 POOR - Requires significant data enhancement"
    
    print(f"🏆 FEASIBILITY ASSESSMENT: {feasibility}")
    
    print(f"\n🤖 DEEP LEARNING ARCHITECTURE RECOMMENDATIONS:")
    
    # Architecture recommendations based on analysis
    if frequency_results['coherence_bandwidth_50'] <= 5:
        cnn_architecture = "1D CNN with small kernels (3-5)"
    elif frequency_results['coherence_bandwidth_50'] <= 10:
        cnn_architecture = "1D CNN with medium kernels (5-7)"
    else:
        cnn_architecture = "1D CNN with large kernels (7-11)"
    
    if spatial_results['discriminability_score'] > 2:
        classification_complexity = "Simple classification network"
    elif spatial_results['discriminability_score'] > 1:
        classification_complexity = "Deep classification network with regularization"
    else:
        classification_complexity = "Very deep network with advanced regularization"
    
    print(f"   🔸 Primary architecture: {cnn_architecture}")
    print(f"   🔸 Classification complexity: {classification_complexity}")
    print(f"   🔸 Input preprocessing: {'Standardization' if frequency_results['dynamic_range']['std'] > 10 else 'Min-Max normalization'}")
    print(f"   🔸 Data augmentation: {'Essential' if overall_score < 4 else 'Recommended'}")
    
    print(f"\n📋 FINAL RECOMMENDATIONS:")
    print(f"   ✅ Dataset is {'excellent' if overall_score >= 4 else 'suitable' if overall_score >= 3 else 'challenging'} for CNN-based indoor localization")
    print(f"   ✅ Expected localization accuracy: {'High (>95%)' if overall_score >= 4 else 'Good (85-95%)' if overall_score >= 3 else 'Moderate (70-85%)'}")
    print(f"   ✅ Recommended model: Hybrid CNN-LSTM or Multi-scale CNN")
    print(f"   ✅ Key success factors: Feature engineering, data augmentation, ensemble methods")

if __name__ == "__main__":
    main()
