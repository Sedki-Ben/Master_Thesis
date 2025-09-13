#!/usr/bin/env python3
"""
CSI-Based Multipath and Environmental Complexity Analysis

This script performs theoretical CSI analysis to understand:
1. Multipath propagation characteristics
2. Environmental complexity (scattering richness)
3. Channel coherence properties
4. Spatial correlation structure
5. Frequency selectivity patterns

THEORETICAL FOUNDATION:
- CSI represents the complex channel response H(f) at each subcarrier
- Multipath creates frequency-selective fading across subcarriers
- Environmental complexity manifests as decorrelation patterns
- Rich scattering environments show specific CSI signatures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
from pathlib import Path
import re
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 11,
    'figure.figsize': (16, 12),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2
})

class CSIMultipathAnalyzer:
    """
    Advanced CSI analysis for multipath and environmental complexity assessment.
    """
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("CSI_Multipath_Analysis")
        self.output_dir.mkdir(exist_ok=True)
        self.csi_data = None
        print(f"🔬 CSI Multipath Analyzer initialized")
        print(f"📁 Results will be saved to: {self.output_dir}")
    
    def load_csi_data(self):
        """
        Load CSI amplitude and phase data for analysis.
        
        THEORY: CSI H(f,t,r) represents the complex channel transfer function
        where f=frequency, t=time, r=spatial position
        """
        print("\n📡 LOADING CSI DATA FOR MULTIPATH ANALYSIS")
        print("=" * 60)
        print("THEORY: CSI = |H(f)|·e^(jφ(f)) where |H(f)| = amplitude, φ(f) = phase")
        
        all_data = []
        
        for file_path in sorted(self.data_dir.glob("*.csv"))[:20]:  # Limit for analysis speed
            match = re.match(r'(\d+),(\d+)\.csv', file_path.name)
            if not match:
                continue
                
            x, y = int(match.group(1)), int(match.group(2))
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    if i >= 100:  # Limit samples per location
                        break
                    
                    amplitude = np.array(json.loads(row['amplitude']))
                    phase = np.array(json.loads(row['phase']))
                    
                    # Reconstruct complex CSI
                    csi_complex = amplitude * np.exp(1j * phase)
                    
                    sample_data = {
                        'x': x, 'y': y,
                        'sample_id': i,
                        'amplitude': amplitude,
                        'phase': phase,
                        'csi_complex': csi_complex,
                        'rssi': float(row['rssi'])
                    }
                    all_data.append(sample_data)
        
        self.csi_data = all_data
        print(f"✅ Loaded {len(self.csi_data)} CSI samples from {len(set([(d['x'], d['y']) for d in self.csi_data]))} locations")
        return len(self.csi_data)
    
    def analyze_frequency_selectivity(self):
        """
        THEORY: Frequency Selectivity Analysis
        
        In multipath environments, different subcarriers experience different fading.
        Key metrics:
        1. Coherence Bandwidth (Bc): Frequency range over which channel is "flat"
        2. RMS Delay Spread (τ_rms): Time dispersion measure, Bc ≈ 1/(2π·τ_rms)
        3. Frequency Correlation Function: R(Δf) = E[H(f)H*(f+Δf)]
        4. Spectral Efficiency: Variation in |H(f)| across frequency
        
        PHYSICAL MEANING:
        - High frequency selectivity → Rich multipath environment
        - Low coherence bandwidth → Large delay spread → Complex environment
        - Decorrelation across frequency → Multiple scatterers
        """
        print("\n📊 FREQUENCY SELECTIVITY ANALYSIS")
        print("=" * 60)
        print("THEORY: Analyzing frequency correlation to understand multipath richness")
        print("METRICS: Coherence bandwidth, RMS delay spread, spectral correlation")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CSI Frequency Selectivity Analysis\n(Multipath Environment Characterization)', 
                     fontsize=16, fontweight='bold')
        
        # Collect all amplitude and phase data
        all_amplitudes = np.array([sample['amplitude'] for sample in self.csi_data])
        all_phases = np.array([sample['phase'] for sample in self.csi_data])
        all_csi_complex = np.array([sample['csi_complex'] for sample in self.csi_data])
        
        # Analysis 1: Frequency Correlation Function
        print("🔍 Computing frequency correlation function...")
        
        n_subcarriers = 52
        freq_separations = range(1, 26)  # Up to half the subcarriers
        correlation_values = []
        
        for delta_f in freq_separations:
            correlations = []
            for sc in range(n_subcarriers - delta_f):
                # Complex correlation between subcarriers separated by delta_f
                h1 = all_csi_complex[:, sc]
                h2 = all_csi_complex[:, sc + delta_f]
                
                # Normalized complex correlation
                corr = np.abs(np.corrcoef(h1.real, h2.real)[0,1] + 
                             1j * np.corrcoef(h1.imag, h2.imag)[0,1])
                
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if correlations:
                correlation_values.append(np.mean(correlations))
            else:
                correlation_values.append(0)
        
        axes[0,0].plot(freq_separations, correlation_values, 'bo-', linewidth=2, markersize=6)
        axes[0,0].axhline(y=0.5, color='red', linestyle='--', label='50% Correlation')
        axes[0,0].axhline(y=0.1, color='orange', linestyle='--', label='10% Correlation')
        axes[0,0].set_xlabel('Frequency Separation (subcarriers)')
        axes[0,0].set_ylabel('|Correlation Coefficient|')
        axes[0,0].set_title('Frequency Correlation Function\n(Coherence Bandwidth Estimation)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Find coherence bandwidth (50% correlation point)
        coherence_bw_50 = next((i for i, corr in enumerate(correlation_values) if corr < 0.5), len(correlation_values))
        coherence_bw_10 = next((i for i, corr in enumerate(correlation_values) if corr < 0.1), len(correlation_values))
        
        print(f"   Coherence bandwidth (50%): {coherence_bw_50 + 1} subcarriers")
        print(f"   Coherence bandwidth (10%): {coherence_bw_10 + 1} subcarriers")
        
        # Analysis 2: Power Delay Profile Estimation
        print("🔍 Estimating power delay profile...")
        
        # Average power spectral density across all samples
        mean_psd = np.mean(np.abs(all_csi_complex)**2, axis=0)
        
        # Inverse FFT to get impulse response (approximate)
        # Pad to increase delay resolution
        padded_psd = np.pad(mean_psd, (0, 208), mode='constant')  # Pad to 260 points
        impulse_response = np.abs(np.fft.ifft(padded_psd))
        
        # Time axis (assuming 20 MHz bandwidth, 52 subcarriers)
        sample_period = 1 / (20e6)  # 50 ns
        delay_axis = np.arange(len(impulse_response)) * sample_period * 1e9  # Convert to ns
        
        axes[0,1].plot(delay_axis[:50], impulse_response[:50], 'g-', linewidth=2)
        axes[0,1].set_xlabel('Delay (ns)')
        axes[0,1].set_ylabel('Normalized Power')
        axes[0,1].set_title('Estimated Power Delay Profile\n(Multipath Components)')
        axes[0,1].grid(True)
        
        # Calculate RMS delay spread
        power_normalized = impulse_response[:50] / np.sum(impulse_response[:50])
        mean_delay = np.sum(delay_axis[:50] * power_normalized)
        rms_delay_spread = np.sqrt(np.sum((delay_axis[:50] - mean_delay)**2 * power_normalized))
        
        axes[0,1].axvline(mean_delay, color='red', linestyle='--', label=f'Mean Delay: {mean_delay:.1f} ns')
        axes[0,1].legend()
        
        print(f"   RMS delay spread: {rms_delay_spread:.2f} ns")
        
        # Analysis 3: Amplitude Variation Across Frequency
        amplitude_variation = np.std(all_amplitudes, axis=0)
        mean_amplitude = np.mean(all_amplitudes, axis=0)
        
        axes[0,2].plot(range(52), mean_amplitude, 'b-', linewidth=2, label='Mean Amplitude')
        axes[0,2].fill_between(range(52), 
                              mean_amplitude - amplitude_variation,
                              mean_amplitude + amplitude_variation,
                              alpha=0.3, label='±1σ Variation')
        axes[0,2].set_xlabel('Subcarrier Index')
        axes[0,2].set_ylabel('CSI Amplitude')
        axes[0,2].set_title('Frequency-Domain Amplitude Profile\n(Spectral Fading Pattern)')
        axes[0,2].legend()
        axes[0,2].grid(True)
        
        # Calculate spectral efficiency metric
        spectral_efficiency = np.std(mean_amplitude) / np.mean(mean_amplitude)
        print(f"   Spectral fading efficiency: {spectral_efficiency:.3f}")
        
        # Analysis 4: Phase Coherence Analysis
        # Complex phasor averaging to measure phase coherence
        phase_coherence = np.abs(np.mean(np.exp(1j * all_phases), axis=0))
        
        axes[1,0].plot(range(52), phase_coherence, 'r-', linewidth=2)
        axes[1,0].set_xlabel('Subcarrier Index')
        axes[1,0].set_ylabel('Phase Coherence')
        axes[1,0].set_title('Phase Coherence Across Subcarriers\n(Multipath Phase Stability)')
        axes[1,0].grid(True)
        axes[1,0].set_ylim(0, 1)
        
        mean_phase_coherence = np.mean(phase_coherence)
        print(f"   Mean phase coherence: {mean_phase_coherence:.3f}")
        
        # Analysis 5: Ricean K-factor Estimation
        print("🔍 Estimating Ricean K-factor...")
        
        # K-factor estimation using method of moments
        k_factors = []
        for sc in range(52):
            amplitude_samples = all_amplitudes[:, sc]
            
            # Method of moments for Ricean distribution
            mean_power = np.mean(amplitude_samples**2)
            var_power = np.var(amplitude_samples**2)
            
            if var_power > 0:
                # K = (mean²/var - 1)/2 for power samples
                k_factor = max(0, (mean_power**2 / var_power - 1) / 2)
                k_factors.append(k_factor)
            else:
                k_factors.append(0)
        
        axes[1,1].plot(range(52), 10*np.log10(np.array(k_factors) + 1e-10), 'purple', linewidth=2)
        axes[1,1].set_xlabel('Subcarrier Index')
        axes[1,1].set_ylabel('K-factor (dB)')
        axes[1,1].set_title('Ricean K-factor Estimation\n(LOS/NLOS Characterization)')
        axes[1,1].grid(True)
        
        mean_k_factor = np.mean(k_factors)
        print(f"   Mean K-factor: {10*np.log10(mean_k_factor + 1e-10):.1f} dB")
        
        # Analysis 6: Spatial-Frequency Joint Analysis
        # Group by location and analyze frequency diversity
        location_diversity = {}
        for sample in self.csi_data:
            loc_key = (sample['x'], sample['y'])
            if loc_key not in location_diversity:
                location_diversity[loc_key] = []
            
            # Calculate frequency diversity metric for this sample
            amp = sample['amplitude']
            freq_div = np.std(amp) / (np.mean(amp) + 1e-10)
            location_diversity[loc_key].append(freq_div)
        
        # Average diversity per location
        locations = list(location_diversity.keys())
        avg_diversity = [np.mean(location_diversity[loc]) for loc in locations]
        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]
        
        scatter = axes[1,2].scatter(x_coords, y_coords, c=avg_diversity, 
                                   s=200, cmap='viridis', alpha=0.8)
        axes[1,2].set_xlabel('X Coordinate (m)')
        axes[1,2].set_ylabel('Y Coordinate (m)')
        axes[1,2].set_title('Spatial Frequency Diversity\n(Environmental Complexity Map)')
        plt.colorbar(scatter, ax=axes[1,2], label='Frequency Diversity')
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'frequency_selectivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'coherence_bw_50': coherence_bw_50 + 1,
            'coherence_bw_10': coherence_bw_10 + 1,
            'rms_delay_spread': rms_delay_spread,
            'spectral_efficiency': spectral_efficiency,
            'mean_phase_coherence': mean_phase_coherence,
            'mean_k_factor_db': 10*np.log10(mean_k_factor + 1e-10),
            'mean_freq_diversity': np.mean(avg_diversity)
        }
    
    def analyze_spatial_correlation(self):
        """
        THEORY: Spatial Correlation Analysis
        
        Spatial correlation measures how CSI changes with position:
        1. Spatial Coherence Distance: Distance over which channel remains correlated
        2. Spatial Diversity: How much channel varies across space
        3. Angular Spread: Related to multipath angular distribution
        4. Spatial Rank: Effective dimensionality of spatial channel
        
        PHYSICAL MEANING:
        - Low spatial correlation → Rich scattering environment
        - High spatial diversity → Complex multipath propagation
        - Short coherence distance → Many independent scatterers
        """
        print("\n🗺️ SPATIAL CORRELATION ANALYSIS")
        print("=" * 60)
        print("THEORY: Analyzing spatial decorrelation to assess scattering richness")
        print("METRICS: Spatial coherence distance, angular spread, spatial diversity")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CSI Spatial Correlation Analysis\n(Scattering Environment Assessment)', 
                     fontsize=16, fontweight='bold')
        
        # Group data by location
        location_data = {}
        for sample in self.csi_data:
            loc_key = (sample['x'], sample['y'])
            if loc_key not in location_data:
                location_data[loc_key] = {'csi_samples': [], 'coordinates': loc_key}
            location_data[loc_key]['csi_samples'].append(sample['csi_complex'])
        
        # Calculate average CSI per location
        location_avg_csi = {}
        for loc, data in location_data.items():
            location_avg_csi[loc] = np.mean(np.array(data['csi_samples']), axis=0)
        
        locations = list(location_avg_csi.keys())
        n_locations = len(locations)
        
        print(f"🔍 Analyzing spatial correlation across {n_locations} locations...")
        
        # Analysis 1: Spatial Correlation vs Distance
        distances = []
        correlations = []
        
        for i in range(n_locations):
            for j in range(i+1, n_locations):
                loc1, loc2 = locations[i], locations[j]
                
                # Euclidean distance
                dist = np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
                
                # Complex correlation between locations
                csi1 = location_avg_csi[loc1]
                csi2 = location_avg_csi[loc2]
                
                # Average correlation across subcarriers
                corr_real = np.corrcoef(csi1.real, csi2.real)[0,1]
                corr_imag = np.corrcoef(csi1.imag, csi2.imag)[0,1]
                
                if not (np.isnan(corr_real) or np.isnan(corr_imag)):
                    avg_corr = np.abs(corr_real + 1j * corr_imag)
                    distances.append(dist)
                    correlations.append(avg_corr)
        
        # Sort by distance for plotting
        sorted_indices = np.argsort(distances)
        distances_sorted = np.array(distances)[sorted_indices]
        correlations_sorted = np.array(correlations)[sorted_indices]
        
        axes[0,0].scatter(distances_sorted, correlations_sorted, alpha=0.6, s=30)
        
        # Fit exponential decay model: R(d) = exp(-d/d_c)
        try:
            # Remove zero distances to avoid log issues
            valid_mask = distances_sorted > 0
            if np.sum(valid_mask) > 5:
                log_corr = np.log(correlations_sorted[valid_mask] + 1e-10)
                slope, intercept, r_value, _, _ = stats.linregress(distances_sorted[valid_mask], log_corr)
                
                d_fit = np.linspace(0, np.max(distances_sorted), 100)
                corr_fit = np.exp(intercept + slope * d_fit)
                axes[0,0].plot(d_fit, corr_fit, 'r-', linewidth=2, 
                              label=f'Fit: R(d)=exp(-d/{-1/slope:.2f})')
                
                coherence_distance = -1/slope if slope < 0 else np.inf
                print(f"   Spatial coherence distance: {coherence_distance:.2f} m")
            else:
                coherence_distance = np.inf
                print("   Insufficient data for coherence distance estimation")
        except:
            coherence_distance = np.inf
            print("   Could not estimate spatial coherence distance")
        
        axes[0,0].set_xlabel('Distance (m)')
        axes[0,0].set_ylabel('Spatial Correlation')
        axes[0,0].set_title('Spatial Correlation vs Distance\n(Coherence Distance Estimation)')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Analysis 2: Spatial Diversity Map
        # Calculate CSI variance at each location (temporal diversity)
        location_variance = {}
        for loc, data in location_data.items():
            if len(data['csi_samples']) > 1:
                csi_array = np.array(data['csi_samples'])
                # Average variance across subcarriers
                variance = np.mean(np.var(np.abs(csi_array), axis=0))
                location_variance[loc] = variance
            else:
                location_variance[loc] = 0
        
        x_coords = [loc[0] for loc in location_variance.keys()]
        y_coords = [loc[1] for loc in location_variance.keys()]
        variances = list(location_variance.values())
        
        scatter = axes[0,1].scatter(x_coords, y_coords, c=variances, 
                                   s=200, cmap='hot', alpha=0.8)
        axes[0,1].set_xlabel('X Coordinate (m)')
        axes[0,1].set_ylabel('Y Coordinate (m)')
        axes[0,1].set_title('Spatial CSI Variance Map\n(Temporal Diversity per Location)')
        plt.colorbar(scatter, ax=axes[0,1], label='CSI Variance')
        axes[0,1].grid(True)
        
        # Analysis 3: Angular Spread Estimation
        # Use eigenvalue decomposition of spatial correlation matrix
        print("🔍 Estimating angular spread...")
        
        # Create spatial correlation matrix
        n_locations = len(locations)
        spatial_corr_matrix = np.zeros((n_locations, n_locations), dtype=complex)
        
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations):
                csi1 = location_avg_csi[loc1]
                csi2 = location_avg_csi[loc2]
                
                # Complex correlation
                corr = np.mean(csi1 * np.conj(csi2)) / (np.sqrt(np.mean(np.abs(csi1)**2) * np.mean(np.abs(csi2)**2)) + 1e-10)
                spatial_corr_matrix[i, j] = corr
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(np.abs(spatial_corr_matrix))
        eigenvals = np.real(eigenvals)
        eigenvals = eigenvals[eigenvals > 0]  # Keep positive eigenvalues
        eigenvals = eigenvals / np.sum(eigenvals)  # Normalize
        
        axes[0,2].plot(range(len(eigenvals)), np.sort(eigenvals)[::-1], 'bo-', linewidth=2)
        axes[0,2].set_xlabel('Eigenmode Index')
        axes[0,2].set_ylabel('Normalized Eigenvalue')
        axes[0,2].set_title('Spatial Correlation Eigenspectrum\n(Angular Spread Indicator)')
        axes[0,2].grid(True)
        
        # Calculate effective rank (90% energy)
        cumsum_eigenvals = np.cumsum(np.sort(eigenvals)[::-1])
        effective_rank = np.argmax(cumsum_eigenvals >= 0.9) + 1
        print(f"   Effective spatial rank (90% energy): {effective_rank}")
        
        # Analysis 4: Subcarrier-Dependent Spatial Correlation
        # Analyze how spatial correlation varies with frequency
        subcarrier_spatial_corr = []
        
        for sc in range(52):
            sc_correlations = []
            
            for i in range(min(10, n_locations)):  # Limit for computation
                for j in range(i+1, min(10, n_locations)):
                    loc1, loc2 = locations[i], locations[j]
                    
                    # Get CSI for this subcarrier
                    csi1_sc = location_avg_csi[loc1][sc]
                    csi2_sc = location_avg_csi[loc2][sc]
                    
                    # Complex correlation for this subcarrier
                    corr = (csi1_sc * np.conj(csi2_sc)) / (np.abs(csi1_sc) * np.abs(csi2_sc) + 1e-10)
                    sc_correlations.append(np.abs(corr))
            
            if sc_correlations:
                subcarrier_spatial_corr.append(np.mean(sc_correlations))
            else:
                subcarrier_spatial_corr.append(0)
        
        axes[1,0].plot(range(52), subcarrier_spatial_corr, 'g-', linewidth=2)
        axes[1,0].set_xlabel('Subcarrier Index')
        axes[1,0].set_ylabel('Mean Spatial Correlation')
        axes[1,0].set_title('Frequency-Dependent Spatial Correlation\n(Dispersive Environment Indicator)')
        axes[1,0].grid(True)
        
        # Analysis 5: Mutual Information Between Locations
        print("🔍 Computing spatial mutual information...")
        
        # Calculate mutual information between CSI at different locations
        mutual_info_matrix = np.zeros((min(10, n_locations), min(10, n_locations)))
        
        for i in range(min(10, n_locations)):
            for j in range(min(10, n_locations)):
                if i != j:
                    loc1, loc2 = locations[i], locations[j]
                    
                    # Discretize CSI amplitudes for MI calculation
                    amp1 = np.abs(location_avg_csi[loc1])
                    amp2 = np.abs(location_avg_csi[loc2])
                    
                    # Convert to discrete bins
                    bins1 = np.digitize(amp1, np.linspace(np.min(amp1), np.max(amp1), 10))
                    bins2 = np.digitize(amp2, np.linspace(np.min(amp2), np.max(amp2), 10))
                    
                    mi = mutual_info_score(bins1, bins2)
                    mutual_info_matrix[i, j] = mi
        
        im = axes[1,1].imshow(mutual_info_matrix, cmap='viridis', aspect='auto')
        axes[1,1].set_xlabel('Location Index')
        axes[1,1].set_ylabel('Location Index')
        axes[1,1].set_title('Spatial Mutual Information Matrix\n(Information Redundancy)')
        plt.colorbar(im, ax=axes[1,1], label='Mutual Information (bits)')
        
        # Analysis 6: Environmental Complexity Score
        # Combine multiple metrics into a complexity score
        complexity_metrics = {
            'spatial_decorrelation': 1 - np.mean(correlations_sorted),
            'temporal_diversity': np.mean(variances),
            'angular_richness': effective_rank / n_locations,
            'frequency_dependence': np.std(subcarrier_spatial_corr),
            'information_diversity': np.mean(mutual_info_matrix[mutual_info_matrix > 0])
        }
        
        # Normalize and combine
        normalized_metrics = {}
        for key, value in complexity_metrics.items():
            if not np.isnan(value) and value > 0:
                normalized_metrics[key] = min(1.0, value / np.mean(list(complexity_metrics.values())))
            else:
                normalized_metrics[key] = 0
        
        overall_complexity = np.mean(list(normalized_metrics.values()))
        
        # Plot complexity radar chart
        categories = list(normalized_metrics.keys())
        values = list(normalized_metrics.values())
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values_plot = values + [values[0]]  # Complete the circle
        angles_plot = np.concatenate([angles, [angles[0]]])
        
        axes[1,2].plot(angles_plot, values_plot, 'bo-', linewidth=2)
        axes[1,2].fill(angles_plot, values_plot, alpha=0.25)
        axes[1,2].set_xticks(angles)
        axes[1,2].set_xticklabels([cat.replace('_', '\n') for cat in categories], fontsize=9)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].set_title(f'Environmental Complexity Score\n(Overall: {overall_complexity:.2f})')
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'spatial_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   Overall environmental complexity: {overall_complexity:.3f}")
        
        return {
            'coherence_distance': coherence_distance,
            'effective_rank': effective_rank,
            'complexity_score': overall_complexity,
            'spatial_decorrelation': complexity_metrics['spatial_decorrelation'],
            'temporal_diversity': complexity_metrics['temporal_diversity']
        }
    
    def analyze_channel_stationarity(self):
        """
        THEORY: Channel Stationarity and Temporal Coherence Analysis
        
        Temporal behavior of CSI reveals:
        1. Coherence Time: Duration channel remains constant
        2. Doppler Spread: Frequency broadening due to motion
        3. Stationarity: Statistical consistency over time
        4. Temporal Correlation Function
        
        PHYSICAL MEANING:
        - Short coherence time → Dynamic environment (people, objects moving)
        - Long coherence time → Static environment
        - Non-stationarity → Time-varying scattering conditions
        """
        print("\n⏱️ CHANNEL STATIONARITY ANALYSIS")
        print("=" * 60)
        print("THEORY: Analyzing temporal CSI behavior for environment dynamics")
        print("METRICS: Coherence time, stationarity, temporal correlation")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CSI Temporal Stationarity Analysis\n(Environmental Dynamics Assessment)', 
                     fontsize=16, fontweight='bold')
        
        # Group by location for temporal analysis
        location_temporal = {}
        for sample in self.csi_data:
            loc_key = (sample['x'], sample['y'])
            if loc_key not in location_temporal:
                location_temporal[loc_key] = []
            location_temporal[loc_key].append({
                'sample_id': sample['sample_id'],
                'csi_complex': sample['csi_complex'],
                'amplitude': sample['amplitude']
            })
        
        # Analysis 1: Temporal Correlation Function
        print("🔍 Computing temporal correlation functions...")
        
        temporal_correlations = []
        max_lag = 20  # Maximum time lag to analyze
        
        for loc, samples in location_temporal.items():
            if len(samples) >= max_lag * 2:
                # Sort by sample_id (time order)
                samples_sorted = sorted(samples, key=lambda x: x['sample_id'])
                
                # Extract CSI time series
                csi_series = np.array([s['csi_complex'] for s in samples_sorted])
                
                # Compute autocorrelation for first subcarrier
                csi_sc0 = csi_series[:, 0]  # First subcarrier
                
                loc_correlations = []
                for lag in range(1, min(max_lag, len(csi_sc0))):
                    if lag < len(csi_sc0):
                        corr = np.abs(np.corrcoef(csi_sc0[:-lag], csi_sc0[lag:])[0,1])
                        if not np.isnan(corr):
                            loc_correlations.append(corr)
                        else:
                            loc_correlations.append(0)
                
                if loc_correlations:
                    temporal_correlations.append(loc_correlations)
        
        # Average temporal correlation across locations
        if temporal_correlations:
            max_len = max(len(tc) for tc in temporal_correlations)
            # Pad shorter sequences
            padded_correlations = []
            for tc in temporal_correlations:
                padded = tc + [0] * (max_len - len(tc))
                padded_correlations.append(padded[:max_len])
            
            mean_temporal_corr = np.mean(padded_correlations, axis=0)
            
            axes[0,0].plot(range(1, len(mean_temporal_corr) + 1), mean_temporal_corr, 'b-', linewidth=2)
            axes[0,0].axhline(y=0.5, color='red', linestyle='--', label='50% Correlation')
            axes[0,0].set_xlabel('Time Lag (samples)')
            axes[0,0].set_ylabel('Temporal Correlation')
            axes[0,0].set_title('Temporal Correlation Function\n(Channel Coherence Time)')
            axes[0,0].legend()
            axes[0,0].grid(True)
            
            # Find coherence time (50% correlation point)
            coherence_time = next((i for i, corr in enumerate(mean_temporal_corr) if corr < 0.5), 
                                 len(mean_temporal_corr))
            print(f"   Temporal coherence time: {coherence_time + 1} samples")
        else:
            coherence_time = 0
            print("   Insufficient temporal data for correlation analysis")
        
        # Analysis 2: CSI Amplitude Stability Over Time
        amplitude_stabilities = []
        
        for loc, samples in location_temporal.items():
            if len(samples) >= 10:
                samples_sorted = sorted(samples, key=lambda x: x['sample_id'])
                
                # Calculate stability metric: 1/CV (coefficient of variation)
                amp_series = [np.mean(s['amplitude']) for s in samples_sorted]
                
                if len(amp_series) > 1:
                    mean_amp = np.mean(amp_series)
                    std_amp = np.std(amp_series)
                    stability = mean_amp / (std_amp + 1e-10)  # Inverse CV
                    amplitude_stabilities.append(stability)
        
        if amplitude_stabilities:
            axes[0,1].hist(amplitude_stabilities, bins=15, alpha=0.7, color='green', edgecolor='black')
            axes[0,1].set_xlabel('Amplitude Stability (1/CV)')
            axes[0,1].set_ylabel('Number of Locations')
            axes[0,1].set_title('CSI Amplitude Stability Distribution\n(Temporal Consistency)')
            axes[0,1].grid(True)
            
            mean_stability = np.mean(amplitude_stabilities)
            print(f"   Mean amplitude stability: {mean_stability:.2f}")
        else:
            mean_stability = 0
        
        # Analysis 3: Stationarity Test (Augmented Dickey-Fuller equivalent)
        # Use variance ratio test for stationarity
        print("🔍 Testing channel stationarity...")
        
        stationarity_scores = []
        
        for loc, samples in location_temporal.items():
            if len(samples) >= 20:
                samples_sorted = sorted(samples, key=lambda x: x['sample_id'])
                
                # Use mean amplitude as test statistic
                amp_series = np.array([np.mean(s['amplitude']) for s in samples_sorted])
                
                # Split into two halves and compare variances
                mid_point = len(amp_series) // 2
                first_half = amp_series[:mid_point]
                second_half = amp_series[mid_point:]
                
                if len(first_half) > 1 and len(second_half) > 1:
                    var1 = np.var(first_half)
                    var2 = np.var(second_half)
                    
                    # Variance ratio (closer to 1 = more stationary)
                    ratio = min(var1, var2) / (max(var1, var2) + 1e-10)
                    stationarity_scores.append(ratio)
        
        if stationarity_scores:
            axes[1,0].hist(stationarity_scores, bins=15, alpha=0.7, color='orange', edgecolor='black')
            axes[1,0].set_xlabel('Stationarity Score (Variance Ratio)')
            axes[1,0].set_ylabel('Number of Locations')
            axes[1,0].set_title('Channel Stationarity Distribution\n(Temporal Consistency Test)')
            axes[1,0].grid(True)
            
            mean_stationarity = np.mean(stationarity_scores)
            print(f"   Mean stationarity score: {mean_stationarity:.3f}")
        else:
            mean_stationarity = 0
        
        # Analysis 4: Environmental Dynamics Summary
        # Create a comprehensive dynamics assessment
        
        dynamics_metrics = {
            'Coherence Time': coherence_time if coherence_time > 0 else 1,
            'Amplitude Stability': mean_stability if mean_stability > 0 else 1,
            'Stationarity': mean_stationarity if mean_stationarity > 0 else 0.5,
            'Temporal Diversity': 1 - (mean_stationarity if mean_stationarity > 0 else 0.5)
        }
        
        # Normalize for radar plot
        normalized_dynamics = {}
        for key, value in dynamics_metrics.items():
            if key == 'Coherence Time':
                normalized_dynamics[key] = min(1.0, value / 10)  # Normalize to 10 samples
            elif key == 'Amplitude Stability':
                normalized_dynamics[key] = min(1.0, value / 20)  # Normalize to stability of 20
            else:
                normalized_dynamics[key] = value
        
        categories = list(normalized_dynamics.keys())
        values = list(normalized_dynamics.values())
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values_plot = values + [values[0]]
        angles_plot = np.concatenate([angles, [angles[0]]])
        
        axes[1,1].plot(angles_plot, values_plot, 'ro-', linewidth=2)
        axes[1,1].fill(angles_plot, values_plot, alpha=0.25, color='red')
        axes[1,1].set_xticks(angles)
        axes[1,1].set_xticklabels([cat.replace(' ', '\n') for cat in categories], fontsize=10)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].set_title('Environmental Dynamics Assessment\n(Temporal Characteristics)')
        axes[1,1].grid(True)
        
        overall_dynamics = np.mean(values)
        print(f"   Overall environmental dynamics score: {overall_dynamics:.3f}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'channel_stationarity_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'coherence_time': coherence_time,
            'amplitude_stability': mean_stability,
            'stationarity_score': mean_stationarity,
            'dynamics_score': overall_dynamics
        }
    
    def generate_comprehensive_assessment(self, freq_results, spatial_results, temporal_results):
        """
        Generate comprehensive multipath and environmental complexity assessment.
        """
        print("\n🎯 COMPREHENSIVE MULTIPATH COMPLEXITY ASSESSMENT")
        print("=" * 80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Comprehensive CSI-Based Environmental Analysis\n(Multipath Complexity Assessment)', 
                     fontsize=16, fontweight='bold')
        
        # Assessment 1: Multipath Richness Score
        multipath_score = (
            (1 / freq_results['coherence_bw_50']) * 0.3 +  # Narrow coherence BW = rich multipath
            freq_results['spectral_efficiency'] * 0.3 +     # High spectral variation
            (1 - freq_results['mean_phase_coherence']) * 0.2 +  # Low phase coherence
            freq_results['rms_delay_spread'] / 100 * 0.2    # Normalized delay spread
        )
        
        # Assessment 2: Environmental Complexity Score  
        complexity_score = (
            spatial_results['spatial_decorrelation'] * 0.25 +
            spatial_results['temporal_diversity'] * 0.25 +
            (spatial_results['effective_rank'] / 10) * 0.25 +  # Normalized rank
            (1 - temporal_results['stationarity_score']) * 0.25
        )
        
        # Assessment 3: Localization Favorability
        localization_score = (
            min(1.0, freq_results['spectral_efficiency'] * 2) * 0.3 +
            min(1.0, spatial_results['spatial_decorrelation'] * 2) * 0.3 +
            min(1.0, temporal_results['amplitude_stability'] / 10) * 0.2 +
            min(1.0, (1 / freq_results['coherence_bw_50']) * 5) * 0.2
        )
        
        # Assessment 4: CNN Optimization Guidance
        cnn_guidance_score = (
            min(1.0, freq_results['spectral_efficiency'] * 3) * 0.4 +  # Feature richness
            min(1.0, complexity_score * 2) * 0.3 +  # Pattern complexity
            min(1.0, temporal_results['amplitude_stability'] / 5) * 0.3  # Training stability
        )
        
        # Create assessment visualization
        assessment_categories = ['Multipath\nRichness', 'Environmental\nComplexity', 
                               'Localization\nFavorability', 'CNN\nOptimization']
        assessment_scores = [multipath_score, complexity_score, localization_score, cnn_guidance_score]
        
        # Normalize scores to 0-1 range
        assessment_scores = [min(1.0, max(0.0, score)) for score in assessment_scores]
        
        bars = axes[0,0].bar(assessment_categories, assessment_scores, 
                            color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        axes[0,0].set_ylabel('Assessment Score (0-1)')
        axes[0,0].set_title('Overall Environmental Assessment\n(CSI-Based Analysis)')
        axes[0,0].set_ylim(0, 1)
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, assessment_scores):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                          f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Technical metrics summary
        technical_summary = f"""
FREQUENCY DOMAIN ANALYSIS:
• Coherence BW (50%): {freq_results['coherence_bw_50']} subcarriers
• RMS Delay Spread: {freq_results['rms_delay_spread']:.2f} ns
• Spectral Efficiency: {freq_results['spectral_efficiency']:.3f}
• Phase Coherence: {freq_results['mean_phase_coherence']:.3f}
• K-factor: {freq_results['mean_k_factor_db']:.1f} dB

SPATIAL ANALYSIS:
• Coherence Distance: {spatial_results['coherence_distance']:.2f} m
• Effective Rank: {spatial_results['effective_rank']}
• Spatial Decorrelation: {spatial_results['spatial_decorrelation']:.3f}
• Complexity Score: {spatial_results['complexity_score']:.3f}

TEMPORAL ANALYSIS:
• Coherence Time: {temporal_results['coherence_time']} samples
• Amplitude Stability: {temporal_results['amplitude_stability']:.2f}
• Stationarity Score: {temporal_results['stationarity_score']:.3f}
• Dynamics Score: {temporal_results['dynamics_score']:.3f}
        """
        
        axes[0,1].text(0.05, 0.95, technical_summary, transform=axes[0,1].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[0,1].set_xlim(0, 1)
        axes[0,1].set_ylim(0, 1)
        axes[0,1].axis('off')
        axes[0,1].set_title('Technical Metrics Summary\n(Quantitative Analysis)')
        
        # CNN Architecture Recommendations
        if freq_results['coherence_bw_50'] <= 3:
            kernel_size = "3-5 subcarriers (narrow coherence)"
            depth_rec = "Deep (6+ layers) for rich features"
        elif freq_results['coherence_bw_50'] <= 7:
            kernel_size = "5-9 subcarriers (moderate coherence)"
            depth_rec = "Moderate (4-6 layers)"
        else:
            kernel_size = "7-11 subcarriers (wide coherence)"
            depth_rec = "Shallow (2-4 layers)"
        
        if complexity_score > 0.7:
            training_rec = "Complex: High regularization, larger dataset"
        elif complexity_score > 0.4:
            training_rec = "Moderate: Standard training approach"
        else:
            training_rec = "Simple: May need data augmentation"
        
        cnn_recommendations = f"""
CNN ARCHITECTURE RECOMMENDATIONS:

KERNEL DESIGN:
• Optimal kernel size: {kernel_size}
• Rationale: Based on {freq_results['coherence_bw_50']}-subcarrier coherence

NETWORK DEPTH:
• Recommended depth: {depth_rec}
• Rationale: Complexity score = {complexity_score:.3f}

TRAINING STRATEGY:
• Approach: {training_rec}
• Expected accuracy: {85 + localization_score * 15:.0f}%

FEATURE ENGINEERING:
• Amplitude dominance: {freq_results['spectral_efficiency']:.3f}
• Phase utility: {'High' if freq_results['mean_phase_coherence'] < 0.5 else 'Moderate'}
• Temporal stability: {'Good' if temporal_results['stationarity_score'] > 0.6 else 'Variable'}

OPTIMIZATION GUIDANCE:
• Batch size: {32 if complexity_score > 0.5 else 64}
• Learning rate: {'Conservative (1e-4)' if complexity_score > 0.7 else 'Standard (1e-3)'}
• Regularization: {'High (0.3-0.5)' if complexity_score > 0.6 else 'Moderate (0.1-0.3)'}
        """
        
        axes[1,0].text(0.05, 0.95, cnn_recommendations, transform=axes[1,0].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1,0].set_xlim(0, 1)
        axes[1,0].set_ylim(0, 1)
        axes[1,0].axis('off')
        axes[1,0].set_title('CNN Development Guidance\n(Architecture & Training)')
        
        # Environmental interpretation
        if multipath_score > 0.7:
            env_type = "RICH MULTIPATH"
            env_desc = "Complex indoor environment with multiple reflectors"
        elif multipath_score > 0.4:
            env_type = "MODERATE MULTIPATH"
            env_desc = "Typical indoor environment with some scattering"
        else:
            env_type = "LIMITED MULTIPATH" 
            env_desc = "Simple environment, possibly line-of-sight dominant"
        
        physical_interpretation = f"""
PHYSICAL ENVIRONMENT INTERPRETATION:

ENVIRONMENT TYPE: {env_type}
{env_desc}

MULTIPATH CHARACTERISTICS:
• Delay spread: {freq_results['rms_delay_spread']:.1f} ns
• Interpretation: {'Rich scattering' if freq_results['rms_delay_spread'] > 50 else 'Moderate scattering' if freq_results['rms_delay_spread'] > 20 else 'Limited scattering'}
• Dominant propagation: {'NLOS' if freq_results['mean_k_factor_db'] < 5 else 'Mixed LOS/NLOS' if freq_results['mean_k_factor_db'] < 15 else 'LOS-dominant'}

SPATIAL CHARACTERISTICS:
• Scattering richness: {'High' if spatial_results['effective_rank'] > 8 else 'Moderate' if spatial_results['effective_rank'] > 4 else 'Low'}
• Coherence distance: {spatial_results['coherence_distance']:.1f} m
• Environment size perception: {'Large' if spatial_results['coherence_distance'] > 2 else 'Medium' if spatial_results['coherence_distance'] > 1 else 'Small/Cluttered'}

LOCALIZATION POTENTIAL:
• CNN feasibility: {'EXCELLENT' if localization_score > 0.8 else 'GOOD' if localization_score > 0.6 else 'MODERATE' if localization_score > 0.4 else 'CHALLENGING'}
• Expected accuracy: {85 + localization_score * 15:.0f}%
• Key advantages: {'Freq. diversity' if freq_results['spectral_efficiency'] > 0.2 else ''} {'Spatial diversity' if spatial_results['spatial_decorrelation'] > 0.5 else ''} {'Temporal stability' if temporal_results['stationarity_score'] > 0.6 else ''}
        """
        
        axes[1,1].text(0.05, 0.95, physical_interpretation, transform=axes[1,1].transAxes,
                      fontsize=9, verticalalignment='top', fontfamily='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        axes[1,1].set_title('Physical Environment Interpretation\n(Propagation Analysis)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comprehensive_assessment.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Final assessment summary
        overall_score = np.mean(assessment_scores)
        
        print(f"\n🏆 FINAL ASSESSMENT SUMMARY:")
        print(f"   Overall environment score: {overall_score:.3f}/1.0")
        print(f"   Environment classification: {env_type}")
        print(f"   CNN localization potential: {'EXCELLENT' if localization_score > 0.8 else 'GOOD' if localization_score > 0.6 else 'MODERATE'}")
        print(f"   Expected accuracy: {85 + localization_score * 15:.0f}%")
        
        return {
            'overall_score': overall_score,
            'multipath_score': multipath_score,
            'complexity_score': complexity_score,
            'localization_score': localization_score,
            'cnn_guidance_score': cnn_guidance_score,
            'environment_type': env_type
        }

def main():
    """
    Main function to perform comprehensive CSI multipath complexity analysis.
    """
    print("🔬 CSI-BASED MULTIPATH AND ENVIRONMENTAL COMPLEXITY ANALYSIS")
    print("=" * 80)
    print("OBJECTIVE: Understand propagation environment through CSI characteristics")
    print("METHODS: Frequency selectivity, spatial correlation, temporal stationarity")
    print("OUTPUT: Multipath assessment and CNN optimization guidance")
    
    analyzer = CSIMultipathAnalyzer("CSI Dataset 750 Samples")
    
    # Load CSI data
    sample_count = analyzer.load_csi_data()
    
    # Perform frequency domain analysis
    freq_results = analyzer.analyze_frequency_selectivity()
    
    # Perform spatial correlation analysis  
    spatial_results = analyzer.analyze_spatial_correlation()
    
    # Perform temporal stationarity analysis
    temporal_results = analyzer.analyze_channel_stationarity()
    
    # Generate comprehensive assessment
    final_assessment = analyzer.generate_comprehensive_assessment(freq_results, spatial_results, temporal_results)
    
    print(f"\n✅ COMPREHENSIVE CSI ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: CSI_Multipath_Analysis/")
    print(f"🎯 Environment assessed as: {final_assessment['environment_type']}")
    print(f"🧠 CNN potential: {final_assessment['localization_score']:.3f}/1.0")

if __name__ == "__main__":
    main()
