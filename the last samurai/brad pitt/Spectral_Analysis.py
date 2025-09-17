#!/usr/bin/env python3
"""
Spectral Analysis of CSI Data for Indoor Localization
=====================================================

This file performs comprehensive spectral analysis of CSI amplitude and phase data
to understand frequency domain characteristics that could improve CNN-based localization.

Key Analysis Areas:
1. Power Delay Profile (PDP) computation from CSI frequency response
2. RMS Delay Spread (τ_rms) calculation
3. Coherence Bandwidth (Bc) estimation
4. Rician K-factor estimation (LOS vs NLOS characterization)
5. Frequency correlation analysis
6. Spectral features for different spatial locations

Physical Parameters:
- 40 MHz OFDM system with 128 FFT bins
- Subcarrier spacing: Δf = 312.5 kHz
- 52 reported CSI bins (centered around DC)
- Delay resolution: ~25 ns (unpadded) or ~3.125 ns (zero-padded to 1024)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
from scipy.interpolate import interp1d
from scipy import signal
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import coordinates
from coordinates_config import get_training_points, get_validation_points, get_testing_points

# ========================================================================
# PHYSICAL PARAMETERS
# ========================================================================
N_REPORTED = 52            # CSI bins reported per snapshot
PHY_FFT = 128             # True OFDM FFT size for 40 MHz
DELTA_F = 312.5e3         # Hz (40MHz/128)
NFFT_PAD = 1024           # Zero-padded FFT for finer delay resolution
SPEED_OF_LIGHT = 3e8      # m/s

class CSI_SpectralAnalyzer:
    """
    Comprehensive spectral analysis of CSI data for indoor localization
    """
    
    def __init__(self, output_dir="spectral_analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load coordinate points
        self.training_points = get_training_points()
        self.validation_points = get_validation_points()
        self.testing_points = get_testing_points()
        self.all_points = self.training_points + self.validation_points + self.testing_points
        
        print("🔬 CSI Spectral Analyzer Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Total analysis points: {len(self.all_points)}")
        print(f"⚡ OFDM Parameters:")
        print(f"   • FFT Size: {PHY_FFT}")
        print(f"   • Subcarrier Spacing: {DELTA_F/1e3:.1f} kHz")
        print(f"   • Total Bandwidth: {DELTA_F * PHY_FFT / 1e6:.1f} MHz")
        print(f"   • Delay Resolution: {1/(NFFT_PAD * DELTA_F) * 1e9:.2f} ns")
    
    # ========================================================================
    # CORE SPECTRAL PROCESSING FUNCTIONS
    # ========================================================================
    
    def complex_from_amp_phase(self, amps, phases):
        """Convert amplitude and phase arrays to complex CSI"""
        a = np.asarray(amps, dtype=float)
        p = np.asarray(phases, dtype=float)
        return a * np.exp(1j * p)
    
    def embed_into_grid(self, H52, phy_fft=PHY_FFT, nfft_pad=NFFT_PAD):
        """
        Embed 52 CSI samples into proper OFDM frequency grid
        Places H52 (ordered negative->positive freq) into zero-initialized array
        with DC at center, ready for IFFT processing
        """
        if len(H52) != N_REPORTED:
            raise ValueError(f"H52 must be length {N_REPORTED}")
        
        # 1) Create physical grid with DC at center (shifted form)
        phy_shifted = np.zeros(phy_fft, dtype=complex)
        center = phy_fft // 2
        half = N_REPORTED // 2
        start = center - half
        phy_shifted[start:start + N_REPORTED] = H52
        
        # 2) Embed into larger padded grid (also shifted)
        if nfft_pad < phy_fft:
            raise ValueError("nfft_pad must be >= phy_fft")
        
        pad_shifted = np.zeros(nfft_pad, dtype=complex)
        pad_center = nfft_pad // 2
        pad_start = pad_center - (phy_fft // 2)
        pad_shifted[pad_start:pad_start + phy_fft] = phy_shifted
        
        return pad_shifted
    
    def compute_pdp(self, H_full_shifted, delta_f=DELTA_F, nfft_pad=NFFT_PAD):
        """
        Compute Power Delay Profile from frequency response
        Returns normalized PDP, delay axis, and complex impulse response
        """
        # Unshift and perform IFFT
        H_for_ifft = np.fft.ifftshift(H_full_shifted)
        h_time = np.fft.ifft(H_for_ifft, n=nfft_pad)
        
        # Power delay profile
        P = np.abs(h_time)**2
        P_sum = P.sum()
        
        if P_sum == 0:
            return P, None, h_time
        
        P_norm = P / P_sum
        
        # Delay axis
        delta_tau = 1.0 / (nfft_pad * delta_f)
        tau = np.arange(nfft_pad) * delta_tau
        
        return P_norm, tau, h_time
    
    def compute_rms_delay(self, P_norm, tau):
        """Compute RMS delay spread from normalized PDP"""
        tau_mean = np.sum(P_norm * tau)
        tau2 = np.sum(P_norm * tau**2)
        tau_rms = np.sqrt(max(0.0, tau2 - tau_mean**2))
        return tau_rms, tau_mean
    
    def compute_coherence_bandwidth(self, H52, delta_f=DELTA_F, threshold=0.5):
        """
        Compute coherence bandwidth from frequency autocorrelation
        Returns Bc in Hz and in subcarriers
        """
        H = np.asarray(H52)
        N = len(H)
        energy = np.sum(np.abs(H)**2)
        
        if energy == 0:
            return 0.0, 0.0, None, None
        
        max_lag = N - 1
        lags = np.arange(0, max_lag + 1)
        R = np.zeros_like(lags, dtype=float)
        
        for d in lags:
            valid = np.arange(0, N - d)
            if valid.size == 0:
                R[d] = 0.0
            else:
                R[d] = np.abs(np.sum(H[valid] * np.conj(H[valid + d]))) / energy
        
        # Normalize to R[0] = 1
        R_norm = R / (R[0] if R[0] != 0 else 1.0)
        freq_offsets = lags * delta_f  # Hz
        
        # Interpolate for finer estimate
        if len(freq_offsets) > 1:
            interp = interp1d(freq_offsets, R_norm, kind='linear', 
                            bounds_error=False, fill_value=(R_norm[0], R_norm[-1]))
            fgrid = np.linspace(freq_offsets[0], freq_offsets[-1], 2000)
            rg = interp(fgrid)
            
            # Find first frequency where correlation <= threshold
            idx = np.where(rg <= threshold)[0]
            if idx.size == 0:
                Bc = fgrid[-1]
            else:
                Bc = fgrid[idx[0]]
        else:
            Bc = freq_offsets[-1] if len(freq_offsets) > 0 else 0.0
        
        Bc_sub = Bc / delta_f
        return Bc, Bc_sub, freq_offsets, R_norm
    
    def compute_rician_K_pdp(self, P_norm):
        """Compute Rician K-factor from PDP (strongest tap vs rest)"""
        los = np.max(P_norm)
        scattered = np.sum(P_norm) - los
        
        if scattered <= 0:
            return np.inf
        
        K_lin = los / scattered
        K_db = 10.0 * np.log10(K_lin)
        return K_db
    
    def compute_rician_K_moments(self, H_samples):
        """Compute Rician K-factor using moment-based estimator"""
        H = np.asarray(H_samples).ravel()
        mu = np.mean(H)
        E2 = np.mean(np.abs(H)**2)
        denom = E2 - np.abs(mu)**2
        
        if denom <= 0:
            return np.inf
        
        K_lin = (np.abs(mu)**2) / denom
        return 10.0 * np.log10(K_lin)
    
    # ========================================================================
    # DATA LOADING AND PROCESSING
    # ========================================================================
    
    def load_csi_data(self, point, dataset_size=250, max_samples=None):
        """Load CSI data for a specific point"""
        x, y = point
        
        # Determine folder based on point type
        if point in self.testing_points:
            folder = "Testing Points Dataset 750 Samples"
        else:
            folder = f"CSI Dataset {dataset_size} Samples"
        
        file_path = Path(folder) / f"{x},{y}.csv"
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            return None, None, None
        
        amplitudes, phases, rssi_values = [], [], []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if max_samples and i >= max_samples:
                        break
                    
                    try:
                        amps = json.loads(row['amplitude'])
                        phases_data = json.loads(row['phase'])
                        rssi = float(row['rssi'])
                        
                        if len(amps) == 52 and len(phases_data) == 52:
                            amplitudes.append(amps)
                            phases.append(phases_data)
                            rssi_values.append(rssi)
                    except:
                        continue
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None, None, None
        
        print(f"✅ Loaded {len(amplitudes)} samples from point ({x}, {y})")
        return np.array(amplitudes), np.array(phases), np.array(rssi_values)
    
    def analyze_single_snapshot(self, amps, phases, phy_fft=PHY_FFT, 
                               nfft_pad=NFFT_PAD, delta_f=DELTA_F):
        """Analyze a single CSI snapshot"""
        # Convert to complex
        H52 = self.complex_from_amp_phase(amps, phases)
        
        # Embed into proper grid
        H_full_shifted = self.embed_into_grid(H52, phy_fft=phy_fft, nfft_pad=nfft_pad)
        
        # Compute PDP
        P_norm, tau, h_time = self.compute_pdp(H_full_shifted, delta_f=delta_f, nfft_pad=nfft_pad)
        
        # Compute RMS delay
        if tau is not None:
            tau_rms, tau_mean = self.compute_rms_delay(P_norm, tau)
        else:
            tau_rms, tau_mean = None, None
        
        # Compute coherence bandwidth
        Bc_hz, Bc_sub, freq_offsets, R_norm = self.compute_coherence_bandwidth(H52, delta_f=delta_f)
        
        # Compute Rician K-factors
        K_pdp = self.compute_rician_K_pdp(P_norm)
        K_mom = self.compute_rician_K_moments(H52)
        
        return {
            "tau_rms_s": tau_rms,
            "tau_mean_s": tau_mean,
            "Bc_hz": Bc_hz,
            "Bc_subcarriers": Bc_sub,
            "K_pdp_dB": K_pdp,
            "K_moments_dB": K_mom,
            "PDP": P_norm,
            "tau_axis_s": tau,
            "freq_lags_Hz": freq_offsets,
            "freq_corr": R_norm,
            "h_time": h_time,
            "H52": H52
        }
    
    def analyze_point_snapshots(self, point, dataset_size=250, max_samples=100):
        """Analyze multiple snapshots for a single point"""
        print(f"\n🔍 Analyzing point ({point[0]}, {point[1]})...")
        
        # Load data
        amplitudes, phases, rssi_values = self.load_csi_data(point, dataset_size, max_samples)
        
        if amplitudes is None:
            return None
        
        # Analyze each snapshot
        results = []
        for i, (amps, phases) in enumerate(zip(amplitudes, phases)):
            try:
                result = self.analyze_single_snapshot(amps, phases)
                result['snapshot_idx'] = i
                result['rssi'] = rssi_values[i]
                results.append(result)
            except Exception as e:
                print(f"⚠️  Error analyzing snapshot {i}: {e}")
                continue
        
        if not results:
            return None
        
        # Aggregate statistics
        df = pd.DataFrame([{
            "tau_rms_s": r["tau_rms_s"],
            "tau_mean_s": r["tau_mean_s"],
            "Bc_hz": r["Bc_hz"],
            "Bc_sub": r["Bc_subcarriers"],
            "K_pdp_dB": r["K_pdp_dB"],
            "K_mom_dB": r["K_moments_dB"],
            "rssi": r["rssi"]
        } for r in results if r["tau_rms_s"] is not None])
        
        if df.empty:
            return None
        
        stats = df.agg(["mean", "std", "median", "min", "max"])
        
        return {
            'point': point,
            'results': results,
            'dataframe': df,
            'statistics': stats,
            'n_samples': len(results)
        }
    
    # ========================================================================
    # ANALYSIS AND VISUALIZATION
    # ========================================================================
    
    def analyze_all_points(self, dataset_size=250, max_samples_per_point=50):
        """Analyze spectral characteristics for all points"""
        print(f"\n🚀 Starting comprehensive spectral analysis...")
        print(f"📊 Dataset size: {dataset_size} samples")
        print(f"🔢 Max samples per point: {max_samples_per_point}")
        
        all_results = {}
        summary_data = []
        
        # Analyze each point
        for point in self.all_points:
            point_result = self.analyze_point_snapshots(point, dataset_size, max_samples_per_point)
            
            if point_result is not None:
                all_results[f"{point[0]}_{point[1]}"] = point_result
                
                # Add to summary
                stats = point_result['statistics']
                summary_data.append({
                    'x': point[0],
                    'y': point[1],
                    'point_type': self.get_point_type(point),
                    'n_samples': point_result['n_samples'],
                    'tau_rms_mean_ns': stats.loc['mean', 'tau_rms_s'] * 1e9,
                    'tau_rms_std_ns': stats.loc['std', 'tau_rms_s'] * 1e9,
                    'Bc_mean_MHz': stats.loc['mean', 'Bc_hz'] / 1e6,
                    'Bc_std_MHz': stats.loc['std', 'Bc_hz'] / 1e6,
                    'K_pdp_mean_dB': stats.loc['mean', 'K_pdp_dB'],
                    'K_pdp_std_dB': stats.loc['std', 'K_pdp_dB'],
                    'rssi_mean_dBm': stats.loc['mean', 'rssi'],
                    'rssi_std_dBm': stats.loc['std', 'rssi']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        print(f"✅ Analysis complete! Processed {len(all_results)} points")
        
        return all_results, summary_df
    
    def get_point_type(self, point):
        """Determine if point is training, validation, or testing"""
        if point in self.training_points:
            return 'training'
        elif point in self.validation_points:
            return 'validation'
        elif point in self.testing_points:
            return 'testing'
        else:
            return 'unknown'
    
    def plot_spatial_characteristics(self, summary_df):
        """Plot spatial distribution of spectral characteristics"""
        print("📊 Creating spatial characteristic plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Define point type colors
        colors = {'training': 'blue', 'validation': 'orange', 'testing': 'green'}
        
        # 1. RMS Delay Spread
        ax = axes[0, 0]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                scatter = ax.scatter(summary_df[mask]['x'], summary_df[mask]['y'], 
                                   c=summary_df[mask]['tau_rms_mean_ns'], 
                                   s=100, alpha=0.8, cmap='viridis', 
                                   edgecolors=colors[pt_type], linewidth=2,
                                   label=f'{pt_type.title()} ({mask.sum()})')
        
        ax.set_title('RMS Delay Spread (ns)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='τ_rms (ns)')
        
        # 2. Coherence Bandwidth
        ax = axes[0, 1]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                scatter = ax.scatter(summary_df[mask]['x'], summary_df[mask]['y'], 
                                   c=summary_df[mask]['Bc_mean_MHz'], 
                                   s=100, alpha=0.8, cmap='plasma',
                                   edgecolors=colors[pt_type], linewidth=2)
        
        ax.set_title('Coherence Bandwidth (MHz)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Bc (MHz)')
        
        # 3. Rician K-factor
        ax = axes[0, 2]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                scatter = ax.scatter(summary_df[mask]['x'], summary_df[mask]['y'], 
                                   c=summary_df[mask]['K_pdp_mean_dB'], 
                                   s=100, alpha=0.8, cmap='coolwarm',
                                   edgecolors=colors[pt_type], linewidth=2)
        
        ax.set_title('Rician K-factor (dB)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='K (dB)')
        
        # 4. RSSI Distribution
        ax = axes[1, 0]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                scatter = ax.scatter(summary_df[mask]['x'], summary_df[mask]['y'], 
                                   c=summary_df[mask]['rssi_mean_dBm'], 
                                   s=100, alpha=0.8, cmap='RdYlBu_r',
                                   edgecolors=colors[pt_type], linewidth=2)
        
        ax.set_title('RSSI (dBm)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='RSSI (dBm)')
        
        # 5. Tau RMS vs Coherence Bandwidth
        ax = axes[1, 1]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                ax.scatter(summary_df[mask]['tau_rms_mean_ns'], 
                          summary_df[mask]['Bc_mean_MHz'],
                          c=colors[pt_type], s=60, alpha=0.7, label=pt_type.title())
        
        ax.set_xlabel('RMS Delay Spread (ns)')
        ax.set_ylabel('Coherence Bandwidth (MHz)')
        ax.set_title('Tau RMS vs Coherence Bandwidth', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 6. K-factor vs RSSI
        ax = axes[1, 2]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                ax.scatter(summary_df[mask]['rssi_mean_dBm'], 
                          summary_df[mask]['K_pdp_mean_dB'],
                          c=colors[pt_type], s=60, alpha=0.7, label=pt_type.title())
        
        ax.set_xlabel('RSSI (dBm)')
        ax.set_ylabel('Rician K-factor (dB)')
        ax.set_title('K-factor vs RSSI', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "spatial_spectral_characteristics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved spatial characteristics plot: {output_path}")
        
        plt.show()
    
    def plot_example_pdp_analysis(self, all_results):
        """Plot example PDP analysis for different point types"""
        print("📊 Creating example PDP analysis plots...")
        
        # Select representative points
        example_points = {
            'training': (1, 1),    # Training point
            'validation': (3, 3),  # Validation point  
            'testing': (2.5, 2.5)  # Testing point
        }
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        for row, (pt_type, point) in enumerate(example_points.items()):
            point_key = f"{point[0]}_{point[1]}"
            
            if point_key not in all_results:
                print(f"⚠️  No data for example point {point}")
                continue
            
            point_data = all_results[point_key]
            
            # Get first valid result
            valid_results = [r for r in point_data['results'] if r['tau_axis_s'] is not None]
            if not valid_results:
                continue
            
            result = valid_results[0]  # Use first snapshot
            
            # Plot 1: PDP
            ax = axes[row, 0]
            tau_ns = result['tau_axis_s'] * 1e9  # Convert to ns
            pdp_db = 10 * np.log10(result['PDP'] + 1e-12)  # Convert to dB
            ax.plot(tau_ns, pdp_db, 'b-', linewidth=1.5)
            ax.set_xlabel('Delay (ns)')
            ax.set_ylabel('Power (dB)')
            ax.set_title(f'{pt_type.title()} Point {point}\nPower Delay Profile')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 200)  # Show first 200 ns
            
            # Plot 2: Frequency Correlation
            ax = axes[row, 1]
            if result['freq_lags_Hz'] is not None and result['freq_corr'] is not None:
                freq_MHz = result['freq_lags_Hz'] / 1e6
                ax.plot(freq_MHz, result['freq_corr'], 'r-', linewidth=1.5)
                ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.7, label='Bc threshold')
                ax.set_xlabel('Frequency Offset (MHz)')
                ax.set_ylabel('Normalized Correlation')
                ax.set_title(f'Frequency Correlation\nBc = {result["Bc_hz"]/1e6:.2f} MHz')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Plot 3: CSI Magnitude and Phase
            ax = axes[row, 2]
            subcarriers = np.arange(52)
            H_mag = np.abs(result['H52'])
            H_phase = np.angle(result['H52'])
            
            ax2 = ax.twinx()
            line1 = ax.plot(subcarriers, H_mag, 'b-', linewidth=1.5, label='Magnitude')
            line2 = ax2.plot(subcarriers, H_phase, 'r-', linewidth=1.5, label='Phase')
            
            ax.set_xlabel('Subcarrier Index')
            ax.set_ylabel('CSI Magnitude', color='b')
            ax2.set_ylabel('CSI Phase (rad)', color='r')
            ax.set_title(f'CSI Frequency Response\nK = {result["K_pdp_dB"]:.1f} dB')
            ax.grid(True, alpha=0.3)
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "example_pdp_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved example PDP analysis: {output_path}")
        
        plt.show()
    
    def plot_statistical_distributions(self, summary_df):
        """Plot statistical distributions of spectral parameters"""
        print("📊 Creating statistical distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Define colors for point types
        colors = {'training': 'blue', 'validation': 'orange', 'testing': 'green'}
        
        # 1. RMS Delay Spread Distribution
        ax = axes[0, 0]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                data = summary_df[mask]['tau_rms_mean_ns']
                ax.hist(data, bins=15, alpha=0.6, color=colors[pt_type], 
                       label=f'{pt_type.title()} (n={mask.sum()})', density=True)
        
        ax.set_xlabel('RMS Delay Spread (ns)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of RMS Delay Spread')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Coherence Bandwidth Distribution
        ax = axes[0, 1]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                data = summary_df[mask]['Bc_mean_MHz']
                ax.hist(data, bins=15, alpha=0.6, color=colors[pt_type], 
                       label=f'{pt_type.title()} (n={mask.sum()})', density=True)
        
        ax.set_xlabel('Coherence Bandwidth (MHz)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Coherence Bandwidth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. K-factor Distribution
        ax = axes[1, 0]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                data = summary_df[mask]['K_pdp_mean_dB']
                # Filter out infinite values
                data_finite = data[np.isfinite(data)]
                if len(data_finite) > 0:
                    ax.hist(data_finite, bins=15, alpha=0.6, color=colors[pt_type], 
                           label=f'{pt_type.title()} (n={len(data_finite)})', density=True)
        
        ax.set_xlabel('Rician K-factor (dB)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Rician K-factor')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. RSSI Distribution
        ax = axes[1, 1]
        for pt_type in ['training', 'validation', 'testing']:
            mask = summary_df['point_type'] == pt_type
            if mask.any():
                data = summary_df[mask]['rssi_mean_dBm']
                ax.hist(data, bins=15, alpha=0.6, color=colors[pt_type], 
                       label=f'{pt_type.title()} (n={mask.sum()})', density=True)
        
        ax.set_xlabel('RSSI (dBm)')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of RSSI')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "statistical_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved statistical distributions: {output_path}")
        
        plt.show()
    
    def save_results(self, summary_df, all_results):
        """Save analysis results to files"""
        print("💾 Saving analysis results...")
        
        # Save summary DataFrame
        summary_path = self.output_dir / "spectral_analysis_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Saved summary data: {summary_path}")
        
        # Save detailed statistics
        stats_path = self.output_dir / "detailed_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write("SPECTRAL ANALYSIS DETAILED STATISTICS\n")
            f.write("="*50 + "\n\n")
            
            for point_type in ['training', 'validation', 'testing']:
                mask = summary_df['point_type'] == point_type
                if mask.any():
                    f.write(f"{point_type.upper()} POINTS (n={mask.sum()}):\n")
                    f.write("-" * 30 + "\n")
                    
                    subset = summary_df[mask]
                    
                    f.write(f"RMS Delay Spread (ns):\n")
                    f.write(f"  Mean ± Std: {subset['tau_rms_mean_ns'].mean():.2f} ± {subset['tau_rms_mean_ns'].std():.2f}\n")
                    f.write(f"  Range: {subset['tau_rms_mean_ns'].min():.2f} - {subset['tau_rms_mean_ns'].max():.2f}\n\n")
                    
                    f.write(f"Coherence Bandwidth (MHz):\n")
                    f.write(f"  Mean ± Std: {subset['Bc_mean_MHz'].mean():.2f} ± {subset['Bc_mean_MHz'].std():.2f}\n")
                    f.write(f"  Range: {subset['Bc_mean_MHz'].min():.2f} - {subset['Bc_mean_MHz'].max():.2f}\n\n")
                    
                    k_finite = subset['K_pdp_mean_dB'][np.isfinite(subset['K_pdp_mean_dB'])]
                    if len(k_finite) > 0:
                        f.write(f"Rician K-factor (dB):\n")
                        f.write(f"  Mean ± Std: {k_finite.mean():.2f} ± {k_finite.std():.2f}\n")
                        f.write(f"  Range: {k_finite.min():.2f} - {k_finite.max():.2f}\n\n")
                    
                    f.write(f"RSSI (dBm):\n")
                    f.write(f"  Mean ± Std: {subset['rssi_mean_dBm'].mean():.2f} ± {subset['rssi_mean_dBm'].std():.2f}\n")
                    f.write(f"  Range: {subset['rssi_mean_dBm'].min():.2f} - {subset['rssi_mean_dBm'].max():.2f}\n\n")
                    
        print(f"✅ Saved detailed statistics: {stats_path}")
    
    def generate_insights_report(self, summary_df):
        """Generate insights report for localization implications"""
        print("📝 Generating insights report...")
        
        report_path = self.output_dir / "localization_insights_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Spectral Analysis Insights for CNN-Based Indoor Localization\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write("This report analyzes spectral characteristics of CSI data to identify potential ")
            f.write("improvements for CNN-based indoor localization systems.\n\n")
            
            f.write("## Key Findings\n\n")
            
            # Spatial variation analysis
            f.write("### 1. Spatial Variation of Channel Characteristics\n\n")
            
            tau_range = summary_df['tau_rms_mean_ns'].max() - summary_df['tau_rms_mean_ns'].min()
            bc_range = summary_df['Bc_mean_MHz'].max() - summary_df['Bc_mean_MHz'].min()
            
            f.write(f"- **RMS Delay Spread**: Varies from {summary_df['tau_rms_mean_ns'].min():.1f} to ")
            f.write(f"{summary_df['tau_rms_mean_ns'].max():.1f} ns across locations ({tau_range:.1f} ns range)\n")
            f.write(f"- **Coherence Bandwidth**: Varies from {summary_df['Bc_mean_MHz'].min():.1f} to ")
            f.write(f"{summary_df['Bc_mean_MHz'].max():.1f} MHz across locations ({bc_range:.1f} MHz range)\n\n")
            
            # Point type analysis
            f.write("### 2. Point Type Characteristics\n\n")
            for pt_type in ['training', 'validation', 'testing']:
                mask = summary_df['point_type'] == pt_type
                if mask.any():
                    subset = summary_df[mask]
                    f.write(f"**{pt_type.title()} Points (n={mask.sum()})**:\n")
                    f.write(f"- Mean RMS Delay: {subset['tau_rms_mean_ns'].mean():.1f} ± {subset['tau_rms_mean_ns'].std():.1f} ns\n")
                    f.write(f"- Mean Coherence BW: {subset['Bc_mean_MHz'].mean():.1f} ± {subset['Bc_mean_MHz'].std():.1f} MHz\n")
                    f.write(f"- Mean RSSI: {subset['rssi_mean_dBm'].mean():.1f} ± {subset['rssi_mean_dBm'].std():.1f} dBm\n\n")
            
            f.write("## Implications for CNN Localization\n\n")
            
            f.write("### 1. Feature Engineering Opportunities\n\n")
            f.write("- **Spectral Features**: RMS delay spread and coherence bandwidth show spatial variation\n")
            f.write("- **Channel State**: Rician K-factor indicates LOS/NLOS conditions\n")
            f.write("- **Multi-scale Processing**: Different coherence bandwidths suggest need for adaptive filtering\n\n")
            
            f.write("### 2. Architecture Recommendations\n\n")
            f.write("- **Attention Mechanisms**: Focus on frequency bins with highest coherence\n")
            f.write("- **Multi-Resolution CNNs**: Process different frequency scales\n")
            f.write("- **Physics-Informed Loss**: Incorporate delay spread constraints\n\n")
            
            f.write("### 3. Data Augmentation Strategies\n\n")
            f.write("- **Coherent Augmentation**: Preserve channel coherence properties\n")
            f.write("- **Delay-Based Augmentation**: Simulate realistic multipath scenarios\n")
            f.write("- **Frequency-Selective Fading**: Model based on measured coherence bandwidth\n\n")
            
            f.write("### 4. Missing Variables & Potential Improvements\n\n")
            f.write("- **Doppler Spread**: Consider mobility effects\n")
            f.write("- **Antenna Pattern**: Account for directional effects\n")
            f.write("- **Environmental Context**: Temperature, humidity effects\n")
            f.write("- **Multi-Antenna Processing**: Spatial diversity\n\n")
            
        print(f"✅ Generated insights report: {report_path}")
    
    def run_comprehensive_analysis(self, dataset_size=250, max_samples_per_point=50):
        """Run complete spectral analysis pipeline"""
        print("🚀 Starting Comprehensive Spectral Analysis Pipeline")
        print("=" * 60)
        
        # 1. Analyze all points
        all_results, summary_df = self.analyze_all_points(dataset_size, max_samples_per_point)
        
        if summary_df.empty:
            print("❌ No valid results obtained. Check data paths and formats.")
            return
        
        # 2. Create visualizations
        self.plot_spatial_characteristics(summary_df)
        self.plot_example_pdp_analysis(all_results)
        self.plot_statistical_distributions(summary_df)
        
        # 3. Save results
        self.save_results(summary_df, all_results)
        
        # 4. Generate insights report
        self.generate_insights_report(summary_df)
        
        print("\n🎉 Comprehensive Spectral Analysis Complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        print("\n📊 Summary Statistics:")
        print(f"   • Points analyzed: {len(all_results)}")
        print(f"   • Total snapshots: {summary_df['n_samples'].sum()}")
        print(f"   • Mean RMS delay: {summary_df['tau_rms_mean_ns'].mean():.1f} ± {summary_df['tau_rms_mean_ns'].std():.1f} ns")
        print(f"   • Mean coherence BW: {summary_df['Bc_mean_MHz'].mean():.1f} ± {summary_df['Bc_mean_MHz'].std():.1f} MHz")
        
        return all_results, summary_df

def main():
    """Main execution function"""
    print("🔬 CSI Spectral Analysis for Indoor Localization")
    print("=" * 50)
    
    # Questions for user (in comments - you can uncomment to make interactive)
    """
    Key Questions to Consider:
    1. Should we analyze all available samples or limit for computational efficiency?
    2. Which dataset size should we focus on (250, 500, 750)?
    3. Are there specific spatial patterns we should investigate?
    4. Should we include temporal variation analysis?
    """
    
    # Initialize analyzer
    analyzer = CSI_SpectralAnalyzer()
    
    # Run comprehensive analysis
    all_results, summary_df = analyzer.run_comprehensive_analysis(
        dataset_size=250,           # Start with 250 samples dataset
        max_samples_per_point=50    # Limit for computational efficiency
    )
    
    return analyzer, all_results, summary_df

if __name__ == "__main__":
    analyzer, all_results, summary_df = main()
