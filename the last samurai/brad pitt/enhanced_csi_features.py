#!/usr/bin/env python3
"""
Enhanced CSI Feature Engineering Based on Spectral Analysis
Implements physics-informed features discovered from spectral analysis
"""

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

class EnhancedCSIFeatures:
    """
    Extract physics-informed features from CSI based on spectral analysis insights
    """
    
    def __init__(self, n_subcarriers=52, phy_fft=128, delta_f=312.5e3):
        self.n_subcarriers = n_subcarriers
        self.phy_fft = phy_fft
        self.delta_f = delta_f
        self.nfft_pad = 1024  # For fine delay resolution
    
    def compute_delay_features(self, amplitudes, phases):
        """Compute RMS delay spread and mean delay"""
        # Convert to complex CSI
        H = amplitudes * np.exp(1j * phases)
        
        # Embed into proper OFDM grid
        H_padded = self.embed_csi(H)
        
        # Compute PDP
        h_time = np.fft.ifft(np.fft.ifftshift(H_padded), n=self.nfft_pad)
        P = np.abs(h_time)**2
        P_norm = P / (P.sum() + 1e-12)
        
        # Delay axis
        delta_tau = 1.0 / (self.nfft_pad * self.delta_f)
        tau = np.arange(self.nfft_pad) * delta_tau
        
        # Compute moments
        tau_mean = np.sum(P_norm * tau)
        tau2 = np.sum(P_norm * tau**2)
        tau_rms = np.sqrt(max(0.0, tau2 - tau_mean**2))
        
        return {
            'tau_rms_ns': tau_rms * 1e9,
            'tau_mean_ns': tau_mean * 1e9,
            'pdp_max': np.max(P_norm),
            'pdp_entropy': -np.sum(P_norm * np.log(P_norm + 1e-12))
        }
    
    def compute_coherence_features(self, amplitudes, phases):
        """Compute coherence bandwidth and frequency correlation features"""
        H = amplitudes * np.exp(1j * phases)
        energy = np.sum(np.abs(H)**2)
        
        if energy == 0:
            return {'Bc_MHz': 0, 'freq_corr_50': 0, 'freq_corr_90': 0}
        
        # Frequency autocorrelation
        max_lag = len(H) - 1
        lags = np.arange(0, max_lag + 1)
        R = np.zeros_like(lags, dtype=float)
        
        for d in lags:
            valid = np.arange(0, len(H) - d)
            if valid.size > 0:
                R[d] = np.abs(np.sum(H[valid] * np.conj(H[valid + d]))) / energy
        
        R_norm = R / (R[0] if R[0] != 0 else 1.0)
        freq_offsets = lags * self.delta_f
        
        # Find coherence bandwidths at different thresholds
        Bc_50 = self.find_coherence_bandwidth(freq_offsets, R_norm, 0.5)
        Bc_90 = self.find_coherence_bandwidth(freq_offsets, R_norm, 0.9)
        
        return {
            'Bc_50_MHz': Bc_50 / 1e6,
            'Bc_90_MHz': Bc_90 / 1e6,
            'freq_corr_slope': np.polyfit(freq_offsets[:10], R_norm[:10], 1)[0] if len(freq_offsets) > 10 else 0
        }
    
    def compute_rician_k_factor(self, amplitudes, phases):
        """Compute Rician K-factor from PDP"""
        H = amplitudes * np.exp(1j * phases)
        H_padded = self.embed_csi(H)
        
        # Compute PDP
        h_time = np.fft.ifft(np.fft.ifftshift(H_padded), n=self.nfft_pad)
        P = np.abs(h_time)**2
        P_norm = P / (P.sum() + 1e-12)
        
        # K-factor: strongest tap vs rest
        los = np.max(P_norm)
        scattered = np.sum(P_norm) - los
        
        if scattered <= 0:
            return {'K_factor_dB': 50}  # Very high LOS
        
        K_lin = los / scattered
        return {'K_factor_dB': 10.0 * np.log10(K_lin)}
    
    def compute_spectral_features(self, amplitudes, phases):
        """Compute spectral shape features"""
        # Spectral centroid
        freq_bins = np.arange(len(amplitudes))
        spectral_centroid = np.sum(freq_bins * amplitudes**2) / (np.sum(amplitudes**2) + 1e-12)
        
        # Spectral spread
        spectral_spread = np.sqrt(np.sum((freq_bins - spectral_centroid)**2 * amplitudes**2) / 
                                 (np.sum(amplitudes**2) + 1e-12))
        
        # Spectral rolloff
        cumsum = np.cumsum(amplitudes**2)
        total = cumsum[-1]
        rolloff_85 = np.where(cumsum >= 0.85 * total)[0][0] if total > 0 else 0
        
        # Phase linearity
        phase_diff = np.diff(phases)
        phase_linearity = 1.0 / (np.std(phase_diff) + 1e-6)
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_spread': spectral_spread,
            'spectral_rolloff_85': rolloff_85,
            'phase_linearity': phase_linearity
        }
    
    def embed_csi(self, H):
        """Embed CSI into proper OFDM frequency grid"""
        phy_shifted = np.zeros(self.phy_fft, dtype=complex)
        center = self.phy_fft // 2
        half = len(H) // 2
        start = center - half
        phy_shifted[start:start + len(H)] = H
        
        # Embed into padded grid
        pad_shifted = np.zeros(self.nfft_pad, dtype=complex)
        pad_center = self.nfft_pad // 2
        pad_start = pad_center - (self.phy_fft // 2)
        pad_shifted[pad_start:pad_start + self.phy_fft] = phy_shifted
        
        return pad_shifted
    
    def find_coherence_bandwidth(self, freq_offsets, R_norm, threshold):
        """Find coherence bandwidth at given threshold"""
        if len(freq_offsets) <= 1:
            return freq_offsets[-1] if len(freq_offsets) > 0 else 0
        
        # Interpolate for finer resolution
        interp = interp1d(freq_offsets, R_norm, kind='linear', 
                         bounds_error=False, fill_value=(R_norm[0], R_norm[-1]))
        fgrid = np.linspace(freq_offsets[0], freq_offsets[-1], 1000)
        rg = interp(fgrid)
        
        # Find first frequency where correlation <= threshold
        idx = np.where(rg <= threshold)[0]
        return fgrid[idx[0]] if idx.size > 0 else fgrid[-1]
    
    def extract_all_features(self, amplitudes, phases):
        """Extract all enhanced features"""
        features = {}
        
        # Original features
        features.update({
            'csi_amp_mean': np.mean(amplitudes),
            'csi_amp_std': np.std(amplitudes),
            'csi_phase_mean': np.mean(phases),
            'csi_phase_std': np.std(phases)
        })
        
        # Enhanced physics-informed features
        features.update(self.compute_delay_features(amplitudes, phases))
        features.update(self.compute_coherence_features(amplitudes, phases))
        features.update(self.compute_rician_k_factor(amplitudes, phases))
        features.update(self.compute_spectral_features(amplitudes, phases))
        
        return features

# Example usage
def demonstrate_enhanced_features():
    """Demonstrate the enhanced feature extraction"""
    print("🔬 Enhanced CSI Feature Engineering Demo")
    
    # Simulate CSI data
    np.random.seed(42)
    amplitudes = np.random.lognormal(0, 0.5, 52)
    phases = np.random.uniform(-np.pi, np.pi, 52)
    
    # Extract features
    extractor = EnhancedCSIFeatures()
    features = extractor.extract_all_features(amplitudes, phases)
    
    print("\n📊 Extracted Features:")
    for key, value in features.items():
        print(f"  {key}: {value:.4f}")
    
    return features

if __name__ == "__main__":
    demonstrate_enhanced_features()
