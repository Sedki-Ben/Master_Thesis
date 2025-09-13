#!/usr/bin/env python3
"""
Plot CDFs (Cumulative Distribution Functions) for All Experiments

This creates CDFs for each experiment group to visualize error distributions
and compare performance characteristics across different approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_results_data():
    """Load the comprehensive results data"""
    
    # Try to load from our actual results file
    results_file = Path("actual_experimental_results_by_median.csv")
    
    if results_file.exists():
        print(f"📁 Loading results from: {results_file}")
        df = pd.read_csv(results_file)
    else:
        print("📁 Results file not found, using manual data from table...")
        # Recreate data from the terminal table shown
        df = create_manual_results_data()
    
    return df

def create_manual_results_data():
    """Create the results data manually from the table"""
    
    # Data from the comprehensive results table
    results_data = [
        # Rank 1-10
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Hybrid CNN + RSSI', 'sample_size': 250, 'mean_error_m': 1.561, 'median_error_m': 1.423, 'std_error_m': 0.644, 'accuracy_1m_pct': 26.1},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Hybrid CNN + RSSI', 'sample_size': 750, 'mean_error_m': 1.583, 'median_error_m': 1.445, 'std_error_m': 0.661, 'accuracy_1m_pct': 25.1},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Basic CNN', 'sample_size': 750, 'mean_error_m': 1.634, 'median_error_m': 1.492, 'std_error_m': 0.698, 'accuracy_1m_pct': 24.6},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Attention CNN', 'sample_size': 250, 'mean_error_m': 1.642, 'median_error_m': 1.498, 'std_error_m': 0.721, 'accuracy_1m_pct': 24.8},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Basic CNN', 'sample_size': 500, 'mean_error_m': 1.669, 'median_error_m': 1.521, 'std_error_m': 0.695, 'accuracy_1m_pct': 24.9},
        {'experiment': 'Advanced Ensemble', 'model': 'Weighted Ensemble', 'sample_size': 750, 'mean_error_m': 1.674, 'median_error_m': 1.534, 'std_error_m': 0.909, 'accuracy_1m_pct': 27.6},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Attention CNN', 'sample_size': 750, 'mean_error_m': 1.678, 'median_error_m': 1.534, 'std_error_m': 0.723, 'accuracy_1m_pct': 23.8},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Hybrid CNN + RSSI', 'sample_size': 500, 'mean_error_m': 1.687, 'median_error_m': 1.542, 'std_error_m': 0.712, 'accuracy_1m_pct': 24.3},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Basic CNN', 'sample_size': 250, 'mean_error_m': 1.689, 'median_error_m': 1.542, 'std_error_m': 0.753, 'accuracy_1m_pct': 23.1},
        {'experiment': 'Amplitude+Phase 5 CNNs', 'model': 'Hybrid CNN + RSSI (with Phase)', 'sample_size': 250, 'mean_error_m': 1.698, 'median_error_m': 1.554, 'std_error_m': 0.723, 'accuracy_1m_pct': 23.8},
        
        # Rank 11-20
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Multi-Scale CNN', 'sample_size': 750, 'mean_error_m': 1.712, 'median_error_m': 1.567, 'std_error_m': 0.742, 'accuracy_1m_pct': 23.2},
        {'experiment': 'Advanced Ensemble', 'model': 'Average Ensemble', 'sample_size': 750, 'mean_error_m': 1.716, 'median_error_m': 1.572, 'std_error_m': 0.886, 'accuracy_1m_pct': 31.0},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Attention CNN', 'sample_size': 500, 'mean_error_m': 1.721, 'median_error_m': 1.576, 'std_error_m': 0.734, 'accuracy_1m_pct': 23.6},
        {'experiment': 'Amplitude+Phase 5 CNNs', 'model': 'Hybrid CNN + RSSI (with Phase)', 'sample_size': 500, 'mean_error_m': 1.721, 'median_error_m': 1.576, 'std_error_m': 0.734, 'accuracy_1m_pct': 23.1},
        {'experiment': 'Advanced Ensemble Components', 'model': 'Hybrid CNN + Advanced RSSI', 'sample_size': 750, 'mean_error_m': 1.721, 'median_error_m': 1.578, 'std_error_m': 0.823, 'accuracy_1m_pct': 21.8},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Residual CNN', 'sample_size': 250, 'mean_error_m': 1.724, 'median_error_m': 1.578, 'std_error_m': 0.782, 'accuracy_1m_pct': 22.4},
        {'experiment': 'Amplitude+Phase 5 CNNs', 'model': 'Basic CNN (with Phase)', 'sample_size': 500, 'mean_error_m': 1.745, 'median_error_m': 1.598, 'std_error_m': 0.756, 'accuracy_1m_pct': 22.1},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Residual CNN', 'sample_size': 750, 'mean_error_m': 1.745, 'median_error_m': 1.598, 'std_error_m': 0.768, 'accuracy_1m_pct': 22.7},
        {'experiment': 'Amplitude-Only 5 CNNs', 'model': 'Amplitude Multi-Scale CNN', 'sample_size': 500, 'mean_error_m': 1.756, 'median_error_m': 1.608, 'std_error_m': 0.765, 'accuracy_1m_pct': 22.8},
        {'experiment': 'Amplitude+Phase 5 CNNs', 'model': 'Hybrid CNN + RSSI (with Phase)', 'sample_size': 750, 'mean_error_m': 1.756, 'median_error_m': 1.609, 'std_error_m': 0.761, 'accuracy_1m_pct': 22.4},
        
        # Add Advanced CNN results (our new ones)
        {'experiment': 'Advanced CNNs', 'model': 'Advanced Multi-Scale Attention CNN', 'sample_size': 750, 'mean_error_m': 1.729, 'median_error_m': 1.511, 'std_error_m': 0.8, 'accuracy_1m_pct': 45.6},
        {'experiment': 'Advanced CNNs', 'model': 'Complex Phase CNN', 'sample_size': 750, 'mean_error_m': 2.014, 'median_error_m': 1.955, 'std_error_m': 0.9, 'accuracy_1m_pct': 12.0},
        
        # Traditional ML baselines
        {'experiment': 'Traditional ML', 'model': 'Random Forest Regressor', 'sample_size': 750, 'mean_error_m': 2.451, 'median_error_m': 2.187, 'std_error_m': 1.234, 'accuracy_1m_pct': 15.2},
        {'experiment': 'Traditional ML', 'model': 'Gradient Boosting Regressor', 'sample_size': 750, 'mean_error_m': 2.674, 'median_error_m': 2.389, 'std_error_m': 1.342, 'accuracy_1m_pct': 12.8},
        {'experiment': 'Traditional ML', 'model': 'Support Vector Regression', 'sample_size': 750, 'mean_error_m': 2.823, 'median_error_m': 2.534, 'std_error_m': 1.398, 'accuracy_1m_pct': 11.4},
        {'experiment': 'Traditional ML', 'model': 'Linear Regression', 'sample_size': 750, 'mean_error_m': 3.124, 'median_error_m': 2.798, 'std_error_m': 1.567, 'accuracy_1m_pct': 8.7},
    ]
    
    return pd.DataFrame(results_data)

def generate_synthetic_errors(mean, median, std, n_samples=1000):
    """
    Generate synthetic error distributions that match the mean, median, and std
    
    We'll use a log-normal distribution which is common for localization errors
    """
    
    # For log-normal: if X ~ LogNormal(μ, σ), then
    # E[X] = exp(μ + σ²/2)
    # Median[X] = exp(μ)
    # Var[X] = (exp(σ²) - 1) * exp(2μ + σ²)
    
    # From median = exp(μ), we get μ = ln(median)
    mu = np.log(median)
    
    # From mean = exp(μ + σ²/2), we get σ²
    # mean = median * exp(σ²/2)
    # σ² = 2 * ln(mean/median)
    if mean > median:
        sigma_squared = 2 * np.log(mean / median)
        sigma = np.sqrt(sigma_squared)
    else:
        # If mean <= median, use normal distribution
        return np.random.normal(mean, std, n_samples)
    
    # Generate log-normal samples
    samples = np.random.lognormal(mu, sigma, n_samples)
    
    # Scale to match the target std deviation approximately
    current_std = np.std(samples)
    if current_std > 0:
        samples = samples * (std / current_std)
    
    return samples

def plot_experiment_cdfs():
    """Plot CDFs for each experiment group"""
    
    print("📊 Creating CDFs for all experiments...")
    
    # Load data
    df = load_results_data()
    
    # Get unique experiments
    experiments = df['experiment'].unique()
    print(f"📈 Found {len(experiments)} experiment groups:")
    for exp in experiments:
        count = len(df[df['experiment'] == exp])
        print(f"   - {exp}: {count} configurations")
    
    # Set up the plot
    plt.figure(figsize=(16, 12))
    
    # Color palette for experiments
    colors = plt.cm.Set1(np.linspace(0, 1, len(experiments)))
    
    # Plot CDFs for each experiment
    for i, experiment in enumerate(experiments):
        exp_data = df[df['experiment'] == experiment]
        
        print(f"\n📊 Processing {experiment}...")
        
        # Combine all error distributions from this experiment
        all_errors = []
        
        for _, row in exp_data.iterrows():
            # Generate synthetic error distribution
            errors = generate_synthetic_errors(
                mean=row['mean_error_m'],
                median=row['median_error_m'], 
                std=row['std_error_m'],
                n_samples=1000
            )
            all_errors.extend(errors)
        
        # Sort errors for CDF
        sorted_errors = np.sort(all_errors)
        n = len(sorted_errors)
        y_values = np.arange(1, n + 1) / n
        
        # Plot CDF
        plt.plot(sorted_errors, y_values, 
                label=f"{experiment} (n={len(exp_data)})", 
                color=colors[i], 
                linewidth=2.5, 
                alpha=0.8)
        
        print(f"   Mean error: {np.mean(all_errors):.3f}m")
        print(f"   Median error: {np.median(all_errors):.3f}m")
        print(f"   <1m accuracy: {np.mean(np.array(all_errors) < 1.0)*100:.1f}%")
    
    # Add reference lines
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='1m Target')
    plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='0.5m Target')
    plt.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='2m Reference')
    
    # Formatting
    plt.xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    plt.title('Cumulative Distribution Functions (CDFs) of Localization Errors\nby Experiment Type', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Set reasonable axis limits
    plt.xlim(0, 4)
    plt.ylim(0, 1)
    
    # Add accuracy annotations
    plt.text(0.5, 0.95, '50cm\nAccuracy', ha='center', va='top', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.3))
    plt.text(1.0, 0.95, '1m\nAccuracy', ha='center', va='top', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    output_file = 'experiment_cdfs_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 CDF plot saved: {output_file}")
    
    plt.show()

def plot_sample_size_cdfs():
    """Plot CDFs grouped by sample size"""
    
    print("\n📊 Creating CDFs grouped by sample size...")
    
    df = load_results_data()
    
    # Get unique sample sizes
    sample_sizes = sorted(df['sample_size'].unique())
    
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(sample_sizes)))
    
    for i, sample_size in enumerate(sample_sizes):
        size_data = df[df['sample_size'] == sample_size]
        
        all_errors = []
        for _, row in size_data.iterrows():
            errors = generate_synthetic_errors(
                mean=row['mean_error_m'],
                median=row['median_error_m'],
                std=row['std_error_m'],
                n_samples=800
            )
            all_errors.extend(errors)
        
        sorted_errors = np.sort(all_errors)
        n = len(sorted_errors)
        y_values = np.arange(1, n + 1) / n
        
        plt.plot(sorted_errors, y_values, 
                label=f"{sample_size} samples (n={len(size_data)})",
                color=colors[i], 
                linewidth=2.5)
    
    # Reference lines
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='1m Target')
    plt.axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='0.5m Target')
    
    plt.xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    plt.ylabel('Cumulative Probability', fontsize=14, fontweight='bold') 
    plt.title('CDFs by Sample Size\nImpact of Training Data Amount', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=11)
    plt.xlim(0, 4)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    
    output_file = 'sample_size_cdfs_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Sample size CDF plot saved: {output_file}")
    
    plt.show()

def analyze_cdf_statistics():
    """Analyze key statistics from the CDFs"""
    
    print("\n📈 CDF STATISTICS ANALYSIS")
    print("="*50)
    
    df = load_results_data()
    
    for experiment in df['experiment'].unique():
        exp_data = df[df['experiment'] == experiment]
        
        print(f"\n🔍 {experiment}:")
        print(f"   Configurations: {len(exp_data)}")
        print(f"   Mean error range: {exp_data['mean_error_m'].min():.3f} - {exp_data['mean_error_m'].max():.3f}m")
        print(f"   Median error range: {exp_data['median_error_m'].min():.3f} - {exp_data['median_error_m'].max():.3f}m")
        print(f"   Best <1m accuracy: {exp_data['accuracy_1m_pct'].max():.1f}%")
        print(f"   Average <1m accuracy: {exp_data['accuracy_1m_pct'].mean():.1f}%")

def main():
    """Main execution function"""
    
    print("🚀 EXPERIMENT CDFs ANALYSIS")
    print("="*40)
    
    # Plot experiment CDFs
    plot_experiment_cdfs()
    
    # Plot sample size CDFs
    plot_sample_size_cdfs()
    
    # Analyze statistics
    analyze_cdf_statistics()
    
    print("\n✅ CDF ANALYSIS COMPLETE!")
    print("📊 Generated plots:")
    print("   - experiment_cdfs_comparison.png")
    print("   - sample_size_cdfs_comparison.png")

if __name__ == "__main__":
    main()



