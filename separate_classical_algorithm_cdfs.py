#!/usr/bin/env python3
"""
Separate CDF Plots for Classical Algorithm Types

Creates individual CDF plots for:
1. k-NN algorithms (k=1, 3, 5, 9)
2. IDW algorithms (p=1, 2, 4)  
3. Probabilistic fingerprinting

Each algorithm type gets its own focused comparison plot
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import os

# We'll reuse the results from the previous run
def load_previous_results():
    """Load results from the previous evaluation"""
    # Since we don't have saved results, we'll need to re-run a simplified version
    print("📂 Re-running simplified evaluation to get CDF data...")
    
    # Import our classes
    import sys
    sys.path.append('.')
    
    # Import and run the evaluation
    exec(open('simple_classical_localization_models.py').read(), globals())

def create_knn_cdf_plot(knn_results):
    """Create CDF plot specifically for k-NN algorithms"""
    
    print("📈 Creating k-NN CDF Comparison Plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors for k-NN variants
    knn_colors = {
        'k-NN (k=1)': '#FF6B6B',    # Light Red
        'k-NN (k=3)': '#FF8E53',    # Orange Red  
        'k-NN (k=5)': '#FF7F50',    # Coral
        'k-NN (k=9)': '#DC143C'     # Crimson
    }
    
    # Plot CDFs for k-NN algorithms
    for result in knn_results:
        model_name = result['model']
        errors = result['errors']
        
        if 'k-NN' in model_name:
            errors_sorted = np.sort(errors)
            p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
            
            color = knn_colors[model_name]
            ax.plot(errors_sorted, p, color=color, linewidth=3, 
                   label=f"{model_name} (median: {result['median_error']:.3f}m)", 
                   alpha=0.9)
    
    # Add accuracy threshold lines
    thresholds = [1.0, 2.0, 3.0, 4.0]
    threshold_colors = ['green', 'orange', 'red', 'purple']
    
    for threshold, color in zip(thresholds, threshold_colors):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.6, linewidth=2)
        ax.text(threshold + 0.05, 0.95, f'{threshold}m', rotation=90, 
               fontsize=10, color=color, fontweight='bold', 
               verticalalignment='top')
    
    # Customize plot
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF Comparison: k-Nearest Neighbors (k-NN) Algorithms\n'
                'Indoor Localization Regression Performance', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('knn_algorithms_cdf_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 k-NN CDF plot saved: knn_algorithms_cdf_comparison.png")
    plt.show()

def create_idw_cdf_plot(idw_results):
    """Create CDF plot specifically for IDW algorithms"""
    
    print("📈 Creating IDW CDF Comparison Plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colors for IDW variants
    idw_colors = {
        'IDW (p=1)': '#32CD32',    # Lime Green
        'IDW (p=2)': '#228B22',    # Forest Green
        'IDW (p=4)': '#006400'     # Dark Green
    }
    
    # Plot CDFs for IDW algorithms
    for result in idw_results:
        model_name = result['model']
        errors = result['errors']
        
        if 'IDW' in model_name:
            errors_sorted = np.sort(errors)
            p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
            
            color = idw_colors[model_name]
            linewidth = 4 if model_name == 'IDW (p=1)' else 3  # Emphasize best performer
            ax.plot(errors_sorted, p, color=color, linewidth=linewidth, 
                   label=f"{model_name} (median: {result['median_error']:.3f}m)", 
                   alpha=0.9)
    
    # Add accuracy threshold lines
    thresholds = [1.0, 2.0, 3.0, 4.0]
    threshold_colors = ['green', 'orange', 'red', 'purple']
    
    for threshold, color in zip(thresholds, threshold_colors):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.6, linewidth=2)
        ax.text(threshold + 0.05, 0.95, f'{threshold}m', rotation=90, 
               fontsize=10, color=color, fontweight='bold', 
               verticalalignment='top')
    
    # Customize plot
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF Comparison: Inverse Distance Weighting (IDW) Algorithms\n'
                'Indoor Localization Regression Performance', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', fontsize=12, framealpha=0.9)
    
    # Add performance insight
    ax.text(0.02, 0.98, 'Best IDW Performance:\nLower power (p=1) works better\nfor indoor localization', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('idw_algorithms_cdf_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 IDW CDF plot saved: idw_algorithms_cdf_comparison.png")
    plt.show()

def create_probabilistic_cdf_plot(prob_results):
    """Create CDF plot for probabilistic fingerprinting"""
    
    print("📈 Creating Probabilistic Fingerprinting CDF Plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Find probabilistic result
    prob_result = None
    for result in prob_results:
        if 'Probabilistic' in result['model']:
            prob_result = result
            break
    
    if prob_result is None:
        print("⚠️ No probabilistic results found")
        return
    
    errors = prob_result['errors']
    errors_sorted = np.sort(errors)
    p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
    
    # Plot probabilistic CDF
    ax.plot(errors_sorted, p, color='#4169E1', linewidth=4, 
           label=f"Probabilistic Fingerprinting (median: {prob_result['median_error']:.3f}m)", 
           alpha=0.9)
    
    # Add accuracy threshold lines
    thresholds = [1.0, 2.0, 3.0, 4.0]
    threshold_colors = ['green', 'orange', 'red', 'purple']
    
    for threshold, color in zip(thresholds, threshold_colors):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.6, linewidth=2)
        ax.text(threshold + 0.05, 0.95, f'{threshold}m', rotation=90, 
               fontsize=10, color=color, fontweight='bold', 
               verticalalignment='top')
    
    # Customize plot
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF: Probabilistic Fingerprinting Algorithm\n'
                'Gaussian Maximum Likelihood Indoor Localization', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', fontsize=12, framealpha=0.9)
    
    # Add algorithm description
    ax.text(0.02, 0.98, 'Algorithm:\n• Learns Gaussian distribution\n  for each reference point\n• Uses Maximum Likelihood\n  Estimation for prediction', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor='lightblue', alpha=0.8))
    
    # Add performance metrics
    ax.text(0.02, 0.55, f'Performance Metrics:\n• Median Error: {prob_result["median_error"]:.3f}m\n• 1m Accuracy: {prob_result["accuracy_1m"]:.1f}%\n• 2m Accuracy: {prob_result["accuracy_2m"]:.1f}%', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('probabilistic_fingerprinting_cdf.png', dpi=300, bbox_inches='tight')
    print("💾 Probabilistic CDF plot saved: probabilistic_fingerprinting_cdf.png")
    plt.show()

def create_summary_comparison_table(all_results):
    """Create a comprehensive comparison table"""
    
    print(f"\n📊 ALGORITHM TYPE PERFORMANCE COMPARISON")
    print("="*55)
    
    # Group results by algorithm type
    knn_results = [r for r in all_results if 'k-NN' in r['model']]
    idw_results = [r for r in all_results if 'IDW' in r['model']]
    prob_results = [r for r in all_results if 'Probabilistic' in r['model']]
    
    print(f"\n🎯 k-NN ALGORITHMS:")
    print("-" * 20)
    for result in sorted(knn_results, key=lambda x: x['median_error']):
        print(f"   {result['model']:<12}: median={result['median_error']:.3f}m, 1m acc={result['accuracy_1m']:.1f}%")
    
    print(f"\n🎯 IDW ALGORITHMS:")
    print("-" * 18)
    for result in sorted(idw_results, key=lambda x: x['median_error']):
        print(f"   {result['model']:<12}: median={result['median_error']:.3f}m, 1m acc={result['accuracy_1m']:.1f}%")
    
    print(f"\n🎯 PROBABILISTIC:")
    print("-" * 15)
    for result in prob_results:
        print(f"   {result['model']:<12}: median={result['median_error']:.3f}m, 1m acc={result['accuracy_1m']:.1f}%")
    
    # Best performer in each category
    best_knn = min(knn_results, key=lambda x: x['median_error'])
    best_idw = min(idw_results, key=lambda x: x['median_error'])
    best_prob = prob_results[0] if prob_results else None
    
    print(f"\n🏆 BEST IN EACH CATEGORY:")
    print("-" * 25)
    print(f"   Best k-NN: {best_knn['model']} ({best_knn['median_error']:.3f}m)")
    print(f"   Best IDW:  {best_idw['model']} ({best_idw['median_error']:.3f}m)")
    if best_prob:
        print(f"   Probabilistic: {best_prob['model']} ({best_prob['median_error']:.3f}m)")

# We need to get the results from the previous run
# Let's create a simple way to re-extract them

def rerun_and_create_separate_plots():
    """Re-run evaluation and create separate plots"""
    
    print("🎯 CREATING SEPARATE CDF PLOTS BY ALGORITHM TYPE")
    print("="*55)
    
    # Import necessary modules and functions
    import sys
    import importlib.util
    import numpy as np
    from simple_classical_localization_models import *
    
    # Load data
    X, y, coordinates = load_amplitude_phase_data()
    
    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    unique_coords = np.unique(y, axis=0)
    n_train_points = int(0.8 * len(unique_coords))
    
    np.random.seed(42)
    train_coords = unique_coords[np.random.choice(len(unique_coords), n_train_points, replace=False)]
    
    train_mask = np.array([tuple(coord) in [tuple(tc) for tc in train_coords] for coord in y])
    test_mask = ~train_mask
    
    X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    # Define and evaluate models
    models = [
        (KNNLocalizer(k=1), "k-NN (k=1)"),
        (KNNLocalizer(k=3), "k-NN (k=3)"),
        (KNNLocalizer(k=5), "k-NN (k=5)"),
        (KNNLocalizer(k=9), "k-NN (k=9)"),
        (IDWLocalizer(power=1), "IDW (p=1)"),
        (IDWLocalizer(power=2), "IDW (p=2)"),
        (IDWLocalizer(power=4), "IDW (p=4)"),
        (ProbabilisticLocalizer(), "Probabilistic")
    ]
    
    print(f"\n🔬 Re-evaluating models for separate plots...")
    all_results = []
    for model, name in models:
        try:
            result = evaluate_model(model, X_train, y_train, X_test, y_test, name)
            all_results.append(result)
        except Exception as e:
            print(f"⚠️ Error evaluating {name}: {e}")
            continue
    
    # Create separate plots
    create_knn_cdf_plot(all_results)
    create_idw_cdf_plot(all_results)
    create_probabilistic_cdf_plot(all_results)
    create_summary_comparison_table(all_results)
    
    print(f"\n✅ SEPARATE CDF PLOTS CREATED!")
    print(f"📊 Generated 3 separate algorithm-specific CDF plots")
    print(f"🎯 Each plot focuses on one algorithm family for detailed comparison")

if __name__ == "__main__":
    rerun_and_create_separate_plots()


