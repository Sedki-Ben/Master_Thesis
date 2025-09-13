#!/usr/bin/env python3
"""
Improved Performance vs Model Complexity Analysis for Indoor Localization

Creates a cleaner IEEE-style plot with more accurate complexity assessment
based on:
- Number of parameters
- Computational operations
- Training complexity
- Implementation difficulty
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from pathlib import Path

def create_improved_performance_complexity_plot():
    """Create improved performance vs complexity analysis plot"""
    
    print("📊 Creating Improved Performance vs Model Complexity Analysis")
    print("="*60)
    
    # Set up the plot style for IEEE publications
    plt.style.use('default')
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # More accurate complexity assessment based on:
    # - Parameters count, computational cost, training complexity
    # - k-NN: O(n) for inference, no training
    # - IDW: O(n) for inference, no training  
    # - Probabilistic: O(n) for inference, O(n) for training (computing means/covs)
    # - CNNs: Various parameter counts and computational complexity
    
    models_data = {
        # Classical Methods (Lower complexity)
        'k-NN (k=3)': {
            'complexity': 1.0, 
            'median_error': 2.55, 
            'category': 'Classical', 
            'color': '#1f77b4',
            'params': '0',
            'ops': 'O(n)'
        },
        'k-NN (k=5)': {
            'complexity': 1.1, 
            'median_error': 2.509, 
            'category': 'Classical', 
            'color': '#1f77b4',
            'params': '0',
            'ops': 'O(n)'
        },
        'k-NN (k=9)': {
            'complexity': 1.2, 
            'median_error': 2.373, 
            'category': 'Classical', 
            'color': '#1f77b4',
            'params': '0',
            'ops': 'O(n)'
        },
        'IDW (power=4)': {
            'complexity': 1.5, 
            'median_error': 2.327, 
            'category': 'Classical', 
            'color': '#ff7f0e',
            'params': '1',
            'ops': 'O(n)'
        },
        'Probabilistic': {
            'complexity': 2.0, 
            'median_error': 1.927, 
            'category': 'Classical', 
            'color': '#2ca02c',
            'params': '~100',
            'ops': 'O(n×d²)'
        },
        
        # Deep Learning Methods (Higher complexity)
        # Basic CNN: ~50K parameters (2 conv + 2 dense layers)
        'Basic CNN': {
            'complexity': 4.0, 
            'median_error': 1.598, 
            'category': 'Deep Learning', 
            'color': '#d62728',
            'params': '~50K',
            'ops': 'O(n×k×f)'
        },
        # Multi-Scale: ~80K parameters (3 parallel conv paths)
        'Multi-Scale CNN': {
            'complexity': 5.5, 
            'median_error': 1.567, 
            'category': 'Deep Learning', 
            'color': '#8c564b',
            'params': '~80K',
            'ops': 'O(3×n×k×f)'
        },
        # Residual: ~70K parameters (skip connections, deeper)
        'Residual CNN': {
            'complexity': 6.0, 
            'median_error': 1.578, 
            'category': 'Deep Learning', 
            'color': '#9467bd',
            'params': '~70K',
            'ops': 'O(n×k×f×L)'
        },
        # Attention: ~120K parameters (attention mechanism)
        'Attention CNN': {
            'complexity': 7.5, 
            'median_error': 1.498, 
            'category': 'Deep Learning', 
            'color': '#e377c2',
            'params': '~120K',
            'ops': 'O(n²×d+n×k×f)'
        },
        # Hybrid: ~90K parameters (dual input branches)
        'Hybrid CNN + RSSI': {
            'complexity': 6.5, 
            'median_error': 1.423, 
            'category': 'Deep Learning', 
            'color': '#7f7f7f',
            'params': '~90K',
            'ops': 'O(2×n×k×f)'
        }
    }
    
    # Separate data by category
    classical_models = {k: v for k, v in models_data.items() if v['category'] == 'Classical'}
    dl_models = {k: v for k, v in models_data.items() if v['category'] == 'Deep Learning'}
    
    # Plot classical methods
    classical_x = [data['complexity'] for data in classical_models.values()]
    classical_y = [data['median_error'] for data in classical_models.values()]
    classical_colors = [data['color'] for data in classical_models.values()]
    
    # Plot deep learning methods  
    dl_x = [data['complexity'] for data in dl_models.values()]
    dl_y = [data['median_error'] for data in dl_models.values()]
    dl_colors = [data['color'] for data in dl_models.values()]
    
    # Create scatter plots with larger points to accommodate text
    scatter1 = ax.scatter(classical_x, classical_y, c=classical_colors, s=200, 
                         marker='o', alpha=0.8, edgecolors='black', linewidth=2,
                         label='Classical Methods')
    
    scatter2 = ax.scatter(dl_x, dl_y, c=dl_colors, s=250, 
                         marker='^', alpha=0.8, edgecolors='black', linewidth=2,
                         label='Deep Learning Methods')
    
    # Add values directly on the points (white text for visibility)
    for model, data in classical_models.items():
        ax.text(data['complexity'], data['median_error'], f'{data["median_error"]:.2f}m', 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                color='white', path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
    
    for model, data in dl_models.items():
        ax.text(data['complexity'], data['median_error'], f'{data["median_error"]:.2f}m', 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                color='white', path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
    
    # Add model name labels with improved positioning
    label_offsets = {
        'k-NN (k=3)': (0.0, 0.12),
        'k-NN (k=5)': (0.0, -0.12),
        'k-NN (k=9)': (0.0, 0.12),
        'IDW (power=4)': (0.0, -0.12),
        'Probabilistic': (0.0, 0.12),
        'Basic CNN': (0.0, -0.12),
        'Multi-Scale CNN': (0.0, 0.12),
        'Residual CNN': (0.0, -0.12),
        'Attention CNN': (0.0, 0.12),
        'Hybrid CNN + RSSI': (0.0, -0.12)
    }
    
    for model, data in models_data.items():
        offset_x, offset_y = label_offsets[model]
        
        # Choose background color based on category
        bg_color = 'lightblue' if data['category'] == 'Deep Learning' else 'lightgray'
        
        ax.annotate(model, 
                   (data['complexity'], data['median_error']),
                   xytext=(data['complexity'] + offset_x, data['median_error'] + offset_y),
                   fontsize=9, fontweight='bold', ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.8, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.7, lw=1))
    
    # Add trend lines
    # Classical methods trend
    classical_x_sorted, classical_y_sorted = zip(*sorted(zip(classical_x, classical_y)))
    ax.plot(classical_x_sorted, classical_y_sorted, '--', color='blue', alpha=0.6, linewidth=2, 
           label='Classical Trend')
    
    # Deep learning methods trend (sort by complexity for proper trend line)
    dl_sorted = sorted(zip(dl_x, dl_y))
    dl_x_sorted, dl_y_sorted = zip(*dl_sorted)
    ax.plot(dl_x_sorted, dl_y_sorted, '--', color='red', alpha=0.6, linewidth=2,
           label='Deep Learning Trend')
    
    # Performance zones (subtle background)
    ax.axhspan(0, 1.5, alpha=0.05, color='green')
    ax.axhspan(1.5, 2.0, alpha=0.05, color='yellow') 
    ax.axhspan(2.0, 3.0, alpha=0.05, color='orange')
    
    # Add performance zone labels on the right
    ax.text(8.2, 1.25, 'Excellent\n(<1.5m)', fontsize=9, ha='center', 
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.7))
    ax.text(8.2, 1.75, 'Good\n(1.5-2.0m)', fontsize=9, ha='center',
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.7))
    ax.text(8.2, 2.25, 'Moderate\n(2.0-3.0m)', fontsize=9, ha='center',
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.7))
    
    # Customize the plot
    ax.set_xlabel('Model Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Median Error (meters)', fontsize=14, fontweight='bold')
    ax.set_title('Performance vs Model Complexity: Indoor Localization Systems', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits and ticks
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(1.0, 2.8)
    
    # Custom x-axis labels
    complexity_ticks = [1, 2, 3, 4, 5, 6, 7, 8]
    complexity_labels = ['Simple\nk-NN', 'Basic\nML', 'Statistical\nML', 'Basic\nCNN', 
                        'Multi-path\nCNN', 'Advanced\nCNN', 'Attention\nCNN', 'Complex\nCNN']
    ax.set_xticks(complexity_ticks)
    ax.set_xticklabels(complexity_labels, fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'improved_performance_vs_complexity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Improved Performance vs Complexity plot saved: {output_file}")
    
    plt.show()
    
    return models_data

def print_complexity_justification():
    """Print justification for complexity values"""
    
    print("\n🔍 MODEL COMPLEXITY JUSTIFICATION")
    print("="*45)
    
    complexity_rationale = {
        1.0: "k-NN (k=3): No training, simple distance computation, O(n) inference",
        1.1: "k-NN (k=5): Slightly more neighbors to check",
        1.2: "k-NN (k=9): More neighbors, higher computational cost",
        1.5: "IDW: Distance weighting with power function, O(n) with extra operations",
        2.0: "Probabilistic: Requires computing/storing means & covariances per location",
        4.0: "Basic CNN: ~50K parameters, 2 conv + 2 dense layers, standard backprop",
        5.5: "Multi-Scale CNN: ~80K parameters, 3 parallel conv paths, more complex architecture",
        6.0: "Residual CNN: ~70K parameters, skip connections, deeper network",
        6.5: "Hybrid CNN+RSSI: ~90K parameters, dual input branches, fusion complexity", 
        7.5: "Attention CNN: ~120K parameters, attention mechanism adds O(n²) operations"
    }
    
    for complexity, rationale in complexity_rationale.items():
        print(f"   {complexity:.1f}: {rationale}")
    
    print(f"\n📊 Key Complexity Factors:")
    print(f"   • Parameter Count: More parameters = higher complexity")
    print(f"   • Computational Operations: Attention > Multi-path > Basic")
    print(f"   • Training Requirements: DL requires extensive training vs. classical")
    print(f"   • Implementation Difficulty: Attention mechanisms most complex")

def main():
    """Main execution function"""
    
    print("🎯 IMPROVED PERFORMANCE vs COMPLEXITY ANALYSIS")
    print("Indoor Localization Systems - Accurate Complexity Assessment")
    print("="*65)
    
    # Create the improved plot
    models_data = create_improved_performance_complexity_plot()
    
    # Justify complexity values
    print_complexity_justification()
    
    # Quick analysis
    classical_errors = [2.55, 2.509, 2.373, 2.327, 1.927]
    dl_errors = [1.598, 1.567, 1.578, 1.498, 1.423]
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    print(f"   Best Classical: {min(classical_errors):.3f}m (Probabilistic)")
    print(f"   Best Deep Learning: {min(dl_errors):.3f}m (Hybrid CNN+RSSI)")
    print(f"   Performance Gain: {((min(classical_errors) - min(dl_errors)) / min(classical_errors) * 100):.1f}%")
    
    print(f"\n✅ IMPROVED ANALYSIS COMPLETE!")
    print(f"📊 Removed info box, added values on points, refined complexity assessment")

if __name__ == "__main__":
    main()
