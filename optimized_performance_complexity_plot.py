#!/usr/bin/env python3
"""
Optimized Performance vs Model Complexity Analysis for Indoor Localization

Creates an optimized IEEE-style plot with:
- Performance zone labels positioned inside the plot
- Multi-level vertical positioning for algorithm labels
- Enhanced legibility and professional spacing
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from pathlib import Path

def create_optimized_performance_complexity_plot():
    """Create optimized performance vs complexity analysis plot"""
    
    print("📊 Creating Optimized Performance vs Model Complexity Analysis")
    print("="*58)
    
    # Set up the plot style for IEEE publications
    plt.style.use('default')
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    
    # Model data with abbreviated names
    models_data = {
        # Classical Methods
        'k-NN (k=3)': {
            'complexity': 1.0, 
            'median_error': 2.55, 
            'category': 'Classical', 
            'color': '#1f77b4'
        },
        'k-NN (k=5)': {
            'complexity': 1.1, 
            'median_error': 2.509, 
            'category': 'Classical', 
            'color': '#1f77b4'
        },
        'k-NN (k=9)': {
            'complexity': 1.2, 
            'median_error': 2.373, 
            'category': 'Classical', 
            'color': '#1f77b4'
        },
        'IDW': {
            'complexity': 1.5, 
            'median_error': 2.327, 
            'category': 'Classical', 
            'color': '#ff7f0e'
        },
        'Probabilistic': {
            'complexity': 2.0, 
            'median_error': 1.927, 
            'category': 'Classical', 
            'color': '#2ca02c'
        },
        
        # Deep Learning Methods (Abbreviated)
        'CNN': {
            'complexity': 4.0, 
            'median_error': 1.598, 
            'category': 'Deep Learning', 
            'color': '#d62728'
        },
        'MS-CNN': {
            'complexity': 5.5, 
            'median_error': 1.567, 
            'category': 'Deep Learning', 
            'color': '#8c564b'
        },
        'R-CNN': {
            'complexity': 6.0, 
            'median_error': 1.578, 
            'category': 'Deep Learning', 
            'color': '#9467bd'
        },
        'H-CNN': {
            'complexity': 6.5, 
            'median_error': 1.423, 
            'category': 'Deep Learning', 
            'color': '#7f7f7f'
        },
        'A-CNN': {
            'complexity': 7.5, 
            'median_error': 1.498, 
            'category': 'Deep Learning', 
            'color': '#e377c2'
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
    
    # Create scatter plots
    scatter1 = ax.scatter(classical_x, classical_y, c=classical_colors, s=180, 
                         marker='o', alpha=0.8, edgecolors='black', linewidth=2,
                         label='Classical Methods')
    
    scatter2 = ax.scatter(dl_x, dl_y, c=dl_colors, s=200, 
                         marker='^', alpha=0.8, edgecolors='black', linewidth=2,
                         label='Deep Learning Methods')
    
    # Multi-level vertical positioning for algorithm labels (to avoid overlaps)
    # Format: (label_offset_x, label_offset_y, value_offset_x, value_offset_y)
    positioning = {
        # Classical methods - staggered levels to avoid overlaps with increased distances
        'k-NN (k=3)': (0.0, 0.18, 0.0, -0.08),        # High level
        'k-NN (k=5)': (-0.25, -0.20, 0.12, 0.10),     # Low level, further offset left
        'k-NN (k=9)': (0.25, 0.20, -0.12, -0.10),     # Mid level, further offset right
        'IDW': (0.0, -0.18, 0.0, 0.08),                # Low level
        'Probabilistic': (0.0, 0.15, 0.0, -0.08),      # High level
        
        # Deep learning methods - well spaced
        'CNN': (0.0, -0.15, 0.0, 0.08),
        'MS-CNN': (0.0, 0.15, 0.0, -0.08),
        'R-CNN': (0.0, -0.15, 0.0, 0.08),
        'H-CNN': (0.0, 0.15, 0.0, -0.08),
        'A-CNN': (0.0, -0.15, 0.0, 0.08)
    }
    
    # Add algorithm labels and values
    for model, data in models_data.items():
        label_offset_x, label_offset_y, value_offset_x, value_offset_y = positioning[model]
        
        # Choose background color based on category
        bg_color = 'lightblue' if data['category'] == 'Deep Learning' else 'lightgray'
        
        # Add algorithm name label (abbreviated)
        ax.annotate(model, 
                   (data['complexity'], data['median_error']),
                   xytext=(data['complexity'] + label_offset_x, data['median_error'] + label_offset_y),
                   fontsize=10, fontweight='bold', ha='center',
                   bbox=dict(boxstyle="round,pad=0.25", facecolor=bg_color, alpha=0.85, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.7, lw=1))
        
        # Add median error value closer to point
        ax.text(data['complexity'] + value_offset_x, data['median_error'] + value_offset_y, 
                f'{data["median_error"]:.2f}m', 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                color='white', 
                path_effects=[path_effects.withStroke(linewidth=2.5, foreground='black')])
    
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
    
    # Performance zones (very subtle background)
    ax.axhspan(1.0, 1.5, alpha=0.04, color='green')
    ax.axhspan(1.5, 2.0, alpha=0.04, color='yellow') 
    ax.axhspan(2.0, 2.9, alpha=0.04, color='orange')
    
    # Add performance zone labels on the RIGHT side
    ax.text(7.6, 1.25, 'Excellent\n(<1.5m)', fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen'),
           fontweight='bold')
    ax.text(7.6, 1.75, 'Good\n(1.5-2.0m)', fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8, edgecolor='orange'),
           fontweight='bold')
    ax.text(7.6, 2.4, 'Moderate\n(2.0-3.0m)', fontsize=9, ha='center', va='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8, edgecolor='darkred'),
           fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Model Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Median Error (meters)', fontsize=14, fontweight='bold')
    ax.set_title('Performance vs Model Complexity: Indoor Localization Systems', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits with optimal spacing (more space at top for labels)
    ax.set_xlim(0.5, 8.0)
    ax.set_ylim(1.0, 2.9)
    
    # Custom x-axis labels
    complexity_ticks = [1, 2, 3, 4, 5, 6, 7, 8]
    complexity_labels = ['Simple\nk-NN', 'Basic\nML', 'Statistical\nML', 'Basic\nCNN', 
                        'Multi-path\nCNN', 'Advanced\nCNN', 'Attention\nCNN', 'Complex\nCNN']
    ax.set_xticks(complexity_ticks)
    ax.set_xticklabels(complexity_labels, fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Create legend with smaller markers
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=7, markeredgecolor='black', label='Classical Methods'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                  markersize=8, markeredgecolor='black', label='Deep Learning Methods'),
        plt.Line2D([0], [0], color='blue', linestyle='--', alpha=0.6, 
                  linewidth=2, label='Classical Trend'),
        plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.6, 
                  linewidth=2, label='Deep Learning Trend')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                      framealpha=0.9, borderpad=0.4, handletextpad=0.4, 
                      columnspacing=0.4, handlelength=1.2)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'optimized_performance_vs_complexity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Optimized Performance vs Complexity plot saved: {output_file}")
    
    plt.show()
    
    return models_data

def print_positioning_improvements():
    """Print summary of positioning improvements"""
    
    print("\n🔧 POSITIONING IMPROVEMENTS")
    print("="*30)
    
    print("✅ Performance Zone Labels:")
    print("   • Positioned on the RIGHT side of the plot")
    print("   • Enhanced visibility with colored borders")
    print("   • Proper positioning within plot boundaries")
    
    print("\n✅ Algorithm Label Positioning:")
    print("   • Multi-level vertical spacing for k-NN variants")
    print("   • Staggered positioning to prevent overlaps")
    print("   • Extended distances from points for clarity")
    print("   • Strategic horizontal offsets for k-NN clustering")
    
    print("\n📏 Specific k-NN Improvements:")
    print("   • k-NN (k=3): High level positioning")
    print("   • k-NN (k=5): Low level, FURTHER offset left (-0.25)")  
    print("   • k-NN (k=9): Mid level, FURTHER offset right (+0.25)")
    print("   • Increased distances prevent label overlaps")

def main():
    """Main execution function"""
    
    print("🎯 OPTIMIZED PERFORMANCE vs COMPLEXITY ANALYSIS")
    print("Indoor Localization - Enhanced Positioning and Legibility")
    print("="*60)
    
    # Create the optimized plot
    models_data = create_optimized_performance_complexity_plot()
    
    # Print positioning improvements
    print_positioning_improvements()
    
    print(f"\n✅ OPTIMIZED PLOT COMPLETE!")
    print(f"📊 Enhanced legibility with improved label positioning")
    print(f"🎯 Performance zones now contained within plot boundaries")

if __name__ == "__main__":
    main()
