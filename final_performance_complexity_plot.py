#!/usr/bin/env python3
"""
Final Performance vs Model Complexity Analysis for Indoor Localization

Creates a polished IEEE-style plot with:
- Smaller legend symbols to prevent overlap
- Strategic label positioning to avoid overlaps
- Values on opposite sides of algorithm labels
- Professional spacing and layout
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from pathlib import Path

def create_final_performance_complexity_plot():
    """Create final polished performance vs complexity analysis plot"""
    
    print("📊 Creating Final Performance vs Model Complexity Analysis")
    print("="*58)
    
    # Set up the plot style for IEEE publications
    plt.style.use('default')
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 9))
    
    # Model data with accurate complexity assessment
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
        'IDW (power=4)': {
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
        
        # Deep Learning Methods
        'Basic CNN': {
            'complexity': 4.0, 
            'median_error': 1.598, 
            'category': 'Deep Learning', 
            'color': '#d62728'
        },
        'Multi-Scale CNN': {
            'complexity': 5.5, 
            'median_error': 1.567, 
            'category': 'Deep Learning', 
            'color': '#8c564b'
        },
        'Residual CNN': {
            'complexity': 6.0, 
            'median_error': 1.578, 
            'category': 'Deep Learning', 
            'color': '#9467bd'
        },
        'Hybrid CNN + RSSI': {
            'complexity': 6.5, 
            'median_error': 1.423, 
            'category': 'Deep Learning', 
            'color': '#7f7f7f'
        },
        'Attention CNN': {
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
    
    # Create scatter plots with appropriate sizes
    scatter1 = ax.scatter(classical_x, classical_y, c=classical_colors, s=180, 
                         marker='o', alpha=0.8, edgecolors='black', linewidth=2,
                         label='Classical Methods')
    
    scatter2 = ax.scatter(dl_x, dl_y, c=dl_colors, s=200, 
                         marker='^', alpha=0.8, edgecolors='black', linewidth=2,
                         label='Deep Learning Methods')
    
    # Strategic positioning for algorithm labels and values
    # Format: (label_offset_x, label_offset_y, value_offset_x, value_offset_y, label_position)
    positioning = {
        'k-NN (k=3)': (0.0, 0.15, 0.0, -0.12, 'above'),
        'k-NN (k=5)': (0.0, -0.15, 0.0, 0.12, 'below'),
        'k-NN (k=9)': (0.0, 0.15, 0.0, -0.12, 'above'),
        'IDW (power=4)': (0.0, -0.15, 0.0, 0.12, 'below'),
        'Probabilistic': (0.0, 0.15, 0.0, -0.12, 'above'),
        'Basic CNN': (0.0, -0.18, 0.0, 0.15, 'below'),
        'Multi-Scale CNN': (0.0, 0.18, 0.0, -0.15, 'above'),
        'Residual CNN': (0.0, -0.18, 0.0, 0.15, 'below'),
        'Hybrid CNN + RSSI': (0.0, 0.18, 0.0, -0.15, 'above'),
        'Attention CNN': (0.0, -0.18, 0.0, 0.15, 'below')
    }
    
    # Add algorithm labels
    for model, data in models_data.items():
        label_offset_x, label_offset_y, value_offset_x, value_offset_y, position = positioning[model]
        
        # Choose background color based on category
        bg_color = 'lightblue' if data['category'] == 'Deep Learning' else 'lightgray'
        
        # Add algorithm name label
        ax.annotate(model, 
                   (data['complexity'], data['median_error']),
                   xytext=(data['complexity'] + label_offset_x, data['median_error'] + label_offset_y),
                   fontsize=9, fontweight='bold', ha='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.85, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', color='black', alpha=0.7, lw=1))
        
        # Add median error value on opposite side
        ax.text(data['complexity'] + value_offset_x, data['median_error'] + value_offset_y, 
                f'{data["median_error"]:.2f}m', 
                ha='center', va='center', fontsize=10, fontweight='bold', 
                color='white', 
                path_effects=[path_effects.withStroke(linewidth=3, foreground='black')])
    
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
    ax.axhspan(0, 1.5, alpha=0.03, color='green')
    ax.axhspan(1.5, 2.0, alpha=0.03, color='yellow') 
    ax.axhspan(2.0, 3.0, alpha=0.03, color='orange')
    
    # Add performance zone labels on the right (smaller and less intrusive)
    ax.text(8.1, 1.25, 'Excellent\n(<1.5m)', fontsize=8, ha='center', 
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.6))
    ax.text(8.1, 1.75, 'Good\n(1.5-2.0m)', fontsize=8, ha='center',
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.6))
    ax.text(8.1, 2.25, 'Moderate\n(2.0-3.0m)', fontsize=8, ha='center',
           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.6))
    
    # Customize the plot
    ax.set_xlabel('Model Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Median Error (meters)', fontsize=14, fontweight='bold')
    ax.set_title('Performance vs Model Complexity: Indoor Localization Systems', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits with more space for labels
    ax.set_xlim(0.5, 8.3)
    ax.set_ylim(1.0, 2.8)
    
    # Custom x-axis labels
    complexity_ticks = [1, 2, 3, 4, 5, 6, 7, 8]
    complexity_labels = ['Simple\nk-NN', 'Basic\nML', 'Statistical\nML', 'Basic\nCNN', 
                        'Multi-path\nCNN', 'Advanced\nCNN', 'Attention\nCNN', 'Complex\nCNN']
    ax.set_xticks(complexity_ticks)
    ax.set_xticklabels(complexity_labels, fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Create legend with smaller markers and better positioning
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=8, markeredgecolor='black', label='Classical Methods'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', 
                  markersize=9, markeredgecolor='black', label='Deep Learning Methods'),
        plt.Line2D([0], [0], color='blue', linestyle='--', alpha=0.6, 
                  linewidth=2, label='Classical Trend'),
        plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.6, 
                  linewidth=2, label='Deep Learning Trend')
    ]
    
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
                      framealpha=0.9, borderpad=0.5, handletextpad=0.5, 
                      columnspacing=0.5, handlelength=1.5)
    
    # Adjust legend marker sizes to be smaller
    for handle in legend.legend_handles:
        if hasattr(handle, 'set_markersize'):
            handle.set_markersize(7)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'final_performance_vs_complexity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Final Performance vs Complexity plot saved: {output_file}")
    
    plt.show()
    
    return models_data

def print_final_summary():
    """Print final summary of the analysis"""
    
    print("\n📊 FINAL ANALYSIS SUMMARY")
    print("="*30)
    
    print("✅ Plot Improvements:")
    print("   • Smaller legend symbols (no overlap)")
    print("   • Strategic label positioning (no algorithm name overlaps)")
    print("   • Values positioned opposite to labels for clarity")
    print("   • Increased plot size for better spacing")
    print("   • Professional IEEE-style formatting")
    
    print("\n🎯 Key Findings:")
    print("   • Classical methods: 1.927m to 2.55m median error")
    print("   • Deep Learning: 1.423m to 1.598m median error")
    print("   • Best performer: Hybrid CNN+RSSI (1.423m)")
    print("   • 26.2% improvement over best classical method")
    print("   • Clear complexity-performance trade-off visible")

def main():
    """Main execution function"""
    
    print("🎯 FINAL PERFORMANCE vs COMPLEXITY ANALYSIS")
    print("Indoor Localization - Polished IEEE-Style Visualization")
    print("="*60)
    
    # Create the final polished plot
    models_data = create_final_performance_complexity_plot()
    
    # Print summary
    print_final_summary()
    
    print(f"\n✅ FINAL PLOT COMPLETE!")
    print(f"📊 Professional visualization ready for IEEE publication")

if __name__ == "__main__":
    main()
