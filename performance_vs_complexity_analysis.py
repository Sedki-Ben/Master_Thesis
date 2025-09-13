#!/usr/bin/env python3
"""
Performance vs Model Complexity Analysis for Indoor Localization

Creates an IEEE-style plot showing the trade-off between model complexity 
and localization performance (median error) across different approaches:
- Classical methods: k-NN, IDW, Probabilistic Fingerprinting
- Deep Learning: Various CNN architectures

This visualization is crucial for understanding the cost-benefit analysis
of different localization approaches.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_performance_complexity_plot():
    """Create performance vs complexity analysis plot"""
    
    print("📊 Creating Performance vs Model Complexity Analysis")
    print("="*55)
    
    # Set up the plot style for IEEE publications
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Define model data
    # Complexity is subjective but follows: simpler algorithms < complex algorithms
    # We'll use a logarithmic-like scale for complexity
    
    models_data = {
        # Classical Methods (Lower complexity)
        'k-NN (k=3)': {'complexity': 1, 'median_error': 2.55, 'category': 'Classical', 'color': '#1f77b4'},
        'k-NN (k=5)': {'complexity': 1.2, 'median_error': 2.509, 'category': 'Classical', 'color': '#1f77b4'},
        'k-NN (k=9)': {'complexity': 1.5, 'median_error': 2.373, 'category': 'Classical', 'color': '#1f77b4'},
        'IDW (power=4)': {'complexity': 2, 'median_error': 2.327, 'category': 'Classical', 'color': '#ff7f0e'},
        'Probabilistic': {'complexity': 3, 'median_error': 1.927, 'category': 'Classical', 'color': '#2ca02c'},
        
        # Deep Learning Methods (Higher complexity)
        'Basic CNN': {'complexity': 5, 'median_error': 1.598, 'category': 'Deep Learning', 'color': '#d62728'},
        'Residual CNN': {'complexity': 6, 'median_error': 1.578, 'category': 'Deep Learning', 'color': '#9467bd'},
        'Multi-Scale CNN': {'complexity': 7, 'median_error': 1.567, 'category': 'Deep Learning', 'color': '#8c564b'},
        'Attention CNN': {'complexity': 8, 'median_error': 1.498, 'category': 'Deep Learning', 'color': '#e377c2'},
        'Hybrid CNN + RSSI': {'complexity': 9, 'median_error': 1.423, 'category': 'Deep Learning', 'color': '#7f7f7f'}
    }
    
    # Separate data by category
    classical_models = {k: v for k, v in models_data.items() if v['category'] == 'Classical'}
    dl_models = {k: v for k, v in models_data.items() if v['category'] == 'Deep Learning'}
    
    # Plot classical methods
    classical_x = [data['complexity'] for data in classical_models.values()]
    classical_y = [data['median_error'] for data in classical_models.values()]
    classical_colors = [data['color'] for data in classical_models.values()]
    classical_labels = list(classical_models.keys())
    
    # Plot deep learning methods  
    dl_x = [data['complexity'] for data in dl_models.values()]
    dl_y = [data['median_error'] for data in dl_models.values()]
    dl_colors = [data['color'] for data in dl_models.values()]
    dl_labels = list(dl_models.keys())
    
    # Create scatter plots
    scatter1 = ax.scatter(classical_x, classical_y, c=classical_colors, s=120, 
                         marker='o', alpha=0.8, edgecolors='black', linewidth=1.5,
                         label='Classical Methods')
    
    scatter2 = ax.scatter(dl_x, dl_y, c=dl_colors, s=150, 
                         marker='^', alpha=0.8, edgecolors='black', linewidth=1.5,
                         label='Deep Learning Methods')
    
    # Add model labels
    for i, (model, data) in enumerate(classical_models.items()):
        # Adjust label positions to avoid overlap
        offset_x = 0.1
        offset_y = 0.05 if i % 2 == 0 else -0.08
        
        ax.annotate(model, 
                   (data['complexity'], data['median_error']),
                   xytext=(data['complexity'] + offset_x, data['median_error'] + offset_y),
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
    
    for i, (model, data) in enumerate(dl_models.items()):
        # Adjust label positions to avoid overlap
        offset_x = 0.1
        offset_y = 0.05 if i % 2 == 0 else -0.08
        
        ax.annotate(model, 
                   (data['complexity'], data['median_error']),
                   xytext=(data['complexity'] + offset_x, data['median_error'] + offset_y),
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6))
    
    # Add trend lines
    # Classical methods trend
    classical_x_sorted, classical_y_sorted = zip(*sorted(zip(classical_x, classical_y)))
    ax.plot(classical_x_sorted, classical_y_sorted, '--', color='blue', alpha=0.5, linewidth=2, 
           label='Classical Trend')
    
    # Deep learning methods trend
    dl_x_sorted, dl_y_sorted = zip(*sorted(zip(dl_x, dl_y)))
    ax.plot(dl_x_sorted, dl_y_sorted, '--', color='red', alpha=0.5, linewidth=2,
           label='Deep Learning Trend')
    
    # Performance zones
    ax.axhspan(0, 1.5, alpha=0.1, color='green', label='Excellent Performance (<1.5m)')
    ax.axhspan(1.5, 2.0, alpha=0.1, color='yellow', label='Good Performance (1.5-2.0m)')
    ax.axhspan(2.0, 3.0, alpha=0.1, color='orange', label='Moderate Performance (2.0-3.0m)')
    
    # Customize the plot
    ax.set_xlabel('Model Complexity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Median Error (meters)', fontsize=14, fontweight='bold')
    ax.set_title('Performance vs Model Complexity\nIndoor Localization Systems', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits and ticks
    ax.set_xlim(0, 10)
    ax.set_ylim(1.0, 3.0)
    
    # Custom x-axis labels
    complexity_labels = ['', 'Simple\n(k-NN)', '', 'Classical\nML', '', 'Basic\nDL', '', 'Advanced\nDL', '', 'Complex\nDL']
    ax.set_xticks(range(10))
    ax.set_xticklabels(complexity_labels, fontsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    legend1 = ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Add statistical annotations
    best_classical = min(classical_y)
    best_dl = min(dl_y)
    improvement = ((best_classical - best_dl) / best_classical) * 100
    
    text_box = f"""Key Insights:
    
• Best Classical: {best_classical:.3f}m (Probabilistic)
• Best Deep Learning: {best_dl:.3f}m (Hybrid CNN+RSSI)
• Improvement: {improvement:.1f}% error reduction
• Complexity Trade-off: {best_dl/best_classical:.2f}× better performance
  at ~3× computational complexity"""
    
    ax.text(0.02, 0.98, text_box, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', horizontalalignment='left',
           bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'performance_vs_complexity_indoor_localization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Performance vs Complexity plot saved: {output_file}")
    
    plt.show()
    
    return models_data

def analyze_complexity_performance_relationship():
    """Analyze the relationship between complexity and performance"""
    
    print("\n🔍 COMPLEXITY-PERFORMANCE ANALYSIS")
    print("="*40)
    
    # Classical methods analysis
    classical_results = {
        'k-NN (k=3)': 2.55,
        'k-NN (k=5)': 2.509, 
        'k-NN (k=9)': 2.373,
        'IDW (power=4)': 2.327,
        'Probabilistic': 1.927
    }
    
    # Deep learning results
    dl_results = {
        'Basic CNN': 1.598,
        'Residual CNN': 1.578,
        'Multi-Scale CNN': 1.567,
        'Attention CNN': 1.498,
        'Hybrid CNN + RSSI': 1.423
    }
    
    print("📊 Classical Methods Performance:")
    for method, error in classical_results.items():
        print(f"   {method:20s}: {error:.3f}m")
    
    print(f"\n   Best Classical: Probabilistic ({min(classical_results.values()):.3f}m)")
    print(f"   Worst Classical: k-NN k=3 ({max(classical_results.values()):.3f}m)")
    print(f"   Classical Range: {max(classical_results.values()) - min(classical_results.values()):.3f}m")
    
    print("\n🧠 Deep Learning Methods Performance:")
    for method, error in dl_results.items():
        print(f"   {method:20s}: {error:.3f}m")
    
    print(f"\n   Best DL: Hybrid CNN+RSSI ({min(dl_results.values()):.3f}m)")
    print(f"   Worst DL: Basic CNN ({max(dl_results.values()):.3f}m)")
    print(f"   DL Range: {max(dl_results.values()) - min(dl_results.values()):.3f}m")
    
    # Overall analysis
    best_classical = min(classical_results.values())
    best_dl = min(dl_results.values())
    improvement = ((best_classical - best_dl) / best_classical) * 100
    
    print(f"\n🎯 OVERALL PERFORMANCE ANALYSIS:")
    print(f"   Best Overall: {min(best_classical, best_dl):.3f}m")
    print(f"   DL vs Classical Improvement: {improvement:.1f}%")
    print(f"   Performance Gain: {best_classical - best_dl:.3f}m")
    
    # Complexity insights
    print(f"\n⚖️  COMPLEXITY-PERFORMANCE TRADE-OFF:")
    print(f"   • Classical methods: Simple, interpretable, fast")
    print(f"   • Deep Learning: Complex, requires training, higher accuracy")
    print(f"   • Sweet Spot: Hybrid CNN+RSSI balances complexity and performance")
    print(f"   • ROI: {improvement:.1f}% accuracy improvement for ~3x complexity increase")

def main():
    """Main execution function"""
    
    print("🎯 PERFORMANCE vs COMPLEXITY ANALYSIS")
    print("Indoor Localization Systems Comparison")
    print("="*50)
    
    # Create the main plot
    models_data = create_performance_complexity_plot()
    
    # Perform detailed analysis
    analyze_complexity_performance_relationship()
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📊 Generated comprehensive performance vs complexity visualization")
    print(f"🎯 Key finding: Deep Learning achieves 26.2% better accuracy than classical methods")

if __name__ == "__main__":
    main()


