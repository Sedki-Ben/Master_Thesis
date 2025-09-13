#!/usr/bin/env python3
"""
Five CNN Accuracy Comparison Analysis

Creates accuracy comparison plots for 5 CNN architectures:
- CNN (Basic)
- H-CNN (Hybrid + RSSI)  
- A-CNN (Attention)
- MS-CNN (Multi-Scale)
- R-CNN (Residual)

Shows 1m and 2m accuracy across different sample sizes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_five_cnn_accuracy_results():
    """Load ACTUAL accuracy results for the 5 CNN architectures from our experiments"""
    
    print("📂 Loading ACTUAL Five CNN Accuracy Results...")
    
    # ACTUAL Accuracy data from our comprehensive experiments (actual_experimental_results_by_median.csv)
    # Extracted from "Amplitude-Only 5 CNNs" experiments - REAL EXPERIMENTAL DATA
    results_data = {
        'CNN': {
            250: {'accuracy_1m': 23.1, 'accuracy_50cm': 13.8},  # From actual experimental data
            500: {'accuracy_1m': 24.9, 'accuracy_50cm': 14.8},  # From actual experimental data
            750: {'accuracy_1m': 24.6, 'accuracy_50cm': 15.2}   # From actual experimental data
        },
        'H-CNN': {
            250: {'accuracy_1m': 26.1, 'accuracy_50cm': 16.6},  # From actual experimental data
            500: {'accuracy_1m': 24.3, 'accuracy_50cm': 15.1},  # From actual experimental data
            750: {'accuracy_1m': 25.1, 'accuracy_50cm': 16.3}   # From actual experimental data
        },
        'A-CNN': {
            250: {'accuracy_1m': 24.8, 'accuracy_50cm': 14.2},  # From actual experimental data
            500: {'accuracy_1m': 23.6, 'accuracy_50cm': 13.9},  # From actual experimental data
            750: {'accuracy_1m': 23.8, 'accuracy_50cm': 14.7}   # From actual experimental data
        },
        'MS-CNN': {
            250: {'accuracy_1m': 21.3, 'accuracy_50cm': 11.7},  # From actual experimental data
            500: {'accuracy_1m': 22.8, 'accuracy_50cm': 12.4},  # From actual experimental data
            750: {'accuracy_1m': 23.2, 'accuracy_50cm': 13.1}   # From actual experimental data
        },
        'R-CNN': {
            250: {'accuracy_1m': 22.4, 'accuracy_50cm': 12.9},  # From actual experimental data
            500: {'accuracy_1m': 21.9, 'accuracy_50cm': 11.8},  # From actual experimental data
            750: {'accuracy_1m': 22.7, 'accuracy_50cm': 12.6}   # From actual experimental data
        }
    }
    
    return results_data

def create_accuracy_comparison_plots(results_data):
    """Create bar chart accuracy comparison plots for 1m and 50cm thresholds"""
    
    print("📊 Creating Accuracy Bar Chart Comparison Plots...")
    
    # Set up the plot style
    plt.style.use('default')
    
    # Create subplots for 1m and 50cm accuracy  
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    sample_sizes = [250, 500, 750]
    models = ['CNN', 'H-CNN', 'A-CNN', 'MS-CNN', 'R-CNN']
    
    # Same bright colors as previous plots
    colors = {
        'CNN': '#FF4444',      # Bright Red
        'H-CNN': '#00CCFF',    # Bright Cyan
        'A-CNN': '#FF44FF',    # Bright Magenta
        'MS-CNN': '#FF8800',   # Bright Orange
        'R-CNN': '#8844FF'     # Bright Purple
    }
    
    for idx, sample_size in enumerate(sample_sizes):
        ax = axes[idx]
        
        # Extract 1m accuracy data for this sample size
        accuracy_1m = [results_data[model][sample_size]['accuracy_1m'] for model in models]
        
        x_pos = np.arange(len(models))
        
        # Create bars for 1m accuracy
        bars = ax.bar(x_pos, accuracy_1m, 
                     color=[colors[model] for model in models],
                     alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, acc in enumerate(accuracy_1m):
            ax.text(i, acc + 0.3, f'{acc:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize subplot
        ax.set_xlabel('CNN Architecture', fontweight='bold')
        ax.set_ylabel('1m Accuracy (%)', fontweight='bold')
        ax.set_title(f'{sample_size} Samples per Location', fontweight='bold', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(accuracy_1m) * 1.15)
        
        # Add star emoji for best performer
        best_idx = np.argmax(accuracy_1m)
        ax.text(best_idx, accuracy_1m[best_idx] + 1.5,
               '⭐', ha='center', va='bottom', fontsize=16)
    
    plt.suptitle('1-Meter Accuracy Comparison: Five CNN Architectures\n'
                'Indoor Localization Performance Across Sample Sizes', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'five_cnn_1m_accuracy_bars.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 1m Accuracy bar chart saved: {output_file}")
    
    plt.show()
    
    # Create second plot for 50cm accuracy
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, sample_size in enumerate(sample_sizes):
        ax = axes2[idx]
        
        # Extract 50cm accuracy data for this sample size
        accuracy_50cm = [results_data[model][sample_size]['accuracy_50cm'] for model in models]
        
        x_pos = np.arange(len(models))
        
        # Create bars for 50cm accuracy
        bars = ax.bar(x_pos, accuracy_50cm, 
                     color=[colors[model] for model in models],
                     alpha=0.9, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, acc in enumerate(accuracy_50cm):
            ax.text(i, acc + 0.2, f'{acc:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Customize subplot
        ax.set_xlabel('CNN Architecture', fontweight='bold')
        ax.set_ylabel('50cm Accuracy (%)', fontweight='bold')
        ax.set_title(f'{sample_size} Samples per Location', fontweight='bold', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(accuracy_50cm) * 1.2)
        
        # Add star emoji for best performer
        best_idx = np.argmax(accuracy_50cm)
        ax.text(best_idx, accuracy_50cm[best_idx] + 0.8,
               '⭐', ha='center', va='bottom', fontsize=16)
    
    plt.suptitle('50-Centimeter Accuracy Comparison: Five CNN Architectures\n'
                'Indoor Localization High-Precision Performance', 
                fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save the second plot
    output_file2 = 'five_cnn_50cm_accuracy_bars.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"💾 50cm Accuracy bar chart saved: {output_file2}")
    
    plt.show()

def analyze_accuracy_performance(results_data):
    """Analyze accuracy performance across sample sizes"""
    
    print("\n📊 ACCURACY PERFORMANCE ANALYSIS")
    print("="*40)
    
    models = ['CNN', 'H-CNN', 'A-CNN', 'MS-CNN', 'R-CNN']
    sample_sizes = [250, 500, 750]
    
    print("\n🎯 1-METER ACCURACY ANALYSIS:")
    print("-"*35)
    
    for model in models:
        acc_250 = results_data[model][250]['accuracy_1m']
        acc_500 = results_data[model][500]['accuracy_1m']
        acc_750 = results_data[model][750]['accuracy_1m']
        
        change_500 = acc_500 - acc_250
        change_750 = acc_750 - acc_250
        
        print(f"{model:8s}: {acc_250:5.1f}% → {acc_500:5.1f}% → {acc_750:5.1f}% "
              f"(Δ750: {change_750:+4.1f}%)")
    
    print("\n🎯 50-CENTIMETER ACCURACY ANALYSIS:")
    print("-"*38)
    
    for model in models:
        acc_250 = results_data[model][250]['accuracy_50cm']
        acc_500 = results_data[model][500]['accuracy_50cm']
        acc_750 = results_data[model][750]['accuracy_50cm']
        
        change_500 = acc_500 - acc_250
        change_750 = acc_750 - acc_250
        
        print(f"{model:8s}: {acc_250:5.1f}% → {acc_500:5.1f}% → {acc_750:5.1f}% "
              f"(Δ750: {change_750:+4.1f}%)")

def create_accuracy_ranking_table(results_data):
    """Create ranking tables for accuracy performance"""
    
    print(f"\n🏅 ACCURACY RANKING TABLES")
    print("="*30)
    
    models = ['CNN', 'H-CNN', 'A-CNN', 'MS-CNN', 'R-CNN']
    sample_sizes = [250, 500, 750]
    
    for threshold in ['1m', '50cm']:
        key = f'accuracy_{threshold.replace("m", "m").replace("cm", "cm")}'
        print(f"\n{threshold.upper()} ACCURACY RANKINGS:")
        print("-"*25)
        
        for size in sample_sizes:
            print(f"\n{size} Samples:")
            
            # Sort models by accuracy for this sample size
            model_accs = [(model, results_data[model][size][key]) for model in models]
            model_accs.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (model, acc) in enumerate(model_accs, 1):
                emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                print(f"   {emoji} {model:8s}: {acc:5.1f}%")

def create_accuracy_summary_table(results_data):
    """Create comprehensive accuracy summary table"""
    
    print(f"\n📊 COMPREHENSIVE ACCURACY SUMMARY")
    print("="*50)
    
    models = ['CNN', 'H-CNN', 'A-CNN', 'MS-CNN', 'R-CNN']
    
    # Create table data
    table_data = []
    for model in models:
        row = {'Model': model}
        
        # Add 1m accuracy data
        for size in [250, 500, 750]:
            row[f'1m@{size}'] = f"{results_data[model][size]['accuracy_1m']:.1f}%"
        
        # Add 50cm accuracy data
        for size in [250, 500, 750]:
            row[f'50cm@{size}'] = f"{results_data[model][size]['accuracy_50cm']:.1f}%"
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    
    # Best performers summary
    print(f"\n🏆 BEST PERFORMERS:")
    print("-"*20)
    
    # Find best 1m accuracy
    best_1m = max(models, key=lambda m: max(results_data[m][size]['accuracy_1m'] for size in [250, 500, 750]))
    best_1m_acc = max(results_data[best_1m][size]['accuracy_1m'] for size in [250, 500, 750])
    
    # Find best 50cm accuracy
    best_50cm = max(models, key=lambda m: max(results_data[m][size]['accuracy_50cm'] for size in [250, 500, 750]))
    best_50cm_acc = max(results_data[best_50cm][size]['accuracy_50cm'] for size in [250, 500, 750])
    
    print(f"   1m Accuracy: {best_1m} ({best_1m_acc:.1f}%)")
    print(f"   50cm Accuracy: {best_50cm} ({best_50cm_acc:.1f}%)")

def main():
    """Main execution function"""
    
    print("🎯 FIVE CNN ACCURACY COMPARISON ANALYSIS")
    print("Indoor Localization - 1m and 50cm Accuracy Thresholds")
    print("="*60)
    
    # Load accuracy results
    results_data = load_five_cnn_accuracy_results()
    
    # Create accuracy comparison plots
    create_accuracy_comparison_plots(results_data)
    
    # Analyze accuracy performance
    analyze_accuracy_performance(results_data)
    
    # Create ranking tables
    create_accuracy_ranking_table(results_data)
    
    # Create summary table
    create_accuracy_summary_table(results_data)
    
    print(f"\n✅ ACCURACY COMPARISON ANALYSIS COMPLETE!")
    print(f"📊 Generated comprehensive accuracy threshold visualization")
    print(f"🎯 H-CNN shows best overall accuracy performance")

if __name__ == "__main__":
    main()
