#!/usr/bin/env python3
"""
Visualize CNN Model Testing Results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the results
df = pd.read_csv('cnn_models_test_results_750_samples.csv')

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('CNN Models Performance on 750 Sample Test Dataset', fontsize=16, fontweight='bold')

# 1. Median Error Comparison
ax1.bar(df['Model'], df['Median_Error_m'], color=['#FF6B35', '#1B998B', '#2E86AB', '#A23B72'])
ax1.set_title('Median Error Comparison', fontweight='bold')
ax1.set_ylabel('Median Error (meters)')
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(df['Median_Error_m']):
    ax1.text(i, v + 0.05, f'{v:.3f}m', ha='center', fontweight='bold')

# 2. Accuracy Comparison (1m and 2m)
x = np.arange(len(df['Model']))
width = 0.35
ax2.bar(x - width/2, df['Accuracy_1m'], width, label='<1m Accuracy', color='#FF6B35', alpha=0.8)
ax2.bar(x + width/2, df['Accuracy_2m'], width, label='<2m Accuracy', color='#1B998B', alpha=0.8)
ax2.set_title('Accuracy Comparison: 1m vs 2m Thresholds', fontweight='bold')
ax2.set_ylabel('Accuracy (%)')
ax2.set_xticks(x)
ax2.set_xticklabels(df['Model'], rotation=45)
ax2.legend()

# 3. Model Complexity vs Performance
ax3.scatter(df['Parameters'], df['Median_Error_m'], s=100, color=['#FF6B35', '#1B998B', '#2E86AB', '#A23B72'])
for i, model in enumerate(df['Model']):
    ax3.annotate(model, (df['Parameters'].iloc[i], df['Median_Error_m'].iloc[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax3.set_title('Model Complexity vs Performance', fontweight='bold')
ax3.set_xlabel('Number of Parameters')
ax3.set_ylabel('Median Error (meters)')
ax3.grid(True, alpha=0.3)

# 4. Accuracy Across Multiple Thresholds
thresholds = ['Accuracy_50cm', 'Accuracy_1m', 'Accuracy_2m', 'Accuracy_3m']
threshold_labels = ['50cm', '1m', '2m', '3m']
colors = ['#FF6B35', '#1B998B', '#2E86AB', '#A23B72']

for i, model in enumerate(df['Model']):
    values = [df[threshold].iloc[i] for threshold in thresholds]
    ax4.plot(threshold_labels, values, marker='o', linewidth=2, label=model, color=colors[i])

ax4.set_title('Accuracy Across Different Distance Thresholds', fontweight='bold')
ax4.set_ylabel('Accuracy (%)')
ax4.set_xlabel('Distance Threshold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cnn_models_test_results_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print(">>> Visualization saved as 'cnn_models_test_results_visualization.png'")

# Print summary table
print("\n" + "="*80)
print("FINAL CNN MODELS TEST RESULTS SUMMARY")
print("="*80)
print(f"{'Model':<18} {'Median (m)':<10} {'<1m (%)':<8} {'<2m (%)':<8} {'Parameters':<10}")
print("-" * 80)

for _, row in df.iterrows():
    print(f"{row['Model']:<18} {row['Median_Error_m']:<10.3f} "
          f"{row['Accuracy_1m']:<8.1f} {row['Accuracy_2m']:<8.1f} "
          f"{row['Parameters']:<10,}")

best_model = df.loc[df['Median_Error_m'].idxmin()]
print(f"\n>>> BEST MODEL: {best_model['Model']}")
print(f"    Median Error: {best_model['Median_Error_m']:.3f} meters")
print(f"    Accuracy <1m: {best_model['Accuracy_1m']:.1f}%")
print(f"    Accuracy <2m: {best_model['Accuracy_2m']:.1f}%")
print(f"    Parameters: {best_model['Parameters']:,}")

