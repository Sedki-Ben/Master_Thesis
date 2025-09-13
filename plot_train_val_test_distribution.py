#!/usr/bin/env python3
"""
Plot Train/Validation/Test Point Distribution

Creates a spatial distribution plot showing the 27/7/5 train/val/test split
using different shapes and colors for each group.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_train_val_test_distribution():
    """Plot the spatial distribution of train/validation/test points"""
    
    print(f"📊 Creating train/validation/test point distribution plot...")
    
    # Define the point splits (from our comprehensive evaluation)
    test_points = [(0.5, 0.5), (1.5, 4.5), (2.5, 2.5), (3.5, 1.5), (5.5, 3.5)]
    validation_points = [(4, 5), (5, 1), (0, 3), (0, 6), (6, 4), (2, 1), (3, 3)]
    
    # All reference points
    all_reference_points = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
        (1, 0), (1, 1), (1, 4), (1, 5),
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
        (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
        (4, 0), (4, 1), (4, 4), (4, 5),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
        (6, 3), (6, 4)
    ]
    
    # Training points (remaining after removing validation points)
    training_points = [p for p in all_reference_points if p not in validation_points]
    
    print(f"   📍 Training points: {len(training_points)}")
    print(f"   📍 Validation points: {len(validation_points)}")
    print(f"   📍 Test points: {len(test_points)}")
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Plot training points (blue circles)
    train_x = [p[0] for p in training_points]
    train_y = [p[1] for p in training_points]
    scatter_train = plt.scatter(train_x, train_y, 
                               s=600, c='blue', alpha=0.8, 
                               edgecolors='black', linewidth=2.5,
                               marker='o', label='Training Points')
    
    # Plot validation points (green triangles)
    val_x = [p[0] for p in validation_points]
    val_y = [p[1] for p in validation_points]
    scatter_val = plt.scatter(val_x, val_y, 
                             s=600, c='green', alpha=0.8, 
                             edgecolors='black', linewidth=2.5,
                             marker='^', label='Validation Points')
    
    # Plot test points (red squares)
    test_x = [p[0] for p in test_points]
    test_y = [p[1] for p in test_points]
    scatter_test = plt.scatter(test_x, test_y, 
                              s=600, c='red', alpha=0.8, 
                              edgecolors='black', linewidth=2.5,
                              marker='s', label='Test Points')
    
    # No coordinate labels on points - just shapes
    
    # Set up the plot
    plt.xlabel('X Coordinate (meters)', fontsize=14, fontweight='bold')
    plt.ylabel('Y Coordinate (meters)', fontsize=14, fontweight='bold')
    plt.title('Train/Validation/Test Point Distribution\n27 Training / 7 Validation / 5 Test Points', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits and ticks
    plt.xlim(-0.5, 6.5)
    plt.ylim(-0.5, 6.5)
    plt.xticks(range(7), fontsize=12)
    plt.yticks(range(7), fontsize=12)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Create legend with proper shapes
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Training Points (27)',
                   markerfacecolor='blue', markersize=15, markeredgecolor='black',
                   markeredgewidth=2),
        plt.Line2D([0], [0], marker='^', color='w', label='Validation Points (7)',
                   markerfacecolor='green', markersize=15, markeredgecolor='black',
                   markeredgewidth=2),
        plt.Line2D([0], [0], marker='s', color='w', label='Test Points (5)',
                   markerfacecolor='red', markersize=15, markeredgecolor='black',
                   markeredgewidth=2)
    ]
    
    plt.legend(handles=legend_elements, loc='upper right', fontsize=14, 
              frameon=True, fancybox=True, shadow=True, 
              bbox_to_anchor=(0.98, 0.98))
    
    # Make the plot look professional
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('train_val_test_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   ✅ Plot saved: train_val_test_distribution.png")
    
    # Print summary
    print(f"\n📋 POINT DISTRIBUTION SUMMARY:")
    print(f"   🔵 Training Points ({len(training_points)}): {training_points}")
    print(f"   🔺 Validation Points ({len(validation_points)}): {validation_points}")
    print(f"   🔴 Test Points ({len(test_points)}): {test_points}")
    
    print(f"\n📊 SPATIAL CHARACTERISTICS:")
    print(f"   • Training points cover the full grid systematically")
    print(f"   • Validation points are distributed across different areas")
    print(f"   • Test points are at half-meter coordinates (interpolation challenge)")
    print(f"   • No overlap between sets (proper evaluation)")

def main():
    """Main execution function"""
    print(f"📊 TRAIN/VALIDATION/TEST POINT DISTRIBUTION PLOT")
    print(f"="*55)
    
    plot_train_val_test_distribution()
    
    print(f"\n✅ Plot generation complete!")
    print(f"   📍 Shows spatial distribution of 27/7/5 split")
    print(f"   🎨 Blue circles: Training | Green triangles: Validation | Red squares: Test")

if __name__ == "__main__":
    main()
