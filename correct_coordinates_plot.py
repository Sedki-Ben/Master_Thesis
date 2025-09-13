#!/usr/bin/env python3
"""
Plot the correct train/validation/test point distribution
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_correct_coordinates():
    """Plot the corrected coordinate distribution"""
    
    # Correct coordinates as provided
    validation_points = [
        (0, 3), (2, 1), (0, 6), (5, 1), (3, 3), (4, 5), (6, 4)
    ]
    
    testing_points = [
        (0.5, 0.5), (1.5, 4.5), (2.5, 2.5), (3.5, 1.5), (5.5, 3.5)
    ]
    
    training_points = [
        (0, 0), (0, 1), (0, 2), (0, 4), (0, 5), (1, 0), (1, 1), (1, 4), (1, 5), 
        (2, 0), (2, 2), (2, 3), (2, 4), (2, 5), (3, 0), (3, 1), (3, 2), (3, 4), 
        (3, 5), (4, 0), (4, 1), (4, 4), (5, 4), (5, 0), (5, 2), (5, 3), (6, 3)
    ]
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot training points (blue circles)
    train_x = [p[0] for p in training_points]
    train_y = [p[1] for p in training_points]
    plt.scatter(train_x, train_y, c='blue', s=150, marker='o', 
               label=f'Training Points ({len(training_points)})', alpha=0.8, edgecolors='darkblue')
    
    # Plot validation points (green triangles)
    val_x = [p[0] for p in validation_points]
    val_y = [p[1] for p in validation_points]
    plt.scatter(val_x, val_y, c='green', s=200, marker='^', 
               label=f'Validation Points ({len(validation_points)})', alpha=0.8, edgecolors='darkgreen')
    
    # Plot testing points (red stars)
    test_x = [p[0] for p in testing_points]
    test_y = [p[1] for p in testing_points]
    plt.scatter(test_x, test_y, c='red', s=300, marker='*', 
               label=f'Testing Points ({len(testing_points)})', alpha=0.9, edgecolors='darkred')
    
    # Add coordinate labels for all points
    for i, (x, y) in enumerate(training_points):
        plt.annotate(f'({x},{y})', (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, ha='left', color='blue', fontweight='bold')
    
    for i, (x, y) in enumerate(validation_points):
        plt.annotate(f'({x},{y})', (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, ha='left', color='green', fontweight='bold')
    
    for i, (x, y) in enumerate(testing_points):
        plt.annotate(f'({x},{y})', (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=9, ha='left', color='red', fontweight='bold')
    
    # Customize the plot
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.xlabel('X Coordinate (meters)', fontsize=14, fontweight='bold')
    plt.ylabel('Y Coordinate (meters)', fontsize=14, fontweight='bold')
    plt.title('Indoor Localization: Correct Train/Validation/Test Point Distribution', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits and ticks
    plt.xlim(-0.5, 6.5)
    plt.ylim(-0.5, 6.5)
    plt.xticks(range(0, 7))
    plt.yticks(range(0, 7))
    
    # Add legend
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add summary text box
    summary_text = f"""Summary:
Training: {len(training_points)} points
Validation: {len(validation_points)} points  
Testing: {len(testing_points)} points
Total: {len(training_points) + len(validation_points) + len(testing_points)} points
Environment: 7×7 meter indoor lab"""
    
    plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('correct_train_val_test_distribution.png', dpi=300, bbox_inches='tight')
    print(">>> Plot saved as 'correct_train_val_test_distribution.png'")
    
    plt.show()
    
    # Print verification
    print("\n" + "="*60)
    print("COORDINATE VERIFICATION")
    print("="*60)
    print(f"Training points ({len(training_points)}):")
    print(training_points)
    print(f"\nValidation points ({len(validation_points)}):")
    print(validation_points)
    print(f"\nTesting points ({len(testing_points)}):")
    print(testing_points)
    print(f"\nTotal points: {len(training_points) + len(validation_points) + len(testing_points)}")
    
    return training_points, validation_points, testing_points

if __name__ == "__main__":
    train, val, test = plot_correct_coordinates()

