#!/usr/bin/env python3
"""
Plot ground truth grid with training/validation/testing points,
plus 5 random predictions for each test point from the best HybridCNN model,
with 1m and 2m accuracy circles.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    print(">>> TensorFlow imported successfully")
except ImportError:
    print("ERROR: TensorFlow not found. Please install with: pip install tensorflow")
    exit(1)

# Import correct coordinates
from coordinates_config import get_training_points, get_validation_points, get_testing_points

class GroundTruthPredictionVisualizer:
    """Visualize ground truth vs predictions with accuracy analysis"""
    
    def __init__(self):
        self.training_points = get_training_points()
        self.validation_points = get_validation_points()
        self.testing_points = get_testing_points()
        
        print(f"🎯 Ground Truth vs Prediction Visualizer Initialized")
        print(f"📊 Training points: {len(self.training_points)}")
        print(f"📊 Validation points: {len(self.validation_points)}")
        print(f"📊 Testing points: {len(self.testing_points)}")
        
    def load_test_data(self, dataset_size=250):
        """Load test data for the specified dataset size"""
        print(f">>> Loading test data for {dataset_size} samples...")
        
        # Load training data to fit scalers (same as model training)
        train_data = self.load_data_by_coordinates(dataset_size, "training")
        test_data = self.load_data_by_coordinates(750, "testing")  # Always 750 for testing
        
        # Fit scalers on training data (same as model training)
        amp_scaler = StandardScaler()
        phase_scaler = StandardScaler()
        rssi_scaler = StandardScaler()
        coord_scaler = MinMaxScaler(feature_range=(0, 1))
        
        amp_scaler.fit(train_data['amplitudes'])
        phase_scaler.fit(train_data['phases'])
        rssi_scaler.fit(train_data['rssi'].reshape(-1, 1))
        coord_scaler.fit(train_data['coordinates'])
        
        # Transform test data
        test_amp_norm = amp_scaler.transform(test_data['amplitudes'])
        test_phase_norm = phase_scaler.transform(test_data['phases'])
        test_rssi_norm = rssi_scaler.transform(test_data['rssi'].reshape(-1, 1)).flatten()
        
        # Create CNN format for HybridCNN (needs both CSI and RSSI)
        test_csi = np.stack([test_amp_norm, test_phase_norm], axis=1)
        
        return {
            'test_csi': test_csi,
            'test_rssi': test_rssi_norm,
            'original_coords': test_data['coordinates'],
            'coord_scaler': coord_scaler,
            'raw_data': test_data
        }
    
    def load_data_by_coordinates(self, dataset_size, point_type="training"):
        """Load data for specific coordinates and dataset size"""
        
        if point_type == "training":
            points = self.training_points
            folder = f"CSI Dataset {dataset_size} Samples"
        elif point_type == "validation":
            points = self.validation_points  
            folder = f"CSI Dataset {dataset_size} Samples"
        elif point_type == "testing":
            points = self.testing_points
            folder = "Testing Points Dataset 750 Samples"
        else:
            raise ValueError("point_type must be 'training', 'validation', or 'testing'")
        
        amplitudes, phases, rssi_values, coordinates = [], [], [], []
        
        for x, y in points:
            file_path = Path(folder) / f"{x},{y}.csv"
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            amps = json.loads(row['amplitude'])
                            phases_data = json.loads(row['phase'])
                            rssi = float(row['rssi'])
                            
                            if len(amps) == 52 and len(phases_data) == 52:
                                amplitudes.append(amps)
                                phases.append(phases_data)
                                rssi_values.append(rssi)
                                coordinates.append([x, y])
                        except:
                            continue
            else:
                print(f"    Warning: File not found: {file_path}")
        
        return {
            'amplitudes': np.array(amplitudes),
            'phases': np.array(phases),
            'rssi': np.array(rssi_values),
            'coordinates': np.array(coordinates)
        }
    
    def load_best_model(self):
        """Load the best HybridCNN model"""
        model_path = Path("the last samurai/tom cruise/hybridcnn_improved_250_samples_improved.h5")
        
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return None
        
        try:
            # Try loading without compilation first
            model = keras.models.load_model(model_path, compile=False)
            print(f"✅ Successfully loaded HybridCNN model: {model_path.name}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            try:
                model = keras.models.load_model(model_path)
                print(f"✅ Successfully loaded HybridCNN model (with compilation): {model_path.name}")
                return model
            except Exception as e2:
                print(f"❌ Error loading model (second attempt): {e2}")
                return None
    
    def get_random_predictions(self, model, test_data, num_predictions=5):
        """Get random predictions for each test point"""
        print(f">>> Getting {num_predictions} random predictions for each test point...")
        
        predictions_by_point = {}
        
        # Group test data by coordinates
        unique_coords = np.unique(test_data['original_coords'], axis=0)
        
        for coord in unique_coords:
            coord_tuple = tuple(coord)
            
            # Find all samples for this coordinate
            mask = np.all(test_data['original_coords'] == coord, axis=1)
            coord_indices = np.where(mask)[0]
            
            if len(coord_indices) >= num_predictions:
                # Randomly select samples
                random.seed(42)  # For reproducibility
                selected_indices = random.sample(list(coord_indices), num_predictions)
            else:
                # Use all available samples
                selected_indices = list(coord_indices)
            
            # Get predictions for selected samples
            selected_csi = test_data['test_csi'][selected_indices]
            selected_rssi = test_data['test_rssi'][selected_indices]
            
            # Make predictions
            predictions_norm = model.predict([selected_csi, selected_rssi], verbose=0)
            predictions_denorm = test_data['coord_scaler'].inverse_transform(predictions_norm)
            
            predictions_by_point[coord_tuple] = {
                'predictions': predictions_denorm,
                'ground_truth': coord,
                'sample_indices': selected_indices,
                'num_samples': len(selected_indices)
            }
            
            # Calculate errors
            errors = np.sqrt(np.sum((coord - predictions_denorm)**2, axis=1))
            mean_error = np.mean(errors)
            
            print(f"    Point {coord}: {len(selected_indices)} predictions, mean error: {mean_error:.3f}m")
        
        return predictions_by_point
    
    def create_comprehensive_plot(self, predictions_by_point):
        """Create comprehensive plot with ground truth and predictions"""
        print(">>> Creating comprehensive visualization...")
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        
        # Define colors and markers
        colors = {
            'training': '#1f77b4',      # Blue
            'validation': '#ff7f0e',    # Orange  
            'testing': '#2ca02c',       # Green
            'predictions': ['#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']  # Different colors for each prediction
        }
        
        # Plot training points
        train_x = [p[0] for p in self.training_points]
        train_y = [p[1] for p in self.training_points]
        ax.scatter(train_x, train_y, c=colors['training'], marker='s', s=80, 
                  label=f'Training Points ({len(self.training_points)})', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Plot validation points
        val_x = [p[0] for p in self.validation_points]
        val_y = [p[1] for p in self.validation_points]
        ax.scatter(val_x, val_y, c=colors['validation'], marker='^', s=100, 
                  label=f'Validation Points ({len(self.validation_points)})', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Plot testing points (ground truth)
        test_x = [p[0] for p in self.testing_points]
        test_y = [p[1] for p in self.testing_points]
        ax.scatter(test_x, test_y, c=colors['testing'], marker='*', s=200, 
                  label=f'Testing Points ({len(self.testing_points)})', alpha=1.0, edgecolor='black', linewidth=2)
        
        # Plot predictions for each test point
        legend_added = {}
        for i, (coord_tuple, pred_info) in enumerate(predictions_by_point.items()):
            ground_truth = pred_info['ground_truth']
            predictions = pred_info['predictions']
            
            # Plot each prediction
            for j, pred in enumerate(predictions):
                color = colors['predictions'][j % len(colors['predictions'])]
                marker = 'o'
                
                # Add to legend only once per prediction number
                label = f'Prediction {j+1}' if j not in legend_added else None
                if label:
                    legend_added[j] = True
                
                ax.scatter(pred[0], pred[1], c=color, marker=marker, s=50, 
                          alpha=0.8, label=label, edgecolor='white', linewidth=1)
                
                # Draw line from ground truth to prediction
                ax.plot([ground_truth[0], pred[0]], [ground_truth[1], pred[1]], 
                       color=color, alpha=0.5, linewidth=1, linestyle='--')
        
        # Draw accuracy circles for each test point
        circle_colors = ['red', 'orange']
        circle_radii = [1.0, 2.0]
        circle_labels = ['1m Accuracy', '2m Accuracy']
        
        for coord_tuple in predictions_by_point.keys():
            ground_truth = predictions_by_point[coord_tuple]['ground_truth']
            
            for radius, color, label in zip(circle_radii, circle_colors, circle_labels):
                circle = plt.Circle((ground_truth[0], ground_truth[1]), radius, 
                                  fill=False, color=color, linewidth=2, alpha=0.7)
                ax.add_patch(circle)
        
        # Add accuracy circle labels (only once)
        for radius, color, label in zip(circle_radii, circle_colors, circle_labels):
            ax.plot([], [], color=color, linewidth=2, label=label)
        
        # Customize plot
        ax.set_xlim(-0.5, 7)
        ax.set_ylim(-0.5, 7)
        ax.set_xlabel('X Coordinate (meters)', fontsize=12)
        ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
        ax.set_title('Ground Truth vs HybridCNN Predictions (Best Model: 1.193m Median Error)\n'
                    'Training/Validation/Testing Points with 5 Random Predictions per Test Point', 
                    fontsize=14, fontweight='bold')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add statistics text
        total_predictions = sum(len(pred_info['predictions']) for pred_info in predictions_by_point.values())
        stats_text = f"Model: HybridCNN (250 samples)\n"
        stats_text += f"Median Error: 1.193m\n"
        stats_text += f"Accuracy <1m: 45.5%\n"
        stats_text += f"Accuracy <2m: 66.1%\n"
        stats_text += f"Total Predictions: {total_predictions}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_path = "ground_truth_vs_hybridcnn_predictions_with_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {output_path}")
        
        plt.show()
        return output_path
    
    def calculate_prediction_statistics(self, predictions_by_point):
        """Calculate and display prediction statistics"""
        print("\n" + "="*80)
        print("📊 PREDICTION ANALYSIS STATISTICS")
        print("="*80)
        
        all_errors = []
        
        for coord_tuple, pred_info in predictions_by_point.items():
            ground_truth = pred_info['ground_truth']
            predictions = pred_info['predictions']
            
            # Calculate errors for this point
            errors = np.sqrt(np.sum((ground_truth - predictions)**2, axis=1))
            all_errors.extend(errors)
            
            print(f"\n📍 Test Point {ground_truth}:")
            print(f"   Predictions: {len(predictions)}")
            print(f"   Mean Error: {np.mean(errors):.3f}m")
            print(f"   Min Error: {np.min(errors):.3f}m")
            print(f"   Max Error: {np.max(errors):.3f}m")
            print(f"   <1m: {np.mean(errors < 1.0) * 100:.1f}%")
            print(f"   <2m: {np.mean(errors < 2.0) * 100:.1f}%")
            
            # Show individual predictions
            for i, (pred, error) in enumerate(zip(predictions, errors)):
                print(f"      Pred {i+1}: ({pred[0]:.3f}, {pred[1]:.3f}) - Error: {error:.3f}m")
        
        # Overall statistics
        all_errors = np.array(all_errors)
        print(f"\n🎯 OVERALL STATISTICS:")
        print(f"   Total Predictions: {len(all_errors)}")
        print(f"   Mean Error: {np.mean(all_errors):.3f}m")
        print(f"   Median Error: {np.median(all_errors):.3f}m")
        print(f"   Std Error: {np.std(all_errors):.3f}m")
        print(f"   Accuracy <1m: {np.mean(all_errors < 1.0) * 100:.1f}%")
        print(f"   Accuracy <2m: {np.mean(all_errors < 2.0) * 100:.1f}%")
        print(f"   Accuracy <3m: {np.mean(all_errors < 3.0) * 100:.1f}%")
    
    def run_visualization(self):
        """Run the complete visualization process"""
        print("🎯 Starting Ground Truth vs Prediction Visualization...")
        
        # Load the best model
        model = self.load_best_model()
        if model is None:
            print("❌ Cannot proceed without a valid model")
            return
        
        # Load test data
        test_data = self.load_test_data(dataset_size=250)
        
        # Get predictions
        predictions_by_point = self.get_random_predictions(model, test_data, num_predictions=5)
        
        # Calculate statistics
        self.calculate_prediction_statistics(predictions_by_point)
        
        # Create visualization
        output_path = self.create_comprehensive_plot(predictions_by_point)
        
        print("\n" + "="*80)
        print("✅ VISUALIZATION COMPLETE!")
        print("="*80)
        print(f"📊 Generated comprehensive plot: {output_path}")
        print(f"📍 Shows ground truth grid with training/validation/testing points")
        print(f"🎯 Displays 5 random predictions per test point from best HybridCNN")
        print(f"⭕ Includes 1m and 2m accuracy circles around ground truth points")
        
        return output_path

def main():
    """Main execution function"""
    visualizer = GroundTruthPredictionVisualizer()
    output_path = visualizer.run_visualization()
    return visualizer, output_path

if __name__ == "__main__":
    visualizer, output_path = main()
