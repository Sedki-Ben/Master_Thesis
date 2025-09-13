#!/usr/bin/env python3
"""
Simple Classical Localization Models: k-NN, IDW, Probabilistic

Implements and evaluates simple classical algorithms for indoor localization regression:
1. k-Nearest Neighbors (k-NN) with different k values
2. Inverse Distance Weighting (IDW) with different power parameters
3. Probabilistic fingerprinting with Gaussian distributions

Plots CDFs comparing all models on the same dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import glob
import os
from pathlib import Path

def load_amplitude_phase_data():
    """Load amplitude and phase data from all reference points"""
    
    print("📂 Loading Amplitude and Phase Data...")
    
    # Get all CSV files from Amplitude Phase Data folder
    data_files = glob.glob("Amplitude Phase Data Single/*.csv")
    
    all_data = []
    coordinates = []
    
    for file_path in data_files:
        # Extract coordinates from filename (e.g., "0,0.csv" -> (0, 0))
        filename = os.path.basename(file_path)
        coord_str = filename.replace('.csv', '')
        try:
            x, y = map(int, coord_str.split(','))
            coordinates.append((x, y))
            
            # Load data
            df = pd.read_csv(file_path)
            
            # Extract features (amplitude + RSSI)
            features = []
            for _, row in df.iterrows():
                # Parse amplitude array - it's stored as a string representation of a list
                amp_str = row['amplitude'].strip('[]"')
                amplitudes = [float(x.strip()) for x in amp_str.split(',')]
                
                # Add RSSI
                rssi = row['rssi']
                
                # Combine features
                feature_vector = amplitudes + [rssi]
                features.append(feature_vector)
            
            # Add to dataset
            for feature_vector in features:
                all_data.append({
                    'features': feature_vector,
                    'x': x,
                    'y': y
                })
                
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_data)} samples from {len(coordinates)} reference points")
    
    # Convert to arrays
    X = np.array([item['features'] for item in all_data])
    y = np.array([[item['x'], item['y']] for item in all_data])
    
    return X, y, coordinates

class KNNLocalizer:
    """k-Nearest Neighbors localization regressor"""
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X.copy()
        self.y_train = y.copy()
        
    def predict(self, X):
        """Predict locations using k-NN regression"""
        predictions = []
        
        for x_test in X:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            
            # Find k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Average their coordinates
            pred_coord = np.mean(self.y_train[k_indices], axis=0)
            predictions.append(pred_coord)
            
        return np.array(predictions)

class IDWLocalizer:
    """Inverse Distance Weighting localization regressor"""
    
    def __init__(self, power=2, epsilon=1e-6):
        self.power = power
        self.epsilon = epsilon  # Small value to avoid division by zero
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        """Store training data"""
        self.X_train = X.copy()
        self.y_train = y.copy()
        
    def predict(self, X):
        """Predict locations using IDW"""
        predictions = []
        
        for x_test in X:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            
            # Add small epsilon to avoid division by zero
            distances = distances + self.epsilon
            
            # Calculate inverse distance weights
            weights = 1 / (distances ** self.power)
            
            # Weighted average of coordinates
            weighted_coords = np.sum(weights.reshape(-1, 1) * self.y_train, axis=0)
            total_weight = np.sum(weights)
            
            pred_coord = weighted_coords / total_weight
            predictions.append(pred_coord)
            
        return np.array(predictions)

class ProbabilisticLocalizer:
    """Probabilistic fingerprinting with Gaussian distributions"""
    
    def __init__(self, smoothing=1e-6):
        self.smoothing = smoothing
        self.reference_points = {}
        
    def fit(self, X, y):
        """Learn Gaussian distributions for each reference point"""
        
        # Group samples by reference point
        unique_coords = np.unique(y, axis=0)
        
        for coord in unique_coords:
            # Find all samples for this reference point
            mask = (y == coord).all(axis=1)
            samples = X[mask]
            
            if len(samples) > 1:
                # Calculate mean and covariance
                mean = np.mean(samples, axis=0)
                cov = np.cov(samples, rowvar=False)
                
                # Add smoothing to diagonal for numerical stability
                cov += self.smoothing * np.eye(cov.shape[0])
                
                self.reference_points[tuple(coord)] = {
                    'mean': mean,
                    'cov': cov,
                    'coord': coord
                }
            elif len(samples) == 1:
                # Single sample - use small covariance
                mean = samples[0]
                cov = self.smoothing * np.eye(len(mean))
                
                self.reference_points[tuple(coord)] = {
                    'mean': mean,
                    'cov': cov,
                    'coord': coord
                }
        
        print(f"📊 Learned distributions for {len(self.reference_points)} reference points")
        
    def predict(self, X):
        """Predict locations using maximum likelihood estimation"""
        predictions = []
        
        for x_test in X:
            max_likelihood = -np.inf
            best_coord = None
            
            # Calculate likelihood for each reference point
            for coord_tuple, ref_data in self.reference_points.items():
                try:
                    # Calculate likelihood
                    likelihood = multivariate_normal.logpdf(
                        x_test, 
                        ref_data['mean'], 
                        ref_data['cov']
                    )
                    
                    if likelihood > max_likelihood:
                        max_likelihood = likelihood
                        best_coord = ref_data['coord']
                        
                except Exception as e:
                    # Skip problematic distributions
                    continue
            
            if best_coord is not None:
                predictions.append(best_coord)
            else:
                # Fallback to origin if no valid prediction
                predictions.append([0, 0])
                
        return np.array(predictions)

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Evaluate a localization model"""
    
    print(f"🔬 Evaluating {model_name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate Euclidean distances (localization errors)
    errors = np.sqrt(np.sum((y_test - y_pred)**2, axis=1))
    
    # Calculate metrics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    std_error = np.std(errors)
    
    # Calculate accuracy metrics
    acc_1m = np.mean(errors <= 1.0) * 100
    acc_2m = np.mean(errors <= 2.0) * 100
    acc_3m = np.mean(errors <= 3.0) * 100
    
    results = {
        'model': model_name,
        'mean_error': mean_error,
        'median_error': median_error,
        'std_error': std_error,
        'accuracy_1m': acc_1m,
        'accuracy_2m': acc_2m,
        'accuracy_3m': acc_3m,
        'errors': errors
    }
    
    print(f"   📊 {model_name}: median={median_error:.3f}m, mean={mean_error:.3f}m, 1m acc={acc_1m:.1f}%")
    
    return results

def create_cdf_comparison_plot(all_results):
    """Create CDF comparison plot for all classical models"""
    
    print("📈 Creating CDF Comparison Plot...")
    
    # Set up the plot
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Colors for different models
    colors = {
        'k-NN (k=1)': '#FF6B6B',      # Light Red
        'k-NN (k=3)': '#FF8E53',      # Orange Red  
        'k-NN (k=5)': '#FF7F50',      # Coral
        'k-NN (k=9)': '#DC143C',      # Crimson
        'IDW (p=1)': '#32CD32',       # Lime Green
        'IDW (p=2)': '#228B22',       # Forest Green
        'IDW (p=4)': '#006400',       # Dark Green
        'Probabilistic': '#4169E1'    # Royal Blue
    }
    
    # Plot CDFs for each model
    for result in all_results:
        model_name = result['model']
        errors = result['errors']
        
        # Sort errors for CDF
        errors_sorted = np.sort(errors)
        p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
        
        # Plot CDF
        color = colors.get(model_name, '#808080')
        linestyle = '--' if 'IDW' in model_name else '-'
        linewidth = 3 if model_name == 'Probabilistic' else 2.5
        
        ax.plot(errors_sorted, p, color=color, linewidth=linewidth, 
               label=f"{model_name} (median: {result['median_error']:.3f}m)", 
               linestyle=linestyle, alpha=0.9)
    
    # Add accuracy threshold lines
    accuracy_thresholds = [1.0, 2.0, 3.0]
    threshold_colors = ['green', 'orange', 'red']
    threshold_labels = ['1m accuracy', '2m accuracy', '3m accuracy']
    
    for threshold, color, label in zip(accuracy_thresholds, threshold_colors, threshold_labels):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.7, linewidth=2)
        ax.text(threshold + 0.05, 0.95, label, rotation=90, 
               fontsize=10, color=color, fontweight='bold', 
               verticalalignment='top')
    
    # Customize the plot
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF Comparison: Simple Classical Localization Models\n'
                'k-NN, IDW, and Probabilistic Fingerprinting', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Set axis limits
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Create legend
    ax.legend(loc='center right', fontsize=10, framealpha=0.9, ncol=1)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = 'simple_classical_models_cdf_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 CDF comparison plot saved: {output_file}")
    
    plt.show()

def create_performance_summary_table(all_results):
    """Create comprehensive performance summary table"""
    
    print(f"\n📊 SIMPLE CLASSICAL MODELS PERFORMANCE SUMMARY")
    print("="*55)
    
    # Sort by median error
    sorted_results = sorted(all_results, key=lambda x: x['median_error'])
    
    # Create summary table
    print(f"{'Rank':<4} {'Model':<15} {'Median (m)':<10} {'Mean (m)':<9} {'1m Acc':<7} {'2m Acc':<7} {'3m Acc':<7}")
    print("-" * 65)
    
    for rank, result in enumerate(sorted_results, 1):
        print(f"{rank:<4} {result['model']:<15} {result['median_error']:<10.3f} "
              f"{result['mean_error']:<9.3f} {result['accuracy_1m']:<7.1f} "
              f"{result['accuracy_2m']:<7.1f} {result['accuracy_3m']:<7.1f}")
    
    # Performance insights
    best_model = sorted_results[0]
    
    print(f"\n🏆 BEST PERFORMER:")
    print(f"   Model: {best_model['model']}")
    print(f"   Median Error: {best_model['median_error']:.3f}m")
    print(f"   1m Accuracy: {best_model['accuracy_1m']:.1f}%")
    print(f"   2m Accuracy: {best_model['accuracy_2m']:.1f}%")

def main():
    """Main execution function"""
    
    print("🎯 SIMPLE CLASSICAL LOCALIZATION MODELS")
    print("k-NN, IDW, and Probabilistic Fingerprinting")
    print("="*50)
    
    # Load data
    X, y, coordinates = load_amplitude_phase_data()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data (stratified by location to ensure all reference points in training)
    print(f"\n📊 Splitting Data...")
    
    # Create location-based split
    unique_coords = np.unique(y, axis=0)
    n_train_points = int(0.8 * len(unique_coords))  # 80% of reference points for training
    
    np.random.seed(42)  # For reproducibility
    train_coords = unique_coords[np.random.choice(len(unique_coords), n_train_points, replace=False)]
    
    # Create masks for train/test split
    train_mask = np.array([tuple(coord) in [tuple(tc) for tc in train_coords] for coord in y])
    test_mask = ~train_mask
    
    X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"   Training: {len(X_train)} samples from {n_train_points} reference points")
    print(f"   Testing: {len(X_test)} samples from {len(unique_coords) - n_train_points} reference points")
    
    # Define models to evaluate
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
    
    # Evaluate all models
    print(f"\n🔬 EVALUATING MODELS...")
    print("-" * 25)
    
    all_results = []
    for model, name in models:
        try:
            result = evaluate_model(model, X_train, y_train, X_test, y_test, name)
            all_results.append(result)
        except Exception as e:
            print(f"⚠️ Error evaluating {name}: {e}")
            continue
    
    # Create CDF comparison plot
    create_cdf_comparison_plot(all_results)
    
    # Create performance summary
    create_performance_summary_table(all_results)
    
    print(f"\n✅ SIMPLE CLASSICAL MODELS EVALUATION COMPLETE!")
    print(f"📊 Generated CDF comparison for {len(all_results)} models")
    print(f"🎯 Results show performance differences across classical approaches")

if __name__ == "__main__":
    main()
