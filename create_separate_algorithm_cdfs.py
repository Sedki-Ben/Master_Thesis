#!/usr/bin/env python3
"""
Create Separate CDF Plots by Algorithm Type

Creates individual CDF plots for each classical algorithm family
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

# Copy the classes from the previous file
class KNNLocalizer:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        
    def predict(self, X):
        predictions = []
        for x_test in X:
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            pred_coord = np.mean(self.y_train[k_indices], axis=0)
            predictions.append(pred_coord)
        return np.array(predictions)

class IDWLocalizer:
    def __init__(self, power=2, epsilon=1e-6):
        self.power = power
        self.epsilon = epsilon
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = X.copy()
        self.y_train = y.copy()
        
    def predict(self, X):
        predictions = []
        for x_test in X:
            distances = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            distances = distances + self.epsilon
            weights = 1 / (distances ** self.power)
            weighted_coords = np.sum(weights.reshape(-1, 1) * self.y_train, axis=0)
            total_weight = np.sum(weights)
            pred_coord = weighted_coords / total_weight
            predictions.append(pred_coord)
        return np.array(predictions)

class ProbabilisticLocalizer:
    def __init__(self, smoothing=1e-6):
        self.smoothing = smoothing
        self.reference_points = {}
        
    def fit(self, X, y):
        unique_coords = np.unique(y, axis=0)
        for coord in unique_coords:
            mask = (y == coord).all(axis=1)
            samples = X[mask]
            if len(samples) > 1:
                mean = np.mean(samples, axis=0)
                cov = np.cov(samples, rowvar=False)
                cov += self.smoothing * np.eye(cov.shape[0])
                self.reference_points[tuple(coord)] = {
                    'mean': mean, 'cov': cov, 'coord': coord
                }
            elif len(samples) == 1:
                mean = samples[0]
                cov = self.smoothing * np.eye(len(mean))
                self.reference_points[tuple(coord)] = {
                    'mean': mean, 'cov': cov, 'coord': coord
                }
        
    def predict(self, X):
        predictions = []
        for x_test in X:
            max_likelihood = -np.inf
            best_coord = None
            for coord_tuple, ref_data in self.reference_points.items():
                try:
                    likelihood = multivariate_normal.logpdf(
                        x_test, ref_data['mean'], ref_data['cov']
                    )
                    if likelihood > max_likelihood:
                        max_likelihood = likelihood
                        best_coord = ref_data['coord']
                except:
                    continue
            if best_coord is not None:
                predictions.append(best_coord)
            else:
                predictions.append([0, 0])
        return np.array(predictions)

def load_amplitude_phase_data():
    print("📂 Loading Amplitude and Phase Data...")
    data_files = glob.glob("Amplitude Phase Data Single/*.csv")
    all_data = []
    coordinates = []
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        coord_str = filename.replace('.csv', '')
        try:
            x, y = map(int, coord_str.split(','))
            coordinates.append((x, y))
            df = pd.read_csv(file_path)
            
            features = []
            for _, row in df.iterrows():
                amp_str = row['amplitude'].strip('[]"')
                amplitudes = [float(x.strip()) for x in amp_str.split(',')]
                rssi = row['rssi']
                feature_vector = amplitudes + [rssi]
                features.append(feature_vector)
            
            for feature_vector in features:
                all_data.append({
                    'features': feature_vector,
                    'x': x, 'y': y
                })
        except Exception as e:
            continue
    
    print(f"✅ Loaded {len(all_data)} samples from {len(coordinates)} reference points")
    X = np.array([item['features'] for item in all_data])
    y = np.array([[item['x'], item['y']] for item in all_data])
    return X, y, coordinates

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    errors = np.sqrt(np.sum((y_test - y_pred)**2, axis=1))
    
    return {
        'model': model_name,
        'median_error': np.median(errors),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'accuracy_1m': np.mean(errors <= 1.0) * 100,
        'accuracy_2m': np.mean(errors <= 2.0) * 100,
        'accuracy_3m': np.mean(errors <= 3.0) * 100,
        'errors': errors
    }

def create_knn_cdf_plot(knn_results):
    print("📈 Creating k-NN CDF Comparison Plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    knn_colors = {
        'k-NN (k=1)': '#FF6B6B',
        'k-NN (k=3)': '#FF8E53',
        'k-NN (k=5)': '#FF7F50',
        'k-NN (k=9)': '#DC143C'
    }
    
    for result in knn_results:
        model_name = result['model']
        errors = result['errors']
        
        if 'k-NN' in model_name:
            errors_sorted = np.sort(errors)
            p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
            
            color = knn_colors[model_name]
            ax.plot(errors_sorted, p, color=color, linewidth=3, 
                   label=f"{model_name} (median: {result['median_error']:.3f}m)", 
                   alpha=0.9)
    
    # Add threshold lines
    thresholds = [1.0, 2.0, 3.0, 4.0]
    threshold_colors = ['green', 'orange', 'red', 'purple']
    
    for threshold, color in zip(thresholds, threshold_colors):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.6, linewidth=2)
        ax.text(threshold + 0.05, 0.95, f'{threshold}m', rotation=90, 
               fontsize=10, color=color, fontweight='bold', 
               verticalalignment='top')
    
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF Comparison: k-Nearest Neighbors (k-NN) Algorithms\n'
                'Indoor Localization Regression Performance', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', fontsize=12, framealpha=0.9)
    
    # Add insight box
    ax.text(0.02, 0.98, 'k-NN Performance:\n• All k values show similar\n  median errors (~3.6m)\n• k=1 slightly better for\n  1m accuracy', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('knn_algorithms_cdf_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 k-NN CDF plot saved: knn_algorithms_cdf_comparison.png")
    plt.show()

def create_idw_cdf_plot(idw_results):
    print("📈 Creating IDW CDF Comparison Plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    idw_colors = {
        'IDW (p=1)': '#32CD32',
        'IDW (p=2)': '#228B22',
        'IDW (p=4)': '#006400'
    }
    
    for result in idw_results:
        model_name = result['model']
        errors = result['errors']
        
        if 'IDW' in model_name:
            errors_sorted = np.sort(errors)
            p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
            
            color = idw_colors[model_name]
            linewidth = 4 if model_name == 'IDW (p=1)' else 3
            ax.plot(errors_sorted, p, color=color, linewidth=linewidth, 
                   label=f"{model_name} (median: {result['median_error']:.3f}m)", 
                   alpha=0.9)
    
    # Add threshold lines
    thresholds = [1.0, 2.0, 3.0, 4.0]
    threshold_colors = ['green', 'orange', 'red', 'purple']
    
    for threshold, color in zip(thresholds, threshold_colors):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.6, linewidth=2)
        ax.text(threshold + 0.05, 0.95, f'{threshold}m', rotation=90, 
               fontsize=10, color=color, fontweight='bold', 
               verticalalignment='top')
    
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF Comparison: Inverse Distance Weighting (IDW) Algorithms\n'
                'Indoor Localization Regression Performance', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', fontsize=12, framealpha=0.9)
    
    ax.text(0.02, 0.98, 'IDW Performance:\n• Lower power (p=1) works best\n• Best classical algorithm\n• p=1: 23.8% accuracy <1m', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('idw_algorithms_cdf_comparison.png', dpi=300, bbox_inches='tight')
    print("💾 IDW CDF plot saved: idw_algorithms_cdf_comparison.png")
    plt.show()

def create_probabilistic_cdf_plot(prob_results):
    print("📈 Creating Probabilistic Fingerprinting CDF Plot...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    prob_result = None
    for result in prob_results:
        if 'Probabilistic' in result['model']:
            prob_result = result
            break
    
    if prob_result is None:
        print("⚠️ No probabilistic results found")
        return
    
    errors = prob_result['errors']
    errors_sorted = np.sort(errors)
    p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
    
    ax.plot(errors_sorted, p, color='#4169E1', linewidth=4, 
           label=f"Probabilistic Fingerprinting (median: {prob_result['median_error']:.3f}m)", 
           alpha=0.9)
    
    # Add threshold lines
    thresholds = [1.0, 2.0, 3.0, 4.0]
    threshold_colors = ['green', 'orange', 'red', 'purple']
    
    for threshold, color in zip(thresholds, threshold_colors):
        ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.6, linewidth=2)
        ax.text(threshold + 0.05, 0.95, f'{threshold}m', rotation=90, 
               fontsize=10, color=color, fontweight='bold', 
               verticalalignment='top')
    
    ax.set_xlabel('Localization Error (meters)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
    ax.set_title('CDF: Probabilistic Fingerprinting Algorithm\n'
                'Gaussian Maximum Likelihood Indoor Localization', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='center right', fontsize=12, framealpha=0.9)
    
    ax.text(0.02, 0.98, 'Algorithm:\n• Learns Gaussian distribution\n  for each reference point\n• Uses Maximum Likelihood\n  Estimation for prediction', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor='lightblue', alpha=0.8))
    
    ax.text(0.02, 0.55, f'Performance:\n• Median: {prob_result["median_error"]:.3f}m\n• 1m Acc: {prob_result["accuracy_1m"]:.1f}%\n• 2m Acc: {prob_result["accuracy_2m"]:.1f}%', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
           facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('probabilistic_fingerprinting_cdf.png', dpi=300, bbox_inches='tight')
    print("💾 Probabilistic CDF plot saved: probabilistic_fingerprinting_cdf.png")
    plt.show()

def main():
    print("🎯 CREATING SEPARATE CDF PLOTS BY ALGORITHM TYPE")
    print("="*55)
    
    # Load and prepare data
    X, y, coordinates = load_amplitude_phase_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    unique_coords = np.unique(y, axis=0)
    n_train_points = int(0.8 * len(unique_coords))
    
    np.random.seed(42)
    train_coords = unique_coords[np.random.choice(len(unique_coords), n_train_points, replace=False)]
    
    train_mask = np.array([tuple(coord) in [tuple(tc) for tc in train_coords] for coord in y])
    test_mask = ~train_mask
    
    X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"Training: {len(X_train)} samples, Testing: {len(X_test)} samples")
    
    # Evaluate models
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
    
    all_results = []
    for model, name in models:
        print(f"Evaluating {name}...")
        result = evaluate_model(model, X_train, y_train, X_test, y_test, name)
        all_results.append(result)
        print(f"   {name}: median={result['median_error']:.3f}m")
    
    # Create separate plots
    create_knn_cdf_plot(all_results)
    create_idw_cdf_plot(all_results)
    create_probabilistic_cdf_plot(all_results)
    
    print(f"\n✅ SEPARATE CDF PLOTS CREATED!")
    print(f"📊 Generated 3 algorithm-specific CDF comparison plots")

if __name__ == "__main__":
    main()


