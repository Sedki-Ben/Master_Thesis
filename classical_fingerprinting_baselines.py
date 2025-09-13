#!/usr/bin/env python3
"""
Classical Fingerprinting Baselines for Indoor Localization

Implements traditional fingerprinting methods to establish whether complex CNNs 
actually improve over classical baselines:
1. k-NN (Euclidean distance in CSI space)
2. Weighted k-NN 
3. Probabilistic fingerprinting (Gaussian)
4. Inverse Distance Weighting (IDW) interpolation
5. Simple MLP
6. RSSI-only baselines

This provides crucial baseline comparison for our CNN models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

class ClassicalFingerprintingBaselines:
    """Classical fingerprinting methods for indoor localization"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.training_data = None
        self.reference_points = None
        
    def load_csi_data(self, sample_size=750):
        """Load CSI data for baseline comparison"""
        
        print(f"📂 Loading CSI data (sample size: {sample_size})...")
        
        # Define our reference points (training locations)
        reference_points = [
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
            (1, 0), (1, 1), (1, 4), (1, 5),
            (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
            (4, 0), (4, 1), (4, 4), (4, 5),
            (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
            (6, 3), (6, 4)
        ]
        
        # Test points (for evaluation)
        test_points = [(0.5, 0.5), (1.5, 2.5), (2.5, 4.5), (3.5, 1.5), (5.5, 3.5)]
        
        # Load training data
        train_data = []
        train_coords = []
        
        dataset_folder = f"CSI Dataset {sample_size} Samples"
        
        for x, y in reference_points:
            file_path = Path(dataset_folder) / f"{x},{y}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                # Extract features
                rssi_values = df['rssi'].values
                amplitude_data = df['amplitude'].apply(eval).tolist()  # Convert string to list
                
                # Create feature vectors
                for i in range(len(df)):
                    features = {
                        'rssi': rssi_values[i],
                        'amplitude': np.array(amplitude_data[i]),
                        'coordinates': np.array([x, y])
                    }
                    train_data.append(features)
                    train_coords.append([x, y])
                
                print(f"   ✅ Loaded {len(df)} samples from ({x}, {y})")
            else:
                print(f"   ❌ Missing file: {file_path}")
        
        # Load test data
        test_data = []
        test_coords = []
        
        test_dataset_folder = f"Testing Points Dataset {sample_size} Samples"
        
        for x, y in test_points:
            file_path = Path(test_dataset_folder) / f"{x},{y}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                
                rssi_values = df['rssi'].values
                amplitude_data = df['amplitude'].apply(eval).tolist()
                
                for i in range(len(df)):
                    features = {
                        'rssi': rssi_values[i],
                        'amplitude': np.array(amplitude_data[i]),
                        'coordinates': np.array([x, y])
                    }
                    test_data.append(features)
                    test_coords.append([x, y])
                
                print(f"   ✅ Loaded {len(df)} test samples from ({x}, {y})")
        
        print(f"📊 Dataset Summary:")
        print(f"   Training samples: {len(train_data)}")
        print(f"   Test samples: {len(test_data)}")
        print(f"   Reference points: {len(reference_points)}")
        print(f"   Test points: {len(test_points)}")
        
        return train_data, test_data, reference_points, test_points
    
    def prepare_feature_matrices(self, train_data, test_data):
        """Prepare different feature matrices for various baselines"""
        
        print("🔄 Preparing feature matrices...")
        
        features = {}
        
        # Extract raw features
        train_rssi = np.array([d['rssi'] for d in train_data])
        train_amplitude = np.array([d['amplitude'] for d in train_data])
        train_coords = np.array([d['coordinates'] for d in train_data])
        
        test_rssi = np.array([d['rssi'] for d in test_data])
        test_amplitude = np.array([d['amplitude'] for d in test_data])
        test_coords = np.array([d['coordinates'] for d in test_data])
        
        # 1. RSSI-only features
        features['rssi_only'] = {
            'X_train': train_rssi.reshape(-1, 1),
            'X_test': test_rssi.reshape(-1, 1),
            'y_train': train_coords,
            'y_test': test_coords
        }
        
        # 2. Amplitude-only features (52 subcarriers)
        features['amplitude_only'] = {
            'X_train': train_amplitude,
            'X_test': test_amplitude,
            'y_train': train_coords,
            'y_test': test_coords
        }
        
        # 3. Combined features (RSSI + Amplitude)
        train_combined = np.column_stack([train_rssi, train_amplitude])
        test_combined = np.column_stack([test_rssi, test_amplitude])
        
        features['combined'] = {
            'X_train': train_combined,
            'X_test': test_combined,
            'y_train': train_coords,
            'y_test': test_coords
        }
        
        # 4. Statistical features (mean, std, min, max of amplitude)
        train_stats = np.column_stack([
            train_rssi,
            np.mean(train_amplitude, axis=1),
            np.std(train_amplitude, axis=1),
            np.min(train_amplitude, axis=1),
            np.max(train_amplitude, axis=1)
        ])
        
        test_stats = np.column_stack([
            test_rssi,
            np.mean(test_amplitude, axis=1),
            np.std(test_amplitude, axis=1),
            np.min(test_amplitude, axis=1),
            np.max(test_amplitude, axis=1)
        ])
        
        features['statistical'] = {
            'X_train': train_stats,
            'X_test': test_stats,
            'y_train': train_coords,
            'y_test': test_coords
        }
        
        print(f"   ✅ Prepared {len(features)} feature sets")
        for name, data in features.items():
            print(f"   📊 {name}: {data['X_train'].shape} -> {data['y_train'].shape}")
        
        return features
    
    def knn_baseline(self, X_train, y_train, X_test, y_test, feature_name):
        """k-NN baseline with Euclidean distance in feature space"""
        
        print(f"🔍 Running k-NN baseline ({feature_name})...")
        
        results = {}
        k_values = [1, 3, 5, 7, 9, 15, 25]
        
        for k in k_values:
            start_time = time.time()
            
            # Standard k-NN
            knn = KNeighborsRegressor(n_neighbors=k, metric='euclidean')
            knn.fit(X_train, y_train)
            
            predictions = knn.predict(X_test)
            
            # Calculate errors
            errors = np.sqrt(np.sum((predictions - y_test) ** 2, axis=1))
            
            results[f'k={k}'] = {
                'predictions': predictions,
                'errors': errors,
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'training_time': time.time() - start_time,
                'accuracy_1m': np.mean(errors <= 1.0) * 100,
                'accuracy_2m': np.mean(errors <= 2.0) * 100
            }
            
            print(f"   k={k}: {results[f'k={k}']['median_error']:.3f}m median error")
        
        # Find best k
        best_k = min(results.keys(), key=lambda k: results[k]['median_error'])
        print(f"   🥇 Best k-NN: {best_k} ({results[best_k]['median_error']:.3f}m)")
        
        return results, best_k
    
    def weighted_knn_baseline(self, X_train, y_train, X_test, y_test, feature_name):
        """Weighted k-NN baseline (inverse distance weighting)"""
        
        print(f"🎯 Running Weighted k-NN baseline ({feature_name})...")
        
        results = {}
        k_values = [3, 5, 7, 9, 15, 25]
        
        for k in k_values:
            start_time = time.time()
            
            # Weighted k-NN
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='euclidean')
            knn.fit(X_train, y_train)
            
            predictions = knn.predict(X_test)
            
            # Calculate errors
            errors = np.sqrt(np.sum((predictions - y_test) ** 2, axis=1))
            
            results[f'weighted_k={k}'] = {
                'predictions': predictions,
                'errors': errors,
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'training_time': time.time() - start_time,
                'accuracy_1m': np.mean(errors <= 1.0) * 100,
                'accuracy_2m': np.mean(errors <= 2.0) * 100
            }
        
        # Find best k
        best_k = min(results.keys(), key=lambda k: results[k]['median_error'])
        print(f"   🥇 Best Weighted k-NN: {best_k} ({results[best_k]['median_error']:.3f}m)")
        
        return results, best_k
    
    def probabilistic_fingerprinting(self, X_train, y_train, X_test, y_test, feature_name):
        """Probabilistic fingerprinting using Gaussian models"""
        
        print(f"📊 Running Probabilistic Fingerprinting ({feature_name})...")
        
        start_time = time.time()
        
        # Group training data by location
        location_models = {}
        unique_locations = np.unique(y_train, axis=0)
        
        for loc in unique_locations:
            loc_mask = np.all(y_train == loc, axis=1)
            loc_features = X_train[loc_mask]
            
            if len(loc_features) > 1:
                # Fit Gaussian model
                mean = np.mean(loc_features, axis=0)
                
                if loc_features.shape[1] == 1:
                    # Handle 1D case (like RSSI-only)
                    cov = np.var(loc_features) + 1e-6
                else:
                    # Handle multi-dimensional case
                    cov = np.cov(loc_features.T)
                    # Regularize covariance matrix
                    cov += np.eye(cov.shape[0]) * 1e-6
                
                location_models[tuple(loc)] = {
                    'mean': mean,
                    'cov': cov,
                    'samples': len(loc_features)
                }
        
        # Make predictions
        predictions = []
        
        for test_sample in X_test:
            likelihoods = []
            locations = []
            
            for loc, model in location_models.items():
                try:
                    # Calculate likelihood
                    if np.isscalar(model['cov']):
                        # 1D case (like RSSI-only)
                        from scipy.stats import norm
                        likelihood = norm.pdf(test_sample[0], model['mean'][0], np.sqrt(model['cov']))
                    else:
                        # Multi-dimensional case
                        likelihood = multivariate_normal.pdf(test_sample, 
                                                           model['mean'], 
                                                           model['cov'])
                    likelihoods.append(likelihood)
                    locations.append(loc)
                except:
                    # Handle numerical issues
                    likelihoods.append(1e-10)
                    locations.append(loc)
            
            # Weighted average based on likelihood
            likelihoods = np.array(likelihoods)
            if np.sum(likelihoods) > 0:
                weights = likelihoods / np.sum(likelihoods)
                predicted_loc = np.average(locations, axis=0, weights=weights)
            else:
                # Fallback to mean location
                predicted_loc = np.mean(locations, axis=0)
            
            predictions.append(predicted_loc)
        
        predictions = np.array(predictions)
        
        # Calculate errors
        errors = np.sqrt(np.sum((predictions - y_test) ** 2, axis=1))
        
        result = {
            'predictions': predictions,
            'errors': errors,
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'training_time': time.time() - start_time,
            'accuracy_1m': np.mean(errors <= 1.0) * 100,
            'accuracy_2m': np.mean(errors <= 2.0) * 100,
            'n_models': len(location_models)
        }
        
        print(f"   📈 Probabilistic: {result['median_error']:.3f}m median error")
        print(f"   🏷️  Created {result['n_models']} location models")
        
        return result
    
    def idw_interpolation(self, X_train, y_train, X_test, y_test, feature_name):
        """Inverse Distance Weighting (IDW) interpolation"""
        
        print(f"🌐 Running IDW Interpolation ({feature_name})...")
        
        power_values = [1, 2, 3, 4]
        results = {}
        
        for power in power_values:
            start_time = time.time()
            
            predictions = []
            
            for test_sample in X_test:
                # Calculate distances to all training samples
                distances = cdist([test_sample], X_train, metric='euclidean')[0]
                
                # Avoid division by zero
                distances = np.maximum(distances, 1e-10)
                
                # Calculate weights
                weights = 1.0 / (distances ** power)
                weights = weights / np.sum(weights)
                
                # Weighted average of coordinates
                predicted_coord = np.average(y_train, axis=0, weights=weights)
                predictions.append(predicted_coord)
            
            predictions = np.array(predictions)
            
            # Calculate errors
            errors = np.sqrt(np.sum((predictions - y_test) ** 2, axis=1))
            
            results[f'power={power}'] = {
                'predictions': predictions,
                'errors': errors,
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'training_time': time.time() - start_time,
                'accuracy_1m': np.mean(errors <= 1.0) * 100,
                'accuracy_2m': np.mean(errors <= 2.0) * 100
            }
        
        # Find best power
        best_power = min(results.keys(), key=lambda p: results[p]['median_error'])
        print(f"   🥇 Best IDW: {best_power} ({results[best_power]['median_error']:.3f}m)")
        
        return results, best_power
    
    def simple_mlp_baseline(self, X_train, y_train, X_test, y_test, feature_name):
        """Simple MLP baseline"""
        
        print(f"🧠 Running Simple MLP baseline ({feature_name})...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define MLP configurations
        configs = [
            {'hidden_layer_sizes': (64,), 'name': 'MLP_64'},
            {'hidden_layer_sizes': (128,), 'name': 'MLP_128'},
            {'hidden_layer_sizes': (64, 32), 'name': 'MLP_64_32'},
            {'hidden_layer_sizes': (128, 64), 'name': 'MLP_128_64'},
            {'hidden_layer_sizes': (256, 128, 64), 'name': 'MLP_256_128_64'}
        ]
        
        results = {}
        
        for config in configs:
            start_time = time.time()
            
            mlp = MLPRegressor(
                hidden_layer_sizes=config['hidden_layer_sizes'],
                activation='relu',
                solver='adam',
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            
            mlp.fit(X_train_scaled, y_train)
            predictions = mlp.predict(X_test_scaled)
            
            # Calculate errors
            errors = np.sqrt(np.sum((predictions - y_test) ** 2, axis=1))
            
            results[config['name']] = {
                'predictions': predictions,
                'errors': errors,
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'training_time': time.time() - start_time,
                'accuracy_1m': np.mean(errors <= 1.0) * 100,
                'accuracy_2m': np.mean(errors <= 2.0) * 100,
                'n_iter': mlp.n_iter_
            }
            
            print(f"   {config['name']}: {results[config['name']]['median_error']:.3f}m")
        
        # Find best MLP
        best_mlp = min(results.keys(), key=lambda m: results[m]['median_error'])
        print(f"   🥇 Best MLP: {best_mlp} ({results[best_mlp]['median_error']:.3f}m)")
        
        return results, best_mlp
    
    def run_comprehensive_baseline_comparison(self, sample_size=750):
        """Run comprehensive baseline comparison"""
        
        print("🚀 COMPREHENSIVE CLASSICAL FINGERPRINTING BASELINE COMPARISON")
        print("="*80)
        
        # Load data
        train_data, test_data, ref_points, test_points = self.load_csi_data(sample_size)
        
        # Prepare features
        feature_sets = self.prepare_feature_matrices(train_data, test_data)
        
        # Store for later use
        self.training_data = train_data
        self.reference_points = ref_points
        
        all_results = {}
        
        # Test each feature set with each method
        for feature_name, features in feature_sets.items():
            print(f"\n🔍 TESTING FEATURE SET: {feature_name.upper()}")
            print("-" * 50)
            
            X_train = features['X_train']
            y_train = features['y_train']
            X_test = features['X_test']
            y_test = features['y_test']
            
            feature_results = {}
            
            # 1. k-NN baseline
            knn_results, best_knn = self.knn_baseline(X_train, y_train, X_test, y_test, feature_name)
            feature_results['knn'] = knn_results[best_knn]
            feature_results['knn']['method'] = f"k-NN ({best_knn})"
            
            # 2. Weighted k-NN
            weighted_knn_results, best_weighted = self.weighted_knn_baseline(X_train, y_train, X_test, y_test, feature_name)
            feature_results['weighted_knn'] = weighted_knn_results[best_weighted]
            feature_results['weighted_knn']['method'] = f"Weighted k-NN ({best_weighted})"
            
            # 3. Probabilistic fingerprinting
            prob_result = self.probabilistic_fingerprinting(X_train, y_train, X_test, y_test, feature_name)
            feature_results['probabilistic'] = prob_result
            feature_results['probabilistic']['method'] = "Probabilistic"
            
            # 4. IDW interpolation
            idw_results, best_idw = self.idw_interpolation(X_train, y_train, X_test, y_test, feature_name)
            feature_results['idw'] = idw_results[best_idw]
            feature_results['idw']['method'] = f"IDW ({best_idw})"
            
            # 5. Simple MLP
            mlp_results, best_mlp = self.simple_mlp_baseline(X_train, y_train, X_test, y_test, feature_name)
            feature_results['mlp'] = mlp_results[best_mlp]
            feature_results['mlp']['method'] = f"MLP ({best_mlp})"
            
            all_results[feature_name] = feature_results
        
        # Save results
        self.results = all_results
        
        return all_results
    
    def create_results_summary_table(self):
        """Create comprehensive results summary table"""
        
        print("\n📊 COMPREHENSIVE BASELINE RESULTS SUMMARY")
        print("="*80)
        
        # Prepare data for table
        table_data = []
        
        for feature_name, feature_results in self.results.items():
            for method_name, result in feature_results.items():
                table_data.append([
                    feature_name.replace('_', ' ').title(),
                    result['method'],
                    f"{result['mean_error']:.3f}",
                    f"{result['median_error']:.3f}",
                    f"{result['std_error']:.3f}",
                    f"{result['accuracy_1m']:.1f}%",
                    f"{result['accuracy_2m']:.1f}%",
                    f"{result['training_time']:.1f}s"
                ])
        
        # Sort by median error
        table_data.sort(key=lambda x: float(x[2]))
        
        # Create DataFrame for nice display
        df = pd.DataFrame(table_data, columns=[
            'Feature Set', 'Method', 'Mean Error (m)', 'Median Error (m)', 
            'Std Error (m)', '<1m Acc', '<2m Acc', 'Time (s)'
        ])
        
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv('classical_fingerprinting_results.csv', index=False)
        print(f"\n💾 Results saved to: classical_fingerprinting_results.csv")
        
        # Key insights
        print(f"\n🎯 KEY INSIGHTS:")
        best_overall = table_data[0]
        print(f"   🥇 Best Overall: {best_overall[1]} with {best_overall[0]} ({best_overall[3]} median error)")
        
        # Best by feature type
        feature_best = {}
        for feature_name in self.results.keys():
            feature_data = [row for row in table_data if row[0].lower().replace(' ', '_') == feature_name]
            if feature_data:
                feature_best[feature_name] = feature_data[0]
        
        print(f"\n   📊 Best by Feature Set:")
        for feature_name, best_result in feature_best.items():
            print(f"      {feature_name}: {best_result[1]} ({best_result[3]} error)")
        
        return df

def main():
    """Main execution function"""
    
    baseline_system = ClassicalFingerprintingBaselines()
    
    # Run comprehensive comparison for 750 samples
    results = baseline_system.run_comprehensive_baseline_comparison(sample_size=750)
    
    # Create summary table
    summary_df = baseline_system.create_results_summary_table()
    
    print(f"\n✅ CLASSICAL FINGERPRINTING BASELINE COMPARISON COMPLETE!")
    print(f"📊 Tested {len(results)} feature sets with 5 classical methods each")
    print(f"📈 Total configurations: {sum(len(methods) for methods in results.values())}")
    print(f"\n🔍 This establishes whether complex CNNs actually improve over classical baselines!")

if __name__ == "__main__":
    main()
