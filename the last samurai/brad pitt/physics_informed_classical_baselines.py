#!/usr/bin/env python3
"""
Physics-Informed Classical Baselines
===================================

Enhanced classical fingerprinting methods using physics-informed features
discovered from spectral analysis. This provides a fair comparison against
our advanced CNN approaches.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import json
import csv
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import coordinates and enhanced features
import sys
sys.path.append('..')
sys.path.append('../..')
from coordinates_config import get_training_points, get_validation_points, get_testing_points
from enhanced_csi_features import EnhancedCSIFeatures

class PhysicsInformedClassicalBaselines:
    """
    Classical localization methods enhanced with physics-informed features
    """
    
    def __init__(self, output_dir="brad_pitt_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load coordinates
        self.training_points = get_training_points()
        self.validation_points = get_validation_points()
        self.testing_points = get_testing_points()
        
        # Initialize enhanced feature extractor
        self.feature_extractor = EnhancedCSIFeatures()
        
        print("🔬 Physics-Informed Classical Baselines Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        
    def load_classical_data(self, dataset_size, point_type="training"):
        """Load data for classical methods with enhanced features"""
        
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
        
        print(f"📊 Loading {point_type} data for classical methods...")
        
        # Feature storage
        raw_features = []  # Raw CSI amplitude + phase + RSSI
        physics_features = []  # Enhanced physics features
        combined_features = []  # All features combined
        coordinates = []
        
        for x, y in points:
            file_path = Path("../..") / folder / f"{x},{y}.csv"
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            amps = json.loads(row['amplitude'])
                            phases_data = json.loads(row['phase'])
                            rssi = float(row['rssi'])
                            
                            if len(amps) == 52 and len(phases_data) == 52:
                                # Raw features (traditional approach)
                                raw_feat = amps + phases_data + [rssi]  # 52 + 52 + 1 = 105 features
                                raw_features.append(raw_feat)
                                
                                # Enhanced physics features
                                enhanced_features = self.feature_extractor.extract_all_features(
                                    np.array(amps), np.array(phases_data)
                                )
                                physics_feat = list(enhanced_features.values())
                                physics_features.append(physics_feat)
                                
                                # Combined features
                                combined_feat = raw_feat + physics_feat
                                combined_features.append(combined_feat)
                                
                                coordinates.append([x, y])
                                
                        except Exception as e:
                            continue
            else:
                print(f"    Warning: File not found: {file_path}")
        
        print(f"✅ Loaded {len(coordinates)} samples")
        print(f"   • Raw features: {len(raw_features[0]) if raw_features else 0}")
        print(f"   • Physics features: {len(physics_features[0]) if physics_features else 0}")
        print(f"   • Combined features: {len(combined_features[0]) if combined_features else 0}")
        
        return {
            'raw_features': np.array(raw_features),
            'physics_features': np.array(physics_features),
            'combined_features': np.array(combined_features),
            'coordinates': np.array(coordinates)
        }
    
    def train_knn_variants(self, train_data, test_data, dataset_size):
        """Train k-NN variants with different feature sets"""
        print(f"🔍 Training k-NN variants ({dataset_size} samples)...")
        
        results = []
        
        # Feature sets to test
        feature_sets = {
            'raw': train_data['raw_features'],
            'physics': train_data['physics_features'], 
            'combined': train_data['combined_features']
        }
        
        test_feature_sets = {
            'raw': test_data['raw_features'],
            'physics': test_data['physics_features'],
            'combined': test_data['combined_features']
        }
        
        # k values to test
        k_values = [1, 3, 5, 7, 10]
        
        for feature_type, train_features in feature_sets.items():
            test_features = test_feature_sets[feature_type]
            
            # Scale features
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
            for k in k_values:
                # Train k-NN
                knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
                knn.fit(train_features_scaled, train_data['coordinates'])
                
                # Predict
                predictions = knn.predict(test_features_scaled)
                
                # Calculate errors
                errors = np.sqrt(np.sum((test_data['coordinates'] - predictions)**2, axis=1))
                
                result = {
                    'method': f'kNN_k{k}',
                    'features': feature_type,
                    'dataset_size': dataset_size,
                    'n_features': train_features.shape[1],
                    'mean_error_m': np.mean(errors),
                    'median_error_m': np.median(errors),
                    'std_error_m': np.std(errors),
                    'min_error_m': np.min(errors),
                    'max_error_m': np.max(errors),
                    'accuracy_50cm': np.mean(errors < 0.5) * 100,
                    'accuracy_1m': np.mean(errors < 1.0) * 100,
                    'accuracy_2m': np.mean(errors < 2.0) * 100,
                    'accuracy_3m': np.mean(errors < 3.0) * 100
                }
                
                results.append(result)
                
                print(f"   • k-NN (k={k}, {feature_type}): {np.median(errors):.3f}m median error")
        
        return results
    
    def train_random_forest(self, train_data, test_data, dataset_size):
        """Train Random Forest with different feature sets"""
        print(f"🌲 Training Random Forest ({dataset_size} samples)...")
        
        results = []
        
        feature_sets = {
            'raw': train_data['raw_features'],
            'physics': train_data['physics_features'], 
            'combined': train_data['combined_features']
        }
        
        test_feature_sets = {
            'raw': test_data['raw_features'],
            'physics': test_data['physics_features'],
            'combined': test_data['combined_features']
        }
        
        # Random Forest parameters
        rf_configs = [
            {'n_estimators': 100, 'max_depth': None},
            {'n_estimators': 200, 'max_depth': 10},
            {'n_estimators': 500, 'max_depth': 15}
        ]
        
        for feature_type, train_features in feature_sets.items():
            test_features = test_feature_sets[feature_type]
            
            # Scale features
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
            for i, config in enumerate(rf_configs):
                # Train Random Forest
                rf = RandomForestRegressor(
                    n_estimators=config['n_estimators'],
                    max_depth=config['max_depth'],
                    random_state=42,
                    n_jobs=-1
                )
                rf.fit(train_features_scaled, train_data['coordinates'])
                
                # Predict
                predictions = rf.predict(test_features_scaled)
                
                # Calculate errors
                errors = np.sqrt(np.sum((test_data['coordinates'] - predictions)**2, axis=1))
                
                result = {
                    'method': f'RandomForest_config{i+1}',
                    'features': feature_type,
                    'dataset_size': dataset_size,
                    'n_features': train_features.shape[1],
                    'mean_error_m': np.mean(errors),
                    'median_error_m': np.median(errors),
                    'std_error_m': np.std(errors),
                    'min_error_m': np.min(errors),
                    'max_error_m': np.max(errors),
                    'accuracy_50cm': np.mean(errors < 0.5) * 100,
                    'accuracy_1m': np.mean(errors < 1.0) * 100,
                    'accuracy_2m': np.mean(errors < 2.0) * 100,
                    'accuracy_3m': np.mean(errors < 3.0) * 100
                }
                
                results.append(result)
                
                print(f"   • RF ({config['n_estimators']} trees, {feature_type}): {np.median(errors):.3f}m median error")
        
        return results
    
    def train_enhanced_mlp(self, train_data, test_data, dataset_size):
        """Train MLP with different feature sets"""
        print(f"🧠 Training Enhanced MLP ({dataset_size} samples)...")
        
        results = []
        
        feature_sets = {
            'raw': train_data['raw_features'],
            'physics': train_data['physics_features'], 
            'combined': train_data['combined_features']
        }
        
        test_feature_sets = {
            'raw': test_data['raw_features'],
            'physics': test_data['physics_features'],
            'combined': test_data['combined_features']
        }
        
        # MLP configurations
        mlp_configs = [
            {'hidden_layer_sizes': (100,), 'alpha': 0.001},
            {'hidden_layer_sizes': (100, 50), 'alpha': 0.001},
            {'hidden_layer_sizes': (200, 100, 50), 'alpha': 0.01}
        ]
        
        for feature_type, train_features in feature_sets.items():
            test_features = test_feature_sets[feature_type]
            
            # Scale features
            scaler = StandardScaler()
            train_features_scaled = scaler.fit_transform(train_features)
            test_features_scaled = scaler.transform(test_features)
            
            for i, config in enumerate(mlp_configs):
                # Train MLP
                mlp = MLPRegressor(
                    hidden_layer_sizes=config['hidden_layer_sizes'],
                    alpha=config['alpha'],
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
                
                try:
                    mlp.fit(train_features_scaled, train_data['coordinates'])
                    
                    # Predict
                    predictions = mlp.predict(test_features_scaled)
                    
                    # Calculate errors
                    errors = np.sqrt(np.sum((test_data['coordinates'] - predictions)**2, axis=1))
                    
                    result = {
                        'method': f'MLP_config{i+1}',
                        'features': feature_type,
                        'dataset_size': dataset_size,
                        'n_features': train_features.shape[1],
                        'mean_error_m': np.mean(errors),
                        'median_error_m': np.median(errors),
                        'std_error_m': np.std(errors),
                        'min_error_m': np.min(errors),
                        'max_error_m': np.max(errors),
                        'accuracy_50cm': np.mean(errors < 0.5) * 100,
                        'accuracy_1m': np.mean(errors < 1.0) * 100,
                        'accuracy_2m': np.mean(errors < 2.0) * 100,
                        'accuracy_3m': np.mean(errors < 3.0) * 100
                    }
                    
                    results.append(result)
                    
                    print(f"   • MLP ({config['hidden_layer_sizes']}, {feature_type}): {np.median(errors):.3f}m median error")
                    
                except Exception as e:
                    print(f"   ⚠️  MLP training failed: {e}")
                    continue
        
        return results
    
    def run_classical_comparison(self, dataset_sizes=[250, 500, 750]):
        """Run comprehensive classical baseline comparison"""
        print("🔬 Physics-Informed Classical Baselines Comparison")
        print("=" * 60)
        
        all_results = []
        
        for dataset_size in dataset_sizes:
            print(f"\n📊 Evaluating on {dataset_size} samples...")
            
            # Load data
            train_data = self.load_classical_data(dataset_size, "training")
            test_data = self.load_classical_data(750, "testing")  # Always test on 750
            
            if len(train_data['coordinates']) == 0 or len(test_data['coordinates']) == 0:
                print(f"⚠️  No data available for {dataset_size} samples")
                continue
            
            # Train different methods
            knn_results = self.train_knn_variants(train_data, test_data, dataset_size)
            rf_results = self.train_random_forest(train_data, test_data, dataset_size)
            mlp_results = self.train_enhanced_mlp(train_data, test_data, dataset_size)
            
            all_results.extend(knn_results + rf_results + mlp_results)
        
        # Save results
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_path = self.output_dir / "physics_informed_classical_results.csv"
            results_df.to_csv(results_path, index=False)
            
            print(f"\n✅ Classical baseline comparison complete!")
            print(f"📁 Results saved to: {results_path}")
            
            # Find best methods
            print(f"\n🏆 TOP CLASSICAL METHODS:")
            best_overall = results_df.loc[results_df['median_error_m'].idxmin()]
            print(f"   • Best Overall: {best_overall['method']} ({best_overall['features']}) - {best_overall['median_error_m']:.3f}m")
            
            # Best by feature type
            for feature_type in ['raw', 'physics', 'combined']:
                subset = results_df[results_df['features'] == feature_type]
                if not subset.empty:
                    best_feat = subset.loc[subset['median_error_m'].idxmin()]
                    print(f"   • Best {feature_type}: {best_feat['method']} - {best_feat['median_error_m']:.3f}m")
            
            return results_df
        else:
            print("❌ No results obtained")
            return None

def main():
    """Main execution function"""
    print("🔬 Physics-Informed Classical Baselines")
    print("=" * 40)
    
    evaluator = PhysicsInformedClassicalBaselines()
    results = evaluator.run_classical_comparison([250, 500, 750])
    
    return evaluator, results

if __name__ == "__main__":
    evaluator, results = main()
