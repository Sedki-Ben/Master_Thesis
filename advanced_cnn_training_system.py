#!/usr/bin/env python3
"""
Complete Advanced CNN Training System with All Improvements

This implements the complete training and evaluation pipeline with all suggested improvements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print("✅ TensorFlow imported successfully")
except ImportError:
    print("❌ TensorFlow not found. Please install with: pip install tensorflow")
    exit(1)

# Import our advanced components
from advanced_cnn_improvements_system import (
    AdvancedCSIPreprocessor, 
    AdvancedCNNArchitectures, 
    RobustAugmentation
)

class LeaveOneReferenceOutEvaluator:
    """
    Leave-one-reference-out validation protocol
    More robust than random splits - prevents location leakage
    """
    
    def __init__(self, reference_points):
        self.reference_points = reference_points
        self.n_refs = len(reference_points)
        
    def generate_splits(self):
        """Generate leave-one-out splits for robust validation"""
        print(f"🔄 Generating leave-one-reference-out splits ({self.n_refs} iterations)...")
        
        splits = []
        for i, test_ref in enumerate(self.reference_points):
            train_refs = [ref for j, ref in enumerate(self.reference_points) if j != i]
            
            splits.append({
                'iteration': i + 1,
                'train_references': train_refs,
                'test_reference': test_ref,
                'test_ref_name': f"{test_ref[0]},{test_ref[1]}"
            })
            
        return splits
    
    def evaluate_model_robustness(self, model, data_loader, splits):
        """Evaluate model across all leave-one-out splits"""
        print("📊 Evaluating model robustness with leave-one-reference-out...")
        
        all_errors = []
        split_results = []
        
        for split in splits:
            print(f"   Testing on reference {split['test_ref_name']} ({split['iteration']}/{len(splits)})...")
            
            # Load data for this split
            train_data = data_loader.load_references(split['train_references'])
            test_data = data_loader.load_references([split['test_reference']])
            
            # Predict on test reference
            predictions = model.predict([test_data['amplitude'], test_data['phase'], test_data['rssi']], verbose=0)
            
            # Calculate errors (in meter space)
            errors = np.sqrt(np.sum((test_data['coordinates'] - predictions)**2, axis=1))
            
            split_result = {
                'test_reference': split['test_ref_name'],
                'mean_error': np.mean(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'max_error': np.max(errors),
                'n_samples': len(errors),
                'accuracy_1m': np.mean(errors < 1.0) * 100
            }
            
            split_results.append(split_result)
            all_errors.extend(errors)
        
        # Overall robustness metrics
        robustness_metrics = {
            'overall_mean_error': np.mean(all_errors),
            'overall_median_error': np.median(all_errors),
            'overall_std_error': np.std(all_errors),
            'mean_of_means': np.mean([r['mean_error'] for r in split_results]),
            'std_of_means': np.std([r['mean_error'] for r in split_results]),
            'worst_reference_error': max([r['mean_error'] for r in split_results]),
            'best_reference_error': min([r['mean_error'] for r in split_results]),
            'split_results': split_results
        }
        
        return robustness_metrics

class AdvancedDataLoader:
    """Advanced data loader with temporal context and improved preprocessing"""
    
    def __init__(self, sample_size=750):
        self.sample_size = sample_size
        self.preprocessor = AdvancedCSIPreprocessor()
        self.augmenter = RobustAugmentation()
        
        # Fixed reference points (27 for training, 7 for validation)
        self.train_references = [
            (0,0), (0,1), (0,2), (0,4), (0,5), (1,0), (1,1), (1,4), (1,5),
            (2,0), (2,2), (2,4), (2,5), (3,0), (3,1), (3,2), (3,4), (3,5),
            (4,0), (4,1), (4,4), (4,5), (5,0), (5,1), (5,2), (5,3), (5,4)
        ]
        
        self.val_references = [
            (4,5), (5,1), (0,3), (0,6), (6,4), (2,1), (3,3)
        ]
        
        self.test_references = [
            (0.5, 0.5), (1.5, 4.5), (2.5, 2.5), (3.5, 1.5), (5.5, 3.5)
        ]
    
    def load_references(self, reference_list, is_test=False):
        """Load data for specific reference points"""
        
        amplitudes, phases, rssi_values, coordinates = [], [], [], []
        
        for ref_point in reference_list:
            x, y = ref_point
            
            # Determine file path based on test vs train
            if is_test:
                file_path = Path(f"Testing Points Dataset {self.sample_size} Samples") / f"{x},{y}.csv"
            else:
                file_path = Path(f"CSI Dataset {self.sample_size} Samples") / f"{int(x)},{int(y)}.csv"
            
            if not file_path.exists():
                print(f"   ⚠️ File not found: {file_path}")
                continue
                
            print(f"   📁 Loading: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                sample_count = 0
                
                for row in reader:
                    if sample_count >= self.sample_size:
                        break
                        
                    try:
                        amplitude = json.loads(row['amplitude'])
                        phase = json.loads(row['phase'])
                        rssi_val = float(row['rssi'])
                        
                        if len(amplitude) == 52 and len(phase) == 52:
                            amplitudes.append(amplitude)
                            phases.append(phase)
                            rssi_values.append(rssi_val)
                            coordinates.append([x, y])
                            sample_count += 1
                            
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
        
        return {
            'amplitude': np.array(amplitudes),
            'phase': np.array(phases),
            'rssi': np.array(rssi_values),
            'coordinates': np.array(coordinates)
        }
    
    def load_all_data(self):
        """Load all training, validation, and test data"""
        print(f"🔄 Loading all data (sample size: {self.sample_size})...")
        
        # Load training data
        print("📂 Loading training data...")
        train_data = self.load_references(self.train_references)
        
        # Load validation data  
        print("📂 Loading validation data...")
        val_data = self.load_references(self.val_references)
        
        # Load test data
        print("📂 Loading test data...")
        test_data = self.load_references(self.test_references, is_test=True)
        
        print(f"✅ Data loaded:")
        print(f"   Training: {len(train_data['coordinates'])} samples from {len(self.train_references)} references")
        print(f"   Validation: {len(val_data['coordinates'])} samples from {len(self.val_references)} references")
        print(f"   Test: {len(test_data['coordinates'])} samples from {len(self.test_references)} references")
        
        return train_data, val_data, test_data
    
    def preprocess_data(self, train_data, val_data, test_data, apply_augmentation=True):
        """Apply advanced preprocessing to all data"""
        print("🔧 Applying advanced preprocessing...")
        
        # Fit preprocessor on training data
        train_processed = self.preprocessor.fit_transform_training(
            train_data['amplitude'], 
            train_data['phase'], 
            train_data['rssi']
        )
        
        # Transform validation and test data
        val_processed = self.preprocessor.transform_test(
            val_data['amplitude'], 
            val_data['phase'], 
            val_data['rssi']
        )
        
        test_processed = self.preprocessor.transform_test(
            test_data['amplitude'], 
            test_data['phase'], 
            test_data['rssi']
        )
        
        # Apply augmentation to training data
        if apply_augmentation:
            print("🔧 Applying robust augmentation to training data...")
            aug_amplitude, aug_phase = self.augmenter.apply_augmentation(
                train_processed['amplitude'], 
                train_processed['phase']
            )
            train_processed['amplitude'] = aug_amplitude
            train_processed['phase'] = aug_phase
        
        # Prepare final data structures
        processed_data = {
            'train': {
                'amplitude': train_processed['amplitude'],
                'phase': train_processed['phase'], 
                'rssi': train_processed['rssi'],
                'coordinates': train_data['coordinates']  # Keep in meter space!
            },
            'val': {
                'amplitude': val_processed['amplitude'],
                'phase': val_processed['phase'],
                'rssi': val_processed['rssi'], 
                'coordinates': val_data['coordinates']   # Keep in meter space!
            },
            'test': {
                'amplitude': test_processed['amplitude'],
                'phase': test_processed['phase'],
                'rssi': test_processed['rssi'],
                'coordinates': test_data['coordinates']  # Keep in meter space!
            }
        }
        
        return processed_data

class AdvancedModelTrainer:
    """Advanced model trainer with all improvements"""
    
    def __init__(self):
        self.architectures = AdvancedCNNArchitectures()
        self.results = []
        
    def train_model_with_improvements(self, model, train_data, val_data, model_name, sample_size):
        """Train model with advanced loss functions and callbacks"""
        print(f"🎯 Training {model_name} (sample size: {sample_size})...")
        
        # Compile with meter-space loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=self.architectures.meter_space_loss,  # NO target scaling!
            metrics=[self.architectures.huber_loss_meters]
        )
        
        # Prepare training inputs
        if model_name.startswith('Advanced'):
            train_inputs = [train_data['amplitude'], train_data['phase'], train_data['rssi']]
            val_inputs = [val_data['amplitude'], val_data['phase'], val_data['rssi']]
        else:
            train_inputs = [train_data['amplitude'], train_data['phase']]
            val_inputs = [val_data['amplitude'], val_data['phase']]
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                patience=25, 
                restore_best_weights=True, 
                monitor='val_loss',
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                factor=0.5, 
                patience=12, 
                min_lr=1e-7,
                monitor='val_loss'
            ),
            ModelCheckpoint(
                f'best_{model_name.lower()}_{sample_size}.h5',
                save_best_only=True, 
                monitor='val_loss'
            )
        ]
        
        print(f"      Model Parameters: {model.count_params():,}")
        print(f"      Training samples: {len(train_data['coordinates'])}")
        print(f"      Validation samples: {len(val_data['coordinates'])}")
        
        start_time = time.time()
        
        # Train with meter-space targets (no scaling!)
        history = model.fit(
            train_inputs, 
            train_data['coordinates'],  # Meter space coordinates
            validation_data=(val_inputs, val_data['coordinates']),  # Meter space coordinates
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        elapsed = time.time() - start_time
        print(f"      ✅ Training completed in {elapsed:.1f}s")
        
        return model, history, elapsed
    
    def evaluate_model_comprehensive(self, model, test_data, model_name, sample_size):
        """Comprehensive model evaluation in meter space"""
        print(f"📊 Evaluating {model_name} (sample size: {sample_size})...")
        
        # Prepare test inputs
        if model_name.startswith('Advanced'):
            test_inputs = [test_data['amplitude'], test_data['phase'], test_data['rssi']]
        else:
            test_inputs = [test_data['amplitude'], test_data['phase']]
        
        # Make predictions (already in meter space)
        predictions = model.predict(test_inputs, verbose=0)
        
        # Calculate errors in meters
        euclidean_errors = np.sqrt(np.sum((test_data['coordinates'] - predictions)**2, axis=1))
        
        results = {
            'model_name': model_name,
            'sample_size': sample_size,
            'mean_error_m': np.mean(euclidean_errors),
            'median_error_m': np.median(euclidean_errors),
            'std_error_m': np.std(euclidean_errors),
            'max_error_m': np.max(euclidean_errors),
            'min_error_m': np.min(euclidean_errors),
            'accuracy_50cm_pct': np.mean(euclidean_errors < 0.5) * 100,
            'accuracy_1m_pct': np.mean(euclidean_errors < 1.0) * 100,
            'accuracy_2m_pct': np.mean(euclidean_errors < 2.0) * 100,
            'errors': euclidean_errors,
            'predictions': predictions,
            'targets': test_data['coordinates']
        }
        
        print(f"      📏 Mean Error: {results['mean_error_m']:.3f}m")
        print(f"      📏 Median Error: {results['median_error_m']:.3f}m")
        print(f"      📏 Std Error: {results['std_error_m']:.3f}m")
        print(f"      🎯 <1m Accuracy: {results['accuracy_1m_pct']:.1f}%")
        print(f"      🎯 <50cm Accuracy: {results['accuracy_50cm_pct']:.1f}%")
        
        return results

def run_comprehensive_evaluation(sample_sizes=[250, 500, 750]):
    """Run comprehensive evaluation with all improvements"""
    
    print("🚀 COMPREHENSIVE ADVANCED CNN EVALUATION")
    print("="*60)
    print("🎯 Testing all improvements on 3 sample sizes")
    print(f"📊 Sample sizes: {sample_sizes}")
    
    all_results = []
    
    for sample_size in sample_sizes:
        print(f"\n{'='*20} SAMPLE SIZE: {sample_size} {'='*20}")
        
        # Initialize components
        data_loader = AdvancedDataLoader(sample_size=sample_size)
        trainer = AdvancedModelTrainer()
        
        # Load and preprocess data
        train_data, val_data, test_data = data_loader.load_all_data()
        processed_data = data_loader.preprocess_data(train_data, val_data, test_data)
        
        # Model architectures to test
        models_to_test = [
            {
                'name': 'Advanced Multi-Scale Attention CNN',
                'builder': trainer.architectures.build_advanced_multiscale_attention_cnn,
                'params': {'amplitude_shape': (52,), 'phase_shape': (52,), 'use_rssi': True}
            },
            {
                'name': 'Complex Phase CNN',
                'builder': trainer.architectures.build_complex_phase_cnn,
                'params': {'amplitude_shape': (52,), 'phase_shape': (52,), 'use_rssi': True}
            }
        ]
        
        for model_config in models_to_test:
            print(f"\n🔵 Testing: {model_config['name']}")
            
            # Build model
            model = model_config['builder'](**model_config['params'])
            
            # Train model
            trained_model, history, training_time = trainer.train_model_with_improvements(
                model, 
                processed_data['train'], 
                processed_data['val'], 
                model_config['name'],
                sample_size
            )
            
            # Evaluate model
            results = trainer.evaluate_model_comprehensive(
                trained_model,
                processed_data['test'],
                model_config['name'],
                sample_size
            )
            
            results['training_time_s'] = training_time
            all_results.append(results)
    
    return all_results

def main():
    """Main execution function"""
    
    print("🚀 ADVANCED CNN IMPROVEMENTS - COMPLETE SYSTEM")
    print("="*60)
    
    # Run comprehensive evaluation
    results = run_comprehensive_evaluation()
    
    # Analyze results
    print("\n🏆 FINAL RESULTS SUMMARY:")
    print("="*80)
    
    # Sort by median error
    results.sort(key=lambda x: x['median_error_m'])
    
    print(f"{'Model':<35} {'Samples':<8} {'Mean (m)':<10} {'Median (m)':<12} {'<1m %':<8} {'Time (s)':<10}")
    print("-"*90)
    
    for result in results:
        print(f"{result['model_name']:<35} {result['sample_size']:<8} "
              f"{result['mean_error_m']:<10.3f} {result['median_error_m']:<12.3f} "
              f"{result['accuracy_1m_pct']:<8.1f} {result['training_time_s']:<10.0f}")
    
    # Best result
    best_result = results[0]
    print(f"\n🥇 BEST MODEL: {best_result['model_name']} ({best_result['sample_size']} samples)")
    print(f"   📏 Median Error: {best_result['median_error_m']:.3f}m")
    print(f"   🎯 <1m Accuracy: {best_result['accuracy_1m_pct']:.1f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('advanced_cnn_improvements_results.csv', index=False)
    print(f"\n📁 Results saved: advanced_cnn_improvements_results.csv")
    
    return results

if __name__ == "__main__":
    main()


