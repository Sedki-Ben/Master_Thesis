#!/usr/bin/env python3
"""
Focused Advanced CNN Evaluation - Test improvements on 750 sample size first

This runs a focused evaluation to demonstrate the improvements work,
then can be extended to all sample sizes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print("✅ TensorFlow imported successfully")
except ImportError:
    print("❌ TensorFlow not found")
    exit(1)

from advanced_cnn_improvements_system import AdvancedCSIPreprocessor, AdvancedCNNArchitectures, RobustAugmentation

class FocusedAdvancedEvaluator:
    """Focused evaluator for advanced improvements"""
    
    def __init__(self, sample_size=750):
        self.sample_size = sample_size
        self.preprocessor = AdvancedCSIPreprocessor()
        self.architectures = AdvancedCNNArchitectures()
        self.augmenter = RobustAugmentation()
        
        # Fixed splits
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
    
    def load_reference_data(self, reference_list, is_test=False):
        """Load data for specific reference points"""
        
        amplitudes, phases, rssi_values, coordinates = [], [], [], []
        
        for ref_point in reference_list:
            x, y = ref_point
            
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
        """Load training, validation, and test data"""
        print(f"🔄 Loading data (sample size: {self.sample_size})...")
        
        train_data = self.load_reference_data(self.train_references)
        val_data = self.load_reference_data(self.val_references)
        test_data = self.load_reference_data(self.test_references, is_test=True)
        
        print(f"✅ Data loaded:")
        print(f"   Training: {len(train_data['coordinates'])} samples")
        print(f"   Validation: {len(val_data['coordinates'])} samples") 
        print(f"   Test: {len(test_data['coordinates'])} samples")
        
        return train_data, val_data, test_data
    
    def preprocess_data(self, train_data, val_data, test_data):
        """Apply advanced preprocessing"""
        print("🔧 Applying advanced preprocessing with improvements...")
        
        # Fit on training data
        train_processed = self.preprocessor.fit_transform_training(
            train_data['amplitude'],
            train_data['phase'], 
            train_data['rssi']
        )
        
        # Transform validation and test
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
        print("🔧 Applying robust augmentation...")
        aug_amplitude, aug_phase = self.augmenter.apply_augmentation(
            train_processed['amplitude'],
            train_processed['phase']
        )
        
        return {
            'train': {
                'amplitude': aug_amplitude.astype(np.float32),
                'phase': aug_phase.astype(np.float32),
                'rssi': train_processed['rssi'].astype(np.float32),
                'coordinates': train_data['coordinates'].astype(np.float32)  # Keep in meter space
            },
            'val': {
                'amplitude': val_processed['amplitude'].astype(np.float32),
                'phase': val_processed['phase'].astype(np.float32),
                'rssi': val_processed['rssi'].astype(np.float32),
                'coordinates': val_data['coordinates'].astype(np.float32)    # Keep in meter space
            },
            'test': {
                'amplitude': test_processed['amplitude'].astype(np.float32),
                'phase': test_processed['phase'].astype(np.float32),
                'rssi': test_processed['rssi'].astype(np.float32),
                'coordinates': test_data['coordinates'].astype(np.float32)   # Keep in meter space
            }
        }
    
    def train_model(self, model, processed_data, model_name):
        """Train model with advanced techniques"""
        print(f"🎯 Training {model_name} with advanced improvements...")
        
        # Compile with meter-space loss (NO coordinate scaling!)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=self.architectures.meter_space_loss,  # Loss in meters
            metrics=[self.architectures.huber_loss_meters]
        )
        
        # Prepare inputs
        train_inputs = [
            processed_data['train']['amplitude'],
            processed_data['train']['phase'],
            processed_data['train']['rssi']
        ]
        
        val_inputs = [
            processed_data['val']['amplitude'],
            processed_data['val']['phase'],
            processed_data['val']['rssi']
        ]
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=25, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=12, min_lr=1e-7),
            ModelCheckpoint(f'advanced_{model_name.lower()}.h5', save_best_only=True)
        ]
        
        print(f"      Model Parameters: {model.count_params():,}")
        print(f"      Training in METER SPACE (no coordinate scaling)")
        
        start_time = time.time()
        
        # Train with meter-space coordinates
        history = model.fit(
            train_inputs,
            processed_data['train']['coordinates'],  # Raw meter coordinates
            validation_data=(val_inputs, processed_data['val']['coordinates']),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"      ✅ Training completed in {training_time:.1f}s")
        
        return model, history, training_time
    
    def evaluate_model(self, model, processed_data, model_name):
        """Evaluate model in meter space"""
        print(f"📊 Evaluating {model_name}...")
        
        test_inputs = [
            processed_data['test']['amplitude'],
            processed_data['test']['phase'],
            processed_data['test']['rssi']
        ]
        
        # Predict in meter space
        predictions = model.predict(test_inputs, verbose=0)
        targets = processed_data['test']['coordinates']
        
        # Calculate errors in meters
        errors = np.sqrt(np.sum((targets - predictions)**2, axis=1))
        
        results = {
            'model_name': model_name,
            'mean_error_m': np.mean(errors),
            'median_error_m': np.median(errors),
            'std_error_m': np.std(errors),
            'max_error_m': np.max(errors),
            'min_error_m': np.min(errors),
            'accuracy_50cm_pct': np.mean(errors < 0.5) * 100,
            'accuracy_1m_pct': np.mean(errors < 1.0) * 100,
            'accuracy_2m_pct': np.mean(errors < 2.0) * 100,
            'predictions': predictions,
            'targets': targets,
            'errors': errors
        }
        
        print(f"      📏 Mean Error: {results['mean_error_m']:.3f}m")
        print(f"      📏 Median Error: {results['median_error_m']:.3f}m")
        print(f"      🎯 <1m Accuracy: {results['accuracy_1m_pct']:.1f}%")
        print(f"      🎯 <50cm Accuracy: {results['accuracy_50cm_pct']:.1f}%")
        
        return results

def run_focused_evaluation():
    """Run focused evaluation with advanced improvements"""
    
    print("🚀 FOCUSED ADVANCED CNN EVALUATION")
    print("="*50)
    print("🎯 Testing ALL advanced improvements:")
    print("   ✅ Meter-space loss (no target scaling)")
    print("   ✅ Phase sanitization (unwrap, slope/offset removal)")
    print("   ✅ Complex/polar processing")
    print("   ✅ Multi-scale + multi-head attention + residual")
    print("   ✅ FiLM-gated RSSI modulation")
    print("   ✅ Robust augmentation (subcarrier dropout, jitter)")
    print("   ✅ Fixed train/val/test split (27/7/5 points)")
    
    evaluator = FocusedAdvancedEvaluator(sample_size=750)
    
    # Load and preprocess data
    train_data, val_data, test_data = evaluator.load_all_data()
    processed_data = evaluator.preprocess_data(train_data, val_data, test_data)
    
    # Models to test
    models_to_test = [
        {
            'name': 'Advanced Multi-Scale Attention CNN',
            'builder': evaluator.architectures.build_advanced_multiscale_attention_cnn,
            'params': {'amplitude_shape': (52,), 'phase_shape': (52,), 'use_rssi': True}
        },
        {
            'name': 'Complex Phase CNN',
            'builder': evaluator.architectures.build_complex_phase_cnn,
            'params': {'amplitude_shape': (52,), 'phase_shape': (52,), 'use_rssi': True}
        }
    ]
    
    results = []
    
    for model_config in models_to_test:
        print(f"\n{'='*20} {model_config['name']} {'='*20}")
        
        # Build model
        model = model_config['builder'](**model_config['params'])
        
        # Train model
        trained_model, history, training_time = evaluator.train_model(
            model, processed_data, model_config['name']
        )
        
        # Evaluate model
        result = evaluator.evaluate_model(
            trained_model, processed_data, model_config['name']
        )
        
        result['training_time_s'] = training_time
        result['sample_size'] = 750
        results.append(result)
    
    return results

def analyze_results(results):
    """Analyze and compare results"""
    print(f"\n🏆 ADVANCED CNN RESULTS ANALYSIS")
    print("="*50)
    
    # Sort by median error
    results.sort(key=lambda x: x['median_error_m'])
    
    print(f"{'Model':<35} {'Mean (m)':<10} {'Median (m)':<12} {'<1m %':<8} {'<50cm %':<10}")
    print("-"*80)
    
    for result in results:
        print(f"{result['model_name']:<35} {result['mean_error_m']:<10.3f} "
              f"{result['median_error_m']:<12.3f} {result['accuracy_1m_pct']:<8.1f} "
              f"{result['accuracy_50cm_pct']:<10.1f}")
    
    # Best result
    best = results[0]
    print(f"\n🥇 BEST ADVANCED MODEL:")
    print(f"   Model: {best['model_name']}")
    print(f"   📏 Median Error: {best['median_error_m']:.3f}m")
    print(f"   📏 Mean Error: {best['mean_error_m']:.3f}m")
    print(f"   🎯 <1m Accuracy: {best['accuracy_1m_pct']:.1f}%")
    print(f"   🎯 <50cm Accuracy: {best['accuracy_50cm_pct']:.1f}%")
    
    # Compare to our best previous result (1.423m median from amplitude hybrid)
    previous_best = 1.423
    improvement = previous_best - best['median_error_m']
    
    print(f"\n📈 IMPROVEMENT ANALYSIS:")
    print(f"   Previous best median error: {previous_best:.3f}m")
    print(f"   New best median error: {best['median_error_m']:.3f}m")
    print(f"   Improvement: {improvement:.3f}m ({improvement/previous_best*100:.1f}%)")
    
    if best['median_error_m'] < previous_best:
        print(f"   🌟 IMPROVEMENT ACHIEVED! All advanced techniques working!")
    else:
        print(f"   📊 No improvement over previous best (may need more tuning)")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('focused_advanced_results.csv', index=False)
    print(f"\n📁 Results saved: focused_advanced_results.csv")
    
    return results

def main():
    """Main execution"""
    
    print("🚀 FOCUSED ADVANCED CNN EVALUATION - ALL IMPROVEMENTS")
    print("="*60)
    
    # Run evaluation
    results = run_focused_evaluation()
    
    # Analyze results
    analyzed_results = analyze_results(results)
    
    print("\n✅ FOCUSED EVALUATION COMPLETE!")
    print("💡 If improvements are working, extend to all 3 sample sizes")
    
    return analyzed_results

if __name__ == "__main__":
    main()
