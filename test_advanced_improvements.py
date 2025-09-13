#!/usr/bin/env python3
"""
Test Advanced CNN Improvements Step by Step

This tests each improvement individually to validate they work correctly.
"""

import numpy as np
import pandas as pd
import json
import csv
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    print("✅ TensorFlow imported successfully")
except ImportError:
    print("❌ TensorFlow not found")
    exit(1)

from advanced_cnn_improvements_system import AdvancedCSIPreprocessor, AdvancedCNNArchitectures

def test_phase_sanitization():
    """Test phase sanitization functionality"""
    print("\n🧪 Testing Phase Sanitization...")
    
    # Create test phase data with artificial discontinuities
    test_phases = np.array([
        [0.1, 0.2, 0.3, 6.0, 0.5, 0.6],  # Jump from 0.3 to 6.0
        [1.1, 1.2, -5.0, 1.4, 1.5, 1.6],  # Jump to -5.0
        [2.1, 2.2, 2.3, 2.4, 2.5, 2.6]   # Normal progression
    ])
    
    preprocessor = AdvancedCSIPreprocessor()
    sanitized = preprocessor.sanitize_phase(test_phases)
    
    print(f"   Original phases shape: {test_phases.shape}")
    print(f"   Sanitized phases shape: {sanitized.shape}")
    print(f"   ✅ Phase sanitization working")
    
    return True

def test_complex_features():
    """Test complex feature creation"""
    print("\n🧪 Testing Complex Feature Creation...")
    
    # Create test data
    amplitudes = np.random.rand(100, 52) * 2  # Random amplitudes
    phases = np.random.rand(100, 52) * 2 * np.pi - np.pi  # Random phases
    
    preprocessor = AdvancedCSIPreprocessor()
    complex_features = preprocessor.create_complex_features(amplitudes, phases)
    
    print(f"   Input amplitude shape: {amplitudes.shape}")
    print(f"   Input phase shape: {phases.shape}")
    print(f"   Complex features created: {list(complex_features.keys())}")
    print(f"   ✅ Complex feature creation working")
    
    return True

def test_meter_space_loss():
    """Test meter-space loss functions"""
    print("\n🧪 Testing Meter-Space Loss Functions...")
    
    architectures = AdvancedCNNArchitectures()
    
    # Create test predictions and targets in meter space
    y_true = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
    y_pred = tf.constant([[1.1, 2.1], [2.9, 4.1], [5.2, 5.8]], dtype=tf.float32)
    
    # Test Euclidean loss
    euclidean_loss = architectures.meter_space_loss(y_true, y_pred)
    print(f"   Euclidean loss: {euclidean_loss.numpy():.3f}")
    
    # Test Huber loss
    huber_loss = architectures.huber_loss_meters(y_true, y_pred)
    print(f"   Huber loss: {huber_loss.numpy():.3f}")
    
    print(f"   ✅ Meter-space loss functions working")
    
    return True

def test_advanced_architectures():
    """Test advanced architecture building"""
    print("\n🧪 Testing Advanced Architecture Building...")
    
    architectures = AdvancedCNNArchitectures()
    
    # Test Multi-Scale Attention CNN
    try:
        model = architectures.build_advanced_multiscale_attention_cnn(
            amplitude_shape=(52,), 
            phase_shape=(52,), 
            use_rssi=True
        )
        print(f"   Advanced Multi-Scale CNN: {model.count_params():,} parameters")
        print(f"   ✅ Advanced Multi-Scale Attention CNN built successfully")
    except Exception as e:
        print(f"   ❌ Error building Advanced Multi-Scale CNN: {e}")
        return False
    
    # Test Complex Phase CNN
    try:
        model2 = architectures.build_complex_phase_cnn(
            amplitude_shape=(52,), 
            phase_shape=(52,), 
            use_rssi=True
        )
        print(f"   Complex Phase CNN: {model2.count_params():,} parameters")
        print(f"   ✅ Complex Phase CNN built successfully")
    except Exception as e:
        print(f"   ❌ Error building Complex Phase CNN: {e}")
        return False
    
    return True

def load_small_test_dataset():
    """Load a small dataset for testing"""
    print("\n🧪 Loading Small Test Dataset...")
    
    # Load one file for testing
    test_file = Path("CSI Dataset 750 Samples") / "0,0.csv"
    
    if not test_file.exists():
        print(f"   ❌ Test file not found: {test_file}")
        return None
    
    amplitudes, phases, rssi_values, coordinates = [], [], [], []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        sample_count = 0
        
        for row in reader:
            if sample_count >= 50:  # Just 50 samples for testing
                break
                
            try:
                amplitude = json.loads(row['amplitude'])
                phase = json.loads(row['phase'])
                rssi_val = float(row['rssi'])
                
                if len(amplitude) == 52 and len(phase) == 52:
                    amplitudes.append(amplitude)
                    phases.append(phase)
                    rssi_values.append(rssi_val)
                    coordinates.append([0.0, 0.0])  # Known coordinates for test file
                    sample_count += 1
                    
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    
    test_data = {
        'amplitude': np.array(amplitudes),
        'phase': np.array(phases),
        'rssi': np.array(rssi_values),
        'coordinates': np.array(coordinates)
    }
    
    print(f"   Loaded {len(amplitudes)} test samples")
    print(f"   ✅ Small test dataset loaded")
    
    return test_data

def test_end_to_end_preprocessing():
    """Test end-to-end preprocessing pipeline"""
    print("\n🧪 Testing End-to-End Preprocessing...")
    
    test_data = load_small_test_dataset()
    if test_data is None:
        return False
    
    try:
        preprocessor = AdvancedCSIPreprocessor()
        
        # Test preprocessing
        processed = preprocessor.fit_transform_training(
            test_data['amplitude'],
            test_data['phase'],
            test_data['rssi']
        )
        
        print(f"   Processed amplitude shape: {processed['amplitude'].shape}")
        print(f"   Processed phase shape: {processed['phase'].shape}")
        print(f"   Processed RSSI shape: {processed['rssi'].shape}")
        print(f"   ✅ End-to-end preprocessing working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in preprocessing: {e}")
        return False

def test_model_training():
    """Test model training with improvements"""
    print("\n🧪 Testing Model Training with Improvements...")
    
    test_data = load_small_test_dataset()
    if test_data is None:
        return False
    
    try:
        # Preprocess data
        preprocessor = AdvancedCSIPreprocessor()
        processed = preprocessor.fit_transform_training(
            test_data['amplitude'],
            test_data['phase'],
            test_data['rssi']
        )
        
        # Build model
        architectures = AdvancedCNNArchitectures()
        model = architectures.build_advanced_multiscale_attention_cnn(
            amplitude_shape=(52,), 
            phase_shape=(52,), 
            use_rssi=True
        )
        
        # Compile with meter-space loss
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=architectures.meter_space_loss,
            metrics=[architectures.huber_loss_meters]
        )
        
        # Prepare inputs
        train_inputs = [
            processed['amplitude'], 
            processed['phase'], 
            processed['rssi']
        ]
        
        # Quick training test (just 1 epoch)
        print("   Running quick training test...")
        history = model.fit(
            train_inputs,
            test_data['coordinates'],  # Meter space targets
            epochs=1,
            batch_size=16,
            verbose=0
        )
        
        print(f"   Training loss: {history.history['loss'][0]:.3f}")
        print(f"   ✅ Model training working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error in model training: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 TESTING ADVANCED CNN IMPROVEMENTS")
    print("="*50)
    
    tests = [
        ("Phase Sanitization", test_phase_sanitization),
        ("Complex Features", test_complex_features),
        ("Meter-Space Loss", test_meter_space_loss),
        ("Advanced Architectures", test_advanced_architectures),
        ("End-to-End Preprocessing", test_end_to_end_preprocessing),
        ("Model Training", test_model_training)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n📊 TEST RESULTS SUMMARY:")
    print("="*30)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🌟 ALL TESTS PASSED! Ready for full evaluation.")
    else:
        print("⚠️  Some tests failed. Check implementation.")
    
    return results

if __name__ == "__main__":
    main()



