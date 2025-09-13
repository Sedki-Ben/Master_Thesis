#!/usr/bin/env python3
"""
Deep Learning CNN-based Indoor Localization System

Implements sophisticated CNN architectures specifically designed for CSI-based
indoor localization with focus on spatial interpolation and regression accuracy.

Key Design Principles:
1. Handle different scales of amplitude and phase data
2. Capture frequency patterns across 52 subcarriers
3. Learn spatial interpolation between grid points
4. Incorporate multi-scale feature extraction
5. Use attention mechanisms for spatial awareness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow, install if needed
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print("✅ TensorFlow imported successfully")
except ImportError:
    print("❌ TensorFlow not found. Please install with: pip install tensorflow")
    exit(1)

class CNNLocalizationSystem:
    """Advanced CNN-based localization system"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.results = {}
        
    def load_and_prepare_data_verbose(self):
        """Load and prepare data with detailed logging"""
        print("🔄 STEP 1: Loading CSI data for CNN training...")
        
        # Load training data
        train_dir = Path("CSI Dataset 750 Samples")
        train_files = list(train_dir.glob("*.csv"))
        print(f"   📁 Found {len(train_files)} training files")
        
        train_amplitudes, train_phases, train_rssi, train_coords = [], [], [], []
        
        for i, file_path in enumerate(train_files):
            if i % 10 == 0:
                print(f"      Loading training file {i+1}/{len(train_files)}: {file_path.name}")
            
            coords = file_path.stem.split(',')
            x, y = float(coords[0]), float(coords[1])
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        amplitudes = json.loads(row['amplitude'])
                        phases = json.loads(row['phase'])
                        rssi = float(row['rssi'])
                        
                        if len(amplitudes) == 52 and len(phases) == 52:
                            train_amplitudes.append(amplitudes)
                            train_phases.append(phases)
                            train_rssi.append(rssi)
                            train_coords.append([x, y])
                    except:
                        continue
        
        # Load testing data
        test_dir = Path("Testing Points Dataset 750 Samples")
        test_files = list(test_dir.glob("*.csv"))
        print(f"   📁 Found {len(test_files)} testing files")
        
        test_amplitudes, test_phases, test_rssi, test_coords = [], [], [], []
        
        for file_path in test_files:
            print(f"      Loading testing file: {file_path.name}")
            coords = file_path.stem.split(',')
            x, y = float(coords[0]), float(coords[1])
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        amplitudes = json.loads(row['amplitude'])
                        phases = json.loads(row['phase'])
                        rssi = float(row['rssi'])
                        
                        if len(amplitudes) == 52 and len(phases) == 52:
                            test_amplitudes.append(amplitudes)
                            test_phases.append(phases)
                            test_rssi.append(rssi)
                            test_coords.append([x, y])
                    except:
                        continue
        
        # Convert to numpy arrays
        self.train_data = {
            'amplitudes': np.array(train_amplitudes),
            'phases': np.array(train_phases),
            'rssi': np.array(train_rssi),
            'coordinates': np.array(train_coords)
        }
        
        self.test_data = {
            'amplitudes': np.array(test_amplitudes),
            'phases': np.array(test_phases),
            'rssi': np.array(test_rssi),
            'coordinates': np.array(test_coords)
        }
        
        print(f"   ✅ Training samples: {len(train_amplitudes)} from {len(set([tuple(c) for c in train_coords]))} locations")
        print(f"   ✅ Testing samples: {len(test_amplitudes)} from {len(set([tuple(c) for c in test_coords]))} locations")
        
        return self.train_data, self.test_data
    
    def prepare_cnn_inputs(self, train_data, test_data):
        """Prepare CNN inputs with optimal preprocessing"""
        print("🔄 STEP 2: Preparing CNN inputs with advanced preprocessing...")
        
        # Separate normalization for amplitude and phase (different scales)
        print("   📏 Normalizing amplitude and phase separately...")
        amp_scaler = StandardScaler()
        phase_scaler = StandardScaler()
        rssi_scaler = StandardScaler()
        
        # Fit scalers on training data
        train_amp_norm = amp_scaler.fit_transform(train_data['amplitudes'])
        train_phase_norm = phase_scaler.fit_transform(train_data['phases'])
        train_rssi_norm = rssi_scaler.fit_transform(train_data['rssi'].reshape(-1, 1)).flatten()
        
        # Apply to test data
        test_amp_norm = amp_scaler.transform(test_data['amplitudes'])
        test_phase_norm = phase_scaler.transform(test_data['phases'])
        test_rssi_norm = rssi_scaler.transform(test_data['rssi'].reshape(-1, 1)).flatten()
        
        # Create Format 2: (2, 52) - Stacked arrangement (best from analysis)
        print("   🧠 Creating 2×52 stacked CNN input format...")
        train_cnn = np.stack([train_amp_norm, train_phase_norm], axis=1)  # (samples, 2, 52)
        test_cnn = np.stack([test_amp_norm, test_phase_norm], axis=1)
        
        # Normalize coordinates for better training stability
        coord_scaler = MinMaxScaler(feature_range=(0, 1))
        train_coords_norm = coord_scaler.fit_transform(train_data['coordinates'])
        test_coords_norm = coord_scaler.transform(test_data['coordinates'])
        
        print(f"   ✅ CNN input shapes:")
        print(f"      Training CSI: {train_cnn.shape}")
        print(f"      Testing CSI: {test_cnn.shape}")
        print(f"      Training coordinates: {train_coords_norm.shape}")
        print(f"      Testing coordinates: {test_coords_norm.shape}")
        
        # Store scalers
        self.scalers = {
            'amplitude': amp_scaler,
            'phase': phase_scaler,
            'rssi': rssi_scaler,
            'coordinates': coord_scaler
        }
        
        return {
            'train_csi': train_cnn,
            'test_csi': test_cnn,
            'train_rssi': train_rssi_norm,
            'test_rssi': test_rssi_norm,
            'train_coords': train_coords_norm,
            'test_coords': test_coords_norm
        }
    
    def create_data_augmentation(self, train_csi, train_rssi, train_coords):
        """Create data augmentation for better generalization"""
        print("🔄 STEP 3: Implementing data augmentation strategies...")
        
        print("   🎲 Adding Gaussian noise to CSI...")
        # Add noise to CSI (small amount to preserve patterns)
        noise_factor = 0.05
        noisy_csi = train_csi + np.random.normal(0, noise_factor, train_csi.shape)
        
        print("   📶 Adding RSSI variations...")
        # Add small RSSI variations
        rssi_noise = np.random.normal(0, 0.1, train_rssi.shape)
        noisy_rssi = train_rssi + rssi_noise
        
        print("   🎯 Creating synthetic intermediate positions...")
        # Create synthetic data points between existing locations
        synthetic_csi, synthetic_rssi, synthetic_coords = [], [], []
        
        unique_coords = np.unique(train_coords, axis=0)
        print(f"      Generating synthetic data between {len(unique_coords)} unique locations...")
        
        for i in range(min(len(unique_coords), 10)):  # Limit for speed
            for j in range(i+1, min(len(unique_coords), 10)):
                coord1, coord2 = unique_coords[i], unique_coords[j]
                
                # Find samples for these coordinates
                idx1 = np.where((train_coords == coord1).all(axis=1))[0]
                idx2 = np.where((train_coords == coord2).all(axis=1))[0]
                
                if len(idx1) > 0 and len(idx2) > 0:
                    # Interpolate between locations
                    alpha = 0.5  # Midpoint
                    interp_coord = alpha * coord1 + (1 - alpha) * coord2
                    
                    # Take a few samples from each location and interpolate
                    for k in range(min(10, len(idx1), len(idx2))):
                        csi1, csi2 = train_csi[idx1[k]], train_csi[idx2[k]]
                        rssi1, rssi2 = train_rssi[idx1[k]], train_rssi[idx2[k]]
                        
                        interp_csi = alpha * csi1 + (1 - alpha) * csi2
                        interp_rssi = alpha * rssi1 + (1 - alpha) * rssi2
                        
                        synthetic_csi.append(interp_csi)
                        synthetic_rssi.append(interp_rssi)
                        synthetic_coords.append(interp_coord)
        
        if synthetic_csi:
            synthetic_csi = np.array(synthetic_csi)
            synthetic_rssi = np.array(synthetic_rssi)
            synthetic_coords = np.array(synthetic_coords)
            
            print(f"      Generated {len(synthetic_csi)} synthetic samples")
            
            # Combine all augmented data
            augmented_csi = np.concatenate([train_csi, noisy_csi, synthetic_csi])
            augmented_rssi = np.concatenate([train_rssi, noisy_rssi, synthetic_rssi])
            augmented_coords = np.concatenate([train_coords, train_coords, synthetic_coords])
        else:
            augmented_csi = np.concatenate([train_csi, noisy_csi])
            augmented_rssi = np.concatenate([train_rssi, noisy_rssi])
            augmented_coords = np.concatenate([train_coords, train_coords])
        
        print(f"   ✅ Augmented dataset: {len(augmented_csi)} samples (original: {len(train_csi)})")
        
        return augmented_csi, augmented_rssi, augmented_coords
    
    def build_basic_cnn(self, input_shape):
        """Build basic CNN architecture - Baseline"""
        print("   🏗️ Building Basic CNN Architecture...")
        print("      Architecture: Simple 1D Conv → Dense layers")
        print("      Purpose: Baseline CNN for comparison")
        
        # CSI input
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        
        # Reshape for 1D convolution along frequency dimension
        x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)  # (52, 2)
        
        # 1D Convolutions along frequency
        x = layers.Conv1D(32, 5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=csi_input, outputs=output, name='BasicCNN')
        return model
    
    def build_multiscale_cnn(self, input_shape):
        """Build multi-scale CNN - Captures patterns at different scales"""
        print("   🏗️ Building Multi-Scale CNN Architecture...")
        print("      Architecture: Multiple parallel conv paths → Concatenation")
        print("      Purpose: Capture both local and global frequency patterns")
        
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)  # (52, 2)
        
        # Multiple parallel paths with different kernel sizes
        # Path 1: Local patterns (small kernels)
        path1 = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        path1 = layers.BatchNormalization()(path1)
        path1 = layers.MaxPooling1D(2)(path1)
        
        # Path 2: Medium patterns
        path2 = layers.Conv1D(32, 7, activation='relu', padding='same')(x)
        path2 = layers.BatchNormalization()(path2)
        path2 = layers.MaxPooling1D(2)(path2)
        
        # Path 3: Global patterns (large kernels)
        path3 = layers.Conv1D(32, 15, activation='relu', padding='same')(x)
        path3 = layers.BatchNormalization()(path3)
        path3 = layers.MaxPooling1D(2)(path3)
        
        # Concatenate paths
        merged = layers.Concatenate()([path1, path2, path3])
        
        # Additional processing
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=csi_input, outputs=output, name='MultiScaleCNN')
        return model
    
    def build_attention_cnn(self, input_shape):
        """Build CNN with attention mechanism - Focus on important frequencies"""
        print("   🏗️ Building Attention-based CNN Architecture...")
        print("      Architecture: CNN → Self-Attention → Dense")
        print("      Purpose: Learn which subcarriers are most important for localization")
        
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)  # (52, 2)
        
        # Initial convolutions
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Self-attention mechanism
        # Query, Key, Value projections
        query = layers.Dense(64)(x)
        key = layers.Dense(64)(x)
        value = layers.Dense(64)(x)
        
        # Attention scores
        attention_scores = layers.Dot(axes=[2, 2])([query, key])
        attention_scores = layers.Lambda(lambda x: x / np.sqrt(64))(attention_scores)
        attention_weights = layers.Softmax(axis=-1)(attention_scores)
        
        # Apply attention
        attended = layers.Dot(axes=[2, 1])([attention_weights, value])
        
        # Combine with original features
        x = layers.Add()([x, attended])
        x = layers.LayerNormalization()(x)
        
        # Final processing
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=csi_input, outputs=output, name='AttentionCNN')
        return model
    
    def build_hybrid_cnn_rssi(self, input_shape):
        """Build hybrid CNN + RSSI model - Best of both worlds"""
        print("   🏗️ Building Hybrid CNN + RSSI Architecture...")
        print("      Architecture: CNN branch + RSSI branch → Fusion")
        print("      Purpose: Use RSSI for coarse positioning, CNN for fine-tuning")
        
        # CSI branch
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        csi_x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)
        
        # Multi-scale CSI processing
        path1 = layers.Conv1D(32, 3, activation='relu', padding='same')(csi_x)
        path1 = layers.BatchNormalization()(path1)
        path1 = layers.MaxPooling1D(2)(path1)
        
        path2 = layers.Conv1D(32, 7, activation='relu', padding='same')(csi_x)
        path2 = layers.BatchNormalization()(path2)
        path2 = layers.MaxPooling1D(2)(path2)
        
        csi_merged = layers.Concatenate()([path1, path2])
        csi_x = layers.Conv1D(64, 3, activation='relu', padding='same')(csi_merged)
        csi_x = layers.BatchNormalization()(csi_x)
        csi_x = layers.GlobalAveragePooling1D()(csi_x)
        csi_features = layers.Dense(128, activation='relu')(csi_x)
        
        # RSSI branch
        rssi_input = keras.Input(shape=(1,), name='rssi_input')
        rssi_x = layers.Dense(32, activation='relu')(rssi_input)
        rssi_x = layers.Dense(32, activation='relu')(rssi_x)
        rssi_features = layers.Dense(32, activation='relu')(rssi_x)
        
        # Fusion layer
        combined = layers.Concatenate()([csi_features, rssi_features])
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=[csi_input, rssi_input], outputs=output, name='HybridCNN_RSSI')
        return model
    
    def build_residual_cnn(self, input_shape):
        """Build ResNet-inspired CNN - Better gradient flow"""
        print("   🏗️ Building Residual CNN Architecture...")
        print("      Architecture: CNN blocks with skip connections")
        print("      Purpose: Deeper network with better gradient flow and feature reuse")
        
        def residual_block(x, filters, kernel_size=3):
            """Residual block with skip connection"""
            shortcut = x
            
            # Main path
            y = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
            y = layers.BatchNormalization()(y)
            y = layers.Conv1D(filters, kernel_size, padding='same')(y)
            y = layers.BatchNormalization()(y)
            
            # Adjust shortcut if needed
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            # Add skip connection
            out = layers.Add()([shortcut, y])
            out = layers.Activation('relu')(out)
            
            return out
        
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)
        
        # Initial convolution
        x = layers.Conv1D(32, 7, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Residual blocks
        x = residual_block(x, 32, 3)
        x = layers.MaxPooling1D(2)(x)
        
        x = residual_block(x, 64, 3)
        x = layers.MaxPooling1D(2)(x)
        
        x = residual_block(x, 128, 3)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=csi_input, outputs=output, name='ResidualCNN')
        return model
    
    def compile_and_train_model(self, model, train_inputs, train_targets, val_inputs, val_targets, model_name):
        """Compile and train a model with detailed progress"""
        print(f"   🎯 Training {model_name}...")
        
        # Custom loss function focusing on Euclidean distance
        def euclidean_distance_loss(y_true, y_pred):
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1)))
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=euclidean_distance_loss,
            metrics=['mae', 'mse']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
            ModelCheckpoint(f'best_{model_name.lower()}.h5', save_best_only=True, monitor='val_loss')
        ]
        
        print(f"      Model Parameters: {model.count_params():,}")
        print(f"      Training samples: {len(train_targets) if isinstance(train_inputs, np.ndarray) else len(train_inputs[0])}")
        print(f"      Validation samples: {len(val_targets) if isinstance(val_inputs, np.ndarray) else len(val_inputs[0])}")
        
        start_time = time.time()
        
        # Train model
        history = model.fit(
            train_inputs, train_targets,
            validation_data=(val_inputs, val_targets),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        elapsed = time.time() - start_time
        print(f"      ✅ Training completed in {elapsed:.1f}s")
        
        return model, history
    
    def evaluate_model_detailed(self, model, test_inputs, test_targets, model_name):
        """Evaluate model with detailed metrics"""
        print(f"   📊 Evaluating {model_name}...")
        
        # Make predictions
        predictions = model.predict(test_inputs, verbose=0)
        
        # Denormalize coordinates
        predictions_denorm = self.scalers['coordinates'].inverse_transform(predictions)
        test_targets_denorm = self.scalers['coordinates'].inverse_transform(test_targets)
        
        # Calculate metrics
        euclidean_errors = np.sqrt(np.sum((test_targets_denorm - predictions_denorm)**2, axis=1))
        
        results = {
            'model_name': model_name,
            'mean_error': np.mean(euclidean_errors),
            'median_error': np.median(euclidean_errors),
            'std_error': np.std(euclidean_errors),
            'max_error': np.max(euclidean_errors),
            'min_error': np.min(euclidean_errors),
            'accuracy_50cm': np.mean(euclidean_errors < 0.5) * 100,
            'accuracy_1m': np.mean(euclidean_errors < 1.0) * 100,
            'accuracy_2m': np.mean(euclidean_errors < 2.0) * 100,
            'predictions': predictions_denorm,
            'targets': test_targets_denorm,
            'errors': euclidean_errors
        }
        
        print(f"      📏 Mean Error: {results['mean_error']:.3f}m")
        print(f"      📏 Median Error: {results['median_error']:.3f}m")
        print(f"      📏 Std Error: {results['std_error']:.3f}m")
        print(f"      📏 Max Error: {results['max_error']:.3f}m")
        print(f"      🎯 Accuracy <50cm: {results['accuracy_50cm']:.1f}%")
        print(f"      🎯 Accuracy <1m: {results['accuracy_1m']:.1f}%")
        print(f"      🎯 Accuracy <2m: {results['accuracy_2m']:.1f}%")
        
        return results

def main():
    """Main execution function for CNN localization"""
    print("🧠 DEEP LEARNING CNN-BASED INDOOR LOCALIZATION")
    print("=" * 80)
    
    # Check TensorFlow/GPU
    print("🔧 System Check:")
    print(f"   TensorFlow version: {tf.__version__}")
    print(f"   GPU available: {tf.config.list_physical_devices('GPU')}")
    print(f"   CPU cores: {tf.config.threading.get_intra_op_parallelism_threads()}")
    
    # Initialize system
    system = CNNLocalizationSystem()
    
    # Load data
    train_data, test_data = system.load_and_prepare_data_verbose()
    
    # Prepare inputs
    processed_data = system.prepare_cnn_inputs(train_data, test_data)
    
    # Data augmentation
    aug_csi, aug_rssi, aug_coords = system.create_data_augmentation(
        processed_data['train_csi'], 
        processed_data['train_rssi'], 
        processed_data['train_coords']
    )
    
    # Split augmented training data for validation
    train_csi, val_csi, train_rssi, val_rssi, train_coords, val_coords = train_test_split(
        aug_csi, aug_rssi, aug_coords, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Final Dataset Sizes:")
    print(f"   Training: {len(train_coords)} samples")
    print(f"   Validation: {len(val_coords)} samples")
    print(f"   Testing: {len(processed_data['test_coords'])} samples")
    
    # Build and train models
    print("\n🏗️ BUILDING AND TRAINING CNN ARCHITECTURES:")
    print("=" * 60)
    
    input_shape = (2, 52)  # (channels, subcarriers)
    all_results = []
    
    # 1. Basic CNN
    print("\n🔵 CNN Architecture 1: Basic CNN")
    basic_cnn = system.build_basic_cnn(input_shape)
    basic_cnn, basic_history = system.compile_and_train_model(
        basic_cnn, train_csi, train_coords, val_csi, val_coords, "BasicCNN"
    )
    basic_results = system.evaluate_model_detailed(
        basic_cnn, processed_data['test_csi'], processed_data['test_coords'], "Basic CNN"
    )
    all_results.append(basic_results)
    
    # 2. Multi-Scale CNN
    print("\n🟡 CNN Architecture 2: Multi-Scale CNN")
    multiscale_cnn = system.build_multiscale_cnn(input_shape)
    multiscale_cnn, multiscale_history = system.compile_and_train_model(
        multiscale_cnn, train_csi, train_coords, val_csi, val_coords, "MultiScaleCNN"
    )
    multiscale_results = system.evaluate_model_detailed(
        multiscale_cnn, processed_data['test_csi'], processed_data['test_coords'], "Multi-Scale CNN"
    )
    all_results.append(multiscale_results)
    
    # 3. Attention CNN
    print("\n🟢 CNN Architecture 3: Attention CNN")
    attention_cnn = system.build_attention_cnn(input_shape)
    attention_cnn, attention_history = system.compile_and_train_model(
        attention_cnn, train_csi, train_coords, val_csi, val_coords, "AttentionCNN"
    )
    attention_results = system.evaluate_model_detailed(
        attention_cnn, processed_data['test_csi'], processed_data['test_coords'], "Attention CNN"
    )
    all_results.append(attention_results)
    
    # 4. Hybrid CNN + RSSI
    print("\n🟠 CNN Architecture 4: Hybrid CNN + RSSI")
    hybrid_cnn = system.build_hybrid_cnn_rssi(input_shape)
    hybrid_cnn, hybrid_history = system.compile_and_train_model(
        hybrid_cnn, 
        [train_csi, train_rssi], train_coords, 
        [val_csi, val_rssi], val_coords, 
        "HybridCNN"
    )
    hybrid_results = system.evaluate_model_detailed(
        hybrid_cnn, 
        [processed_data['test_csi'], processed_data['test_rssi']], 
        processed_data['test_coords'], 
        "Hybrid CNN + RSSI"
    )
    all_results.append(hybrid_results)
    
    # 5. Residual CNN
    print("\n🔴 CNN Architecture 5: Residual CNN")
    residual_cnn = system.build_residual_cnn(input_shape)
    residual_cnn, residual_history = system.compile_and_train_model(
        residual_cnn, train_csi, train_coords, val_csi, val_coords, "ResidualCNN"
    )
    residual_results = system.evaluate_model_detailed(
        residual_cnn, processed_data['test_csi'], processed_data['test_coords'], "Residual CNN"
    )
    all_results.append(residual_results)
    
    # Results Summary
    print("\n🏆 FINAL CNN RESULTS COMPARISON:")
    print("=" * 80)
    
    # Sort by mean error
    all_results.sort(key=lambda x: x['mean_error'])
    
    print(f"{'Model':<20} {'Mean Error (m)':<15} {'<50cm (%)':<10} {'<1m (%)':<10} {'<2m (%)':<10}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['model_name']:<20} {result['mean_error']:<15.3f} "
              f"{result['accuracy_50cm']:<10.1f} {result['accuracy_1m']:<10.1f} "
              f"{result['accuracy_2m']:<10.1f}")
    
    # Best model analysis
    best_result = all_results[0]
    print(f"\n🥇 BEST CNN MODEL: {best_result['model_name']}")
    print(f"   Mean Euclidean Error: {best_result['mean_error']:.3f} meters")
    print(f"   Median Error: {best_result['median_error']:.3f} meters")
    print(f"   Standard Deviation: {best_result['std_error']:.3f} meters")
    print(f"   Accuracy <1m: {best_result['accuracy_1m']:.1f}%")
    print(f"   Accuracy <50cm: {best_result['accuracy_50cm']:.1f}%")
    
    # Target achievement
    target_achieved = best_result['mean_error'] < 1.0
    print(f"\n🎯 TARGET ACHIEVEMENT:")
    print(f"   Target: <1 meter mean error")
    print(f"   Result: {'✅ ACHIEVED!' if target_achieved else '❌ Not achieved'}")
    print(f"   Performance: {best_result['mean_error']:.3f}m")
    
    if target_achieved:
        print(f"   🌟 EXCELLENT! CNN achieved the target with {best_result['mean_error']:.3f}m accuracy!")
    else:
        improvement = 1.0 - best_result['mean_error']
        print(f"   📈 Close to target! Only {abs(improvement):.3f}m away from 1.0m goal")
    
    print("\n✅ DEEP LEARNING CNN LOCALIZATION COMPLETE!")
    
    return system, all_results

if __name__ == "__main__":
    system, results = main()
