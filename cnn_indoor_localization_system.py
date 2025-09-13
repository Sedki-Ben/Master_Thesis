#!/usr/bin/env python3
"""
CNN-Based Indoor Localization System

Implements and trains 5 CNN architectures for indoor localization:
1. Baseline CNN
2. Hybrid CNN + RSSI
3. Attention CNN 
4. Multi-Scale CNN
5. Residual CNN

Trains with 250/500/750 sample sizes, early stopping, and generates:
- Learning curves
- CDFs 
- Comprehensive testing results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import glob
import os
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_amplitude_rssi_data():
    """Load amplitude and RSSI data from all reference points"""
    
    print("📂 Loading Amplitude and RSSI Data for CNN Training...")
    
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
            
            for _, row in df.iterrows():
                # Parse amplitude array
                amp_str = row['amplitude'].strip('[]"')
                amplitudes = [float(x.strip()) for x in amp_str.split(',')]
                rssi = row['rssi']
                
                all_data.append({
                    'amplitudes': amplitudes,
                    'rssi': rssi,
                    'x': x,
                    'y': y
                })
                
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")
            continue
    
    print(f"✅ Loaded {len(all_data)} samples from {len(set(coordinates))} reference points")
    
    return all_data, list(set(coordinates))

def create_train_val_test_split(all_data, coordinates):
    """Create 27/7/7 train/validation/test split by reference points"""
    
    print("📊 Creating 27/7/7 Train/Validation/Test Split...")
    
    # Sort coordinates for reproducible split
    unique_coords = sorted(list(set(coordinates)))  # Remove duplicates and sort
    np.random.seed(42)
    
    # Shuffle coordinates using numpy but convert back to tuples
    indices = np.random.permutation(len(unique_coords))
    shuffled_coords = [unique_coords[i] for i in indices]
    
    # Split: 27 train, remaining for val/test
    if len(unique_coords) == 34:
        train_coords = set(shuffled_coords[:27])
        remaining_coords = shuffled_coords[27:34]
        # Split remaining 7 points: 4 validation, 3 test
        val_coords = set(remaining_coords[:4])   
        test_coords = set(remaining_coords[4:])  # Last 3 for test
    else:
        # Handle other cases
        n_train = min(27, len(unique_coords) - 7)
        train_coords = set(shuffled_coords[:n_train])
        remaining = shuffled_coords[n_train:]
        n_val = len(remaining) // 2
        val_coords = set(remaining[:n_val])
        test_coords = set(remaining[n_val:])
    
    print(f"   Training points: {len(train_coords)}")
    print(f"   Validation points: {len(val_coords)}")
    print(f"   Test points: {len(test_coords)}")
    
    # Split data by coordinates
    train_data = [item for item in all_data if (item['x'], item['y']) in train_coords]
    val_data = [item for item in all_data if (item['x'], item['y']) in val_coords]
    test_data = [item for item in all_data if (item['x'], item['y']) in test_coords]
    
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    print(f"   Test samples: {len(test_data)}")
    
    return train_data, val_data, test_data, train_coords, val_coords, test_coords

def prepare_data_for_sample_size(train_data, val_data, test_data, sample_size):
    """Prepare data for specific sample size per reference point"""
    
    print(f"🔄 Preparing Data for {sample_size} Samples per Reference Point...")
    
    # Group by coordinates and sample
    def sample_by_coords(data, n_samples):
        coord_groups = {}
        for item in data:
            coord = (item['x'], item['y'])
            if coord not in coord_groups:
                coord_groups[coord] = []
            coord_groups[coord].append(item)
        
        sampled_data = []
        for coord, items in coord_groups.items():
            if len(items) >= n_samples:
                sampled_items = np.random.choice(items, n_samples, replace=False)
            else:
                sampled_items = items  # Use all available if less than n_samples
            sampled_data.extend(sampled_items)
        
        return sampled_data
    
    # Sample data
    np.random.seed(42)  # For reproducible sampling
    train_sampled = sample_by_coords(train_data, sample_size)
    val_sampled = sample_by_coords(val_data, sample_size)
    test_sampled = sample_by_coords(test_data, min(sample_size, 100))  # Limit test size
    
    # Convert to arrays
    def data_to_arrays(data):
        amplitudes = np.array([item['amplitudes'] for item in data])
        # Reshape for Conv1D: (batch, time_steps, features) -> (batch, 52, 1)
        amplitudes = amplitudes.reshape(amplitudes.shape[0], amplitudes.shape[1], 1)
        rssi = np.array([item['rssi'] for item in data]).reshape(-1, 1)
        coordinates = np.array([[item['x'], item['y']] for item in data])
        return amplitudes, rssi, coordinates
    
    X_amp_train, X_rssi_train, y_train = data_to_arrays(train_sampled)
    X_amp_val, X_rssi_val, y_val = data_to_arrays(val_sampled)
    X_amp_test, X_rssi_test, y_test = data_to_arrays(test_sampled)
    
    print(f"   Final train: {len(X_amp_train)} samples")
    print(f"   Final val: {len(X_amp_val)} samples") 
    print(f"   Final test: {len(X_amp_test)} samples")
    
    return (X_amp_train, X_rssi_train, y_train), (X_amp_val, X_rssi_val, y_val), (X_amp_test, X_rssi_test, y_test)

def normalize_data(train_data, val_data, test_data):
    """Normalize amplitude and RSSI data"""
    
    X_amp_train, X_rssi_train, y_train = train_data
    X_amp_val, X_rssi_val, y_val = val_data
    X_amp_test, X_rssi_test, y_test = test_data
    
    # Normalize amplitudes (reshape for scaler, then back to 3D)
    amp_scaler = StandardScaler()
    
    # Flatten for scaling
    X_amp_train_flat = X_amp_train.reshape(X_amp_train.shape[0], -1)
    X_amp_val_flat = X_amp_val.reshape(X_amp_val.shape[0], -1)
    X_amp_test_flat = X_amp_test.reshape(X_amp_test.shape[0], -1)
    
    # Scale
    X_amp_train_flat_norm = amp_scaler.fit_transform(X_amp_train_flat)
    X_amp_val_flat_norm = amp_scaler.transform(X_amp_val_flat)
    X_amp_test_flat_norm = amp_scaler.transform(X_amp_test_flat)
    
    # Reshape back to 3D
    X_amp_train_norm = X_amp_train_flat_norm.reshape(X_amp_train.shape)
    X_amp_val_norm = X_amp_val_flat_norm.reshape(X_amp_val.shape)
    X_amp_test_norm = X_amp_test_flat_norm.reshape(X_amp_test.shape)
    
    # Normalize RSSI
    rssi_scaler = StandardScaler()
    X_rssi_train_norm = rssi_scaler.fit_transform(X_rssi_train)
    X_rssi_val_norm = rssi_scaler.transform(X_rssi_val)
    X_rssi_test_norm = rssi_scaler.transform(X_rssi_test)
    
    # Normalize coordinates for training (but keep original for evaluation)
    coord_scaler = MinMaxScaler()
    y_train_norm = coord_scaler.fit_transform(y_train)
    y_val_norm = coord_scaler.transform(y_val)
    y_test_norm = coord_scaler.transform(y_test)
    
    return {
        'train': (X_amp_train_norm, X_rssi_train_norm, y_train_norm),
        'val': (X_amp_val_norm, X_rssi_val_norm, y_val_norm),
        'test': (X_amp_test_norm, X_rssi_test_norm, y_test_norm),
        'original_coords': {'train': y_train, 'val': y_val, 'test': y_test},
        'scalers': {'amp': amp_scaler, 'rssi': rssi_scaler, 'coord': coord_scaler}
    }

# CNN Architecture Definitions

def build_baseline_cnn(input_shape):
    """Baseline CNN architecture"""
    
    inputs = layers.Input(shape=input_shape)
    
    # 1D Convolutional layers
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer (x, y coordinates)
    outputs = layers.Dense(2, activation='linear')(x)
    
    model = Model(inputs, outputs, name='Baseline_CNN')
    return model

def build_hybrid_cnn_rssi(amp_input_shape, rssi_input_shape):
    """Hybrid CNN with RSSI branch"""
    
    # Amplitude branch
    amp_input = layers.Input(shape=amp_input_shape, name='amplitude')
    amp_x = layers.Conv1D(64, 3, activation='relu', padding='same')(amp_input)
    amp_x = layers.BatchNormalization()(amp_x)
    amp_x = layers.MaxPooling1D(2)(amp_x)
    
    amp_x = layers.Conv1D(128, 3, activation='relu', padding='same')(amp_x)
    amp_x = layers.BatchNormalization()(amp_x)
    amp_x = layers.GlobalAveragePooling1D()(amp_x)
    
    # RSSI branch
    rssi_input = layers.Input(shape=rssi_input_shape, name='rssi')
    rssi_x = layers.Dense(32, activation='relu')(rssi_input)
    rssi_x = layers.Dense(64, activation='relu')(rssi_x)
    
    # Combine branches
    combined = layers.concatenate([amp_x, rssi_x])
    x = layers.Dense(512, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(2, activation='linear')(x)
    
    model = Model([amp_input, rssi_input], outputs, name='Hybrid_CNN_RSSI')
    return model

def build_attention_cnn(input_shape):
    """CNN with attention mechanism"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Convolutional layers
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    attention = layers.Conv1D(128, 1, activation='softmax', padding='same')(x)
    x = layers.multiply([x, attention])
    
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(2, activation='linear')(x)
    
    model = Model(inputs, outputs, name='Attention_CNN')
    return model

def build_multiscale_cnn(input_shape):
    """Multi-scale CNN with parallel convolution paths"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Multiple scales
    scale1 = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    scale1 = layers.BatchNormalization()(scale1)
    
    scale2 = layers.Conv1D(64, 5, activation='relu', padding='same')(inputs)
    scale2 = layers.BatchNormalization()(scale2)
    
    scale3 = layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
    scale3 = layers.BatchNormalization()(scale3)
    
    # Combine scales
    combined = layers.concatenate([scale1, scale2, scale3])
    
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(512, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(2, activation='linear')(x)
    
    model = Model(inputs, outputs, name='MultiScale_CNN')
    return model

def build_residual_cnn(input_shape):
    """Residual CNN with skip connections"""
    
    inputs = layers.Input(shape=input_shape)
    
    # Initial conv
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual block 1
    residual = x
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # Skip connection
    
    x = layers.MaxPooling1D(2)(x)
    
    # Residual block 2
    residual = layers.Conv1D(128, 1, padding='same')(x)  # Match dimensions
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # Skip connection
    
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(2, activation='linear')(x)
    
    model = Model(inputs, outputs, name='Residual_CNN')
    return model

def train_model_with_early_stopping(model, train_data, val_data, model_name, sample_size, epochs=100):
    """Train model with early stopping and return history"""
    
    print(f"🏋️ Training {model_name} with {sample_size} samples...")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Early stopping callback
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    if model_name == 'Hybrid_CNN_RSSI':
        # Hybrid model has two inputs
        X_train = [train_data[0], train_data[1]]
        X_val = [val_data[0], val_data[1]]
    else:
        # Other models use only amplitude data
        X_train = train_data[0]
        X_val = val_data[0]
    
    history = model.fit(
        X_train, train_data[2],
        validation_data=(X_val, val_data[2]),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def evaluate_model(model, test_data, coord_scaler, model_name):
    """Evaluate model and return metrics"""
    
    if model_name == 'Hybrid_CNN_RSSI':
        X_test = [test_data[0], test_data[1]]
    else:
        X_test = test_data[0]
    
    # Make predictions
    y_pred_norm = model.predict(X_test, verbose=0)
    
    # Denormalize predictions and true values
    y_pred = coord_scaler.inverse_transform(y_pred_norm)
    y_true = coord_scaler.inverse_transform(test_data[2])
    
    # Calculate errors
    errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
    
    metrics = {
        'median_error': np.median(errors),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'accuracy_1m': np.mean(errors <= 1.0) * 100,
        'accuracy_2m': np.mean(errors <= 2.0) * 100,
        'accuracy_3m': np.mean(errors <= 3.0) * 100,
        'errors': errors
    }
    
    return metrics

def main():
    """Main training and evaluation pipeline"""
    
    print("🎯 CNN-BASED INDOOR LOCALIZATION SYSTEM")
    print("Training 5 Architectures with 3 Sample Sizes")
    print("="*50)
    
    # Load data
    all_data, coordinates = load_amplitude_rssi_data()
    
    # Create train/val/test split
    train_data, val_data, test_data, train_coords, val_coords, test_coords = create_train_val_test_split(all_data, coordinates)
    
    # Define models and sample sizes
    sample_sizes = [250, 500, 750]
    model_builders = {
        'Baseline_CNN': build_baseline_cnn,
        'Hybrid_CNN_RSSI': build_hybrid_cnn_rssi,
        'Attention_CNN': build_attention_cnn,
        'MultiScale_CNN': build_multiscale_cnn,
        'Residual_CNN': build_residual_cnn
    }
    
    all_results = []
    all_histories = {}
    
    # Train all models
    for sample_size in sample_sizes:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH {sample_size} SAMPLES PER REFERENCE POINT")
        print(f"{'='*60}")
        
        # Prepare data for this sample size
        train_prep, val_prep, test_prep = prepare_data_for_sample_size(
            train_data, val_data, test_data, sample_size
        )
        
        # Normalize data
        normalized_data = normalize_data(train_prep, val_prep, test_prep)
        
        for model_name, builder in model_builders.items():
            print(f"\n{'-'*40}")
            print(f"Model: {model_name}")
            print(f"Sample Size: {sample_size}")
            print(f"{'-'*40}")
            
            try:
                # Build model
                if model_name == 'Hybrid_CNN_RSSI':
                    model = builder((52, 1), (1,))  # amp_shape, rssi_shape
                else:
                    model = builder((52, 1))  # amp_shape only
                
                # Train model
                history = train_model_with_early_stopping(
                    model, 
                    normalized_data['train'],
                    normalized_data['val'],
                    model_name,
                    sample_size
                )
                
                # Evaluate model
                metrics = evaluate_model(
                    model,
                    normalized_data['test'],
                    normalized_data['scalers']['coord'],
                    model_name
                )
                
                # Store results
                result = {
                    'model': model_name,
                    'sample_size': sample_size,
                    'median_error': metrics['median_error'],
                    'mean_error': metrics['mean_error'],
                    'std_error': metrics['std_error'],
                    'accuracy_1m': metrics['accuracy_1m'],
                    'accuracy_2m': metrics['accuracy_2m'],
                    'accuracy_3m': metrics['accuracy_3m'],
                    'errors': metrics['errors'],
                    'epochs_trained': len(history.history['loss'])
                }
                
                all_results.append(result)
                all_histories[f"{model_name}_{sample_size}"] = history.history
                
                print(f"✅ {model_name} ({sample_size}): {metrics['median_error']:.3f}m median error")
                
            except Exception as e:
                print(f"❌ Error training {model_name} with {sample_size} samples: {e}")
                continue
    
    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"📊 Trained {len(all_results)} models successfully")
    
    return all_results, all_histories

if __name__ == "__main__":
    results, histories = main()
