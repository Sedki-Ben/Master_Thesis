#!/usr/bin/env python3
"""
Retrain Existing CNN Models with Proper Train/Val/Test Split

Uses the 5 CNN architectures from cnn_localization_deep_learning.py with:
- 27 training points, 7 validation points, 5 test points
- Sample sizes: 250, 500, 750
- Early stopping after 10 epochs without improvement
- Learning curves, CDFs, and comprehensive results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import glob
import os
import json
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define the specific coordinate splits from our established work
VALIDATION_COORDS = [
    (0, 3), (0, 6), (2, 1), (3, 3), (4, 5), (5, 1), (6, 4)
]

TEST_COORDS = [
    (0.5, 0.5), (2.5, 2.5), (1.5, 4.5), (3.5, 1.5), (5.5, 3.5)
]

# Training coordinates will be all others (calculated from available coordinates)

def load_amplitude_rssi_data():
    """Load amplitude and RSSI data with proper coordinate mapping"""
    
    print("📂 Loading Amplitude and RSSI Data...")
    
    # Load reference points data
    data_files = glob.glob("Amplitude Phase Data Single/*.csv")
    all_data = []
    available_coords = []
    
    for file_path in data_files:
        filename = os.path.basename(file_path)
        coord_str = filename.replace('.csv', '')
        try:
            x, y = map(int, coord_str.split(','))
            available_coords.append((x, y))
            
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
    
    print(f"✅ Loaded {len(all_data)} samples from {len(set(available_coords))} reference points")
    
    # Load testing points data  
    testing_files = glob.glob("Testing Points Amplitude Phase Data Single/*.csv")
    testing_coords_map = {
        "0.5,0.5.csv": (0.5, 0.5),
        "2.5,2.5.csv": (2.5, 2.5), 
        "1.5,4.5.csv": (1.5, 4.5),
        "3.5,1.5.csv": (3.5, 1.5),
        "5.5,3.5.csv": (5.5, 3.5)
    }
    
    for file_path in testing_files:
        filename = os.path.basename(file_path)
        if filename in testing_coords_map:
            x, y = testing_coords_map[filename]
            
            try:
                df = pd.read_csv(file_path)
                
                for _, row in df.iterrows():
                    # Parse amplitude array
                    amp_str = row['amplitude'].strip('[]"')
                    amplitudes = [float(x_val.strip()) for x_val in amp_str.split(',')]
                    rssi = row['rssi']
                    
                    all_data.append({
                        'amplitudes': amplitudes,
                        'rssi': rssi,
                        'x': x,
                        'y': y
                    })
                    
            except Exception as e:
                print(f"⚠️ Error processing testing file {filename}: {e}")
                continue
    
    print(f"✅ Added testing points data to dataset")
    
    # Identify actual coordinates we have
    actual_coords = sorted(list(set(available_coords)))
    print(f"📍 Available coordinates: {actual_coords}")
    
    # Use predefined validation and test coordinates
    actual_val = [coord for coord in VALIDATION_COORDS if coord in actual_coords]
    actual_test = TEST_COORDS  # Test coordinates are fixed from testing points
    
    # Training coordinates are all others
    used_coords = set(actual_val + actual_test)
    actual_train = [coord for coord in actual_coords if coord not in used_coords]
    
    print(f"📊 Final Split:")
    print(f"   Training: {len(actual_train)} points - {actual_train}")
    print(f"   Validation: {len(actual_val)} points - {actual_val}")
    print(f"   Test: {len(actual_test)} points - {actual_test}")
    
    return all_data, actual_train, actual_val, actual_test

def prepare_data_by_coordinates(all_data, train_coords, val_coords, test_coords, sample_size):
    """Prepare data split by coordinates with specific sample size"""
    
    print(f"🔄 Preparing Data for {sample_size} Samples per Reference Point...")
    
    def filter_and_sample_data(data, coords, n_samples):
        filtered_data = [item for item in data if (item['x'], item['y']) in coords]
        
        # Group by coordinates and sample
        coord_groups = {}
        for item in filtered_data:
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
    
    # Sample data for each split
    np.random.seed(42)  # For reproducible sampling
    train_data = filter_and_sample_data(all_data, train_coords, sample_size)
    val_data = filter_and_sample_data(all_data, val_coords, sample_size)
    test_data = filter_and_sample_data(all_data, test_coords, min(sample_size, 100))
    
    # Convert to arrays
    def data_to_arrays(data):
        amplitudes = np.array([item['amplitudes'] for item in data])
        rssi = np.array([item['rssi'] for item in data]).reshape(-1, 1)
        coordinates = np.array([[item['x'], item['y']] for item in data])
        return amplitudes, rssi, coordinates
    
    X_amp_train, X_rssi_train, y_train = data_to_arrays(train_data)
    X_amp_val, X_rssi_val, y_val = data_to_arrays(val_data)
    X_amp_test, X_rssi_test, y_test = data_to_arrays(test_data)
    
    print(f"   Train: {len(X_amp_train)} samples from {len(train_coords)} points")
    print(f"   Val: {len(X_amp_val)} samples from {len(val_coords)} points")
    print(f"   Test: {len(X_amp_test)} samples from {len(test_coords)} points")
    
    return (X_amp_train, X_rssi_train, y_train), (X_amp_val, X_rssi_val, y_val), (X_amp_test, X_rssi_test, y_test)

def normalize_data(train_data, val_data, test_data):
    """Normalize all data using training set statistics"""
    
    X_amp_train, X_rssi_train, y_train = train_data
    X_amp_val, X_rssi_val, y_val = val_data
    X_amp_test, X_rssi_test, y_test = test_data
    
    # Normalize amplitudes
    amp_scaler = StandardScaler()
    X_amp_train_norm = amp_scaler.fit_transform(X_amp_train)
    X_amp_val_norm = amp_scaler.transform(X_amp_val)
    
    # Handle empty test data
    if len(X_amp_test) > 0:
        X_amp_test_norm = amp_scaler.transform(X_amp_test)
    else:
        X_amp_test_norm = np.array([]).reshape(0, X_amp_train.shape[1])
    
    # Normalize RSSI
    rssi_scaler = StandardScaler()
    X_rssi_train_norm = rssi_scaler.fit_transform(X_rssi_train)
    X_rssi_val_norm = rssi_scaler.transform(X_rssi_val)
    
    # Handle empty test RSSI data
    if len(X_rssi_test) > 0:
        X_rssi_test_norm = rssi_scaler.transform(X_rssi_test)
    else:
        X_rssi_test_norm = np.array([]).reshape(0, 1)
    
    # Don't normalize coordinates - keep in meters for evaluation
    
    return {
        'train': (X_amp_train_norm, X_rssi_train_norm, y_train),
        'val': (X_amp_val_norm, X_rssi_val_norm, y_val),
        'test': (X_amp_test_norm, X_rssi_test_norm, y_test),
        'scalers': {'amp': amp_scaler, 'rssi': rssi_scaler}
    }

# CNN Architecture Definitions (from cnn_localization_deep_learning.py)

def build_baseline_cnn():
    """Baseline CNN architecture"""
    inputs = layers.Input(shape=(52,), name='csi_input')
    
    # Dense layers with dropout
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(2, activation='linear', name='coordinates')(x)
    
    model = Model(inputs, outputs, name='Baseline_CNN')
    return model

def build_hybrid_cnn_rssi():
    """Hybrid CNN with RSSI features"""
    # CSI input
    csi_input = layers.Input(shape=(52,), name='csi_input')
    csi_dense = layers.Dense(128, activation='relu')(csi_input)
    csi_dropout = layers.Dropout(0.3)(csi_dense)
    
    # RSSI input  
    rssi_input = layers.Input(shape=(1,), name='rssi_input')
    rssi_dense = layers.Dense(32, activation='relu')(rssi_input)
    
    # Combine features
    combined = layers.concatenate([csi_dropout, rssi_dense])
    x = layers.Dense(128, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(2, activation='linear', name='coordinates')(x)
    
    model = Model([csi_input, rssi_input], outputs, name='Hybrid_CNN_RSSI')
    return model

def build_attention_cnn():
    """CNN with attention mechanism"""
    inputs = layers.Input(shape=(52,), name='csi_input')
    
    # Reshape for attention
    x = layers.Reshape((52, 1))(inputs)
    
    # Attention layer
    attention_weights = layers.Dense(1, activation='softmax')(x)
    attended = layers.multiply([x, attention_weights])
    attended = layers.Flatten()(attended)
    
    # Dense layers
    x = layers.Dense(256, activation='relu')(attended)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(2, activation='linear', name='coordinates')(x)
    
    model = Model(inputs, outputs, name='Attention_CNN')
    return model

def build_multiscale_cnn():
    """Multi-scale CNN with parallel processing"""
    inputs = layers.Input(shape=(52,), name='csi_input')
    
    # Multiple scales
    scale1 = layers.Dense(64, activation='relu')(inputs)
    scale2 = layers.Dense(128, activation='relu')(inputs)
    scale3 = layers.Dense(256, activation='relu')(inputs)
    
    # Combine scales
    combined = layers.concatenate([scale1, scale2, scale3])
    x = layers.Dropout(0.3)(combined)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(2, activation='linear', name='coordinates')(x)
    
    model = Model(inputs, outputs, name='MultiScale_CNN')
    return model

def build_residual_cnn():
    """Residual CNN with skip connections"""
    inputs = layers.Input(shape=(52,), name='csi_input')
    
    # First block
    x1 = layers.Dense(128, activation='relu')(inputs)
    x1 = layers.Dropout(0.2)(x1)
    
    # Residual block
    x2 = layers.Dense(128, activation='relu')(x1)
    x2 = layers.Dropout(0.2)(x2)
    x2_skip = layers.Add()([x1, x2])  # Skip connection
    
    # Final layers
    x = layers.Dense(128, activation='relu')(x2_skip)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    outputs = layers.Dense(2, activation='linear', name='coordinates')(x)
    
    model = Model(inputs, outputs, name='Residual_CNN')
    return model

def train_model_with_callbacks(model, train_data, val_data, model_name, sample_size):
    """Train model with early stopping and callbacks"""
    
    print(f"🏋️ Training {model_name} with {sample_size} samples...")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Prepare inputs based on model type
    if model_name == 'Hybrid_CNN_RSSI':
        X_train = [train_data[0], train_data[1]]  # [amplitudes, rssi]
        X_val = [val_data[0], val_data[1]]
    else:
        X_train = train_data[0]  # amplitudes only
        X_val = val_data[0]
    
    # Train model
    start_time = time.time()
    history = model.fit(
        X_train, train_data[2],  # y_train
        validation_data=(X_val, val_data[2]),  # y_val
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.1f}s ({len(history.history['loss'])} epochs)")
    
    return history, training_time

def evaluate_model(model, test_data, model_name):
    """Evaluate model performance"""
    
    # Check if we have test data
    if len(test_data[0]) == 0:
        print("⚠️ No test data available for evaluation")
        return {
            'median_error': 0.0,
            'mean_error': 0.0,
            'std_error': 0.0,
            'accuracy_50cm': 0.0,
            'accuracy_1m': 0.0,
            'accuracy_2m': 0.0,
            'errors': np.array([]),
            'predictions': np.array([]),
            'true_coords': np.array([])
        }
    
    # Prepare test inputs
    if model_name == 'Hybrid_CNN_RSSI':
        X_test = [test_data[0], test_data[1]]
    else:
        X_test = test_data[0]
    
    # Make predictions
    try:
        y_pred = model.predict(X_test, verbose=0)
        y_true = test_data[2]
        
        # Calculate errors
        errors = np.sqrt(np.sum((y_true - y_pred)**2, axis=1))
        
        metrics = {
            'median_error': np.median(errors),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'accuracy_50cm': np.mean(errors <= 0.5) * 100,
            'accuracy_1m': np.mean(errors <= 1.0) * 100,
            'accuracy_2m': np.mean(errors <= 2.0) * 100,
            'errors': errors,
            'predictions': y_pred,
            'true_coords': y_true
        }
        
        return metrics
        
    except Exception as e:
        print(f"⚠️ Error during model evaluation: {e}")
        return {
            'median_error': float('inf'),
            'mean_error': float('inf'),
            'std_error': 0.0,
            'accuracy_50cm': 0.0,
            'accuracy_1m': 0.0,
            'accuracy_2m': 0.0,
            'errors': np.array([]),
            'predictions': np.array([]),
            'true_coords': np.array([])
        }

def create_learning_curves_plot(all_histories):
    """Create learning curves for all models"""
    
    print("📈 Creating Learning Curves...")
    
    sample_sizes = [250, 500, 750]
    model_names = ['Baseline_CNN', 'Hybrid_CNN_RSSI', 'Attention_CNN', 'MultiScale_CNN', 'Residual_CNN']
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Learning Curves: CNN Indoor Localization Models\nTraining and Validation Loss', 
                 fontsize=16, fontweight='bold')
    
    for i, sample_size in enumerate(sample_sizes):
        for j, model_name in enumerate(model_names):
            ax = axes[i, j]
            
            key = f"{model_name}_{sample_size}"
            if key in all_histories:
                history = all_histories[key]
                
                # Plot training and validation loss
                epochs = range(1, len(history['loss']) + 1)
                ax.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
                ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
                
                ax.set_title(f'{model_name}\n{sample_size} samples', fontsize=10, fontweight='bold')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('MSE Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{model_name}\n{sample_size} samples', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('cnn_learning_curves_all_models.png', dpi=300, bbox_inches='tight')
    print("💾 Learning curves saved: cnn_learning_curves_all_models.png")
    plt.show()

def create_cdf_comparison_plot(all_results):
    """Create CDF comparison for all models"""
    
    print("📈 Creating CDF Comparison...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('CDF Comparison: CNN Indoor Localization Models\nError Distribution by Sample Size', 
                 fontsize=16, fontweight='bold')
    
    sample_sizes = [250, 500, 750]
    colors = ['#FF4444', '#00CCFF', '#FF44FF', '#FF8800', '#8844FF']
    model_names = ['Baseline_CNN', 'Hybrid_CNN_RSSI', 'Attention_CNN', 'MultiScale_CNN', 'Residual_CNN']
    
    for i, sample_size in enumerate(sample_sizes):
        ax = axes[i]
        
        for j, model_name in enumerate(model_names):
            key = f"{model_name}_{sample_size}"
            if key in all_results:
                errors = all_results[key]['errors']
                errors_sorted = np.sort(errors)
                p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
                
                ax.plot(errors_sorted, p, color=colors[j], linewidth=3, 
                       label=f"{model_name} (median: {all_results[key]['median_error']:.3f}m)", 
                       alpha=0.9)
        
        # Add threshold lines
        thresholds = [0.5, 1.0, 2.0]
        threshold_colors = ['green', 'orange', 'red']
        for threshold, color in zip(thresholds, threshold_colors):
            ax.axvline(x=threshold, color=color, linestyle=':', alpha=0.7, linewidth=2)
            ax.text(threshold + 0.02, 0.95, f'{threshold}m', rotation=90, 
                   fontsize=9, color=color, fontweight='bold', 
                   verticalalignment='top')
        
        ax.set_title(f'{sample_size} Samples per Point', fontsize=12, fontweight='bold')
        ax.set_xlabel('Localization Error (meters)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('cnn_cdf_comparison_all_models.png', dpi=300, bbox_inches='tight')
    print("💾 CDF comparison saved: cnn_cdf_comparison_all_models.png")
    plt.show()

def create_results_table(all_results):
    """Create comprehensive results table"""
    
    print("📊 Creating Results Table...")
    
    # Check if we have any results
    if not all_results:
        print("⚠️ No results to display - all models failed to train")
        return pd.DataFrame()
    
    # Prepare data for table
    table_data = []
    for key, result in all_results.items():
        model_name, sample_size = key.rsplit('_', 1)
        table_data.append({
            'Model': model_name,
            'Sample_Size': int(sample_size),
            'Median_Error_m': result['median_error'],
            'Mean_Error_m': result['mean_error'],
            'Std_Error_m': result['std_error'],
            'Accuracy_50cm_pct': result['accuracy_50cm'],
            'Accuracy_1m_pct': result['accuracy_1m'],
            'Accuracy_2m_pct': result['accuracy_2m'],
            'Training_Time_s': result.get('training_time', 0)
        })
    
    # Convert to DataFrame and sort
    df = pd.DataFrame(table_data)
    if not df.empty:
        df = df.sort_values(['Sample_Size', 'Median_Error_m'])
    
    # Save to CSV
    df.to_csv('cnn_comprehensive_results.csv', index=False)
    print("💾 Results table saved: cnn_comprehensive_results.csv")
    
    # Display table
    print("\n📊 COMPREHENSIVE RESULTS TABLE")
    print("="*80)
    print(df.to_string(index=False, float_format='%.3f'))
    
    return df

def main():
    """Main training and evaluation pipeline"""
    
    print("🎯 RETRAIN EXISTING CNN MODELS")
    print("5 Architectures × 3 Sample Sizes = 15 Models")
    print("="*50)
    
    # Load data and coordinate splits
    all_data, train_coords, val_coords, test_coords = load_amplitude_rssi_data()
    
    # Define models and sample sizes
    sample_sizes = [250, 500, 750]
    model_builders = {
        'Baseline_CNN': build_baseline_cnn,
        'Hybrid_CNN_RSSI': build_hybrid_cnn_rssi,
        'Attention_CNN': build_attention_cnn,
        'MultiScale_CNN': build_multiscale_cnn,
        'Residual_CNN': build_residual_cnn
    }
    
    all_results = {}
    all_histories = {}
    
    # Train all combinations
    for sample_size in sample_sizes:
        print(f"\n{'='*60}")
        print(f"TRAINING WITH {sample_size} SAMPLES PER REFERENCE POINT")
        print(f"{'='*60}")
        
        # Prepare data for this sample size
        train_data, val_data, test_data = prepare_data_by_coordinates(
            all_data, train_coords, val_coords, test_coords, sample_size
        )
        
        # Normalize data
        normalized_data = normalize_data(train_data, val_data, test_data)
        
        for model_name, builder in model_builders.items():
            print(f"\n{'-'*40}")
            print(f"Model: {model_name}")
            print(f"Sample Size: {sample_size}")
            print(f"{'-'*40}")
            
            try:
                # Build model
                model = builder()
                
                # Train model
                history, training_time = train_model_with_callbacks(
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
                    model_name
                )
                
                # Store results
                key = f"{model_name}_{sample_size}"
                metrics['training_time'] = training_time
                all_results[key] = metrics
                all_histories[key] = history.history
                
                print(f"✅ {model_name} ({sample_size}): {metrics['median_error']:.3f}m median error")
                
            except Exception as e:
                print(f"❌ Error training {model_name} with {sample_size} samples: {e}")
                continue
    
    # Generate visualizations and results
    print(f"\n🎯 GENERATING RESULTS...")
    create_learning_curves_plot(all_histories)
    create_cdf_comparison_plot(all_results)
    results_df = create_results_table(all_results)
    
    print(f"\n✅ TRAINING AND EVALUATION COMPLETE!")
    print(f"📊 Successfully trained {len(all_results)} models")
    print(f"📈 Generated learning curves, CDFs, and comprehensive results")
    
    return all_results, all_histories, results_df

if __name__ == "__main__":
    results, histories, df = main()
