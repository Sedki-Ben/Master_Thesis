#!/usr/bin/env python3
"""
Tom Cruise: Improved CNN Training System
Fixes all generalization issues identified in the diagnosis:
- Eliminates data leakage in preprocessing
- Reduces learning rate for better convergence  
- Increases regularization (dropout + L2)
- Uses standard MSE loss instead of custom Euclidean
- Removes data augmentation as requested
- Saves models in 'the last samurai/tom cruise' folder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import time
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print(">>> TensorFlow imported successfully")
except ImportError:
    print("ERROR: TensorFlow not found. Please install with: pip install tensorflow")
    exit(1)

# Import correct coordinates
from coordinates_config import get_training_points, get_validation_points, get_testing_points

class TomCruiseImprovedCNNSystem:
    """Tom Cruise: Fixed CNN training system with proper generalization"""
    
    def __init__(self, output_dir="the last samurai/tom cruise"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.results = {}
        
        # Load correct coordinates
        self.training_points = get_training_points()
        self.validation_points = get_validation_points()
        self.testing_points = get_testing_points()
        
        print(f"🎬 Tom Cruise Improved CNN System Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Coordinates loaded:")
        print(f"    Training points: {len(self.training_points)}")
        print(f"    Validation points: {len(self.validation_points)}")
        print(f"    Testing points: {len(self.testing_points)}")
        
    def load_data_by_coordinates(self, dataset_size, point_type="training"):
        """Load data for specific coordinates and dataset size"""
        
        if point_type == "training":
            points = self.training_points
            folder = f"CSI Dataset {dataset_size} Samples"
        elif point_type == "validation":
            points = self.validation_points  
            folder = f"CSI Dataset {dataset_size} Samples"
        elif point_type == "testing":
            points = self.testing_points
            folder = "Testing Points Dataset 750 Samples"  # Always use 750 for testing
        else:
            raise ValueError("point_type must be 'training', 'validation', or 'testing'")
        
        amplitudes, phases, rssi_values, coordinates = [], [], [], []
        
        for x, y in points:
            file_path = Path(folder) / f"{x},{y}.csv"
            
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            amps = json.loads(row['amplitude'])
                            phases_data = json.loads(row['phase'])
                            rssi = float(row['rssi'])
                            
                            if len(amps) == 52 and len(phases_data) == 52:
                                amplitudes.append(amps)
                                phases.append(phases_data)
                                rssi_values.append(rssi)
                                coordinates.append([x, y])
                        except:
                            continue
            else:
                print(f"    Warning: File not found: {file_path}")
        
        return {
            'amplitudes': np.array(amplitudes),
            'phases': np.array(phases),
            'rssi': np.array(rssi_values),
            'coordinates': np.array(coordinates)
        }
    
    def load_complete_dataset(self, dataset_size):
        """Load complete dataset with correct coordinate splits"""
        print(f"\n>>> Loading dataset size: {dataset_size} samples")
        
        # Load training data
        train_data = self.load_data_by_coordinates(dataset_size, "training")
        print(f"    Training: {len(train_data['amplitudes'])} samples from {len(self.training_points)} points")
        
        # Load validation data
        val_data = self.load_data_by_coordinates(dataset_size, "validation")
        print(f"    Validation: {len(val_data['amplitudes'])} samples from {len(self.validation_points)} points")
        
        # Load test data (always 750 samples)
        test_data = self.load_data_by_coordinates(750, "testing")
        print(f"    Testing: {len(test_data['amplitudes'])} samples from {len(self.testing_points)} points")
        
        return train_data, val_data, test_data
    
    def prepare_cnn_inputs_fixed(self, train_data, val_data, test_data, dataset_size):
        """Prepare CNN inputs with FIXED preprocessing - NO DATA LEAKAGE"""
        print(f">>> Fixed preprocessing for {dataset_size} samples (NO data leakage)...")
        
        # FIT SCALERS ONLY ON TRAINING DATA (CRITICAL FIX)
        amp_scaler = StandardScaler()
        phase_scaler = StandardScaler()
        rssi_scaler = StandardScaler()
        coord_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Fit ONLY on training data
        amp_scaler.fit(train_data['amplitudes'])
        phase_scaler.fit(train_data['phases'])
        rssi_scaler.fit(train_data['rssi'].reshape(-1, 1))
        coord_scaler.fit(train_data['coordinates'])
        
        print("    ✅ Scalers fitted ONLY on training data (data leakage eliminated)")
        
        # Transform each dataset using training scalers
        train_amp_norm = amp_scaler.transform(train_data['amplitudes'])
        train_phase_norm = phase_scaler.transform(train_data['phases'])
        train_rssi_norm = rssi_scaler.transform(train_data['rssi'].reshape(-1, 1)).flatten()
        train_coords_norm = coord_scaler.transform(train_data['coordinates'])
        
        val_amp_norm = amp_scaler.transform(val_data['amplitudes'])
        val_phase_norm = phase_scaler.transform(val_data['phases'])
        val_rssi_norm = rssi_scaler.transform(val_data['rssi'].reshape(-1, 1)).flatten()
        val_coords_norm = coord_scaler.transform(val_data['coordinates'])
        
        test_amp_norm = amp_scaler.transform(test_data['amplitudes'])
        test_phase_norm = phase_scaler.transform(test_data['phases'])
        test_rssi_norm = rssi_scaler.transform(test_data['rssi'].reshape(-1, 1)).flatten()
        test_coords_norm = coord_scaler.transform(test_data['coordinates'])
        
        # Create CNN format (2, 52) - stacked arrangement
        train_cnn = np.stack([train_amp_norm, train_phase_norm], axis=1)
        val_cnn = np.stack([val_amp_norm, val_phase_norm], axis=1)
        test_cnn = np.stack([test_amp_norm, test_phase_norm], axis=1)
        
        # Store scalers
        self.scalers[dataset_size] = {
            'amplitude': amp_scaler,
            'phase': phase_scaler,
            'rssi': rssi_scaler,
            'coordinates': coord_scaler
        }
        
        print(f"    CNN input shapes - Train: {train_cnn.shape}, Val: {val_cnn.shape}, Test: {test_cnn.shape}")
        
        return {
            'train_csi': train_cnn,
            'val_csi': val_cnn,
            'test_csi': test_cnn,
            'train_rssi': train_rssi_norm,
            'val_rssi': val_rssi_norm,
            'test_rssi': test_rssi_norm,
            'train_coords': train_coords_norm,
            'val_coords': val_coords_norm,
            'test_coords': test_coords_norm,
            'original_test_coords': test_data['coordinates']
        }
    
    # IMPROVED CNN ARCHITECTURES WITH PROPER REGULARIZATION
    def build_basic_cnn_improved(self, input_shape):
        """Build improved basic CNN with proper regularization"""
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)
        
        # Conv layers with proper regularization
        x = layers.Conv1D(32, 5, activation='relu', padding='same', 
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout1D(0.2)(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.Conv1D(64, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout1D(0.2)(x)
        x = layers.MaxPooling1D(2)(x)
        
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with increased regularization
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.5)(x)  # Increased from 0.3
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.4)(x)  # Increased from 0.2
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        model = keras.Model(inputs=csi_input, outputs=output, name='BasicCNN_Improved')
        return model
    
    def build_multiscale_cnn_improved(self, input_shape):
        """Build improved multi-scale CNN with proper regularization"""
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)
        
        # Multiple parallel paths with regularization
        path1 = layers.Conv1D(32, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(1e-4))(x)
        path1 = layers.BatchNormalization()(path1)
        path1 = layers.SpatialDropout1D(0.2)(path1)
        path1 = layers.MaxPooling1D(2)(path1)
        
        path2 = layers.Conv1D(32, 7, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(1e-4))(x)
        path2 = layers.BatchNormalization()(path2)
        path2 = layers.SpatialDropout1D(0.2)(path2)
        path2 = layers.MaxPooling1D(2)(path2)
        
        path3 = layers.Conv1D(32, 15, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(1e-4))(x)
        path3 = layers.BatchNormalization()(path3)
        path3 = layers.SpatialDropout1D(0.2)(path3)
        path3 = layers.MaxPooling1D(2)(path3)
        
        merged = layers.Concatenate()([path1, path2, path3])
        x = layers.Conv1D(128, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(merged)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with increased regularization
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.4)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        model = keras.Model(inputs=csi_input, outputs=output, name='MultiScaleCNN_Improved')
        return model
    
    def build_attention_cnn_improved(self, input_shape):
        """Build improved CNN with attention mechanism and proper regularization"""
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)
        
        x = layers.Conv1D(64, 5, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout1D(0.2)(x)
        
        x = layers.Conv1D(64, 3, activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout1D(0.2)(x)
        
        # Simplified self-attention mechanism
        query = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
        key = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
        value = layers.Dense(64, kernel_regularizer=regularizers.l2(1e-4))(x)
        
        attention_scores = layers.Dot(axes=[2, 2])([query, key])
        attention_scores = layers.Lambda(lambda x: x / np.sqrt(64))(attention_scores)
        attention_weights = layers.Softmax(axis=-1)(attention_scores)
        attended = layers.Dot(axes=[2, 1])([attention_weights, value])
        
        x = layers.Add()([x, attended])
        x = layers.LayerNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with increased regularization
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.4)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        model = keras.Model(inputs=csi_input, outputs=output, name='AttentionCNN_Improved')
        return model
    
    def build_hybrid_cnn_rssi_improved(self, input_shape):
        """Build improved hybrid CNN + RSSI model with proper regularization"""
        # CSI branch
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        csi_x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)
        
        path1 = layers.Conv1D(32, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(1e-4))(csi_x)
        path1 = layers.BatchNormalization()(path1)
        path1 = layers.SpatialDropout1D(0.2)(path1)
        path1 = layers.MaxPooling1D(2)(path1)
        
        path2 = layers.Conv1D(32, 7, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(1e-4))(csi_x)
        path2 = layers.BatchNormalization()(path2)
        path2 = layers.SpatialDropout1D(0.2)(path2)
        path2 = layers.MaxPooling1D(2)(path2)
        
        csi_merged = layers.Concatenate()([path1, path2])
        csi_x = layers.Conv1D(64, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(1e-4))(csi_merged)
        csi_x = layers.BatchNormalization()(csi_x)
        csi_x = layers.GlobalAveragePooling1D()(csi_x)
        csi_features = layers.Dense(128, activation='relu', 
                                   kernel_regularizer=regularizers.l2(1e-4))(csi_x)
        
        # RSSI branch
        rssi_input = keras.Input(shape=(1,), name='rssi_input')
        rssi_x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(rssi_input)
        rssi_x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(rssi_x)
        rssi_features = layers.Dense(32, activation='relu', 
                                    kernel_regularizer=regularizers.l2(1e-4))(rssi_x)
        
        # Fusion with regularization
        combined = layers.Concatenate()([csi_features, rssi_features])
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(combined)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.3)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        model = keras.Model(inputs=[csi_input, rssi_input], outputs=output, name='HybridCNN_Improved')
        return model
    
    def build_residual_cnn_improved(self, input_shape):
        """Build improved ResNet-inspired CNN with proper regularization"""
        def residual_block(x, filters, kernel_size=3):
            shortcut = x
            y = layers.Conv1D(filters, kernel_size, padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(1e-4))(x)
            y = layers.BatchNormalization()(y)
            y = layers.SpatialDropout1D(0.2)(y)
            y = layers.Conv1D(filters, kernel_size, padding='same',
                             kernel_regularizer=regularizers.l2(1e-4))(y)
            y = layers.BatchNormalization()(y)
            
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv1D(filters, 1, padding='same',
                                        kernel_regularizer=regularizers.l2(1e-4))(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            out = layers.Add()([shortcut, y])
            out = layers.Activation('relu')(out)
            return out
        
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)
        
        x = layers.Conv1D(32, 7, padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.SpatialDropout1D(0.2)(x)
        
        x = residual_block(x, 32, 3)
        x = layers.MaxPooling1D(2)(x)
        
        x = residual_block(x, 64, 3)
        x = layers.MaxPooling1D(2)(x)
        
        x = residual_block(x, 128, 3)
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers with increased regularization
        x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.4)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        model = keras.Model(inputs=csi_input, outputs=output, name='ResidualCNN_Improved')
        return model
    
    def compile_and_train_model_improved(self, model, train_inputs, train_targets, val_inputs, val_targets, model_name, dataset_size):
        """Compile and train model with improved configuration"""
        print(f"    Training {model_name} on {dataset_size} samples (IMPROVED CONFIG)...")
        
        # Use standard MSE loss instead of custom Euclidean
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002),  # Reduced from 0.001
            loss='mse',  # Standard MSE instead of custom Euclidean
            metrics=['mae']
        )
        
        print(f"    ✅ Learning rate: 0.0002 (reduced from 0.001)")
        print(f"    ✅ Loss function: MSE (changed from Euclidean)")
        print(f"    ✅ L2 regularization: 1e-4 added to all layers")
        print(f"    ✅ Dropout increased: 0.5 for dense, 0.2 for spatial")
        
        # Model checkpoint
        checkpoint_path = self.output_dir / f"{model_name.lower()}_{dataset_size}_samples_improved.h5"
        callbacks = [
            EarlyStopping(patience=30, restore_best_weights=True, monitor='val_loss'),  # Increased patience
            ReduceLROnPlateau(factor=0.5, patience=15, min_lr=1e-6),  # Increased patience
            ModelCheckpoint(str(checkpoint_path), save_best_only=True, monitor='val_loss')
        ]
        
        print(f"        Parameters: {model.count_params():,}")
        print(f"        Training samples: {len(train_targets) if isinstance(train_inputs, np.ndarray) else len(train_inputs[0])}")
        print(f"        Validation samples: {len(val_targets) if isinstance(val_inputs, np.ndarray) else len(val_inputs[0])}")
        
        start_time = time.time()
        history = model.fit(
            train_inputs, train_targets,
            validation_data=(val_inputs, val_targets),
            epochs=150,  # Increased from 100
            batch_size=16,  # Reduced from 32 for better gradients
            callbacks=callbacks,
            verbose=1
        )
        
        elapsed = time.time() - start_time
        print(f"        Training completed in {elapsed:.1f}s")
        
        return model, history
    
    def evaluate_model_detailed(self, model, test_inputs, test_targets, original_coords, model_name, dataset_size):
        """Evaluate model with detailed metrics"""
        print(f"    Evaluating {model_name} ({dataset_size} samples)...")
        
        predictions = model.predict(test_inputs, verbose=0)
        
        # Denormalize coordinates
        predictions_denorm = self.scalers[dataset_size]['coordinates'].inverse_transform(predictions)
        
        # Calculate metrics
        euclidean_errors = np.sqrt(np.sum((original_coords - predictions_denorm)**2, axis=1))
        
        results = {
            'model_name': model_name,
            'dataset_size': dataset_size,
            'mean_error': np.mean(euclidean_errors),
            'median_error': np.median(euclidean_errors),
            'std_error': np.std(euclidean_errors),
            'max_error': np.max(euclidean_errors),
            'min_error': np.min(euclidean_errors),
            'accuracy_50cm': np.mean(euclidean_errors < 0.5) * 100,
            'accuracy_1m': np.mean(euclidean_errors < 1.0) * 100,
            'accuracy_2m': np.mean(euclidean_errors < 2.0) * 100,
            'accuracy_3m': np.mean(euclidean_errors < 3.0) * 100,
            'predictions': predictions_denorm,
            'targets': original_coords,
            'errors': euclidean_errors
        }
        
        print(f"        Mean Error: {results['mean_error']:.3f}m")
        print(f"        Median Error: {results['median_error']:.3f}m")
        print(f"        Accuracy <1m: {results['accuracy_1m']:.1f}%")
        print(f"        Accuracy <2m: {results['accuracy_2m']:.1f}%")
        print(f"        Accuracy <3m: {results['accuracy_3m']:.1f}%")
        
        return results
    
    def save_results_summary(self, all_results):
        """Save comprehensive results summary"""
        # Save to CSV
        csv_data = []
        for result in all_results:
            csv_data.append({
                'Model': result['model_name'],
                'Dataset_Size': result['dataset_size'],
                'Mean_Error_m': result['mean_error'],
                'Median_Error_m': result['median_error'],
                'Std_Error_m': result['std_error'],
                'Min_Error_m': result['min_error'],
                'Max_Error_m': result['max_error'],
                'Accuracy_50cm': result['accuracy_50cm'],
                'Accuracy_1m': result['accuracy_1m'],
                'Accuracy_2m': result['accuracy_2m'],
                'Accuracy_3m': result['accuracy_3m'],
                'Test_Samples': len(result['errors'])
            })
        
        df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / 'improved_results.csv'
        df.to_csv(csv_path, index=False)
        print(f">>> Improved results saved to {csv_path}")
        
        return df
    
    def plot_learning_curves(self, all_histories, dataset_sizes):
        """Plot learning curves for all improved models"""
        print(">>> Creating improved learning curves...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        model_names = ['BasicCNN_Improved', 'MultiScaleCNN_Improved', 'AttentionCNN_Improved', 
                      'HybridCNN_Improved', 'ResidualCNN_Improved']
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for i, model_name in enumerate(model_names):
            ax = axes[i]
            
            for j, dataset_size in enumerate(dataset_sizes):
                key = f"{model_name}_{dataset_size}"
                if key in all_histories:
                    history = all_histories[key]
                    epochs = range(1, len(history.history['loss']) + 1)
                    
                    alpha = 0.7 + j * 0.1
                    linestyle = ['-', '--', '-.'][j]
                    
                    ax.plot(epochs, history.history['loss'], 
                           color=colors[i], linestyle=linestyle, alpha=alpha,
                           label=f'Train {dataset_size}', linewidth=2)
                    ax.plot(epochs, history.history['val_loss'], 
                           color=colors[i], linestyle=linestyle, alpha=alpha*0.7,
                           label=f'Val {dataset_size}', linewidth=1)
            
            ax.set_title(f'{model_name} Learning Curves (IMPROVED)', fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss (MSE)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide the last subplot
        axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improved_learning_curves.png', dpi=300, bbox_inches='tight')
        print(f"    Saved improved learning curves to {self.output_dir / 'improved_learning_curves.png'}")
        plt.show()
    
    def plot_accuracy_cdfs(self, all_results):
        """Plot CDFs for improved models"""
        print(">>> Creating improved accuracy CDFs...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        thresholds = [1.0, 2.0, 3.0]
        threshold_names = ['1m', '2m', '3m']
        
        for i, (threshold, name) in enumerate(zip(thresholds, threshold_names)):
            ax = axes[i]
            
            for result in all_results:
                errors = result['errors']
                errors_sorted = np.sort(errors)
                p = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
                
                label = f"{result['model_name']} ({result['dataset_size']})"
                ax.plot(errors_sorted, p, label=label, linewidth=2, alpha=0.8)
            
            ax.axvline(x=threshold, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.set_xlabel('Localization Error (meters)')
            ax.set_ylabel('Cumulative Probability')
            ax.set_title(f'CDF: {name} Accuracy (IMPROVED)', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xlim(0, min(6, max([np.max(r['errors']) for r in all_results])))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'improved_accuracy_cdfs.png', dpi=300, bbox_inches='tight')
        print(f"    Saved improved CDFs to {self.output_dir / 'improved_accuracy_cdfs.png'}")
        plt.show()
    
    def run_improved_training(self):
        """Run improved training for all models and dataset sizes"""
        print("="*80)
        print("🎬 TOM CRUISE: IMPROVED CNN TRAINING WITH FIXES")
        print("="*80)
        print("🚀 Improvements implemented:")
        print("   ✅ Data leakage eliminated (scalers fit only on training)")
        print("   ✅ Learning rate reduced: 0.001 → 0.0002")  
        print("   ✅ L2 regularization added: 1e-4")
        print("   ✅ Dropout increased: 0.5 for dense, 0.2 for spatial")
        print("   ✅ Loss function: MSE (more stable than Euclidean)")
        print("   ✅ Data augmentation removed (as requested)")
        print("   ✅ Batch size reduced: 32 → 16")
        print("   ✅ Training epochs increased: 100 → 150")
        print("="*80)
        
        dataset_sizes = [250, 500, 750]
        model_builders = {
            'BasicCNN_Improved': self.build_basic_cnn_improved,
            'MultiScaleCNN_Improved': self.build_multiscale_cnn_improved,
            'AttentionCNN_Improved': self.build_attention_cnn_improved,
            'HybridCNN_Improved': self.build_hybrid_cnn_rssi_improved,
            'ResidualCNN_Improved': self.build_residual_cnn_improved
        }
        
        all_results = []
        all_histories = {}
        
        # Train models for each dataset size
        for dataset_size in dataset_sizes:
            print(f"\n{'='*20} DATASET SIZE: {dataset_size} SAMPLES {'='*20}")
            
            # Load data
            train_data, val_data, test_data = self.load_complete_dataset(dataset_size)
            processed_data = self.prepare_cnn_inputs_fixed(train_data, val_data, test_data, dataset_size)
            
            # NO DATA AUGMENTATION (as requested)
            print("    ✅ NO data augmentation applied (as requested)")
            
            input_shape = (2, 52)
            
            # Train each model
            for model_name, model_builder in model_builders.items():
                print(f"\n--- {model_name} ---")
                
                # Build model
                model = model_builder(input_shape)
                
                # Prepare inputs based on model type
                if 'Hybrid' in model_name:
                    train_inputs = [processed_data['train_csi'], processed_data['train_rssi']]
                    val_inputs = [processed_data['val_csi'], processed_data['val_rssi']]
                    test_inputs = [processed_data['test_csi'], processed_data['test_rssi']]
                else:
                    train_inputs = processed_data['train_csi']
                    val_inputs = processed_data['val_csi']
                    test_inputs = processed_data['test_csi']
                
                # Train model with improved configuration
                model, history = self.compile_and_train_model_improved(
                    model, train_inputs, processed_data['train_coords'], 
                    val_inputs, processed_data['val_coords'],
                    model_name, dataset_size
                )
                
                # Store history
                all_histories[f"{model_name}_{dataset_size}"] = history
                
                # Evaluate model
                results = self.evaluate_model_detailed(
                    model, test_inputs, processed_data['test_coords'],
                    processed_data['original_test_coords'], model_name, dataset_size
                )
                
                all_results.append(results)
        
        # Save comprehensive results
        results_df = self.save_results_summary(all_results)
        
        # Create visualizations
        self.plot_learning_curves(all_histories, dataset_sizes)
        self.plot_accuracy_cdfs(all_results)
        
        # Print final summary
        print("\n" + "="*80)
        print("🏆 TOM CRUISE IMPROVED RESULTS SUMMARY")
        print("="*80)
        
        # Sort by median error
        all_results.sort(key=lambda x: x['median_error'])
        
        print(f"{'Model':<20} {'Size':<6} {'Median (m)':<10} {'<1m (%)':<8} {'<2m (%)':<8} {'<3m (%)':<8}")
        print("-" * 75)
        
        for result in all_results:
            print(f"{result['model_name']:<20} {result['dataset_size']:<6} "
                  f"{result['median_error']:<10.3f} {result['accuracy_1m']:<8.1f} "
                  f"{result['accuracy_2m']:<8.1f} {result['accuracy_3m']:<8.1f}")
        
        best_result = all_results[0]
        print(f"\n🥇 BEST IMPROVED MODEL: {best_result['model_name']} ({best_result['dataset_size']} samples)")
        print(f"   Median Error: {best_result['median_error']:.3f}m")
        print(f"   Accuracy <1m: {best_result['accuracy_1m']:.1f}%")
        print(f"   Accuracy <2m: {best_result['accuracy_2m']:.1f}%")
        
        # Compare with old results
        print(f"\n📊 IMPROVEMENT COMPARISON:")
        print(f"   Previous best median error: 1.610m")
        print(f"   Current best median error: {best_result['median_error']:.3f}m")
        improvement = ((1.610 - best_result['median_error']) / 1.610) * 100
        print(f"   Improvement: {improvement:+.1f}%")
        
        print("\n🎬 TOM CRUISE IMPROVED TRAINING COMPLETE!")
        print(f"📁 All improved results saved in: {self.output_dir}")
        
        return all_results, all_histories

def main():
    """Main execution function"""
    system = TomCruiseImprovedCNNSystem()
    results, histories = system.run_improved_training()
    return system, results, histories

if __name__ == "__main__":
    system, results, histories = main()

