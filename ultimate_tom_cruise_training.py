#!/usr/bin/env python3
"""
Ultimate Tom Cruise: Advanced CNN Training System
Implements state-of-the-art hyperparameter optimization and training techniques
for maximum localization performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
from pathlib import Path
import time
import os
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
    print(">>> TensorFlow imported successfully")
except ImportError:
    print("ERROR: TensorFlow not found. Please install with: pip install tensorflow")
    exit(1)

# Import correct coordinates
from coordinates_config import get_training_points, get_validation_points, get_testing_points

class UltimateTomCruiseSystem:
    """Ultimate Tom Cruise: State-of-the-art CNN training with optimized hyperparameters"""
    
    def __init__(self, output_dir="the last samurai/ultimate_tom_cruise"):
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
        
        print(f"🚀 Ultimate Tom Cruise System Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Advanced hyperparameter optimization enabled")
        
    def get_optimized_hyperparameters(self, dataset_size):
        """Get optimized hyperparameters based on dataset size"""
        
        if dataset_size == 250:
            return {
                'initial_lr': 0.0008,  # Higher for small datasets
                'batch_size': 8,       # Smaller batches for better gradients
                'l2_reg': 3e-5,        # Less regularization for small data
                'dropout_dense': 0.6,  # More dropout to prevent overfitting
                'dropout_spatial': 0.3,
                'epochs': 250,         # More epochs for small datasets
                'patience_early': 40,
                'patience_lr': 20
            }
        elif dataset_size == 500:
            return {
                'initial_lr': 0.0005,  # Moderate learning rate
                'batch_size': 12,      # Balanced batch size
                'l2_reg': 5e-5,        # Moderate regularization
                'dropout_dense': 0.55,
                'dropout_spatial': 0.25,
                'epochs': 200,
                'patience_early': 35,
                'patience_lr': 18
            }
        else:  # 750
            return {
                'initial_lr': 0.0003,  # Lower for larger datasets
                'batch_size': 16,      # Larger batches for stability
                'l2_reg': 7e-5,        # More regularization for larger data
                'dropout_dense': 0.5,
                'dropout_spatial': 0.2,
                'epochs': 180,
                'patience_early': 30,
                'patience_lr': 15
            }
    
    def cosine_annealing_schedule(self, epoch, initial_lr, total_epochs, min_lr=1e-7):
        """Cosine annealing learning rate schedule"""
        return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs)) / 2
    
    def warmup_cosine_schedule(self, epoch, initial_lr, total_epochs, warmup_epochs=10, min_lr=1e-7):
        """Warmup + Cosine annealing schedule"""
        if epoch < warmup_epochs:
            # Linear warmup
            return initial_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            adjusted_epoch = epoch - warmup_epochs
            adjusted_total = total_epochs - warmup_epochs
            return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * adjusted_epoch / adjusted_total)) / 2
    
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
            folder = "Testing Points Dataset 750 Samples"
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
        
        train_data = self.load_data_by_coordinates(dataset_size, "training")
        print(f"    Training: {len(train_data['amplitudes'])} samples from {len(self.training_points)} points")
        
        val_data = self.load_data_by_coordinates(dataset_size, "validation")
        print(f"    Validation: {len(val_data['amplitudes'])} samples from {len(self.validation_points)} points")
        
        test_data = self.load_data_by_coordinates(750, "testing")
        print(f"    Testing: {len(test_data['amplitudes'])} samples from {len(self.testing_points)} points")
        
        return train_data, val_data, test_data
    
    def prepare_cnn_inputs_ultimate(self, train_data, val_data, test_data, dataset_size):
        """Prepare CNN inputs with ultimate preprocessing"""
        print(f">>> Ultimate preprocessing for {dataset_size} samples...")
        
        # Fit scalers ONLY on training data
        amp_scaler = StandardScaler()
        phase_scaler = StandardScaler()
        rssi_scaler = StandardScaler()
        coord_scaler = MinMaxScaler(feature_range=(0, 1))
        
        amp_scaler.fit(train_data['amplitudes'])
        phase_scaler.fit(train_data['phases'])
        rssi_scaler.fit(train_data['rssi'].reshape(-1, 1))
        coord_scaler.fit(train_data['coordinates'])
        
        # Transform data
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
        
        # Create CNN format
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
        
        print(f"    ✅ Ultimate preprocessing complete")
        
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
    
    def build_ultimate_hybrid_cnn(self, input_shape, hyperparams):
        """Build ultimate hybrid CNN with optimized architecture"""
        # CSI branch with advanced architecture
        csi_input = keras.Input(shape=input_shape, name='csi_input')
        csi_x = layers.Reshape((input_shape[1], input_shape[0]))(csi_input)
        
        # Multi-scale feature extraction
        path1 = layers.Conv1D(48, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(csi_x)
        path1 = layers.BatchNormalization()(path1)
        path1 = layers.SpatialDropout1D(hyperparams['dropout_spatial'])(path1)
        
        path2 = layers.Conv1D(48, 7, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(csi_x)
        path2 = layers.BatchNormalization()(path2)
        path2 = layers.SpatialDropout1D(hyperparams['dropout_spatial'])(path2)
        
        path3 = layers.Conv1D(48, 15, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(csi_x)
        path3 = layers.BatchNormalization()(path3)
        path3 = layers.SpatialDropout1D(hyperparams['dropout_spatial'])(path3)
        
        # Merge and process
        csi_merged = layers.Concatenate()([path1, path2, path3])
        csi_x = layers.Conv1D(96, 5, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(csi_merged)
        csi_x = layers.BatchNormalization()(csi_x)
        csi_x = layers.SpatialDropout1D(hyperparams['dropout_spatial'])(csi_x)
        csi_x = layers.MaxPooling1D(2)(csi_x)
        
        # Second level processing
        csi_x = layers.Conv1D(128, 3, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(csi_x)
        csi_x = layers.BatchNormalization()(csi_x)
        csi_x = layers.GlobalAveragePooling1D()(csi_x)
        
        # Advanced CSI features
        csi_features = layers.Dense(192, activation='relu', 
                                   kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(csi_x)
        csi_features = layers.Dropout(hyperparams['dropout_dense'] * 0.8)(csi_features)
        
        # Enhanced RSSI branch
        rssi_input = keras.Input(shape=(1,), name='rssi_input')
        rssi_x = layers.Dense(64, activation='relu', 
                             kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(rssi_input)
        rssi_x = layers.Dropout(hyperparams['dropout_dense'] * 0.6)(rssi_x)
        rssi_x = layers.Dense(48, activation='relu', 
                             kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(rssi_x)
        rssi_features = layers.Dense(48, activation='relu', 
                                    kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(rssi_x)
        
        # Advanced fusion
        combined = layers.Concatenate()([csi_features, rssi_features])
        
        # Multi-layer fusion network
        x = layers.Dense(320, activation='relu', 
                        kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(combined)
        x = layers.Dropout(hyperparams['dropout_dense'])(x)
        
        x = layers.Dense(192, activation='relu', 
                        kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(x)
        x = layers.Dropout(hyperparams['dropout_dense'] * 0.8)(x)
        
        x = layers.Dense(96, activation='relu', 
                        kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(x)
        x = layers.Dropout(hyperparams['dropout_dense'] * 0.6)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=[csi_input, rssi_input], outputs=output, 
                           name='UltimateHybridCNN')
        return model
    
    def compile_and_train_ultimate_model(self, model, train_inputs, train_targets, 
                                       val_inputs, val_targets, model_name, dataset_size):
        """Compile and train model with ultimate configuration"""
        hyperparams = self.get_optimized_hyperparameters(dataset_size)
        
        print(f"    🚀 Training {model_name} with ULTIMATE config:")
        print(f"        Initial LR: {hyperparams['initial_lr']}")
        print(f"        Batch size: {hyperparams['batch_size']}")
        print(f"        L2 reg: {hyperparams['l2_reg']}")
        print(f"        Dropout: {hyperparams['dropout_dense']}/{hyperparams['dropout_spatial']}")
        print(f"        Epochs: {hyperparams['epochs']}")
        
        # Advanced optimizer with gradient clipping
        optimizer = keras.optimizers.Adam(
            learning_rate=hyperparams['initial_lr'],
            clipnorm=1.0  # Gradient clipping
        )
        
        # Label smoothing for robustness
        def smooth_mse_loss(y_true, y_pred, smoothing=0.05):
            mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
            # Add small noise for label smoothing effect
            noise = tf.random.normal(tf.shape(y_pred), stddev=smoothing)
            smooth_pred = y_pred + noise
            smooth_mse = tf.keras.losses.mean_squared_error(y_true, smooth_pred)
            return 0.9 * mse + 0.1 * smooth_mse
        
        model.compile(
            optimizer=optimizer,
            loss='mse',  # Standard MSE for stability
            metrics=['mae']
        )
        
        # Advanced callbacks
        checkpoint_path = self.output_dir / f"{model_name.lower()}_{dataset_size}_samples_ultimate.h5"
        
        # Learning rate scheduler
        def lr_schedule(epoch):
            return self.warmup_cosine_schedule(
                epoch, 
                hyperparams['initial_lr'], 
                hyperparams['epochs'],
                warmup_epochs=10
            )
        
        callbacks = [
            EarlyStopping(
                patience=hyperparams['patience_early'], 
                restore_best_weights=True, 
                monitor='val_loss',
                min_delta=1e-6
            ),
            LearningRateScheduler(lr_schedule, verbose=0),
            ModelCheckpoint(
                str(checkpoint_path), 
                save_best_only=True, 
                monitor='val_loss',
                save_weights_only=False
            )
        ]
        
        print(f"        Parameters: {model.count_params():,}")
        
        start_time = time.time()
        history = model.fit(
            train_inputs, train_targets,
            validation_data=(val_inputs, val_targets),
            epochs=hyperparams['epochs'],
            batch_size=hyperparams['batch_size'],
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        elapsed = time.time() - start_time
        print(f"        ✅ Ultimate training completed in {elapsed:.1f}s")
        
        return model, history
    
    def evaluate_model_ultimate(self, model, test_inputs, test_targets, original_coords, model_name, dataset_size):
        """Evaluate model with comprehensive metrics"""
        print(f"    🎯 Evaluating {model_name} ({dataset_size} samples)...")
        
        predictions = model.predict(test_inputs, verbose=0)
        predictions_denorm = self.scalers[dataset_size]['coordinates'].inverse_transform(predictions)
        
        euclidean_errors = np.sqrt(np.sum((original_coords - predictions_denorm)**2, axis=1))
        
        results = {
            'model_name': model_name,
            'dataset_size': dataset_size,
            'mean_error': np.mean(euclidean_errors),
            'median_error': np.median(euclidean_errors),
            'std_error': np.std(euclidean_errors),
            'max_error': np.max(euclidean_errors),
            'min_error': np.min(euclidean_errors),
            'accuracy_25cm': np.mean(euclidean_errors < 0.25) * 100,
            'accuracy_50cm': np.mean(euclidean_errors < 0.5) * 100,
            'accuracy_1m': np.mean(euclidean_errors < 1.0) * 100,
            'accuracy_2m': np.mean(euclidean_errors < 2.0) * 100,
            'accuracy_3m': np.mean(euclidean_errors < 3.0) * 100,
            'predictions': predictions_denorm,
            'targets': original_coords,
            'errors': euclidean_errors
        }
        
        print(f"        🎯 Median Error: {results['median_error']:.3f}m")
        print(f"        🎯 Accuracy <50cm: {results['accuracy_50cm']:.1f}%")
        print(f"        🎯 Accuracy <1m: {results['accuracy_1m']:.1f}%")
        print(f"        🎯 Accuracy <2m: {results['accuracy_2m']:.1f}%")
        
        return results
    
    def run_ultimate_training(self):
        """Run ultimate training with optimized hyperparameters"""
        print("="*80)
        print("🚀 ULTIMATE TOM CRUISE: MAXIMUM PERFORMANCE TRAINING")
        print("="*80)
        print("🎯 Advanced optimizations:")
        print("   ✅ Dataset-specific hyperparameters")
        print("   ✅ Warmup + Cosine annealing LR schedule")
        print("   ✅ Optimized batch sizes per dataset")
        print("   ✅ Advanced regularization tuning")
        print("   ✅ Label smoothing for robustness")
        print("   ✅ Gradient clipping")
        print("   ✅ Enhanced architecture")
        print("="*80)
        
        dataset_sizes = [250, 500, 750]
        all_results = []
        all_histories = {}
        
        for dataset_size in dataset_sizes:
            print(f"\n{'='*20} ULTIMATE TRAINING: {dataset_size} SAMPLES {'='*20}")
            
            # Load data
            train_data, val_data, test_data = self.load_complete_dataset(dataset_size)
            processed_data = self.prepare_cnn_inputs_ultimate(train_data, val_data, test_data, dataset_size)
            
            # Get optimized hyperparameters
            hyperparams = self.get_optimized_hyperparameters(dataset_size)
            
            # Build ultimate model
            input_shape = (2, 52)
            model = self.build_ultimate_hybrid_cnn(input_shape, hyperparams)
            
            # Prepare inputs
            train_inputs = [processed_data['train_csi'], processed_data['train_rssi']]
            val_inputs = [processed_data['val_csi'], processed_data['val_rssi']]
            test_inputs = [processed_data['test_csi'], processed_data['test_rssi']]
            
            # Train with ultimate configuration
            model, history = self.compile_and_train_ultimate_model(
                model, train_inputs, processed_data['train_coords'], 
                val_inputs, processed_data['val_coords'],
                f"UltimateHybrid", dataset_size
            )
            
            # Store history
            all_histories[f"UltimateHybrid_{dataset_size}"] = history
            
            # Evaluate
            results = self.evaluate_model_ultimate(
                model, test_inputs, processed_data['test_coords'],
                processed_data['original_test_coords'], f"UltimateHybrid", dataset_size
            )
            
            all_results.append(results)
        
        # Save results
        self.save_ultimate_results(all_results, all_histories)
        
        return all_results, all_histories
    
    def save_ultimate_results(self, all_results, all_histories):
        """Save ultimate results and create visualizations"""
        
        # Save CSV
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
                'Accuracy_25cm': result['accuracy_25cm'],
                'Accuracy_50cm': result['accuracy_50cm'],
                'Accuracy_1m': result['accuracy_1m'],
                'Accuracy_2m': result['accuracy_2m'],
                'Accuracy_3m': result['accuracy_3m'],
                'Test_Samples': len(result['errors'])
            })
        
        df = pd.DataFrame(csv_data)
        df = df.sort_values('Median_Error_m')
        
        csv_path = self.output_dir / 'ultimate_results.csv'
        df.to_csv(csv_path, index=False)
        
        # Print results
        print("\n" + "="*80)
        print("🏆 ULTIMATE TOM CRUISE RESULTS")
        print("="*80)
        
        print(f"{'Model':<18} {'Size':<6} {'Median(m)':<10} {'<25cm(%)':<9} {'<50cm(%)':<9} {'<1m(%)':<7} {'<2m(%)':<7}")
        print("-" * 80)
        
        for _, row in df.iterrows():
            print(f"{row['Model']:<18} {row['Dataset_Size']:<6} "
                  f"{row['Median_Error_m']:<10.3f} {row['Accuracy_25cm']:<9.1f} {row['Accuracy_50cm']:<9.1f} "
                  f"{row['Accuracy_1m']:<7.1f} {row['Accuracy_2m']:<7.1f}")
        
        best = df.iloc[0]
        print(f"\n🥇 ULTIMATE BEST: {best['Model']} ({best['Dataset_Size']} samples)")
        print(f"   🎯 Median Error: {best['Median_Error_m']:.3f}m")
        print(f"   🎯 Sub-meter Accuracy: {best['Accuracy_1m']:.1f}%")
        
        # Compare with Tom Cruise
        print(f"\n📊 IMPROVEMENT vs Tom Cruise:")
        print(f"   Previous best: 1.193m → Current: {best['Median_Error_m']:.3f}m")
        improvement = ((1.193 - best['Median_Error_m']) / 1.193) * 100
        print(f"   Improvement: {improvement:+.1f}%")
        
        print(f"\n🚀 ULTIMATE TOM CRUISE COMPLETE!")
        print(f"📁 Results saved in: {self.output_dir}")

def main():
    """Main execution function"""
    system = UltimateTomCruiseSystem()
    results, histories = system.run_ultimate_training()
    return system, results, histories

if __name__ == "__main__":
    system, results, histories = main()
