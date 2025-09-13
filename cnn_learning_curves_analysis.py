#!/usr/bin/env python3
"""
CNN Learning Curves Analysis

Creates IEEE-style learning curves showing training and validation loss 
over epochs for the 5 main CNN architectures. This reveals:
- Convergence behavior
- Overfitting/underfitting patterns  
- Training stability
- Optimal stopping points

Analyzes: Baseline CNN, Hybrid CNN+RSSI, Attention CNN, Multi-Scale CNN, Residual CNN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print("✅ TensorFlow imported successfully")
except ImportError:
    print("❌ TensorFlow not found. Please install with: pip install tensorflow")
    exit(1)

class CNNLearningCurveAnalysis:
    """Analyze learning curves for CNN architectures"""
    
    def __init__(self, sample_size=250):
        self.sample_size = sample_size
        self.models = {}
        self.histories = {}
        self.training_data = None
        self.validation_data = None
        
    def load_csi_data(self):
        """Load CSI data for training"""
        
        print(f"📂 Loading CSI data (sample size: {self.sample_size})...")
        
        # Training points (27 points as used in our experiments)
        training_points = [
            (0, 0), (0, 1), (0, 2), (0, 4), (0, 5), (0, 6),
            (1, 0), (1, 1), (1, 4), (1, 5),
            (2, 0), (2, 2), (2, 3), (2, 4), (2, 5),
            (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
            (4, 0), (4, 1), (4, 4), (4, 5),
            (5, 2), (5, 3)
        ]
        
        # Validation points (7 points)
        validation_points = [
            (0, 3), (2, 1), (5, 0), (5, 1), (5, 4), (6, 3), (6, 4)
        ]
        
        def load_points_data(points, dataset_folder):
            """Load data for given points"""
            data = []
            coordinates = []
            
            for x, y in points:
                file_path = Path(dataset_folder) / f"{x},{y}.csv"
                
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    
                    # Take only requested sample size
                    df = df.head(self.sample_size)
                    
                    # Extract amplitude data (convert string representation to array)
                    amplitude_data = df['amplitude'].apply(eval).tolist()
                    
                    for amp in amplitude_data:
                        data.append(np.array(amp))
                        coordinates.append([x, y])
                    
                    print(f"   ✅ Loaded {len(df)} samples from ({x}, {y})")
                else:
                    print(f"   ❌ Missing file: {file_path}")
            
            return np.array(data), np.array(coordinates)
        
        # Load training and validation data
        dataset_folder = f"CSI Dataset {self.sample_size} Samples"
        
        X_train, y_train = load_points_data(training_points, dataset_folder)
        X_val, y_val = load_points_data(validation_points, dataset_folder)
        
        print(f"📊 Data Summary:")
        print(f"   Training: {X_train.shape[0]} samples from {len(training_points)} points")
        print(f"   Validation: {X_val.shape[0]} samples from {len(validation_points)} points")
        print(f"   Feature shape: {X_train.shape[1:]} (52 amplitude values)")
        
        # Store data
        self.training_data = (X_train, y_train)
        self.validation_data = (X_val, y_val)
        
        return X_train, y_train, X_val, y_val
    
    def build_baseline_cnn(self):
        """Build baseline CNN architecture"""
        
        input_layer = keras.Input(shape=(52,), name='amplitude_input')
        
        # Reshape for 1D convolution
        x = layers.Reshape((52, 1))(input_layer)
        
        # 1D Convolutions
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
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=input_layer, outputs=output, name='BaselineCNN')
        return model
    
    def build_hybrid_cnn_rssi(self):
        """Build hybrid CNN + RSSI architecture"""
        
        # CSI input
        csi_input = keras.Input(shape=(52,), name='csi_input')
        csi_x = layers.Reshape((52, 1))(csi_input)
        
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
        
        # RSSI input (simulated from amplitude statistics)
        rssi_input = keras.Input(shape=(1,), name='rssi_input')
        rssi_x = layers.Dense(32, activation='relu')(rssi_input)
        rssi_x = layers.Dense(32, activation='relu')(rssi_x)
        rssi_features = layers.Dense(32, activation='relu')(rssi_x)
        
        # Fusion
        combined = layers.Concatenate()([csi_features, rssi_features])
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=[csi_input, rssi_input], outputs=output, name='HybridCNN_RSSI')
        return model
    
    def build_attention_cnn(self):
        """Build attention CNN architecture"""
        
        input_layer = keras.Input(shape=(52,), name='amplitude_input')
        x = layers.Reshape((52, 1))(input_layer)
        
        # Initial convolutions
        x = layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Self-attention mechanism
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
        
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=input_layer, outputs=output, name='AttentionCNN')
        return model
    
    def build_multiscale_cnn(self):
        """Build multi-scale CNN architecture"""
        
        input_layer = keras.Input(shape=(52,), name='amplitude_input')
        x = layers.Reshape((52, 1))(input_layer)
        
        # Multiple parallel paths
        path1 = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
        path1 = layers.BatchNormalization()(path1)
        path1 = layers.MaxPooling1D(2)(path1)
        
        path2 = layers.Conv1D(32, 7, activation='relu', padding='same')(x)
        path2 = layers.BatchNormalization()(path2)
        path2 = layers.MaxPooling1D(2)(path2)
        
        path3 = layers.Conv1D(32, 15, activation='relu', padding='same')(x)
        path3 = layers.BatchNormalization()(path3)
        path3 = layers.MaxPooling1D(2)(path3)
        
        # Concatenate paths
        merged = layers.Concatenate()([path1, path2, path3])
        
        x = layers.Conv1D(128, 3, activation='relu', padding='same')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=input_layer, outputs=output, name='MultiScaleCNN')
        return model
    
    def build_residual_cnn(self):
        """Build residual CNN architecture"""
        
        def residual_block(x, filters, kernel_size=3):
            """Residual block with skip connection"""
            shortcut = x
            
            y = layers.Conv1D(filters, kernel_size, padding='same', activation='relu')(x)
            y = layers.BatchNormalization()(y)
            y = layers.Conv1D(filters, kernel_size, padding='same')(y)
            y = layers.BatchNormalization()(y)
            
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
                shortcut = layers.BatchNormalization()(shortcut)
            
            out = layers.Add()([shortcut, y])
            out = layers.Activation('relu')(out)
            
            return out
        
        input_layer = keras.Input(shape=(52,), name='amplitude_input')
        x = layers.Reshape((52, 1))(input_layer)
        
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
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        output = layers.Dense(2, activation='linear', name='coordinates')(x)
        
        model = keras.Model(inputs=input_layer, outputs=output, name='ResidualCNN')
        return model
    
    def euclidean_distance_loss(self, y_true, y_pred):
        """Custom loss function - Euclidean distance in physical space"""
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1)))
    
    def train_model_with_curves(self, model, model_name, X_train, y_train, X_val, y_val):
        """Train model and capture learning curves"""
        
        print(f"\n🎯 Training {model_name}...")
        
        # Prepare data based on model type
        if model_name == 'Hybrid CNN + RSSI':
            # Simulate RSSI from amplitude statistics
            rssi_train = np.mean(X_train, axis=1).reshape(-1, 1)
            rssi_val = np.mean(X_val, axis=1).reshape(-1, 1)
            
            train_inputs = [X_train, rssi_train]
            val_inputs = [X_val, rssi_val]
        else:
            train_inputs = X_train
            val_inputs = X_val
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=self.euclidean_distance_loss,
            metrics=['mae']
        )
        
        # Training callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-6, monitor='val_loss')
        ]
        
        # Train model
        history = model.fit(
            train_inputs,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(val_inputs, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Store history
        self.histories[model_name] = history.history
        self.models[model_name] = model
        
        # Final evaluation
        val_loss = model.evaluate(val_inputs, y_val, verbose=0)
        print(f"   ✅ {model_name} final validation loss: {val_loss[0]:.4f}")
        
        return history.history
    
    def train_all_models(self):
        """Train all 5 CNN architectures and capture learning curves"""
        
        print("🚀 TRAINING ALL CNN ARCHITECTURES FOR LEARNING CURVES")
        print("="*65)
        
        # Load data
        X_train, y_train, X_val, y_val = self.load_csi_data()
        
        # Build and train each model
        models_to_train = [
            ('Baseline CNN', self.build_baseline_cnn),
            ('Hybrid CNN + RSSI', self.build_hybrid_cnn_rssi),
            ('Attention CNN', self.build_attention_cnn),
            ('Multi-Scale CNN', self.build_multiscale_cnn),
            ('Residual CNN', self.build_residual_cnn)
        ]
        
        for model_name, build_func in models_to_train:
            print(f"\n{'='*50}")
            print(f"🏗️ Building and training {model_name}")
            print('='*50)
            
            # Build model
            model = build_func()
            
            # Train and capture curves
            history = self.train_model_with_curves(
                model, model_name, X_train, y_train, X_val, y_val
            )
        
        print(f"\n✅ ALL MODELS TRAINED SUCCESSFULLY!")
        return self.histories
    
    def plot_ieee_style_learning_curves(self):
        """Create IEEE-style learning curve plots"""
        
        print("📊 Creating IEEE-style Learning Curves...")
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Colors for each model
        colors = {
            'Baseline CNN': '#1f77b4',
            'Hybrid CNN + RSSI': '#ff7f0e', 
            'Attention CNN': '#2ca02c',
            'Multi-Scale CNN': '#d62728',
            'Residual CNN': '#9467bd'
        }
        
        # Plot individual learning curves
        for idx, (model_name, history) in enumerate(self.histories.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            epochs = range(1, len(history['loss']) + 1)
            
            # Plot training and validation loss
            ax.plot(epochs, history['loss'], 
                   color=colors[model_name], linewidth=2, label='Training Loss', alpha=0.8)
            ax.plot(epochs, history['val_loss'], 
                   color=colors[model_name], linewidth=2, linestyle='--', label='Validation Loss', alpha=0.8)
            
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Loss (Euclidean Distance)', fontweight='bold')
            ax.set_title(f'{model_name}', fontweight='bold', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Find best epoch (minimum validation loss)
            best_epoch = np.argmin(history['val_loss']) + 1
            best_val_loss = np.min(history['val_loss'])
            
            # Mark best epoch
            ax.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.7)
            ax.text(best_epoch, best_val_loss, f'Best: {best_epoch}', 
                   rotation=90, verticalalignment='bottom', fontsize=9)
        
        # Remove empty subplot
        if len(self.histories) == 5:
            axes[1, 2].remove()
        
        plt.suptitle(f'IEEE-Style Learning Curves: CNN Architectures ({self.sample_size} samples per location)\n'
                    'Training vs Validation Loss Over Epochs', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save plot
        output_file = f'ieee_learning_curves_{self.sample_size}_samples.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 IEEE-style learning curves saved: {output_file}")
        
        plt.show()
    
    def plot_comparative_learning_curves(self):
        """Create comparative learning curves showing all models together"""
        
        print("📈 Creating Comparative Learning Curves...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        colors = {
            'Baseline CNN': '#1f77b4',
            'Hybrid CNN + RSSI': '#ff7f0e',
            'Attention CNN': '#2ca02c', 
            'Multi-Scale CNN': '#d62728',
            'Residual CNN': '#9467bd'
        }
        
        # Plot 1: Training Loss Comparison
        for model_name, history in self.histories.items():
            epochs = range(1, len(history['loss']) + 1)
            ax1.plot(epochs, history['loss'], 
                    color=colors[model_name], linewidth=2, label=model_name, alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Training Loss', fontweight='bold')
        ax1.set_title('Training Loss Comparison', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Validation Loss Comparison
        for model_name, history in self.histories.items():
            epochs = range(1, len(history['val_loss']) + 1)
            ax2.plot(epochs, history['val_loss'], 
                    color=colors[model_name], linewidth=2, label=model_name, alpha=0.8)
        
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Validation Loss', fontweight='bold')
        ax2.set_title('Validation Loss Comparison', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(f'Comparative Learning Curves: All CNN Architectures ({self.sample_size} samples)\n'
                    'Training Dynamics and Convergence Behavior', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save plot
        output_file = f'comparative_learning_curves_{self.sample_size}_samples.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Comparative learning curves saved: {output_file}")
        
        plt.show()
    
    def analyze_training_dynamics(self):
        """Analyze training dynamics and provide insights"""
        
        print("\n🔍 TRAINING DYNAMICS ANALYSIS")
        print("="*40)
        
        analysis_results = {}
        
        for model_name, history in self.histories.items():
            print(f"\n📊 {model_name}:")
            
            train_loss = history['loss']
            val_loss = history['val_loss']
            
            # Find best epoch
            best_epoch = np.argmin(val_loss) + 1
            best_val_loss = np.min(val_loss)
            final_train_loss = train_loss[best_epoch - 1]
            
            # Calculate overfitting indicators
            final_gap = val_loss[-1] - train_loss[-1]
            best_gap = best_val_loss - final_train_loss
            
            # Convergence analysis
            early_val_loss = np.mean(val_loss[:5]) if len(val_loss) >= 5 else val_loss[0]
            improvement = early_val_loss - best_val_loss
            
            analysis_results[model_name] = {
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss,
                'total_epochs': len(train_loss),
                'final_gap': final_gap,
                'improvement': improvement,
                'converged_early': best_epoch < len(train_loss) * 0.7
            }
            
            print(f"   Best epoch: {best_epoch}/{len(train_loss)}")
            print(f"   Best validation loss: {best_val_loss:.4f}")
            print(f"   Training improvement: {improvement:.4f}")
            print(f"   Final train-val gap: {final_gap:.4f}")
            
            if final_gap > 0.3:
                print("   ⚠️  Potential overfitting detected")
            elif final_gap < 0.1:
                print("   ✅ Good generalization")
            else:
                print("   📊 Moderate overfitting")
            
            if best_epoch < len(train_loss) * 0.5:
                print("   🚀 Fast convergence")
            elif best_epoch < len(train_loss) * 0.8:
                print("   📈 Normal convergence")
            else:
                print("   🐌 Slow convergence")
        
        return analysis_results

def main():
    """Main execution function"""
    
    print("📚 CNN LEARNING CURVES ANALYSIS")
    print("="*40)
    print("Creating IEEE-style learning curves for 5 CNN architectures")
    print("Sample size: 250 per location")
    
    # Initialize analyzer
    analyzer = CNNLearningCurveAnalysis(sample_size=250)
    
    # Train all models and capture learning curves
    histories = analyzer.train_all_models()
    
    # Create IEEE-style plots
    analyzer.plot_ieee_style_learning_curves()
    
    # Create comparative plots
    analyzer.plot_comparative_learning_curves()
    
    # Analyze training dynamics
    analysis = analyzer.analyze_training_dynamics()
    
    print(f"\n✅ LEARNING CURVE ANALYSIS COMPLETE!")
    print(f"📊 Generated IEEE-style learning curves for {len(histories)} models")
    print(f"🎯 This shows convergence behavior, overfitting patterns, and training stability")

if __name__ == "__main__":
    main()


