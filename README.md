# Master Thesis: CNN-Based Indoor Localization Using CSI Data

This repository contains the complete implementation and analysis of CNN-based indoor localization using Channel State Information (CSI) data for my Master's thesis.

## 🎯 Project Overview

This research explores deep learning approaches for indoor localization using WiFi Channel State Information (CSI) data. The project implements and compares multiple CNN architectures for accurate position estimation in indoor environments.

## 📊 Key Results

### Best Performing Models
- **Tom Cruise Best**: HybridCNN (250 samples) - **1.193m median error**, 45.5% accuracy <1m
- **Ultimate Tom Cruise**: HybridCNN (750 samples) - **1.469m median error**, 36.4% accuracy <1m

### CNN Architectures Implemented
1. **Basic CNN** - Simple convolutional architecture
2. **Multi-Scale CNN** - Multiple kernel sizes for feature extraction
3. **Attention CNN** - Self-attention mechanism for feature selection
4. **Hybrid CNN + RSSI** - Combined CSI and RSSI features
5. **Residual CNN** - ResNet-inspired architecture with skip connections

## 📁 Repository Structure

```
Master_Thesis/
├── CSI Dataset 250-750 Samples/          # Training datasets
├── Testing Points Dataset/                # Test datasets  
├── the last samurai/                     # Model experiments
│   ├── tom cruise/                       # Improved models
│   └── ultimate_tom_cruise/              # Final optimized models
├── cnn_localization_deep_learning.py     # Main CNN implementation
├── coordinates_config.py                 # Dataset coordinate configuration
├── classical_fingerprinting_baselines.py # Classical algorithm baselines
├── HYPERPARAMETER_OPTIMIZATION_GUIDE.md  # Hyperparameter tuning guide
└── analysis scripts/                     # Various analysis tools
```

## 🚀 Key Features

### Advanced CNN Implementations
- **Data Leakage Prevention**: Proper train/validation/test splits
- **Hyperparameter Optimization**: Dataset-specific learning rates and batch sizes
- **Advanced Regularization**: L2 regularization, dropout, batch normalization
- **Learning Rate Scheduling**: Warmup + cosine annealing

### Comprehensive Analysis
- **Performance Comparison**: 5 CNN architectures vs classical methods
- **Ablation Studies**: Feature importance and architecture components
- **Hyperparameter Analysis**: Learning rate, batch size, regularization effects
- **Visualization Tools**: Learning curves, CDFs, spatial distribution plots

## 📈 Experimental Results

### CNN vs Classical Methods
- **CNN Best**: 1.193m median error (HybridCNN)
- **Classical Best**: ~2.5m median error (Weighted k-NN)
- **Improvement**: >50% better accuracy with CNNs

### Dataset Size Analysis
- **250 samples**: 1.193m median error (best performance)
- **500 samples**: 1.698m median error  
- **750 samples**: 1.699m median error

## 🛠️ Technical Implementation

### Environment Setup
```bash
# Core dependencies
tensorflow>=2.10.0
scikit-learn>=1.1.0
numpy>=1.21.0
pandas>=1.4.0
matplotlib>=3.5.0
```

### Quick Start
```python
# Load and train a CNN model
from cnn_localization_deep_learning import CNNLocalizationSystem
from coordinates_config import get_training_points, get_validation_points

system = CNNLocalizationSystem()
model = system.build_hybrid_cnn_rssi((2, 52))
# Training and evaluation code...
```

## 📊 Data Format

### CSI Data Structure
- **Input Shape**: (samples, 2, 52) - Amplitude and Phase for 52 subcarriers
- **Output**: (x, y) coordinates in meters
- **Environment**: 7×7 meter indoor laboratory
- **Grid Resolution**: 1 meter spacing

### Coordinate System
- **Training Points**: 27 reference locations
- **Validation Points**: 7 reference locations  
- **Testing Points**: 5 intermediate locations (0.5m offset)

## 🔬 Research Contributions

1. **Comprehensive CNN Architecture Comparison** for CSI-based localization
2. **Advanced Hyperparameter Optimization** techniques for small datasets
3. **Proper Data Handling** to prevent data leakage in localization tasks
4. **Performance Analysis** across different dataset sizes and architectures
5. **Classical vs Deep Learning Comparison** with fair evaluation methodology

## 📝 Key Files

- `cnn_localization_deep_learning.py` - Main CNN implementation
- `tom_cruise_improved_cnn_training.py` - Improved training pipeline
- `ultimate_tom_cruise_training.py` - Final optimized training
- `classical_fingerprinting_baselines.py` - Classical method baselines
- `HYPERPARAMETER_OPTIMIZATION_GUIDE.md` - Comprehensive optimization guide

## 🎯 Future Work

- **Ensemble Methods**: Combining multiple CNN predictions
- **Real-time Implementation**: Optimization for deployment
- **Multi-building Generalization**: Cross-environment validation
- **Advanced Architectures**: Transformer-based models for CSI data

## 📧 Contact

For questions about this research or collaboration opportunities, please reach out through GitHub issues or repository discussions.

---

*This repository represents comprehensive research in deep learning for indoor localization, demonstrating state-of-the-art performance using CNN architectures with CSI data.*
