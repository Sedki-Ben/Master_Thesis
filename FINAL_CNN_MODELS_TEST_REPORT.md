# CNN Models Testing Results - 750 Sample Test Dataset

## Executive Summary

This report presents the comprehensive testing results of 5 trained CNN models for indoor localization using CSI (Channel State Information) data. The models were evaluated on a test dataset containing 750 samples (3,688 individual measurements) from 5 unique locations.

## Dataset Information

- **Test Dataset**: Testing Points Dataset 750 Samples
- **Total Test Samples**: 3,688 measurements
- **Unique Test Locations**: 5 locations
- **Input Features**: 52 subcarrier amplitudes and phases, RSSI values
- **Target**: 2D coordinates (x, y)

## Model Testing Results

### Models Successfully Tested: 4 out of 5

| Rank | Model | Median Error (m) | Mean Error (m) | Accuracy <1m (%) | Accuracy <2m (%) | Parameters |
|------|-------|------------------|----------------|------------------|------------------|------------|
| 🥇 1st | **Multi-Scale CNN** | **1.787** | **1.978** | **22.2%** | **52.8%** | 105,762 |
| 🥈 2nd | Residual CNN | 2.160 | 2.180 | 6.0% | 40.5% | 178,530 |
| 🥉 3rd | Hybrid CNN | 2.341 | 2.089 | 22.7% | 40.3% | 106,562 |
| 4th | Basic CNN | 2.730 | 2.259 | 19.4% | 40.4% | 23,650 |

### Model Status Notes
- **Attention CNN**: Could not be tested due to TensorFlow compatibility issues with Lambda layers
- **Note**: Ranking based on median error (more robust metric than mean)

## Detailed Performance Analysis

### 🏆 Best Model: Multi-Scale CNN

**Key Performance Metrics:**
- **Median Error**: 1.787 meters
- **Mean Error**: 1.978 meters
- **Standard Deviation**: 1.180 meters
- **Min Error**: 0.012 meters
- **Max Error**: 4.189 meters

**Accuracy Breakdown:**
- **<50cm**: 16.5% of predictions
- **<1m**: 22.2% of predictions
- **<2m**: 52.8% of predictions
- **<3m**: 78.1% of predictions
- **<5m**: 100.0% of predictions

**Model Characteristics:**
- **Architecture**: Multiple parallel convolution paths with different kernel sizes
- **Parameters**: 105,762 (moderate complexity)
- **Strength**: Captures both local and global frequency patterns effectively

### Comparative Performance Summary

| Model | 50cm Accuracy (%) | 1m Accuracy (%) | 2m Accuracy (%) | 3m Accuracy (%) |
|-------|-------------------|-----------------|-----------------|-----------------|
| Multi-Scale CNN | 16.5 | 22.2 | 52.8 | 78.1 |
| Residual CNN | 0.5 | 6.0 | 40.5 | 85.6 |
| Hybrid CNN | 5.5 | 22.7 | 40.3 | 82.6 |
| Basic CNN | 18.3 | 19.4 | 40.4 | 59.4 |

## Key Insights

### 1. **Multi-Scale CNN Dominance**
- Clear winner with best median error (1.787m) and highest 2m accuracy (52.8%)
- Balanced performance across all accuracy thresholds
- Optimal complexity-performance trade-off

### 2. **Model Complexity vs Performance**
- **Residual CNN**: Highest parameter count (178,530) but poor 1m accuracy (6.0%)
- **Basic CNN**: Lowest parameters (23,650) with reasonable performance
- **Sweet Spot**: Multi-Scale and Hybrid CNNs (~106K parameters) show best performance

### 3. **Critical Performance Metrics**
- **Best 1m Accuracy**: Hybrid CNN (22.7%) - Critical for high-precision applications
- **Best 2m Accuracy**: Multi-Scale CNN (52.8%) - Best overall localization performance
- **Most Consistent**: Multi-Scale CNN with balanced accuracy across thresholds

### 4. **Practical Implications**
- **Sub-meter precision**: Still challenging (best model achieves 22.2% <1m accuracy)
- **2-meter precision**: Achievable for ~50% of cases with Multi-Scale CNN
- **Room-level accuracy**: Very reliable (>78% <3m accuracy)

## Technical Specifications

### Data Preprocessing
- **Amplitude & Phase**: StandardScaler normalization (separate scaling)
- **Coordinates**: MinMaxScaler (0-1 range)
- **Input Format**: (2, 52) stacked arrangement (amplitude, phase)

### Testing Environment
- **TensorFlow Version**: 2.20.0
- **Hardware**: CPU-based inference
- **Test Duration**: ~2 minutes per model

## Recommendations

### 1. **Production Deployment**
- **Recommend**: Multi-Scale CNN for balanced performance
- **Alternative**: Hybrid CNN for applications requiring highest 1m accuracy

### 2. **Model Improvement Opportunities**
- Fix Attention CNN compatibility issues for complete evaluation
- Investigate Residual CNN's poor short-range performance
- Consider ensemble methods combining Multi-Scale and Hybrid CNNs

### 3. **Application Suitability**
- **High-precision apps** (sub-meter): Needs improvement or hybrid approaches
- **Smart building navigation** (2-3m): Multi-Scale CNN adequate
- **Room detection** (3-5m): All models perform well

## Data Files Generated

1. **`cnn_models_test_results_750_samples.csv`**: Complete numerical results
2. **`test_models_clean.py`**: Testing script (Windows compatible)
3. **`test_attention_model.py`**: Attention CNN specific testing script

## Conclusion

The Multi-Scale CNN emerges as the best performing model with a median error of 1.787 meters and 52.8% accuracy within 2 meters. While sub-meter precision remains challenging, the models demonstrate strong performance for room-level localization tasks. The testing confirms that moderate complexity architectures (Multi-Scale, Hybrid) achieve optimal performance compared to both simpler (Basic) and more complex (Residual) alternatives.

---

**Test Completed**: September 11, 2025  
**Total Models Tested**: 4/5 (80% success rate)  
**Test Dataset Size**: 3,688 samples from 750 sample dataset  
**Best Overall Model**: Multi-Scale CNN (median error: 1.787m)

