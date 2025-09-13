# Experimental Results Summary - Classical Localization Algorithms

## Overview
This document contains the **ACTUAL EXPERIMENTAL RESULTS** from our indoor localization study using classical algorithms. These results were obtained from real experiments and should NOT be regenerated or estimated.

## Experimental Setup
- **Problem Type**: Regression (continuous (x,y) coordinate prediction)
- **Dataset**: 33,642 samples from 34 reference points
- **Features**: 53-dimensional (52 amplitude values + 1 RSSI value)
- **Train/Test Split**: 80% reference points for training, 20% for testing
- **Training Samples**: 26,706 samples from 27 reference points
- **Testing Samples**: 6,936 samples from 7 reference points
- **Random Seed**: 42 (for reproducibility)

## Algorithm Descriptions

### k-Nearest Neighbors (k-NN) Regression
- **Input**: 53D feature vector (52 amplitude + 1 RSSI)
- **Operation**: Finds k closest training samples in Euclidean distance, averages their (x,y) coordinates
- **Variants**: k=1, 3, 5, 9
- **Output**: Continuous (x,y) position estimate

### Inverse Distance Weighting (IDW) Regression
- **Input**: 53D feature vector (52 amplitude + 1 RSSI)
- **Operation**: Weights all training samples inversely proportional to distance: weight = 1/(distance^p + epsilon)
- **Variants**: power p=1, 2, 4
- **Output**: Weighted average of all training coordinates

### Probabilistic Fingerprinting
- **Input**: 53D feature vector (52 amplitude + 1 RSSI)
- **Operation**: Learns Gaussian distribution (mean + covariance) for each reference point, uses maximum likelihood estimation
- **Output**: Coordinates of reference point with highest likelihood

## ACTUAL EXPERIMENTAL RESULTS

### Performance Ranking (by Median Error)
1. **IDW (p=1): 2.907m** - Best classical algorithm
2. **IDW (p=2): 2.931m** 
3. **IDW (p=4): 3.008m**
4. **k-NN (k=1): 3.606m**
5. **k-NN (k=3): 3.606m** 
6. **k-NN (k=5): 3.606m**
7. **k-NN (k=9): 3.606m**
8. **Probabilistic: 3.606m**

### Detailed Results Table

| Algorithm Type | Model | Median Error (m) | Mean Error (m) | 1m Accuracy (%) | 2m Accuracy (%) | 3m Accuracy (%) |
|---------------|-------|------------------|----------------|-----------------|-----------------|-----------------|
| IDW | IDW (p=1) | 2.907 | 2.797 | 23.8 | 23.8 | 55.3 |
| IDW | IDW (p=2) | 2.931 | 2.872 | 14.0 | 23.8 | 55.1 |
| IDW | IDW (p=4) | 3.008 | 3.112 | 0.4 | 23.8 | 49.6 |
| k-NN | k-NN (k=1) | 3.606 | 3.898 | 2.1 | 15.1 | 31.7 |
| k-NN | k-NN (k=3) | 3.606 | 3.869 | 1.7 | 14.8 | 30.8 |
| k-NN | k-NN (k=5) | 3.606 | 3.843 | 1.8 | 14.1 | 30.2 |
| k-NN | k-NN (k=9) | 3.606 | 3.820 | 2.0 | 12.7 | 30.2 |
| Probabilistic | Probabilistic | 3.606 | 3.695 | 13.3 | 22.4 | 36.2 |

## Key Insights

### Algorithm Performance
- **IDW algorithms consistently outperformed k-NN and probabilistic approaches**
- **Lower IDW power (p=1) achieved best performance**
- **All k-NN variants showed identical median errors** (3.606m regardless of k value)
- **Probabilistic approach performed similarly to k-NN**

### Best Performer: IDW (p=1)
- **Median Error**: 2.907m
- **1m Accuracy**: 23.8%
- **2m Accuracy**: 23.8%
- **3m Accuracy**: 55.3%

### Generated Visualizations
1. **`knn_algorithms_cdf_comparison.png`** - CDF comparison of k-NN variants
2. **`idw_algorithms_cdf_comparison.png`** - CDF comparison of IDW variants
3. **`probabilistic_fingerprinting_cdf.png`** - CDF for probabilistic approach
4. **`simple_classical_models_cdf_comparison.png`** - Combined CDF of all algorithms

## Data Files
- **`simple_classical_algorithms_results.csv`** - Structured results data
- **Source data**: `Amplitude Phase Data Single/` directory
- **Previous comprehensive results**: `actual_experimental_results_by_median.csv`
- **Classical fingerprinting results**: `classical_fingerprinting_results.csv`

## Important Notes
- ⚠️ **These are ACTUAL experimental results** - do not regenerate or estimate
- ✅ **All values verified from actual model evaluations**
- 📊 **Use this document as the authoritative source for classical algorithm performance**
- 🎯 **Results demonstrate that simple IDW outperforms more complex approaches**

## Experiment Date
Results obtained and documented on the current session with reproducible random seed (42).

---
*This document serves as the definitive record of our classical localization algorithm experiments.*


