#!/usr/bin/env python3
"""
Generalization Issues Diagnosis
Analyzes the training curves and performance metrics to identify why models aren't generalizing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_results():
    """Analyze the comprehensive results to identify generalization issues"""
    
    print("🔍 GENERALIZATION ISSUES DIAGNOSIS")
    print("=" * 60)
    
    # Load results
    results_path = Path("the last samurai") / "comprehensive_results.csv"
    df = pd.read_csv(results_path)
    
    print("📊 PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    # Group by model and analyze performance across dataset sizes
    models = df['Model'].unique()
    dataset_sizes = df['Dataset_Size'].unique()
    
    print(f"Models: {list(models)}")
    print(f"Dataset sizes: {list(dataset_sizes)}")
    print()
    
    # Analyze each model's performance across dataset sizes
    for model in models:
        model_data = df[df['Model'] == model].sort_values('Dataset_Size')
        
        print(f"\n🤖 {model} Performance Analysis:")
        print(f"{'Size':<6} {'Median Error':<12} {'Accuracy <1m':<12} {'Accuracy <2m':<12}")
        print("-" * 50)
        
        for _, row in model_data.iterrows():
            print(f"{row['Dataset_Size']:<6} {row['Median_Error_m']:<12.3f} "
                  f"{row['Accuracy_1m']:<12.1f}% {row['Accuracy_2m']:<12.1f}%")
        
        # Check if performance improves with more data
        median_errors = model_data['Median_Error_m'].values
        acc_1m = model_data['Accuracy_1m'].values
        
        if len(median_errors) >= 2:
            error_trend = "IMPROVING" if median_errors[-1] < median_errors[0] else "WORSENING"
            acc_trend = "IMPROVING" if acc_1m[-1] > acc_1m[0] else "WORSENING"
            print(f"   📈 Error trend (250→750): {error_trend}")
            print(f"   📈 <1m Accuracy trend (250→750): {acc_trend}")

def identify_key_issues():
    """Identify the main generalization issues"""
    
    print("\n\n🚨 KEY GENERALIZATION ISSUES IDENTIFIED")
    print("=" * 60)
    
    # Load results
    results_path = Path("the last samurai") / "comprehensive_results.csv"
    df = pd.read_csv(results_path)
    
    issues_found = []
    
    # Issue 1: Poor absolute performance
    median_errors = df['Median_Error_m']
    if median_errors.mean() > 2.0:
        issues_found.append("HIGH_ERROR_RATES")
        print("❌ ISSUE 1: HIGH ERROR RATES")
        print(f"   Average median error: {median_errors.mean():.2f}m")
        print(f"   Expected for good localization: <1.0m")
        print(f"   Current best: {median_errors.min():.2f}m")
    
    # Issue 2: Low accuracy rates
    acc_1m = df['Accuracy_1m']
    if acc_1m.mean() < 50:
        issues_found.append("LOW_ACCURACY")
        print("\n❌ ISSUE 2: LOW ACCURACY RATES")
        print(f"   Average <1m accuracy: {acc_1m.mean():.1f}%")
        print(f"   Expected for good localization: >80%")
        print(f"   Current best: {acc_1m.max():.1f}%")
    
    # Issue 3: No improvement with more data
    models = df['Model'].unique()
    no_improvement_count = 0
    
    for model in models:
        model_data = df[df['Model'] == model].sort_values('Dataset_Size')
        if len(model_data) >= 2:
            error_250 = model_data.iloc[0]['Median_Error_m']
            error_750 = model_data.iloc[-1]['Median_Error_m']
            if error_750 >= error_250:  # No improvement or worse
                no_improvement_count += 1
    
    if no_improvement_count >= len(models) * 0.6:  # Most models don't improve
        issues_found.append("NO_SCALING_BENEFIT")
        print(f"\n❌ ISSUE 3: NO BENEFIT FROM MORE DATA")
        print(f"   {no_improvement_count}/{len(models)} models don't improve with more data")
        print(f"   This suggests fundamental training issues")
    
    # Issue 4: High variance in performance
    for model in models:
        model_data = df[df['Model'] == model]
        error_std = model_data['Median_Error_m'].std()
        if error_std > 0.5:
            issues_found.append("HIGH_VARIANCE")
            print(f"\n❌ ISSUE 4: HIGH PERFORMANCE VARIANCE")
            print(f"   {model} error std: {error_std:.3f}m")
            print(f"   Inconsistent performance across dataset sizes")
            break
    
    return issues_found

def analyze_training_configuration():
    """Analyze the training configuration for potential issues"""
    
    print("\n\n🔧 TRAINING CONFIGURATION ANALYSIS")
    print("=" * 60)
    
    config_issues = []
    
    print("📋 Current Training Configuration:")
    print("   • Learning Rate: 0.001 (Adam)")
    print("   • Batch Size: 32")
    print("   • Max Epochs: 100")
    print("   • Early Stopping: patience=20")
    print("   • Learning Rate Reduction: factor=0.5, patience=10")
    print("   • Data Augmentation: Gaussian noise (0.05) + synthetic interpolation")
    print("   • Loss Function: Euclidean distance")
    print("")
    
    print("🚨 POTENTIAL CONFIGURATION ISSUES:")
    
    # Issue 1: Learning rate too high
    print("❌ ISSUE 1: LEARNING RATE TOO HIGH")
    print("   Current: 0.001")
    print("   For regression tasks with small datasets: 0.0001-0.0005 is often better")
    print("   High LR can cause instability and poor convergence")
    config_issues.append("HIGH_LEARNING_RATE")
    
    # Issue 2: Insufficient regularization
    print("\n❌ ISSUE 2: INSUFFICIENT REGULARIZATION")
    print("   Current dropout rates: 0.2-0.3")
    print("   No L1/L2 regularization")
    print("   No weight decay")
    print("   Small dataset + complex models = overfitting risk")
    config_issues.append("INSUFFICIENT_REGULARIZATION")
    
    # Issue 3: Data leakage in preprocessing
    print("\n❌ ISSUE 3: POTENTIAL DATA LEAKAGE")
    print("   Scaling fitted on ALL data (train+val+test)")
    print("   Should only fit on training data")
    print("   This can lead to overly optimistic validation performance")
    config_issues.append("DATA_LEAKAGE")
    
    # Issue 4: Loss function mismatch
    print("\n❌ ISSUE 4: LOSS FUNCTION ISSUES")
    print("   Using custom Euclidean distance loss")
    print("   Standard MSE might be more stable")
    print("   No coordinate-wise weighting considered")
    config_issues.append("LOSS_FUNCTION")
    
    # Issue 5: Validation set issues
    print("\n❌ ISSUE 5: VALIDATION SET PROBLEMS")
    print("   Only 7 validation points vs 27 training points")
    print("   Small validation set = unreliable validation metrics")
    print("   May not represent true generalization performance")
    config_issues.append("SMALL_VALIDATION_SET")
    
    # Issue 6: Data augmentation issues  
    print("\n❌ ISSUE 6: DATA AUGMENTATION CONCERNS")
    print("   Linear interpolation between points")
    print("   May not reflect real CSI physics")
    print("   Could introduce unrealistic patterns")
    config_issues.append("UNREALISTIC_AUGMENTATION")
    
    return config_issues

def suggest_improvements():
    """Suggest specific improvements based on identified issues"""
    
    print("\n\n💡 SUGGESTED IMPROVEMENTS")
    print("=" * 60)
    
    print("🎯 IMMEDIATE FIXES:")
    print("1. REDUCE LEARNING RATE")
    print("   • Change from 0.001 to 0.0002")
    print("   • Add warmup schedule: start at 0.00005, increase to 0.0002")
    print("   • Use cosine annealing for better convergence")
    print("")
    
    print("2. FIX DATA PREPROCESSING")
    print("   • Fit scalers ONLY on training data")
    print("   • Transform val/test using training scalers")
    print("   • This eliminates data leakage")
    print("")
    
    print("3. INCREASE REGULARIZATION")
    print("   • Increase dropout: 0.4-0.5 for dense layers")
    print("   • Add L2 regularization: 1e-4 to 1e-3")
    print("   • Add BatchNormalization to all conv layers")
    print("   • Use SpatialDropout1D for conv layers")
    print("")
    
    print("4. IMPROVE LOSS FUNCTION")
    print("   • Try standard MSE loss")
    print("   • Add coordinate-wise weighting if needed")
    print("   • Consider Huber loss for robustness")
    print("")
    
    print("5. ENHANCE DATA AUGMENTATION")
    print("   • Reduce noise factor: 0.05 → 0.02")
    print("   • Add frequency domain augmentation")
    print("   • Use mixup between samples from same location")
    print("   • Remove unrealistic interpolation")
    print("")
    
    print("6. IMPROVE VALIDATION STRATEGY")
    print("   • Use k-fold cross-validation")
    print("   • Ensure validation points are representative")
    print("   • Monitor training vs validation more carefully")
    print("")
    
    print("🔬 ADVANCED IMPROVEMENTS:")
    print("7. ARCHITECTURE CHANGES")
    print("   • Reduce model complexity (fewer parameters)")
    print("   • Add residual connections")
    print("   • Use depthwise separable convolutions")
    print("   • Implement progressive resizing")
    print("")
    
    print("8. TRAINING STRATEGY")
    print("   • Use curriculum learning (easy→hard samples)")
    print("   • Implement model ensembling")
    print("   • Add pseudo-labeling for unlabeled data")
    print("   • Use transfer learning if possible")
    print("")
    
    print("9. DATA QUALITY")
    print("   • Analyze CSI data quality and outliers")
    print("   • Remove noisy samples")
    print("   • Balance dataset across locations")
    print("   • Collect more validation data")

def create_improved_training_config():
    """Create an improved training configuration"""
    
    print("\n\n📝 IMPROVED TRAINING CONFIGURATION")
    print("=" * 60)
    
    config = """
    # IMPROVED TRAINING HYPERPARAMETERS
    
    OPTIMIZER:
        - Type: Adam
        - Learning Rate: 0.0002 (reduced from 0.001)
        - Beta1: 0.9, Beta2: 0.999
        - Weight Decay: 1e-4 (L2 regularization)
    
    LEARNING RATE SCHEDULE:
        - Warmup: 5 epochs, 0.00005 → 0.0002  
        - Main: Cosine annealing with restarts
        - Min LR: 1e-6
    
    REGULARIZATION:
        - Dropout: 0.5 (dense layers), 0.3 (conv layers)
        - SpatialDropout1D: 0.2 (after conv layers)
        - BatchNormalization: All conv and dense layers
        - L2 Weight Decay: 1e-4
    
    DATA PREPROCESSING:
        - Fit scalers ONLY on training data
        - No data leakage
        - Robust scaling (less sensitive to outliers)
    
    DATA AUGMENTATION:
        - Gaussian noise: 0.02 (reduced from 0.05)
        - RSSI noise: 0.05 (reduced from 0.1)
        - Mixup within same location
        - Remove synthetic interpolation
    
    LOSS FUNCTION:
        - Primary: MSE (more stable than Euclidean)
        - Alternative: Huber loss (robust to outliers)
    
    TRAINING PROCEDURE:
        - Batch Size: 16 (smaller for better gradients)
        - Max Epochs: 150
        - Early Stopping: patience=30, monitor='val_loss'
        - Model Checkpoint: save best validation
        - Validation: Use stratified split or k-fold
    
    MONITORING:
        - Track train/val loss gap
        - Monitor gradient norms
        - Learning rate progression
        - Model complexity metrics
    """
    
    print(config)

def main():
    """Main diagnosis function"""
    
    # Run all analyses
    analyze_results()
    issues = identify_key_issues()
    config_issues = analyze_training_configuration()
    suggest_improvements()
    create_improved_training_config()
    
    # Summary
    print("\n\n📋 DIAGNOSIS SUMMARY")
    print("=" * 60)
    print(f"Performance Issues Found: {len(issues)}")
    print(f"Configuration Issues Found: {len(config_issues)}")
    print("\nKey Problems:")
    for i, issue in enumerate(issues + config_issues, 1):
        print(f"  {i}. {issue}")
    
    print(f"\n🎯 MAIN CONCLUSION:")
    print(f"The models show poor generalization due to:")
    print(f"  • Too high learning rate causing unstable training")
    print(f"  • Data leakage in preprocessing inflating validation performance")
    print(f"  • Insufficient regularization leading to overfitting")
    print(f"  • Small validation set providing unreliable feedback")
    print(f"\n💡 RECOMMENDED ACTION:")
    print(f"Implement the improved training configuration above")
    print(f"and retrain with proper data splitting and regularization.")

if __name__ == "__main__":
    main()

