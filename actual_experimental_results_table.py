#!/usr/bin/env python3
"""
ACTUAL Experimental Results Table - Using ONLY Real Data

This table contains ONLY the actual results from comprehensive_cnn_results.csv
No fabricated or estimated values.
"""

import pandas as pd
import numpy as np

def create_actual_results_table():
    """Create table with ACTUAL experimental results only"""
    
    print(f"📊 ACTUAL EXPERIMENTAL RESULTS - NO FABRICATED DATA")
    print(f"="*70)
    print(f"🔍 Reading from comprehensive_cnn_results.csv (our real experiments)")
    
    # Read the actual results file
    try:
        df = pd.read_csv('comprehensive_cnn_results.csv')
    except FileNotFoundError:
        print("❌ comprehensive_cnn_results.csv not found!")
        return None
    
    # Sort by median error (best first)
    df = df.sort_values('median_error_m').reset_index(drop=True)
    
    # Display the comprehensive table
    print(f"📋 ACTUAL RESULTS TABLE ({len(df)} configurations)")
    print(f"🔍 Ordered by MEDIAN ERROR (most reliable metric)")
    print(f"="*140)
    
    # Create a formatted display
    print(f"{'Rank':<4} {'Experiment':<20} {'Model':<35} {'Samples':<7} {'Mean (m)':<9} {'Median (m)':<10} {'Std (m)':<8} {'<1m %':<7} {'<50cm %':<8} {'Time (s)':<8}")
    print("-" * 140)
    
    for idx, row in df.iterrows():
        rank = idx + 1
        experiment = row['experiment'][:19]
        model = row['model'][:34]
        samples = int(row['sample_size'])
        mean_err = f"{row['mean_error_m']:.3f}"
        median_err = f"{row['median_error_m']:.3f}"
        std_err = f"{row['std_error_m']:.3f}"
        acc_1m = f"{row['accuracy_1m_pct']:.1f}"
        acc_50cm = f"{row['accuracy_50cm_pct']:.1f}"
        time_s = f"{row['training_time_s']:.0f}"
        
        print(f"{rank:<4} {experiment:<20} {model:<35} {samples:<7} {mean_err:<9} {median_err:<10} {std_err:<8} {acc_1m:<7} {acc_50cm:<8} {time_s:<8}")
    
    # Analysis of actual results
    print(f"\n🎯 ACTUAL PERFORMANCE ANALYSIS:")
    print(f"="*60)
    
    best_median = df.iloc[0]
    best_mean = df.loc[df['mean_error_m'].idxmin()]
    best_accuracy = df.loc[df['accuracy_1m_pct'].idxmax()]
    
    print(f"🥇 Best Median Error: {best_median['median_error_m']:.3f}m")
    print(f"   Model: {best_median['model']} ({best_median['sample_size']} samples)")
    print(f"   <1m Accuracy: {best_median['accuracy_1m_pct']:.1f}%")
    
    print(f"\n🥇 Best Mean Error: {best_mean['mean_error_m']:.3f}m")
    print(f"   Model: {best_mean['model']} ({best_mean['sample_size']} samples)")
    print(f"   <1m Accuracy: {best_mean['accuracy_1m_pct']:.1f}%")
    
    print(f"\n🎯 Best <1m Accuracy: {best_accuracy['accuracy_1m_pct']:.1f}%")
    print(f"   Model: {best_accuracy['model']} ({best_accuracy['sample_size']} samples)")
    print(f"   Median Error: {best_accuracy['median_error_m']:.3f}m")
    
    # Check for models with >30% accuracy (realistic threshold)
    high_accuracy_models = df[df['accuracy_1m_pct'] > 30]
    print(f"\n🎯 MODELS WITH >30% <1M ACCURACY:")
    print(f"="*60)
    if len(high_accuracy_models) > 0:
        for _, row in high_accuracy_models.iterrows():
            print(f"   {row['model']} ({row['sample_size']} samples): {row['accuracy_1m_pct']:.1f}% <1m accuracy, {row['median_error_m']:.3f}m median")
    else:
        print(f"   ⚠️  No models achieved >30% <1m accuracy in our experiments")
    
    # Mean vs Median analysis
    print(f"\n📊 MEAN VS MEDIAN ANALYSIS:")
    print(f"="*60)
    df['mean_median_diff'] = abs(df['mean_error_m'] - df['median_error_m'])
    df['mean_median_ratio'] = df['mean_error_m'] / df['median_error_m']
    
    print(f"Average Mean-Median difference: {df['mean_median_diff'].mean():.3f}m")
    print(f"Average Mean/Median ratio: {df['mean_median_ratio'].mean():.3f}")
    print(f"Models with Mean ≈ Median (diff <0.1m): {len(df[df['mean_median_diff'] < 0.1])}")
    print(f"Models with Mean > Median (ratio >1.1): {len(df[df['mean_median_ratio'] > 1.1])}")
    
    return df

def main():
    """Main execution function"""
    
    print(f"📊 ACTUAL EXPERIMENTAL RESULTS TABLE")
    print(f"   🎯 Using ONLY real data from our experiments")
    print(f"   📊 No fabricated or estimated values")
    print(f"   🔍 Based on comprehensive_cnn_results.csv")
    
    # Create actual results table
    results_df = create_actual_results_table()
    
    if results_df is not None:
        print(f"\n✅ ACTUAL RESULTS TABLE GENERATED!")
        print(f"📊 Total configurations: {len(results_df)}")
        print(f"🏆 Best median error: {results_df.iloc[0]['median_error_m']:.3f}m")
        print(f"🎯 Highest <1m accuracy: {results_df['accuracy_1m_pct'].max():.1f}%")
        
        # Save corrected results
        results_df.to_csv('actual_experimental_results_by_median.csv', index=False)
        print(f"📁 Saved: actual_experimental_results_by_median.csv")

if __name__ == "__main__":
    main()



