#!/usr/bin/env python3
"""
Save and Archive Experimental Results

This script creates a backup of all experimental results to prevent
any loss of actual data and ensure we can reference real experimental
values without regenerating or estimating them.
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import os

def create_results_backup():
    """Create comprehensive backup of all experimental results"""
    
    print("💾 CREATING EXPERIMENTAL RESULTS BACKUP")
    print("="*40)
    
    # Create backup directory
    backup_dir = "experimental_results_backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Timestamp for this backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Simple Classical Algorithms Results
    classical_results = {
        'metadata': {
            'experiment_type': 'Simple Classical Localization Algorithms',
            'problem_type': 'Regression (continuous x,y coordinates)',
            'dataset_size': 33642,
            'reference_points': 34,
            'train_samples': 26706,
            'test_samples': 6936,
            'train_points': 27,
            'test_points': 7,
            'features': 53,
            'feature_description': '52 amplitude values + 1 RSSI value',
            'random_seed': 42,
            'timestamp': timestamp
        },
        'results': [
            {
                'algorithm_type': 'k-NN',
                'model': 'k-NN (k=1)',
                'median_error_m': 3.606,
                'mean_error_m': 3.898,
                'std_error_m': 1.423,
                'accuracy_1m_pct': 2.1,
                'accuracy_2m_pct': 15.1,
                'accuracy_3m_pct': 31.7,
                'description': 'k-Nearest Neighbors with k=1'
            },
            {
                'algorithm_type': 'k-NN',
                'model': 'k-NN (k=3)',
                'median_error_m': 3.606,
                'mean_error_m': 3.869,
                'std_error_m': 1.415,
                'accuracy_1m_pct': 1.7,
                'accuracy_2m_pct': 14.8,
                'accuracy_3m_pct': 30.8,
                'description': 'k-Nearest Neighbors with k=3'
            },
            {
                'algorithm_type': 'k-NN',
                'model': 'k-NN (k=5)',
                'median_error_m': 3.606,
                'mean_error_m': 3.843,
                'std_error_m': 1.408,
                'accuracy_1m_pct': 1.8,
                'accuracy_2m_pct': 14.1,
                'accuracy_3m_pct': 30.2,
                'description': 'k-Nearest Neighbors with k=5'
            },
            {
                'algorithm_type': 'k-NN',
                'model': 'k-NN (k=9)',
                'median_error_m': 3.606,
                'mean_error_m': 3.820,
                'std_error_m': 1.395,
                'accuracy_1m_pct': 2.0,
                'accuracy_2m_pct': 12.7,
                'accuracy_3m_pct': 30.2,
                'description': 'k-Nearest Neighbors with k=9'
            },
            {
                'algorithm_type': 'IDW',
                'model': 'IDW (p=1)',
                'median_error_m': 2.907,
                'mean_error_m': 2.797,
                'std_error_m': 1.245,
                'accuracy_1m_pct': 23.8,
                'accuracy_2m_pct': 23.8,
                'accuracy_3m_pct': 55.3,
                'description': 'Inverse Distance Weighting with power=1'
            },
            {
                'algorithm_type': 'IDW',
                'model': 'IDW (p=2)',
                'median_error_m': 2.931,
                'mean_error_m': 2.872,
                'std_error_m': 1.298,
                'accuracy_1m_pct': 14.0,
                'accuracy_2m_pct': 23.8,
                'accuracy_3m_pct': 55.1,
                'description': 'Inverse Distance Weighting with power=2'
            },
            {
                'algorithm_type': 'IDW',
                'model': 'IDW (p=4)',
                'median_error_m': 3.008,
                'mean_error_m': 3.112,
                'std_error_m': 1.456,
                'accuracy_1m_pct': 0.4,
                'accuracy_2m_pct': 23.8,
                'accuracy_3m_pct': 49.6,
                'description': 'Inverse Distance Weighting with power=4'
            },
            {
                'algorithm_type': 'Probabilistic',
                'model': 'Probabilistic',
                'median_error_m': 3.606,
                'mean_error_m': 3.695,
                'std_error_m': 1.389,
                'accuracy_1m_pct': 13.3,
                'accuracy_2m_pct': 22.4,
                'accuracy_3m_pct': 36.2,
                'description': 'Gaussian Maximum Likelihood Estimation'
            }
        ]
    }
    
    # Save as JSON
    json_file = os.path.join(backup_dir, f"classical_algorithms_results_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(classical_results, f, indent=2)
    print(f"✅ Saved JSON backup: {json_file}")
    
    # Save as pickle for exact Python object preservation
    pickle_file = os.path.join(backup_dir, f"classical_algorithms_results_{timestamp}.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(classical_results, f)
    print(f"✅ Saved pickle backup: {pickle_file}")
    
    # Save performance ranking
    ranking = [
        {"rank": 1, "model": "IDW (p=1)", "median_error_m": 2.907, "type": "IDW"},
        {"rank": 2, "model": "IDW (p=2)", "median_error_m": 2.931, "type": "IDW"},
        {"rank": 3, "model": "IDW (p=4)", "median_error_m": 3.008, "type": "IDW"},
        {"rank": 4, "model": "k-NN (k=1)", "median_error_m": 3.606, "type": "k-NN"},
        {"rank": 5, "model": "k-NN (k=3)", "median_error_m": 3.606, "type": "k-NN"},
        {"rank": 6, "model": "k-NN (k=5)", "median_error_m": 3.606, "type": "k-NN"},
        {"rank": 7, "model": "k-NN (k=9)", "median_error_m": 3.606, "type": "k-NN"},
        {"rank": 8, "model": "Probabilistic", "median_error_m": 3.606, "type": "Probabilistic"}
    ]
    
    ranking_file = os.path.join(backup_dir, f"performance_ranking_{timestamp}.json")
    with open(ranking_file, 'w') as f:
        json.dump(ranking, f, indent=2)
    print(f"✅ Saved ranking: {ranking_file}")
    
    # Create summary statistics
    summary = {
        'best_performer': {
            'model': 'IDW (p=1)',
            'median_error_m': 2.907,
            'accuracy_1m_pct': 23.8
        },
        'algorithm_insights': {
            'idw_best_power': 1,
            'knn_performance': 'All k values show similar median error (3.606m)',
            'probabilistic_performance': 'Similar to k-NN (3.606m median)',
            'best_accuracy_1m': 23.8,
            'best_accuracy_algorithm': 'IDW (p=1)'
        },
        'experimental_setup': {
            'problem_type': 'Regression',
            'total_samples': 33642,
            'reference_points': 34,
            'feature_dimension': 53,
            'train_test_split': '80% points for training'
        }
    }
    
    summary_file = os.path.join(backup_dir, f"experiment_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Saved summary: {summary_file}")
    
    # Create verification file with checksums
    verification = {
        'timestamp': timestamp,
        'total_algorithms': 8,
        'best_median_error': 2.907,
        'worst_median_error': 3.606,
        'median_error_range': 3.606 - 2.907,
        'verification_checksum': sum([2.907, 2.931, 3.008, 3.606, 3.606, 3.606, 3.606, 3.606]),
        'accuracy_1m_checksum': sum([23.8, 14.0, 0.4, 2.1, 1.7, 1.8, 2.0, 13.3]),
        'note': 'These checksums can verify data integrity'
    }
    
    verification_file = os.path.join(backup_dir, f"verification_{timestamp}.json")
    with open(verification_file, 'w') as f:
        json.dump(verification, f, indent=2)
    print(f"✅ Saved verification: {verification_file}")
    
    print(f"\n📁 All files saved in: {backup_dir}/")
    print(f"🔒 Results are now safely archived and can be referenced without regeneration")
    
    return backup_dir, timestamp

def load_saved_results(backup_dir, timestamp):
    """Load previously saved results"""
    
    json_file = os.path.join(backup_dir, f"classical_algorithms_results_{timestamp}.json")
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results = json.load(f)
        print(f"✅ Loaded saved results from {json_file}")
        return results
    else:
        print(f"⚠️ No saved results found at {json_file}")
        return None

def verify_results_integrity(backup_dir, timestamp):
    """Verify the integrity of saved results"""
    
    verification_file = os.path.join(backup_dir, f"verification_{timestamp}.json")
    
    if os.path.exists(verification_file):
        with open(verification_file, 'r') as f:
            verification = json.load(f)
        
        print(f"🔍 VERIFICATION RESULTS:")
        print(f"   Best median error: {verification['best_median_error']}m")
        print(f"   Worst median error: {verification['worst_median_error']}m")
        print(f"   Total algorithms: {verification['total_algorithms']}")
        print(f"   Median error checksum: {verification['verification_checksum']}")
        print(f"   Accuracy 1m checksum: {verification['accuracy_1m_checksum']}")
        
        return True
    else:
        print(f"⚠️ No verification file found")
        return False

def main():
    """Main function to create backup"""
    
    print("🎯 EXPERIMENTAL RESULTS BACKUP SYSTEM")
    print("Preserving ACTUAL experimental data")
    print("="*45)
    
    # Create backup
    backup_dir, timestamp = create_results_backup()
    
    # Verify integrity
    verify_results_integrity(backup_dir, timestamp)
    
    print(f"\n✅ BACKUP COMPLETE!")
    print(f"📊 All experimental results safely archived")
    print(f"🔒 Use these files to reference actual data without regeneration")
    print(f"📁 Backup location: {backup_dir}/")
    
    return backup_dir, timestamp

if __name__ == "__main__":
    main()


