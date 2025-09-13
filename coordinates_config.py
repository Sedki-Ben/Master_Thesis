#!/usr/bin/env python3
"""
Official Coordinates Configuration
Use this file to import the correct coordinates in your experiments
"""

# ========================================
# OFFICIAL COORDINATE SETS
# ========================================

# Training Points (27 reference locations)
TRAINING_POINTS = [
    (0, 0), (0, 1), (0, 2), (0, 4), (0, 5), 
    (1, 0), (1, 1), (1, 4), (1, 5), 
    (2, 0), (2, 2), (2, 3), (2, 4), (2, 5), 
    (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), 
    (4, 0), (4, 1), (4, 4), 
    (5, 0), (5, 2), (5, 3), (5, 4), 
    (6, 3)
]

# Validation Points (7 reference locations)
VALIDATION_POINTS = [
    (0, 3), (2, 1), (0, 6), (5, 1), (3, 3), (4, 5), (6, 4)
]

# Testing Points (5 intermediate locations)
TESTING_POINTS = [
    (0.5, 0.5), (1.5, 4.5), (2.5, 2.5), (3.5, 1.5), (5.5, 3.5)
]

# Combined sets
ALL_REFERENCE_POINTS = TRAINING_POINTS + VALIDATION_POINTS
ALL_POINTS = TRAINING_POINTS + VALIDATION_POINTS + TESTING_POINTS

# Summary information
COORDINATE_INFO = {
    'training_count': len(TRAINING_POINTS),
    'validation_count': len(VALIDATION_POINTS),
    'testing_count': len(TESTING_POINTS),
    'total_reference_points': len(ALL_REFERENCE_POINTS),
    'total_points': len(ALL_POINTS),
    'environment_size': '7x7 meters',
    'grid_spacing': '1 meter',
    'test_offset': '0.5 meters'
}

def get_training_points():
    """Return training point coordinates"""
    return TRAINING_POINTS.copy()

def get_validation_points():
    """Return validation point coordinates"""
    return VALIDATION_POINTS.copy()

def get_testing_points():
    """Return testing point coordinates"""
    return TESTING_POINTS.copy()

def get_all_reference_points():
    """Return all reference points (training + validation)"""
    return ALL_REFERENCE_POINTS.copy()

def get_coordinate_info():
    """Return summary information about the coordinate sets"""
    return COORDINATE_INFO.copy()

def print_coordinate_summary():
    """Print a summary of all coordinate sets"""
    print("="*60)
    print("COORDINATE CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Training Points: {len(TRAINING_POINTS)}")
    print(f"Validation Points: {len(VALIDATION_POINTS)}")
    print(f"Testing Points: {len(TESTING_POINTS)}")
    print(f"Total Reference Points: {len(ALL_REFERENCE_POINTS)}")
    print(f"Total Points: {len(ALL_POINTS)}")
    print(f"Environment: {COORDINATE_INFO['environment_size']}")
    print(f"Grid Spacing: {COORDINATE_INFO['grid_spacing']}")
    print(f"Test Offset: {COORDINATE_INFO['test_offset']}")
    print("="*60)

if __name__ == "__main__":
    print_coordinate_summary()
    
    print("\nTraining Points:")
    print(TRAINING_POINTS)
    
    print("\nValidation Points:")
    print(VALIDATION_POINTS)
    
    print("\nTesting Points:")
    print(TESTING_POINTS)

