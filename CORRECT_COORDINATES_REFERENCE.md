# CORRECT COORDINATES REFERENCE

## ✅ **OFFICIAL COORDINATE SETS** ✅

### 🔵 **Training Points (27 points):**
```python
training_points = [
    (0, 0), (0, 1), (0, 2), (0, 4), (0, 5), 
    (1, 0), (1, 1), (1, 4), (1, 5), 
    (2, 0), (2, 2), (2, 3), (2, 4), (2, 5), 
    (3, 0), (3, 1), (3, 2), (3, 4), (3, 5), 
    (4, 0), (4, 1), (4, 4), 
    (5, 0), (5, 2), (5, 3), (5, 4), 
    (6, 3)
]
```

### 🟡 **Validation Points (7 points):**
```python
validation_points = [
    (0, 3), (2, 1), (0, 6), (5, 1), (3, 3), (4, 5), (6, 4)
]
```

### 🎯 **Testing Points (5 points):**
```python
testing_points = [
    (0.5, 0.5), (1.5, 4.5), (2.5, 2.5), (3.5, 1.5), (5.5, 3.5)
]
```

## 📊 **Summary Statistics:**
- **Training Points**: 27 reference locations
- **Validation Points**: 7 reference locations  
- **Testing Points**: 5 intermediate locations
- **Total Points**: 39 locations
- **Environment**: 7×7 meter indoor laboratory
- **Grid Spacing**: 1 meter between reference points
- **Test Offset**: 0.5 meters from grid (interpolation testing)

## 🗺️ **Spatial Layout:**
```
Y-axis (meters)
6  [ ][V][ ][ ][ ][ ][ ]
5  [T][ ][ ][ ][V][ ][ ]  
4  [T][ ][T][T][T][T][ ]
3  [V][ ][T][V][T][T][T]
2  [T][V][T][ ][T][T][ ]
1  [T][T][ ][ ][V][V][ ]
0  [T][T][ ][T][T][T][ ]
   0  1  2  3  4  5  6   X-axis (meters)

Legend:
[T] = Training points (27)
[V] = Validation points (7)  
Test points (5) are at intermediate locations: (0.5,0.5), (1.5,4.5), (2.5,2.5), (3.5,1.5), (5.5,3.5)
```

## 📁 **Data Organization:**
- **Training Data Folder**: `CSI Dataset 750 Samples/`
- **Validation Data**: Same folder as training (split by coordinates)
- **Testing Data Folder**: `Testing Points Dataset 750 Samples/`

## 🎯 **Key Design Features:**
1. **Strategic Distribution**: Points cover entire 7×7m environment
2. **Interpolation Testing**: Test points at 0.5m offsets test sub-meter precision
3. **No Location Leakage**: Clear separation between train/val/test sets
4. **Balanced Coverage**: Good spatial distribution across the environment

## 📈 **Usage in Experiments:**
- **Training**: Learn CSI fingerprints at 27 reference locations
- **Validation**: Hyperparameter tuning and model selection (7 points)
- **Testing**: Final evaluation on interpolation capability (5 intermediate points)

---
**Generated**: Corrected coordinates as specified by user
**Visualization**: `correct_train_val_test_distribution.png`
**Script**: `correct_coordinates_plot.py`

