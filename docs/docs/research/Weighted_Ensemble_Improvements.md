# Balanced Ensemble Improvements

## Overview

The ensemble training method has been updated to use a **balanced ensemble approach** that gives equal weight to all models, ensuring fair contribution from each model regardless of their individual performance scores.

## Key Changes

### 1. **Balanced Weighting**
- **Before**: Dynamic weighting based on cross-validation accuracy scores
- **After**: Equal weights for all models (0.25 each when Prophet is included, 0.333 each for ML models only)
- **Formula**: `equal_weights = 1 / number_of_models`

### 2. **Equal Model Contribution**
- All models contribute equally to the final prediction
- No model is prioritized based on performance
- Ensures diversity in the ensemble

### 3. **Cross-Validation for Evaluation Only**
The new approach:
1. Performs time series cross-validation on each model
2. Calculates accuracy scores (100 - MAPE) for each fold
3. Averages accuracy scores across all folds
4. Uses equal weights for all models regardless of performance
5. Reports CV scores for informational purposes only

### 4. **Balanced Weight Calculation**
```python
# Equal weights for all models
if prophet_predictions is not None:
    # 4 models: Prophet, Random Forest, Ridge, XGBoost
    prophet_weight = 0.25
    rf_weight = 0.25
    ridge_weight = 0.25
    xgb_weight = 0.25
else:
    # 3 ML models only
    rf_weight = 1/3
    ridge_weight = 1/3
    xgb_weight = 1/3
```

## New Functions

### `create_weighted_ensemble_model(train_df, config, cv_splits=None)`
- Creates ensemble model with balanced weights
- Returns models, scaler, features, weights, and CV scores
- Performs cross-validation for evaluation purposes only

### `predict_weighted_ensemble(ensemble_dict, features_df, prophet_predictions=None, prophet_accuracy=None)`
- Makes balanced ensemble predictions using equal weights
- Returns ensemble predictions and individual model predictions
- Prophet accuracy parameter is ignored (kept for compatibility)

## Benefits

### 1. **Fairness**
- All models contribute equally regardless of performance
- No bias towards historically better performing models
- Encourages model diversity

### 2. **Simplicity**
- Simple and transparent weighting scheme
- Easy to understand and explain
- No complex weight calculation logic

### 3. **Robustness**
- Less sensitive to overfitting in cross-validation
- More stable ensemble performance
- Reduces risk of giving too much weight to a single model

### 4. **Transparency**
- Weights are clearly defined and predictable
- CV accuracy scores are still available for analysis
- Individual model contributions are tracked

## Example Output

```
Cross-validation accuracy scores:
Random Forest: 85.23%
Ridge Regression: 82.45%
XGBoost: 87.12%

Using balanced model weights:
Random Forest: 0.250
Ridge Regression: 0.250
XGBoost: 0.250
Note: Prophet weight will be 0.25 when included, ML weights will be adjusted to 0.25 each

Balanced ensemble weights (4 models):
Prophet: 0.250
Random Forest: 0.250
Ridge Regression: 0.250
XGBoost: 0.250
```

## Implementation Details

### Weight Calculation Process
1. **Cross-validation**: Train each model on CV folds
2. **Accuracy calculation**: `accuracy = 100 - MAPE`
3. **Equal weighting**: All models get equal weight regardless of performance
4. **Model count**: 4 models when Prophet included, 3 when ML models only

### Balanced Weighting Mechanism
- Prophet gets equal weight (0.25) when included
- All ML models get equal weight (0.25 each when Prophet included, 0.333 each when Prophet not included)
- Weights always sum to 1.0
- No performance-based adjustments

### Error Handling
- Epsilon added to prevent division by zero
- NaN handling for edge cases
- Graceful handling of missing Prophet predictions

### Performance Considerations
- Cross-validation is performed once during training for evaluation
- Weights are fixed and reused for predictions
- No additional computational overhead during inference

## Usage

The balanced ensemble is automatically used in the main prediction pipeline. The API response now includes:

```json
{
  "ensemble_weights": {
    "prophet_weight": 0.250,
    "rf_weight": 0.250,
    "ridge_weight": 0.250,
    "xgb_weight": 0.250
  },
  "predictions": [
    {
      "model_weights": {
        "prophet_weight": 0.250,
        "rf_weight": 0.250,
        "ridge_weight": 0.250,
        "xgb_weight": 0.250
      }
    }
  ]
}
```

## Testing

The system includes comprehensive testing for balanced weighting:

```python
# Test balanced weighting scenarios
scenarios = [
    ("3 ML Models Only", None),
    ("4 Models with Prophet", prophet_predictions)
]

# All models should get equal weight
assert all(abs(weight - expected_weight) < 1e-6 for weight in weights.values())
```

## Future Enhancements

1. **Configurable Weights**: Allow manual weight specification
2. **Model Selection**: Automatically exclude poorly performing models
3. **Confidence Intervals**: Weight-based uncertainty quantification
4. **Online Learning**: Update weights as new data becomes available
5. **Seasonal Weighting**: Different weights for different seasons/periods 