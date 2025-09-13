#!/usr/bin/env python3
"""
Test script to demonstrate accuracy-based ensemble weighting.
This script shows how the ensemble weights are calculated based on each model's accuracy.
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_training import predict_weighted_ensemble

def test_accuracy_based_weights():
    """Test accuracy-based weighting with different accuracy scenarios"""
    
    print("Accuracy-Based Ensemble Weighting Test")
    print("=" * 60)
    
    # Mock ensemble dictionary with different accuracy scores
    mock_ensemble = {
        'rf_model': None,  # Not used in this test
        'ridge_model': None,  # Not used in this test
        'xgb_model': None,  # Not used in this test
        'scaler': None,  # Not used in this test
        'features': ['feature1', 'feature2'],  # Not used in this test
        'cv_scores': {
            'rf_accuracy': 85.0,      # Random Forest accuracy
            'ridge_accuracy': 82.0,   # Ridge Regression accuracy
            'xgb_accuracy': 88.0      # XGBoost accuracy
        }
    }
    
    # Mock predictions (single value for simplicity)
    mock_features = None  # Not used in this test
    prophet_predictions = np.array([100.0])
    prophet_accuracy = 87.0
    
    print(f"Model Accuracies:")
    print(f"Prophet: {prophet_accuracy:.1f}%")
    print(f"Random Forest: {mock_ensemble['cv_scores']['rf_accuracy']:.1f}%")
    print(f"Ridge Regression: {mock_ensemble['cv_scores']['ridge_accuracy']:.1f}%")
    print(f"XGBoost: {mock_ensemble['cv_scores']['xgb_accuracy']:.1f}%")
    
    print(f"\nCalculating accuracy-based weights...")
    
    # Calculate weights manually to show the process
    accuracies = [
        prophet_accuracy,
        mock_ensemble['cv_scores']['rf_accuracy'],
        mock_ensemble['cv_scores']['ridge_accuracy'],
        mock_ensemble['cv_scores']['xgb_accuracy']
    ]
    
    total_accuracy = sum(accuracies)
    
    prophet_weight = prophet_accuracy / total_accuracy
    rf_weight = mock_ensemble['cv_scores']['rf_accuracy'] / total_accuracy
    ridge_weight = mock_ensemble['cv_scores']['ridge_accuracy'] / total_accuracy
    xgb_weight = mock_ensemble['cv_scores']['xgb_accuracy'] / total_accuracy
    
    print(f"\nManual Weight Calculation:")
    print(f"Total Accuracy: {total_accuracy:.1f}%")
    print(f"Prophet Weight: {prophet_accuracy:.1f} / {total_accuracy:.1f} = {prophet_weight:.3f}")
    print(f"RF Weight: {mock_ensemble['cv_scores']['rf_accuracy']:.1f} / {total_accuracy:.1f} = {rf_weight:.3f}")
    print(f"Ridge Weight: {mock_ensemble['cv_scores']['ridge_accuracy']:.1f} / {total_accuracy:.1f} = {ridge_weight:.3f}")
    print(f"XGB Weight: {mock_ensemble['cv_scores']['xgb_accuracy']:.1f} / {total_accuracy:.1f} = {xgb_weight:.3f}")
    
    print(f"\nFinal Weights:")
    print(f"Prophet: {prophet_weight:.3f} ({prophet_weight*100:.1f}%)")
    print(f"Random Forest: {rf_weight:.3f} ({rf_weight*100:.1f}%)")
    print(f"Ridge Regression: {ridge_weight:.3f} ({ridge_weight*100:.1f}%)")
    print(f"XGBoost: {xgb_weight:.3f} ({xgb_weight*100:.1f}%)")
    print(f"Sum: {prophet_weight + rf_weight + ridge_weight + xgb_weight:.3f}")
    
    # Test different accuracy scenarios
    print(f"\n" + "="*60)
    print("Different Accuracy Scenarios")
    print("="*60)
    
    scenarios = [
        {
            "name": "Balanced Models",
            "accuracies": [85.0, 84.0, 83.0, 86.0]
        },
        {
            "name": "Prophet Dominant",
            "accuracies": [95.0, 75.0, 70.0, 80.0]
        },
        {
            "name": "XGBoost Dominant", 
            "accuracies": [80.0, 75.0, 70.0, 95.0]
        },
        {
            "name": "Close Competition",
            "accuracies": [87.0, 86.0, 85.0, 88.0]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        accuracies = scenario['accuracies']
        total = sum(accuracies)
        
        weights = [acc / total for acc in accuracies]
        
        print(f"  Prophet: {weights[0]:.3f} ({accuracies[0]:.1f}%)")
        print(f"  RF: {weights[1]:.3f} ({accuracies[1]:.1f}%)")
        print(f"  Ridge: {weights[2]:.3f} ({accuracies[2]:.1f}%)")
        print(f"  XGB: {weights[3]:.3f} ({accuracies[3]:.1f}%)")
        print(f"  Sum: {sum(weights):.3f}")

def demonstrate_weight_impact():
    """Demonstrate how weights impact ensemble predictions"""
    
    print(f"\n" + "="*60)
    print("Weight Impact on Ensemble Predictions")
    print("="*60)
    
    # Mock individual model predictions
    prophet_pred = 100.0
    rf_pred = 95.0
    ridge_pred = 105.0
    xgb_pred = 98.0
    
    print(f"Individual Model Predictions:")
    print(f"Prophet: {prophet_pred:.1f}")
    print(f"Random Forest: {rf_pred:.1f}")
    print(f"Ridge Regression: {ridge_pred:.1f}")
    print(f"XGBoost: {xgb_pred:.1f}")
    
    # Test with different weighting schemes
    weighting_schemes = [
        {
            "name": "Equal Weights (Old)",
            "weights": [0.25, 0.25, 0.25, 0.25]
        },
        {
            "name": "Accuracy-Based (New)",
            "weights": [0.254, 0.248, 0.240, 0.258]  # Based on 87, 85, 82, 88 accuracy
        },
        {
            "name": "Prophet Dominant",
            "weights": [0.40, 0.20, 0.20, 0.20]
        },
        {
            "name": "XGBoost Dominant",
            "weights": [0.20, 0.20, 0.20, 0.40]
        }
    ]
    
    for scheme in weighting_schemes:
        ensemble_pred = (
            prophet_pred * scheme['weights'][0] +
            rf_pred * scheme['weights'][1] +
            ridge_pred * scheme['weights'][2] +
            xgb_pred * scheme['weights'][3]
        )
        
        print(f"\n{scheme['name']}:")
        print(f"  Weights: Prophet={scheme['weights'][0]:.3f}, RF={scheme['weights'][1]:.3f}, "
              f"Ridge={scheme['weights'][2]:.3f}, XGB={scheme['weights'][3]:.3f}")
        print(f"  Ensemble Prediction: {ensemble_pred:.2f}")

if __name__ == "__main__":
    test_accuracy_based_weights()
    demonstrate_weight_impact()
    
    print(f"\n" + "="*60)
    print("Summary")
    print("="*60)
    print("The accuracy-based weighting system:")
    print("1. Uses cross-validation accuracy scores for each model")
    print("2. Normalizes weights so they sum to 1.0")
    print("3. Gives higher weight to more accurate models")
    print("4. Automatically adapts when Prophet is included/excluded")
    print("5. Provides more balanced and reliable ensemble predictions") 