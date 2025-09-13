#!/usr/bin/env python3
"""
Test script for balanced ensemble functionality
"""

import pandas as pd
import numpy as np
from model_training import create_weighted_ensemble_model, predict_weighted_ensemble
from data_loader import load_and_prepare_data
from feature_engineering import create_advanced_features
from models import FeatureConfig

def test_balanced_ensemble():
    """Test the balanced ensemble functionality"""
    print("Testing balanced ensemble functionality...")
    
    # Load data
    print("Loading data...")
    energy_df, weather_future_df, official_df = load_and_prepare_data()
    
    # Create features
    print("Creating features...")
    energy_df = create_advanced_features(energy_df)
    
    # Create config
    config = FeatureConfig()
    
    # Test balanced ensemble creation
    print("Creating balanced ensemble model...")
    balanced_ensemble_dict = create_weighted_ensemble_model(energy_df, config)
    
    # Check that weights are properly calculated
    weights = balanced_ensemble_dict['weights']
    cv_scores = balanced_ensemble_dict['cv_scores']
    
    print(f"\nModel weights: {weights}")
    print(f"CV scores: {cv_scores}")
    
    # Verify weights are balanced (0.25 each for ML models)
    expected_weight = 0.25
    for model, weight in weights.items():
        assert abs(weight - expected_weight) < 1e-6, f"Weight for {model} should be {expected_weight}, got {weight}"
    
    # Verify all weights are positive
    for model, weight in weights.items():
        assert weight > 0, f"Weight for {model} should be positive, got {weight}"
    
    # Test prediction functionality
    print("\nTesting prediction functionality...")
    
    # Use a small subset for testing
    test_data = energy_df.head(10)
    
    # Test without Prophet predictions (3 ML models, 0.333 each)
    ensemble_pred, individual_preds = predict_weighted_ensemble(
        balanced_ensemble_dict, 
        test_data
    )
    
    print(f"Ensemble prediction shape: {ensemble_pred.shape}")
    print(f"Individual predictions: {list(individual_preds.keys())}")
    
    # Check weights for 3-model ensemble
    weights_3_models = individual_preds['weights']
    expected_weight_3 = 1/3
    for model, weight in weights_3_models.items():
        assert abs(weight - expected_weight_3) < 1e-6, f"Weight for {model} should be {expected_weight_3}, got {weight}"
    
    # Test with Prophet predictions (4 models, 0.25 each)
    prophet_pred = np.random.random(len(test_data)) * 10  # Mock Prophet predictions
    
    print(f"\nTesting balanced weighting with Prophet (4 models)")
    ensemble_pred_with_prophet, individual_preds_with_prophet = predict_weighted_ensemble(
        balanced_ensemble_dict, 
        test_data,
        prophet_pred
    )
    
    print(f"Ensemble prediction with Prophet shape: {ensemble_pred_with_prophet.shape}")
    
    # Check that all models get equal weight (0.25 each)
    balanced_weights = individual_preds_with_prophet['weights']
    expected_weight_4 = 0.25
    
    print(f"\nBalanced weights (4 models): {balanced_weights}")
    
    for model, weight in balanced_weights.items():
        assert abs(weight - expected_weight_4) < 1e-6, f"Weight for {model} should be {expected_weight_4}, got {weight}"
    
    # Verify weights sum to 1
    total_weight = sum(balanced_weights.values())
    print(f"Total weight: {total_weight:.6f}")
    assert abs(total_weight - 1.0) < 1e-6, f"Weights should sum to 1, got {total_weight}"
    
    # Verify predictions are reasonable
    assert len(ensemble_pred) == len(test_data), "Prediction length should match data length"
    assert len(ensemble_pred_with_prophet) == len(test_data), "Prediction length should match data length"
    
    print("\n✅ All tests passed!")
    
    return balanced_ensemble_dict

def test_balanced_weighting_scenarios():
    """Test different scenarios to ensure balanced weights are maintained"""
    print("\n" + "="*50)
    print("TESTING BALANCED WEIGHTING SCENARIOS")
    print("="*50)
    
    # Load data
    energy_df, weather_future_df, official_df = load_and_prepare_data()
    energy_df = create_advanced_features(energy_df)
    config = FeatureConfig()
    
    # Create ensemble model
    balanced_ensemble_dict = create_weighted_ensemble_model(energy_df, config)
    test_data = energy_df.head(5)
    
    # Test different scenarios
    scenarios = [
        ("3 ML Models Only", None),
        ("4 Models with Prophet", np.random.random(len(test_data)) * 10)
    ]
    
    for scenario_name, prophet_pred in scenarios:
        print(f"\n{scenario_name}:")
        
        if prophet_pred is not None:
            ensemble_pred, individual_preds = predict_weighted_ensemble(
                balanced_ensemble_dict,
                test_data,
                prophet_pred
            )
            expected_weight = 0.25
            expected_count = 4
        else:
            ensemble_pred, individual_preds = predict_weighted_ensemble(
                balanced_ensemble_dict,
                test_data
            )
            expected_weight = 1/3
            expected_count = 3
        
        weights = individual_preds['weights']
        
        print(f"  Number of models: {len(weights)}")
        print(f"  Expected weight per model: {expected_weight:.3f}")
        print(f"  Actual weights: {[f'{w:.3f}' for w in weights.values()]}")
        
        # Verify all weights are equal
        for model, weight in weights.items():
            assert abs(weight - expected_weight) < 1e-6, f"Weight for {model} should be {expected_weight} in {scenario_name}"
        
        # Verify correct number of models
        assert len(weights) == expected_count, f"Should have {expected_count} models in {scenario_name}"
        
        print(f"  ✅ All weights balanced correctly")

if __name__ == "__main__":
    # Run tests
    balanced_ensemble_dict = test_balanced_ensemble()
    test_balanced_weighting_scenarios()
    
    print("\n" + "="*50)
    print("ALL BALANCED ENSEMBLE TESTS PASSED!")
    print("="*50) 