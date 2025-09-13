#!/usr/bin/env python3
"""
Script to predict next 90 days of daily energy totals for all facilities.
Looks for weather.csv in each facility folder and saves predictions to predictions-90.csv
"""

import os
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Add the prediction-api directory to the path
sys.path.append('./prediction-api')

from data_loader import load_energy_data, load_weather_data, load_future_weather_data, prepare_prediction_data
from model_training import create_optimized_prophet_model, create_weighted_ensemble_model, predict_weighted_ensemble
from feature_engineering import create_advanced_features, get_prophet_regressors
from models import FeatureConfig

def load_facility_metadata() -> Dict:
    """Load facility metadata from JSON file"""
    try:
        metadata_path = '../data/facility-metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return {facility["name"]: facility for facility in metadata}
    except Exception as e:
        print(f"Error loading facility metadata: {e}")
        return {}

def get_next_90_days_weather(facility_name: str) -> Optional[pd.DataFrame]:
    """Load weather data for the next 90 days from facility folder"""
    try:
        # Check if weather.csv exists in facility folder
        weather_path = f'../data/{facility_name}/weather.csv'
        if not os.path.exists(weather_path):
            print(f"Weather file not found: {weather_path}")
            return None
        
        # Load weather data
        weather_df = pd.read_csv(weather_path)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Get today's date
        today = datetime.now().date()
        
        # Filter for next 90 days
        end_date = today + timedelta(days=90)
        future_weather = weather_df[
            (weather_df['date'] >= pd.Timestamp(today)) & 
            (weather_df['date'] <= pd.Timestamp(end_date))
        ].copy()
        
        if future_weather.empty:
            print(f"No weather data found for next 90 days for {facility_name}")
            return None
        
        print(f"Found {len(future_weather)} days of weather data for {facility_name}")
        return future_weather
        
    except Exception as e:
        print(f"Error loading weather data for {facility_name}: {e}")
        return None

def create_prediction_dataframe(facility_name: str, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Create prediction dataframe for next 90 days"""
    # Create date range for next 90 days
    today = datetime.now().date()
    date_range = pd.date_range(start=today, end=today + timedelta(days=89), freq='D')
    
    # Create base prediction dataframe
    prediction_df = pd.DataFrame({
        'ds': date_range,
        'actual': [None] * len(date_range)  # No actual data for future predictions
    })
    
    return prediction_df

def train_models_for_facility(facility_name: str) -> tuple:
    """Train models for a specific facility"""
    print(f"\n{'='*50}")
    print(f"TRAINING MODELS FOR {facility_name.upper()}")
    print(f"{'='*50}")
    
    # Load energy data
    energy_df = load_energy_data(facility_name)
    if energy_df is None or energy_df.empty:
        print(f"No energy data found for {facility_name}")
        return None, None, None
    
    # Load weather data
    weather_df = load_weather_data(facility_name)
    if weather_df is None or weather_df.empty:
        print(f"No weather data found for {facility_name}")
        return None, None, None
    
    # Create feature configuration
    config = FeatureConfig(
        facility_name=facility_name,
        time_features=True,
        weather_features=True,
        rolling_features=True,
        lag_features=True,
        interaction_features=True,
        windows=[3, 7, 15, 30, 60],
        lags=[2, 3, 7, 14, 21, 30, 45, 60]
    )
    
    # Create advanced features
    energy_df = create_advanced_features(energy_df)
    
    # Get regressors
    regressor_cols = get_prophet_regressors(config, energy_df)
    available_regressors = [col for col in regressor_cols if col in energy_df.columns]
    
    # Limit regressors to avoid overfitting
    max_regressors = 6
    if len(available_regressors) > max_regressors:
        correlations = {}
        for col in available_regressors:
            if col in energy_df.columns and energy_df[col].notna().sum() > 0:
                corr = abs(energy_df[col].corr(energy_df['y']))
                if not np.isnan(corr):
                    correlations[col] = corr
        
        available_regressors = sorted(correlations.keys(), 
                                    key=lambda x: correlations[x], 
                                    reverse=True)[:max_regressors]
    
    print(f"Using {len(available_regressors)} regressors: {available_regressors}")
    
    # Train Prophet model
    print("Training Prophet model...")
    prophet_model = create_optimized_prophet_model(config)
    for col in available_regressors:
        if col in energy_df.columns:
            prophet_model.add_regressor(col, standardize='auto')
    prophet_model.fit(energy_df)
    
    # Train ensemble models
    print("Training ensemble models...")
    # Create simple CV splits for ensemble training
    energy_df_sorted = energy_df.sort_values('ds').reset_index(drop=True)
    total_samples = len(energy_df_sorted)
    initial_train_size = int(total_samples * 0.7)
    
    cv_splits = []
    for i in range(3):  # 3 CV splits
        train_end_idx = initial_train_size + (i * int(total_samples * 0.1))
        test_start_idx = train_end_idx
        test_end_idx = min(test_start_idx + int(total_samples * 0.1), total_samples)
        
        if test_end_idx > total_samples or test_start_idx >= total_samples:
            break
            
        train_df = energy_df_sorted.iloc[:train_end_idx].copy()
        test_df = energy_df_sorted.iloc[test_start_idx:test_end_idx].copy()
        
        if len(train_df) > 30 and len(test_df) > 5:
            cv_splits.append((train_df, test_df))
    
    weighted_ensemble_dict = create_weighted_ensemble_model(energy_df, config, cv_splits)
    
    return prophet_model, weighted_ensemble_dict, available_regressors

def predict_next_90_days(facility_name: str, prophet_model, weighted_ensemble_dict, available_regressors, weather_df: pd.DataFrame) -> List[Dict]:
    """Generate predictions for next 90 days"""
    print(f"\nGenerating predictions for {facility_name}...")
    
    # Create prediction dataframe
    prediction_df = create_prediction_dataframe(facility_name, weather_df)
    
    # Load energy data for feature preparation
    energy_df = load_energy_data(facility_name)
    if energy_df is None:
        return []
    
    # Create feature configuration
    config = FeatureConfig(
        facility_name=facility_name,
        time_features=True,
        weather_features=True,
        rolling_features=True,
        lag_features=True,
        interaction_features=True,
        windows=[3, 7, 15, 30, 60],
        lags=[2, 3, 7, 14, 21, 30, 45, 60]
    )
    
    # Prepare prediction data with features
    prediction_data = prepare_prediction_data(prediction_df, weather_df, energy_df, config)
    
    # Get Prophet predictions
    prophet_pred = prophet_model.predict(prediction_data)
    prophet_pred_values = prophet_pred['yhat'].values
    
    # Get ensemble predictions
    ensemble_pred, individual_predictions = predict_weighted_ensemble(
        weighted_ensemble_dict, 
        prediction_data, 
        available_regressors
    )
    
    xgb_pred = individual_predictions['xgb_prediction']
    
    # Create results
    predictions = []
    for i in range(len(prediction_data)):
        date = prediction_data.iloc[i]['ds']
        date_str = date.strftime('%Y-%m-%d')
        
        result = {
            "date": date_str,
            "ensemble_prediction": round(float(ensemble_pred[i]), 2),
            "xgboost_prediction": round(float(xgb_pred[i]), 2),
            "prophet_prediction": round(float(prophet_pred_values[i]), 2),
            "temperature": round(float(prediction_data.iloc[i].get('temperature_2m_mean', 0)), 2)
        }
        predictions.append(result)
    
    return predictions

def save_predictions_to_csv(facility_name: str, predictions: List[Dict]):
    """Save predictions to CSV file in facility folder"""
    if not predictions:
        print(f"No predictions to save for {facility_name}")
        return
    
    # Create facility directory if it doesn't exist
    facility_dir = f'../data/{facility_name}'
    os.makedirs(facility_dir, exist_ok=True)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(predictions)
    csv_path = f'{facility_dir}/predictions-90.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"Saved {len(predictions)} predictions to {csv_path}")
    print(f"Date range: {predictions[0]['date']} to {predictions[-1]['date']}")

def main():
    """Main function to predict next 90 days for all facilities"""
    print("="*60)
    print("PREDICTING NEXT 90 DAYS FOR ALL FACILITIES")
    print("="*60)
    
    # Load facility metadata
    facilities = load_facility_metadata()
    if not facilities:
        print("No facilities found in metadata")
        return
    
    print(f"Found {len(facilities)} facilities: {list(facilities.keys())}")
    
    # Process each facility
    for facility_name, facility_info in facilities.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING FACILITY: {facility_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Check if weather data exists for next 90 days
            weather_df = get_next_90_days_weather(facility_name)
            if weather_df is None:
                print(f"Skipping {facility_name} - no weather data available")
                continue
            
            # Train models
            prophet_model, weighted_ensemble_dict, available_regressors = train_models_for_facility(facility_name)
            if prophet_model is None:
                print(f"Failed to train models for {facility_name}")
                continue
            
            # Generate predictions
            predictions = predict_next_90_days(
                facility_name, prophet_model, weighted_ensemble_dict, 
                available_regressors, weather_df
            )
            
            # Save predictions
            save_predictions_to_csv(facility_name, predictions)
            
        except Exception as e:
            print(f"Error processing {facility_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("PREDICTION COMPLETE")
    print("="*60)
    print("Check each facility folder for predictions-90.csv files")

if __name__ == "__main__":
    main() 