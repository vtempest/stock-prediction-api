#!/usr/bin/env python3
"""
Standalone script to generate predictions for custom date ranges and save them to separate files.
This script directly uses the prediction functions without requiring the API to be running.
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
import warnings
import numpy as np

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_and_prepare_data, prepare_prediction_data
from feature_engineering import create_advanced_features, get_prophet_regressors
from model_training import create_optimized_prophet_model, create_weighted_ensemble_model, predict_weighted_ensemble
from models import FeatureConfig
from utils import NumpyEncoder
from config import get_data_folder

warnings.filterwarnings('ignore')

def generate_custom_predictions(facility_name, custom_start_date, custom_end_date, config=None, trained_models=None, weather_future_df=None):
    """
    Generate predictions for a custom date range and save to separate files.
    
    Args:
        facility_name (str): Facility name
        custom_start_date (str): Start date in YYYY-MM-DD format
        custom_end_date (str): End date in YYYY-MM-DD format
        config (FeatureConfig): Feature configuration (optional)
        trained_models (dict): Pre-trained models from main function (optional)
        weather_future_df (DataFrame): Weather data for future dates (optional)
    """
    
    if config is None:
        config = FeatureConfig(
            use_weather_features=True,
            use_temporal_features=True,
            use_lag_features=True,
            use_rolling_features=True,
            use_interaction_features=False,
            custom_start_date=custom_start_date,
            custom_end_date=custom_end_date
        )
    
    print(f"\n{'='*60}")
    print(f"GENERATING CUSTOM PREDICTIONS")
    print(f"Date Range: {custom_start_date} to {custom_end_date}")
    print(f"{'='*60}")
    
    try:
        # Use pre-trained models if provided, otherwise train new ones
        if trained_models is not None:
            print("Using pre-trained models from main function...")
            prophet_model = trained_models['prophet_model']
            ensemble_model = trained_models['ensemble_model']
            available_regressors = trained_models['available_regressors']
            prophet_metrics = trained_models.get('prophet_metrics', {'accuracy': 85.0})
        else:
            print("Training new models (standalone mode)...")
            # Load and prepare data
            print("Loading and preparing data...")
            energy_df, weather_future_df, official_df = load_and_prepare_data(facility_name)
            
            # Create features
            print("Creating advanced features...")
            energy_df = create_advanced_features(energy_df)
            
            # Train Prophet model
            print("Training Prophet model...")
            prophet_model = create_optimized_prophet_model(config)
            
            # Add regressors
            regressor_cols = get_prophet_regressors(config, energy_df)
            available_regressors = [col for col in regressor_cols if col in energy_df.columns]
            
            for col in available_regressors:
                if col in energy_df.columns:
                    prophet_model.add_regressor(col, standardize='auto')
            
            # Fit model
            prophet_model.fit(energy_df)
            
            # Create ensemble model
            print("Creating ensemble model...")
            ensemble_model = create_weighted_ensemble_model(energy_df, config)
            prophet_metrics = {'accuracy': 85.0}
        
        # Prepare prediction data for custom date range
        print(f"Preparing prediction data for {custom_start_date} to {custom_end_date}...")
        
        if weather_future_df is None:
            # Load weather data if not provided
            energy_df, weather_future_df, _ = load_and_prepare_data(facility_name)
        else:
            # Load energy data for feature creation
            energy_df, _, _ = load_and_prepare_data(facility_name)
        
        # Create date range for custom predictions
        start_date = pd.to_datetime(custom_start_date)
        end_date = pd.to_datetime(custom_end_date)
        custom_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        custom_prediction_df = pd.DataFrame({'ds': custom_dates})
        
        # Add temperature data for custom date range
        custom_prediction_df = pd.merge(
            custom_prediction_df,
            weather_future_df[['ds', 'temperature_2m_mean']],
            on='ds',
            how='left'
        )
        
        # Fill missing temperature data with historical mean
        temp_mean = energy_df['temperature_2m_mean'].mean()
        custom_prediction_df['temperature_2m_mean'] = custom_prediction_df['temperature_2m_mean'].fillna(temp_mean)
        
        if custom_prediction_df.empty:
            print("No prediction data available for the specified date range")
            return None
        
        # Create features for custom predictions using the same process as May predictions
        print("Creating advanced features for custom predictions...")
        
        # First, get the historical energy data ready
        historical_df = energy_df.copy()
        historical_df = historical_df.sort_values('ds')
        
        # Create a temporary dataframe combining historical and prediction data
        temp_df = pd.concat([
            historical_df[['ds', 'y']],
            custom_prediction_df[['ds']].assign(y=np.nan)
        ]).sort_values('ds')
        
        # Create lag features
        temp_df['y_lag1'] = temp_df['y'].shift(1)
        temp_df['y_lag2'] = temp_df['y'].shift(2)
        temp_df['y_lag3'] = temp_df['y'].shift(3)
        temp_df['y_lag7'] = temp_df['y'].shift(7)
        temp_df['y_lag14'] = temp_df['y'].shift(14)
        temp_df['y_lag30'] = temp_df['y'].shift(30)
        
        # Create rolling statistics
        temp_df['y_rolling_mean_3'] = temp_df['y'].rolling(3, min_periods=1).mean()
        temp_df['y_rolling_mean_7'] = temp_df['y'].rolling(7, min_periods=1).mean()
        temp_df['y_rolling_mean_14'] = temp_df['y'].rolling(14, min_periods=1).mean()
        temp_df['y_rolling_mean_30'] = temp_df['y'].rolling(30, min_periods=1).mean()
        temp_df['y_rolling_std_7'] = temp_df['y'].rolling(7, min_periods=1).std()
        temp_df['y_rolling_std_14'] = temp_df['y'].rolling(14, min_periods=1).std()
        temp_df['y_rolling_max_7'] = temp_df['y'].rolling(7, min_periods=1).max()
        temp_df['y_rolling_min_7'] = temp_df['y'].rolling(7, min_periods=1).min()
        
        # Create trend features
        temp_df['y_diff1'] = temp_df['y'].diff(1)
        temp_df['y_diff7'] = temp_df['y'].diff(7)
        temp_df['y_pct_change1'] = temp_df['y'].pct_change(1)
        temp_df['y_pct_change7'] = temp_df['y'].pct_change(7)
        
        # Get the lag features for prediction dates
        lag_features = [col for col in temp_df.columns if col.startswith('y_')]
        custom_prediction_df = pd.merge(
            custom_prediction_df,
            temp_df[['ds'] + lag_features].tail(len(custom_prediction_df)),
            on='ds',
            how='left'
        )
        
        # Add advanced features with simplified data structure
        custom_prediction_df = create_advanced_features(custom_prediction_df, energy_csv_path=None)
        
        # Fill any remaining NaN values with the last known value
        for col in lag_features:
            if col in custom_prediction_df.columns:
                custom_prediction_df[col] = custom_prediction_df[col].fillna(method='ffill')
                custom_prediction_df[col] = custom_prediction_df[col].fillna(method='bfill')
                custom_prediction_df[col] = custom_prediction_df[col].fillna(0)  # Fill any remaining NaNs with 0
        
        # Ensure all required regressor columns are present
        missing_regressors = [col for col in available_regressors if col not in custom_prediction_df.columns]
        if missing_regressors:
            print(f"Warning: Missing regressor columns: {missing_regressors}")
            print("Filling missing columns with default values...")
            for col in missing_regressors:
                custom_prediction_df[col] = 0.0  # Default value for missing regressors
        
        print(f"Custom prediction data shape: {custom_prediction_df.shape}")
        print(f"Available columns: {list(custom_prediction_df.columns)}")
        
        # Generate predictions
        print("Generating predictions...")
        custom_predictions = []
        
        print(f"Processing {len(custom_prediction_df)} dates for predictions...")
        
        for i in range(len(custom_prediction_df)):
            date = custom_prediction_df.iloc[i]['ds']
            date_str = date.strftime('%Y-%m-%d')
            
            print(f"Processing date {i+1}/{len(custom_prediction_df)}: {date_str}")
            
            # Prophet prediction
            prophet_future = custom_prediction_df.iloc[i:i+1][['ds'] + available_regressors].copy()
            prophet_forecast = prophet_model.predict(prophet_future)
            prophet_predicted = float(prophet_forecast['yhat'].iloc[0])
            
            # Ensemble prediction
            ensemble_result = predict_weighted_ensemble(
                ensemble_model, 
                custom_prediction_df.iloc[i:i+1],
                prophet_predicted,
                prophet_metrics['accuracy']
            )
            
            # Extract ensemble prediction and details
            ensemble_predicted = ensemble_result[0][0]  # Get first prediction value
            ensemble_details = ensemble_result[1]
            
            # XGBoost prediction (from ensemble details)
            xgb_predicted = ensemble_details['xgb_pred'][0]  # Get first prediction value
            
            result = {
                "date": date_str,
                "prophet_prediction": round(float(prophet_predicted), 2),
                "xgb_prediction": round(float(xgb_predicted), 2),
                "ensemble_prediction": round(float(ensemble_predicted), 2),
                "actual_energy": None,  # No actual data for custom ranges
                "error": None,
                "percent_error": None,
                "ensemble_weights": {
                    "prophet_weight": 0.4,
                    "rf_weight": 0.2,
                    "ridge_weight": 0.2,
                    "xgb_weight": 0.2
                },
                "weather_data": {
                    "temperature_2m_mean": float(custom_prediction_df.iloc[i].get('temperature_2m_mean', 0))
                }
            }
            custom_predictions.append(result)
            
            print(f"{date_str}: Prophet={prophet_predicted:.2f}, Ensemble={ensemble_predicted:.2f}")
        
        print(f"Generated {len(custom_predictions)} predictions successfully")
        
        # Save predictions to separate files
        print("Saving predictions to files...")
        save_custom_predictions_to_files(facility_name, custom_predictions, custom_start_date, custom_end_date)
        
        return custom_predictions
        
    except Exception as e:
        print(f"Error generating custom predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_custom_predictions_to_files(facility_name, predictions, start_date, end_date):
    """Save custom predictions to separate CSV and JSON files"""
    
    if not predictions:
        print("No predictions to save")
        return
    
    # Create output directory
    output_dir = f'{get_data_folder(facility_name)}/custom_predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save as CSV
    csv_filename = f'{output_dir}/custom_predictions_{start_date}_to_{end_date}_{timestamp}.csv'
    
    csv_data = []
    for pred in predictions:
        csv_data.append({
            'date': pred['date'],
            'prophet_predicted_energy': pred['prophet_prediction'],
            'xgb_predicted_energy': pred['xgb_prediction'],
            'ensemble_predicted_energy': float(round(float(pred['ensemble_prediction']), 2))
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_filename, index=False)
    print(f"Saved {len(csv_data)} predictions to CSV: {csv_filename}")
    

def main():
    """Main function to test different custom date ranges"""
    
    print("Custom Date Range Predictions Generator")
    print("=" * 60)
    
    # Test single date range
    facility_name = "example_facility"
    start_date = "2025-06-26"
    end_date = "2025-07-12"
    
    print(f"\n{'='*60}")
    print(f"Generating predictions for date range: {start_date} to {end_date}")
    print(f"{'='*60}")
    
    predictions = generate_custom_predictions(facility_name, start_date, end_date)
    
    if predictions:
        print(f"Successfully generated {len(predictions)} predictions")
    else:
        print(f"Failed to generate predictions")
    
    print(f"\n{'='*60}")
    print("PREDICTION GENERATION COMPLETED")
    print(f"{'='*60}")
    print(f"\nCheck the {get_data_folder(facility_name)}/custom_predictions directory for output files:")
    print("- CSV files with predictions")
    print("- JSON files with detailed data")

if __name__ == "__main__":
    main() 