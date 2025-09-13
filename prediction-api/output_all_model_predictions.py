import pandas as pd
import numpy as np
from prophet import Prophet
import os
from datetime import datetime, timedelta
from data_loader import load_and_prepare_data
from feature_engineering import create_advanced_features, get_prophet_regressors
from model_training import create_optimized_prophet_model, create_ensemble_model, create_weighted_ensemble_model, predict_weighted_ensemble
from config import get_data_folder
from models import FeatureConfig

def create_future_features(future_df, last_known_day=0):
    """Create advanced features for future dataframe without requiring 'y' column"""
    df = future_df.copy()
    
    # Time features
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['quarter'] = df['ds'].dt.quarter
    
    # Cyclical time features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # Binary features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    
    # Shutdown feature - for future predictions, we'll set to 0 (no shutdown)
    df['shutdown'] = 0
    
    # Days since commission - continue from the last known day
    df['days_elapsed'] = range(last_known_day + 1, last_known_day + len(df) + 1)
    
    # For future dates, we'll use mean values for lag and rolling features
    # since we don't have actual historical data
    df['y_lag1'] = 0  # Will be filled with mean
    df['y_lag2'] = 0
    df['y_lag3'] = 0
    df['y_lag7'] = 0
    df['y_lag14'] = 0
    df['y_lag30'] = 0
    df['y_rolling_mean_3'] = 0
    df['y_rolling_mean_7'] = 0
    df['y_rolling_mean_14'] = 0
    df['y_rolling_mean_30'] = 0
    df['y_rolling_std_7'] = 0
    df['y_rolling_std_14'] = 0
    df['y_rolling_max_7'] = 0
    df['y_rolling_min_7'] = 0
    df['y_diff1'] = 0
    df['y_diff7'] = 0
    df['y_pct_change1'] = 0
    df['y_pct_change7'] = 0
    
    return df

def generate_all_model_predictions(start_date, end_date, prophet_model, ensemble_dict, weighted_ensemble_dict, energy_df, weather_future_df, official_df, available_regressors):
    """Generate predictions for all models for a specific date range"""
    print(f"\nGenerating predictions for all models ({start_date} to {end_date})...")
    
    # Create future dates for the specified range
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})
    
    # Add regressors to future dataframe
    for col in available_regressors:
        if col in weather_future_df.columns:
            future_df[col] = weather_future_df[col].values[:len(future_dates)]
        else:
            future_df[col] = energy_df[col].mean()  # Use mean for other features
    
    # Make Prophet predictions
    print("Making Prophet predictions...")
    prophet_forecast = prophet_model.predict(future_df)
    prophet_pred_values = prophet_forecast['yhat'].values
    
    # Calculate the last known day from historical data
    last_known_day = energy_df['days_elapsed'].max() if 'days_elapsed' in energy_df.columns else 0
    
    # Create advanced features for future dataframe
    future_df_with_features = create_future_features(future_df, last_known_day)
    
    # Fill lag and rolling features with mean values from training data
    y_mean = energy_df['y'].mean()
    for col in ['y_lag1', 'y_lag2', 'y_lag3', 'y_lag7', 'y_lag14', 'y_lag30', 
                'y_rolling_mean_3', 'y_rolling_mean_7', 'y_rolling_mean_14', 'y_rolling_mean_30',
                'y_rolling_std_7', 'y_rolling_std_14', 'y_rolling_max_7', 'y_rolling_min_7',
                'y_diff1', 'y_diff7', 'y_pct_change1', 'y_pct_change7']:
        if col in future_df_with_features.columns:
            future_df_with_features[col] = y_mean
    
    # Prepare features for ML models
    range_features = future_df_with_features[ensemble_dict['features']].fillna(0)
    range_features = range_features.replace([np.inf, -np.inf], np.nan)
    range_features = range_features.fillna(range_features.median())
    
    # Make individual ML model predictions
    print("Making Random Forest predictions...")
    rf_pred_values = ensemble_dict['rf_model'].predict(range_features)
    
    print("Making Ridge Regression predictions...")
    ridge_pred_values = ensemble_dict['ridge_model'].predict(ensemble_dict['scaler'].transform(range_features))
    
    print("Making XGBoost predictions...")
    xgb_pred_values = ensemble_dict['xgb_model'].predict(range_features)
    
    # Make weighted ensemble predictions
    print("Making Weighted Ensemble predictions...")
    weighted_ensemble_pred, individual_preds = predict_weighted_ensemble(
        weighted_ensemble_dict, 
        future_df_with_features, 
        prophet_pred_values, 
        prophet_accuracy=95.0  # Assuming Prophet accuracy, could be calculated from validation
    )
    
    # Prepare results
    predictions = []
    for i in range(len(future_dates)):
        date = future_dates[i]
        
        # Get all model predictions
        prophet_predicted = float(prophet_pred_values[i])
        rf_predicted = float(rf_pred_values[i])
        ridge_predicted = float(ridge_pred_values[i])
        xgb_predicted = float(xgb_pred_values[i])
        weighted_ensemble_predicted = float(weighted_ensemble_pred[i])
        
        # Get official value if available
        official_value = None
        if date in official_df['date'].values:
            official_value = float(official_df[official_df['date'] == date]['energy_output'].iloc[0]) 
        
        # Calculate error rates for all models
        prophet_error_rate = abs((prophet_predicted - official_value) / official_value * 100) if official_value is not None else None
        rf_error_rate = abs((rf_predicted - official_value) / official_value * 100) if official_value is not None else None
        ridge_error_rate = abs((ridge_predicted - official_value) / official_value * 100) if official_value is not None else None
        xgb_error_rate = abs((xgb_predicted - official_value) / official_value * 100) if official_value is not None else None
        weighted_ensemble_error_rate = abs((weighted_ensemble_predicted - official_value) / official_value * 100) if official_value is not None else None
        
        predictions.append({
            'date': date.strftime('%Y-%m-%d'),
            'prophet_predicted_energy': round(prophet_predicted, 2),
            'random_forest_predicted_energy': round(rf_predicted, 2),
            'ridge_predicted_energy': round(ridge_predicted, 2),
            'xgboost_predicted_energy': round(xgb_predicted, 2),
            'weighted_ensemble_predicted_energy': round(weighted_ensemble_predicted, 2),
            'official_energy': round(official_value, 2) if official_value is not None else None,
            'prophet_error_rate': round(prophet_error_rate, 2) if prophet_error_rate is not None else None,
            'random_forest_error_rate': round(rf_error_rate, 2) if rf_error_rate is not None else None,
            'ridge_error_rate': round(ridge_error_rate, 2) if ridge_error_rate is not None else None,
            'xgboost_error_rate': round(xgb_error_rate, 2) if xgb_error_rate is not None else None,
            'weighted_ensemble_error_rate': round(weighted_ensemble_error_rate, 2) if weighted_ensemble_error_rate is not None else None
        })
    
    return predictions

def output_all_model_predictions(facility_name, custom_start_date=None, custom_end_date=None):
    """Generate and output predictions from all models to a single CSV file"""
    print("\nGenerating predictions for all models...")
    
    # Create default feature config
    config = FeatureConfig(
        facility_name=facility_name,
        time_features=True,
        weather_features=True,
        lag_features=True,
        rolling_features=True,
        interaction_features=True
    )
    
    # Load and prepare data
    print("Loading and preparing data...")
    energy_df, weather_future_df, _ = load_and_prepare_data(facility_name)
    
    # Create advanced features
    print("Creating advanced features...")
    energy_df = create_advanced_features(energy_df, energy_csv_path=None)
    
    # Create Prophet model
    print("Creating and training Prophet model...")
    prophet_model = create_optimized_prophet_model(config)
    
    # Add regressors
    regressor_cols = get_prophet_regressors(config, energy_df)
    available_regressors = [col for col in regressor_cols if col in energy_df.columns]
    
    for col in available_regressors:
        if col in energy_df.columns:
            prophet_model.add_regressor(col)
    
    # Fit the model
    prophet_model.fit(energy_df)
    
    # Create ensemble model for individual ML models
    print("Creating and training ensemble model...")
    ensemble_dict = create_ensemble_model(energy_df, config)
    
    # Create weighted ensemble model
    print("Creating and training weighted ensemble model...")
    weighted_ensemble_dict = create_weighted_ensemble_model(energy_df, config)
    
    # Load official energy output for comparison
    facility_folder = get_data_folder(facility_name)
    official_df = pd.read_csv(f'{facility_folder}/energy-totals.csv')
    official_df = official_df.rename(columns={'date': 'date', 'calculated_energy': 'energy_output'})
    official_df['date'] = pd.to_datetime(official_df['date'])
    
    # Create results directory if it doesn't exist
    results_dir = f'{facility_folder}/model_predictions'
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate predictions for May 2024
    may_predictions = generate_all_model_predictions('2024-05-01', '2024-05-31', prophet_model, ensemble_dict, weighted_ensemble_dict, energy_df, weather_future_df, official_df, available_regressors)
    
    # Initialize all_predictions with May predictions
    all_predictions = may_predictions.copy()
    
    # Generate predictions for custom date range if provided
    if custom_start_date and custom_end_date:
        print(f"\nGenerating predictions for custom date range: {custom_start_date} to {custom_end_date}")
        custom_predictions = generate_all_model_predictions(custom_start_date, custom_end_date, prophet_model, ensemble_dict, weighted_ensemble_dict, energy_df, weather_future_df, official_df, available_regressors)
        all_predictions.extend(custom_predictions)
    
    # Generate predictions for October-November 2024 (default extended range)
    if not custom_start_date or not custom_end_date:
        oct_nov_predictions = generate_all_model_predictions('2024-10-01', '2024-11-30', prophet_model, ensemble_dict, weighted_ensemble_dict, energy_df, weather_future_df, official_df, available_regressors)
        all_predictions.extend(oct_nov_predictions)
    
    # Save to CSV with all model results
    output_file = f'{results_dir}/all_models_predictions.csv'
    
    # Create DataFrame for easier CSV writing
    df_predictions = pd.DataFrame(all_predictions)
    
    # Save to CSV
    df_predictions.to_csv(output_file, index=False)
    
    print(f"\nSaved all model predictions to {output_file}")
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total days predicted: {len(all_predictions)}")
    print(f"May 2024 predictions: {len(may_predictions)} days")
    
    if custom_start_date and custom_end_date:
        custom_count = len(all_predictions) - len(may_predictions)
        print(f"Custom range predictions: {custom_count} days")
    else:
        oct_nov_count = len(all_predictions) - len(may_predictions)
        print(f"October-November 2024 predictions: {oct_nov_count} days")
    
    print(f"Date range: {all_predictions[0]['date']} to {all_predictions[-1]['date']}")
    
    # Calculate average error rates for days with official values
    models = ['prophet', 'random_forest', 'ridge', 'xgboost', 'weighted_ensemble']
    
    for model in models:
        error_col = f'{model}_error_rate'
        errors = [p[error_col] for p in all_predictions if p[error_col] is not None]
        
        if errors:
            print(f"\n{model.replace('_', ' ').title()} Error Rates:")
            print(f"Average: {np.mean(errors):.2f}%")
            print(f"Min: {min(errors):.2f}%")
            print(f"Max: {max(errors):.2f}%")
    
    # Print model weights from weighted ensemble
    weights = weighted_ensemble_dict['weights']
    print(f"\nWeighted Ensemble Model Weights:")
    print(f"Random Forest: {weights['rf_weight']:.3f}")
    print(f"Ridge Regression: {weights['ridge_weight']:.3f}")
    print(f"XGBoost: {weights['xgb_weight']:.3f}")
    
    return output_file

if __name__ == "__main__":
    # You can call with custom date ranges like:
    # output_all_model_predictions('facility_name', custom_start_date='2024-06-01', custom_end_date='2024-06-30')
    output_all_model_predictions('ssdairy') 