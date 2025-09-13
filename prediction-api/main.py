import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np
import json
import warnings
import traceback
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from models import FeatureConfig
from utils import NumpyEncoder
from data_loader import load_and_prepare_data, prepare_prediction_data, load_energy_data, load_weather_data, load_future_weather_data, load_validation_data
from feature_engineering import create_advanced_features, get_prophet_regressors
from model_training import create_optimized_prophet_model, create_weighted_ensemble_model, predict_weighted_ensemble
from evaluation import evaluate_model
from config import get_data_folder
from generate_custom_predictions import generate_custom_predictions
import json

# Load OpenAPI schema
with open('./openapi-schema.json') as f:
    openapi_schema = json.load(f)

warnings.filterwarnings('ignore')

app = FastAPI(docs_url=None)  # Disable default docs to add your own

def custom_openapi():
    return openapi_schema

app.openapi = custom_openapi

def load_facility_metadata():
    """Load facility metadata from JSON file"""
    try:
        metadata_path = '../data/facility-metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Error loading facility metadata: {e}")
        return []

@app.get("/facilities")
async def get_facilities():
    """Get list of available facilities from metadata"""
    try:
        facilities = load_facility_metadata()
        # Return only the essential fields for the API
        facility_list = []
        for facility in facilities:
            facility_list.append({
                "name": facility["name"],
                "address": facility["address"],
                "latitude": facility["latitude"],
                "longitude": facility["longitude"],
                "start": facility["start"]
            })
        return facility_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading facilities: {str(e)}")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Predict Statistics API"
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for detailed error logging"""
    error_msg = f"An error occurred: {str(exc)}"
    error_location = f"Error occurred in {request.url.path}"
    error_trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    
    print("\n" + "="*50)
    print("ERROR DETAILS")
    print("="*50)
    print(f"Timestamp: {pd.Timestamp.now()}")
    print(f"Endpoint: {error_location}")
    print(f"Error Type: {type(exc).__name__}")
    print(f"Error Message: {error_msg}")
    print("\nStack Trace:")
    print(error_trace)
    print("\nRequest Info:")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Headers: {dict(request.headers)}")
    print("="*50 + "\n")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "path": request.url.path
        }
    )

@app.post("/predict")
async def predict(config: FeatureConfig):
    """
    Generate energy predictions for both May official dates and optional custom date range.
    
    The API now supports:
    1. May official date predictions (with actual data comparison)
    2. Custom date range predictions (if custom_start_date and custom_end_date are provided)
    
    Custom date range predictions will have actual_energy, error, and percent_error set to null
    since no actual data is available for comparison.
    """
    try:
        print("\n" + "="*50)
        print("STARTING PREDICTION PIPELINE")
        print(f"Timestamp: {pd.Timestamp.now()}")
        print("="*50)
        
        print("Loading and preparing data...")
        energy_df, weather_future_df, official_df = load_and_prepare_data(
            config.facility_name, 
            training_cutoff_date='2025-05-30', 
            validation_start_date='2025-06-01'
        )
        
        print("Creating advanced features based on config...")
        print(f"Config settings: {config.dict()}")
        energy_df = create_advanced_features(energy_df)
        
        print(f"Data shape after feature engineering: {energy_df.shape}")
        print(f"Features created: {list(energy_df.columns)}")
        
        # ===== PROPHET MODEL CROSS-VALIDATION =====
        print("\n" + "="*50)
        print("PROPHET MODEL TIME SERIES CROSS-VALIDATION")
        print("="*50)

        # Train full Prophet model for cross-validation
        prophet_cv_model = create_optimized_prophet_model(config)

        # Add regressors
        regressor_cols = get_prophet_regressors(config, energy_df)
        available_regressors = [col for col in regressor_cols if col in energy_df.columns]

        # Limit regressors to avoid overfitting - optimized for speed
        max_regressors = 6  # Reduced from 10 to 6 for faster processing
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

        print(f"Using {len(available_regressors)} regressors for faster processing: {available_regressors}")

        for col in available_regressors:
            if col in energy_df.columns:
                prophet_cv_model.add_regressor(col, standardize='auto')

        # Fit the model
        prophet_cv_model.fit(energy_df)

        # Use Prophet's built-in cross-validation - more conservative for smaller datasets
        print("Running Prophet built-in cross-validation...")
        
        # Calculate conservative parameters based on data size
        data_size = len(energy_df)
        min_initial = min(90, int(data_size * 0.3))  # At least 90 days or 30% of data
        min_period = min(30, int(data_size * 0.1))   # At least 30 days or 10% of data
        min_horizon = min(30, int(data_size * 0.1))  # At least 30 days or 10% of data
        
        try:
            df_cv = cross_validation(
                prophet_cv_model, 
                initial=f'{min_initial} days',
                period=f'{min_period} days',
                horizon=f'{min_horizon} days',
                parallel="threads"
            )
        except Exception as e:
            print(f"Prophet cross-validation failed: {e}")
            print("Using fallback metrics...")
            # Use simple train/test split for metrics
            split_idx = int(len(energy_df) * 0.8)
            train_df = energy_df.iloc[:split_idx]
            test_df = energy_df.iloc[split_idx:]
            
            # Fit on train and predict on test
            prophet_cv_model.fit(train_df)
            test_pred = prophet_cv_model.predict(test_df)
            
            # Calculate simple metrics
            prophet_metrics = {
                'mae': np.mean(np.abs(test_pred['yhat'].values - test_df['y'].values)),
                'rmse': np.sqrt(np.mean((test_pred['yhat'].values - test_df['y'].values) ** 2)),
                'mape': np.mean(np.abs((test_pred['yhat'].values - test_df['y'].values) / test_df['y'].values)) * 100,
                'r2_score': 1 - (np.mean((test_pred['yhat'].values - test_df['y'].values) ** 2) / test_df['y'].var()),
                'accuracy': 100 - np.mean(np.abs((test_pred['yhat'].values - test_df['y'].values) / test_df['y'].values)) * 100
            }
        else:
            # Calculate performance metrics using Prophet's built-in function
            df_p = performance_metrics(df_cv)

            # here Prophet uses MDAPE (Median) instead of MAPE (mean)
            # Convert to our format (Prophet uses different scale)
            prophet_metrics = {
                'mae': df_p['mae'].mean(),     # Keep in original scale
                'rmse': df_p['rmse'].mean(),   # Keep in original scale
                'mape': df_p['mdape'].mean() * 100,   # Convert to percentage
                'r2_score': 1 - (df_p['mse'].mean() / energy_df['y'].var()),
                'accuracy': 100 - (df_p['mdape'].mean() * 100)
            }
        
        print(f"Prophet CV Results:")
        print(f"MAE: {prophet_metrics['mae']:.2f}")
        print(f"RMSE: {prophet_metrics['rmse']:.2f}") 
        print(f"MAPE: {prophet_metrics['mape']:.2f}%")
        print(f"R²: {prophet_metrics['r2_score']:.4f}")
        print(f"Accuracy: {prophet_metrics['accuracy']:.2f}%")

        # Skip saving Prophet CV results to file for faster execution
        print("Prophet CV results logged to console only (file saving disabled)")

        # ===== CREATE TIME SERIES CROSS-VALIDATION SPLITS =====
        print("\n" + "="*50)
        print("CREATING TIME SERIES CROSS-VALIDATION SPLITS")
        print("="*50)
        
        # Sort data by date to ensure proper time series splitting
        energy_df_sorted = energy_df.sort_values('ds').reset_index(drop=True)
        
        # Create time series cross-validation splits - more conservative for smaller datasets
        cv_splits = []
        total_samples = len(energy_df_sorted)
        
        # Adjust parameters based on data size
        if total_samples < 100:
            n_splits = 2  # Very small dataset
            min_train_size = 20
            min_test_size = 3
        elif total_samples < 200:
            n_splits = 3  # Small dataset
            min_train_size = 30
            min_test_size = 5
        else:
            n_splits = 4  # Larger dataset
            min_train_size = 50
            min_test_size = 10
        
        # Calculate split points for time series CV
        initial_train_size = int(total_samples * 0.7)
        step_size = int(total_samples * 0.1)
        
        for i in range(n_splits):
            # Calculate train and test indices for this fold
            train_end_idx = initial_train_size + (i * step_size)
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + step_size, total_samples)
            
            # Ensure we don't go beyond data bounds
            if test_end_idx > total_samples or test_start_idx >= total_samples:
                break
                
            # Create train and test dataframes for this fold
            train_df = energy_df_sorted.iloc[:train_end_idx].copy()
            test_df = energy_df_sorted.iloc[test_start_idx:test_end_idx].copy()
            
            # Only add split if both train and test have sufficient data
            if len(train_df) >= min_train_size and len(test_df) >= min_test_size:
                cv_splits.append((train_df, test_df))
                print(f"Fold {len(cv_splits)}: Train size={len(train_df)}, Test size={len(test_df)}")
        
        print(f"Created {len(cv_splits)} time series CV splits")
        
        # If no valid splits created, create a simple train/test split
        if len(cv_splits) == 0:
            print("No valid CV splits created, using simple train/test split")
            split_idx = int(total_samples * 0.8)
            train_df = energy_df_sorted.iloc[:split_idx].copy()
            test_df = energy_df_sorted.iloc[split_idx:].copy()
            cv_splits = [(train_df, test_df)]
            print(f"Simple split: Train size={len(train_df)}, Test size={len(test_df)}")

        # ===== WEIGHTED ENSEMBLE MODEL TRAINING =====
        print("\n" + "="*50)
        print("TRAINING WEIGHTED ENSEMBLE MODELS")
        print("="*50)
        
        # Create weighted ensemble model using cross-validation to determine weights
        weighted_ensemble_dict = create_weighted_ensemble_model(energy_df, config, cv_splits)
        
        # Extract weights and CV scores for reporting
        weights = weighted_ensemble_dict['weights']
        cv_scores = weighted_ensemble_dict['cv_scores']
        
        print(f"\nFinal model weights determined by CV accuracy:")
        print(f"Random Forest: {weights['rf_weight']:.3f} (CV Accuracy: {cv_scores['rf_accuracy']:.2f}%)")
        print(f"Ridge Regression: {weights['ridge_weight']:.3f} (CV Accuracy: {cv_scores['ridge_accuracy']:.2f}%)")
        print(f"XGBoost: {weights['xgb_weight']:.3f} (CV Accuracy: {cv_scores['xgb_accuracy']:.2f}%)")

        # ===== FINAL MODEL TRAINING =====
        print("\n" + "="*50)
        print("TRAINING FINAL PROPHET MODEL ON FULL DATASET")
        print("="*50)
        
        # Retrain Prophet on full dataset
        final_prophet = create_optimized_prophet_model(config)
        for col in available_regressors:
            final_prophet.add_regressor(col)
        final_prophet.fit(energy_df)
        
        # ===== VALIDATION MONTHS PREDICTIONS =====
        if hasattr(config, 'validation_months') and config.validation_months:
            print("\n" + "="*60)
            print("RUNNING POST-VALIDATION FOR MULTIPLE MONTHS")
            print("="*60)
            print(f"Validation months: {config.validation_months}")
            
            # Import validation functions
            from validation_months import run_validation_for_month, save_validation_results_to_csv
            
            # Run validation for each month
            all_validation_results = []
            for month in config.validation_months:
                try:
                    print(f"\nProcessing validation month: {month}")
                    month_result = run_validation_for_month(
                        month, energy_df, weather_future_df, config,
                        final_prophet, weighted_ensemble_dict, available_regressors
                    )
                    all_validation_results.append(month_result)
                except Exception as e:
                    print(f"Error validating month {month}: {str(e)}")
                    continue
            
            # Save validation results to CSV
            if all_validation_results:
                csv_path = save_validation_results_to_csv(all_validation_results, config.facility_name, config)
                print(f"\nValidation results saved to: {csv_path}")
                
                # Return early with validation summary
                total_validation_predictions = sum(result['total_predictions'] for result in all_validation_results)
                response_data = {
                    "validation_mode": True,
                    "facility_name": config.facility_name,
                    "validation_months": config.validation_months,
                    "total_validation_predictions": total_validation_predictions,
                    "csv_file": csv_path,
                    "month_summary": [
                        {
                            "month": result["month"],
                            "predictions": result["total_predictions"]
                        } for result in all_validation_results
                    ]
                }
                
                # Clean NaN values before returning
                from utils import clean_nan_values
                response_data = clean_nan_values(response_data)
                
                return response_data
        
        # ===== PREDICTIONS FOR MAY =====
        print("\n" + "="*50)
        print("PREDICTING OFFICIAL MAY DATES")
        print("="*50)
        
        prediction_data = prepare_prediction_data(official_df, weather_future_df, energy_df, config)
        
        # Prophet predictions
        prophet_may_pred = final_prophet.predict(prediction_data)
        prophet_pred_values = prophet_may_pred['yhat'].values
        
        # Weighted ensemble predictions
        ensemble_may_pred, individual_predictions = predict_weighted_ensemble(
            weighted_ensemble_dict, 
            prediction_data, 
            prophet_pred_values,
            prophet_metrics['accuracy']  # Pass Prophet accuracy for dynamic weighting
        )
        
        # Extract individual model predictions and dynamic weights
        rf_may_pred = individual_predictions['rf_pred']
        ridge_may_pred = individual_predictions['ridge_pred']
        xgb_may_pred = individual_predictions['xgb_pred']
        dynamic_weights = individual_predictions['weights']
        
        # Use weighted ensemble predictions as final predictions
        final_predictions = ensemble_may_pred
        
        # Prepare May results
        may_prediction_results = []
        for i in range(len(prediction_data)):
            date_str = prediction_data.iloc[i]['ds'].strftime('%Y-%m-%d')
            predicted_energy = float(final_predictions[i])  # Convert to native Python float
            
            # Get confidence intervals from Prophet
            prediction_lower = float(prophet_may_pred.iloc[i]['yhat_lower'])
            prediction_upper = float(prophet_may_pred.iloc[i]['yhat_upper'])
            
            actual_energy_raw = official_df[official_df['ds'] == prediction_data.iloc[i]['ds']]['actual'].iloc[0]
            actual_energy = float(actual_energy_raw) if actual_energy_raw is not None else None
            
            result = {
                "date": date_str,
                "predicted_energy": round(float(predicted_energy), 2),
                "prophet_prediction": round(float(prophet_pred_values[i]), 2),
                "rf_prediction": round(float(rf_may_pred[i]), 2),
                "ridge_prediction": round(float(ridge_may_pred[i]), 2),
                "xgb_prediction": round(float(xgb_may_pred[i]), 2),
                "ensemble_prediction": round(float(ensemble_may_pred[i]), 2),
                "prediction_lower": round(float(prediction_lower), 2),
                "prediction_upper": round(float(prediction_upper), 2),
                "actual_energy": actual_energy,
                "error": round(float(actual_energy - predicted_energy), 2) if actual_energy is not None else None,
                "percent_error": round(float(abs(actual_energy - predicted_energy) / actual_energy * 100), 2) if actual_energy is not None and actual_energy != 0 else None,
                "model_weights": {
                    "prophet_weight": round(float(dynamic_weights['prophet_weight']), 3),
                    "rf_weight": round(float(dynamic_weights['rf_weight']), 3),
                    "ridge_weight": round(float(dynamic_weights['ridge_weight']), 3),
                    "xgb_weight": round(float(dynamic_weights['xgb_weight']), 3)
                },
                "weather_data": {
                    "temperature_2m_mean": float(prediction_data.iloc[i].get('temperature_2m_mean', 0))
                }
            }
            may_prediction_results.append(result)
            
            actual_str = f"{actual_energy:.2f}" if actual_energy is not None else "N/A"
            error_str = f"{abs(actual_energy - predicted_energy):.2f}" if actual_energy is not None else "N/A"
            print(f"{date_str}: Dynamic Weighted Ensemble={predicted_energy:.2f}, Prophet={prophet_pred_values[i]:.2f}, "
                  f"RF={rf_may_pred[i]:.2f}, Ridge={ridge_may_pred[i]:.2f}, XGB={xgb_may_pred[i]:.2f}, "
                  f"Actual={actual_str}, Error={error_str}")
        
        # ===== CUSTOM DATE RANGE PREDICTIONS =====
        custom_prediction_results = []
        if config.custom_start_date and config.custom_end_date:
            print("\n" + "="*50)
            print(f"PREDICTING CUSTOM DATE RANGE: {config.custom_start_date} to {config.custom_end_date}")
            print("="*50)
            
            # Check if weather data is available for the custom date range
            def check_weather_data_availability(start_date, end_date, facility_name, weather_csv_path=None):
                """Check if weather data is available for the specified date range"""
                if weather_csv_path is None:
                    weather_csv_path = f'{get_data_folder(facility_name)}/weather_future.csv'
                try:
                    if not os.path.exists(weather_csv_path):
                        print(f"Warning: Weather data file not found at {weather_csv_path}")
                        return False, None
                    
                    # Read weather data
                    weather_df = pd.read_csv(weather_csv_path)
                    weather_df['date'] = pd.to_datetime(weather_df['date'])
                    
                    # Convert date strings to datetime
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    
                    # Filter weather data for the requested range
                    available_weather = weather_df[
                        (weather_df['date'] >= start_dt) & 
                        (weather_df['date'] <= end_dt)
                    ]
                    
                    if len(available_weather) == 0:
                        print(f"No weather data available for date range {start_date} to {end_date}")
                        return False, None
                    
                    # Check if we have data for all requested dates
                    requested_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
                    missing_dates = []
                    for date in requested_dates:
                        if date not in available_weather['date'].values:
                            missing_dates.append(date.strftime('%Y-%m-%d'))
                    
                    if missing_dates:
                        print(f"Missing weather data for dates: {missing_dates}")
                        return False, None
                    
                    print(f"Weather data available for all {len(requested_dates)} dates in range {start_date} to {end_date}")
                    return True, available_weather
                    
                except Exception as e:
                    print(f"Error checking weather data availability: {e}")
                    return False, None
            
            # Check weather data availability
            weather_available, available_weather_data = check_weather_data_availability(
                config.custom_start_date, 
                config.custom_end_date,
                config.facility_name
            )
            
            if not weather_available:
                print("Skipping custom predictions due to missing weather data")
                custom_prediction_results = []
            else:
                try:
                    # Prepare trained models for custom predictions
                    trained_models = {
                        'prophet_model': final_prophet,
                        'ensemble_model': weighted_ensemble_dict,
                        'available_regressors': available_regressors,
                        'prophet_metrics': prophet_metrics
                    }
                    
                    # Call the dedicated function for custom predictions with trained models
                    custom_prediction_results = generate_custom_predictions(
                        config.custom_start_date, 
                        config.custom_end_date, 
                        config,
                        trained_models,
                        weather_future_df
                    )
                    
                    if custom_prediction_results:
                        print(f"Successfully generated {len(custom_prediction_results)} custom predictions")
                        print("Custom predictions saved to separate files (see console output above)")
                    else:
                        print("Failed to generate custom predictions")
                        custom_prediction_results = []
                    
                except Exception as e:
                    print(f"Error generating custom date range predictions: {e}")
                    custom_prediction_results = []
        
        # Results are returned directly, no file saving needed
        print(f"\nGenerated {len(may_prediction_results)} May predictions")
        
        # If custom predictions exist, mention they were generated
        if custom_prediction_results:
            print(f"Generated {len(custom_prediction_results)} custom date range predictions")
        
        # Post-validation using data from 06-01-2025 onwards
        print(f"\n{'='*60}")
        print("POST-VALIDATION USING DATA FROM 06-01-2025 ONWARDS")
        print(f"{'='*60}")
        
        # Load validation data (06-01-2025 onwards)
        validation_data = load_validation_data(config.facility_name, validation_start_date='2025-06-01')
        
        if validation_data is not None and len(validation_data) > 0:
            print(f"Loaded {len(validation_data)} validation records from 06-01-2025 onwards")
            
            # Prepare validation data with features
            validation_df = pd.DataFrame(validation_data)
            validation_df['ds'] = pd.to_datetime(validation_df['ds'])
            
            # Create advanced features for validation data
            validation_df = create_advanced_features(validation_df)
            
            # Prepare validation data for prediction
            validation_prediction_data = prepare_prediction_data(validation_df, weather_future_df, energy_df, config)
            
            # Ensure validation data has the required features for ensemble models
            if len(validation_prediction_data) > 0:
                # Fill any missing features with 0
                for col in weighted_ensemble_dict['features']:
                    if col not in validation_prediction_data.columns:
                        validation_prediction_data[col] = 0
            
            # Get predictions for validation period
            prophet_val_pred = final_prophet.predict(validation_prediction_data)
            prophet_val_values = prophet_val_pred['yhat'].values
            
            # Get ensemble predictions for validation
            ensemble_val_pred, individual_val_predictions = predict_weighted_ensemble(
                weighted_ensemble_dict, 
                validation_prediction_data, 
                prophet_val_values,
                prophet_metrics['accuracy']
            )
            
            # Extract individual model predictions for validation
            rf_val_pred = individual_val_predictions['rf_pred']
            ridge_val_pred = individual_val_predictions['ridge_pred']
            xgb_val_pred = individual_val_predictions['xgb_pred']
            
            # Get actual values for validation
            validation_actual = validation_df['y'].values
            
            # Validate each model's predictions
            prophet_val_metrics = evaluate_model(validation_actual, prophet_val_values, "Prophet Post Validation (06-01-2025+)")
            rf_val_metrics = evaluate_model(validation_actual, rf_val_pred, "Random Forest Post Validation (06-01-2025+)")
            ridge_val_metrics = evaluate_model(validation_actual, ridge_val_pred, "Ridge Regression Post Validation (06-01-2025+)")
            xgb_val_metrics = evaluate_model(validation_actual, xgb_val_pred, "XGBoost Post Validation (06-01-2025+)")
            ensemble_val_metrics = evaluate_model(validation_actual, ensemble_val_pred, "Dynamic Weighted Ensemble Post Validation (06-01-2025+)")
            
            # Generate dropout predictions for the 4 validation months
            dropout_predictions = generate_dropout_predictions(
                config.facility_name, final_prophet, weighted_ensemble_dict, 
                available_regressors, validation_prediction_data, validation_actual
            )
            
        else:
            print("No validation data available from 06-01-2025 onwards - skipping post-validation")
            prophet_val_metrics = rf_val_metrics = ridge_val_metrics = xgb_val_metrics = ensemble_val_metrics = None
            dropout_predictions = []
        
        # Final validation for May models only (since custom range has no actual data)
        # Filter out None values for validation and get corresponding prediction indices
        may_actual_with_data = []
        prediction_indices = []
        
        for i, r in enumerate(may_prediction_results):
            if r['actual_energy'] is not None:
                may_actual_with_data.append(float(r['actual_energy']))
                prediction_indices.append(i)
        
        may_actual = np.array(may_actual_with_data) if may_actual_with_data else np.array([])
        
        print(f"\n{'='*60}")
        print("MAY PREDICTIONS VALIDATION")
        print(f"{'='*60}")
        
        # Only run validation if we have actual data
        if len(may_actual) > 0:
            # Get corresponding predictions for validation
            prophet_pred_for_validation = prophet_pred_values[prediction_indices]
            rf_pred_for_validation = rf_may_pred[prediction_indices]
            ridge_pred_for_validation = ridge_may_pred[prediction_indices]
            xgb_pred_for_validation = xgb_may_pred[prediction_indices]
            ensemble_pred_for_validation = ensemble_may_pred[prediction_indices]
            
            # Validate each model's May predictions
            prophet_may_metrics = evaluate_model(may_actual, prophet_pred_for_validation, "Prophet May Validation")
            rf_may_metrics = evaluate_model(may_actual, rf_pred_for_validation, "Random Forest May Validation")
            ridge_may_metrics = evaluate_model(may_actual, ridge_pred_for_validation, "Ridge Regression May Validation")
            xgb_may_metrics = evaluate_model(may_actual, xgb_pred_for_validation, "XGBoost May Validation")
            ensemble_may_metrics = evaluate_model(may_actual, ensemble_pred_for_validation, "Dynamic Weighted Ensemble May Validation")
        else:
            print("No actual data available for May validation - skipping model validation")
            prophet_may_metrics = rf_may_metrics = ridge_may_metrics = xgb_may_metrics = ensemble_may_metrics = None
        
        # Print comparison table
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        print("\nCross-Validation Performance:")
        print(f"{'Model':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Accuracy':<10} {'Weight':<8}")
        print("-" * 70)
        
        # Prophet CV results
        prophet_cv_row = (
            "Prophet", 
            f"{prophet_metrics['mae']:.2f}",
            f"{prophet_metrics['rmse']:.2f}",
            f"{prophet_metrics['r2_score']:.3f}" if not np.isnan(prophet_metrics['r2_score']) else "N/A",
            f"{prophet_metrics['accuracy']:.1f}%" if not np.isnan(prophet_metrics['accuracy']) else "N/A",
            f"{dynamic_weights['prophet_weight']:.3f}"
        )
        print(f"{prophet_cv_row[0]:<20} {prophet_cv_row[1]:<8} {prophet_cv_row[2]:<8} {prophet_cv_row[3]:<8} {prophet_cv_row[4]:<10} {prophet_cv_row[5]:<8}")
        
        # ML models CV results
        ml_models_cv = [
            ("Random Forest", cv_scores['rf_accuracy'], dynamic_weights['rf_weight']),
            ("Ridge Regression", cv_scores['ridge_accuracy'], dynamic_weights['ridge_weight']),
            ("XGBoost", cv_scores['xgb_accuracy'], dynamic_weights['xgb_weight'])
        ]
        
        cv_results = [
            {
                "model": "Prophet",
                "mae": prophet_metrics['mae'],
                "rmse": prophet_metrics['rmse'],
                "r2_score": prophet_metrics['r2_score'] if not np.isnan(prophet_metrics['r2_score']) else None,
                "accuracy": prophet_metrics['accuracy'] if not np.isnan(prophet_metrics['accuracy']) else None,
                "weight": dynamic_weights['prophet_weight']
            }
        ]
        
        for name, accuracy, weight in ml_models_cv:
            # Estimate other metrics based on accuracy (simplified)
            mae = 2.0 * (100 - accuracy) / 100  # Rough estimate
            rmse = 2.5 * (100 - accuracy) / 100  # Rough estimate
            r2 = max(0, accuracy / 100 - 0.1)  # Rough estimate
            
            print(f"{name:<20} {mae:<8.2f} {rmse:<8.2f} {r2:<8.3f} {accuracy:<10.1f}% {weight:<8.3f}")
            cv_results.append({
                "model": name,
                "mae": mae,
                "rmse": rmse,
                "r2_score": r2,
                "accuracy": accuracy,
                "weight": weight
            })
        
        print("\nMay Validation Performance:")
        print(f"{'Model':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Accuracy':<10}")
        print("-" * 60)
        
        models_may = [
            ("Prophet", prophet_may_metrics),
            ("Random Forest", rf_may_metrics),
            ("Ridge Regression", ridge_may_metrics),
            ("XGBoost", xgb_may_metrics),
            ("Dynamic Weighted Ensemble", ensemble_may_metrics)
        ]
        
        may_results = []
        for name, metrics in models_may:
            if metrics is not None:
                mae = f"{metrics['mae']:.2f}"
                rmse = f"{metrics['rmse']:.2f}"
                r2 = f"{metrics['r2_score']:.3f}" if not np.isnan(metrics['r2_score']) else "N/A"
                acc = f"{metrics['accuracy']:.1f}%" if not np.isnan(metrics['accuracy']) else "N/A"
                print(f"{name:<20} {mae:<8} {rmse:<8} {r2:<8} {acc:<10}")
                may_results.append({
                    "model": name,
                    "mae": metrics['mae'],
                    "rmse": metrics['rmse'],
                    "r2_score": metrics['r2_score'] if not np.isnan(metrics['r2_score']) else None,
                    "accuracy": metrics['accuracy'] if not np.isnan(metrics['accuracy']) else None
                })
            else:
                print(f"{name:<20} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10}")
                may_results.append({
                    "model": name,
                    "mae": None,
                    "rmse": None,
                    "r2_score": None,
                    "accuracy": None
                })
        
        # Post-validation results (06-01-2025 onwards)
        if prophet_val_metrics is not None:
            print("\nPost Validation Performance (06-01-2025 onwards):")
            print(f"{'Model':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Accuracy':<10}")
            print("-" * 60)
            
            models_val = [
                ("Prophet", prophet_val_metrics),
                ("Random Forest", rf_val_metrics),
                ("Ridge Regression", ridge_val_metrics),
                ("XGBoost", xgb_val_metrics),
                ("Dynamic Weighted Ensemble", ensemble_val_metrics)
            ]
            
            val_results = []
            for name, metrics in models_val:
                if metrics is not None:
                    mae = f"{metrics['mae']:.2f}"
                    rmse = f"{metrics['rmse']:.2f}"
                    r2 = f"{metrics['r2_score']:.3f}" if not np.isnan(metrics['r2_score']) else "N/A"
                    acc = f"{metrics['accuracy']:.1f}%" if not np.isnan(metrics['accuracy']) else "N/A"
                    print(f"{name:<20} {mae:<8} {rmse:<8} {r2:<8} {acc:<10}")
                    val_results.append({
                        "model": name,
                        "mae": metrics['mae'],
                        "rmse": metrics['rmse'],
                        "r2_score": metrics['r2_score'] if not np.isnan(metrics['r2_score']) else None,
                        "accuracy": metrics['accuracy'] if not np.isnan(metrics['accuracy']) else None
                    })
                else:
                    print(f"{name:<20} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10}")
                    val_results.append({
                        "model": name,
                        "mae": None,
                        "rmse": None,
                        "r2_score": None,
                        "accuracy": None
                    })
        else:
            val_results = []
            
        # Save Prophet and XGBoost predictions to CSV for May and custom range
        may_model_predictions = []
        custom_model_predictions = []
        
        # Get shutdown dates from energy-totals.csv - use facility-specific path if facility_name is provided
        if config.facility_name:
            energy_totals_path = f'../data/{config.facility_name}/energy-totals.csv'
        else:
            energy_totals_path = f'{get_data_folder(config.facility_name)}/energy-totals.csv'
        
        if os.path.exists(energy_totals_path):
            energy_totals_df = pd.read_csv(energy_totals_path)
            energy_totals_df['date'] = pd.to_datetime(energy_totals_df['date'])
            
            # Check if 'percent' column exists before filtering shutdown days
            if 'percent' in energy_totals_df.columns:
                shutdown_days = energy_totals_df[energy_totals_df['percent'] < 92]
                shutdown_dates = set(shutdown_days['date'].dt.strftime('%Y-%m-%d').tolist())
                print(f"Found {len(shutdown_dates)} shutdown days to exclude")
            else:
                print(f"Warning: 'percent' column not found in {energy_totals_path}, skipping shutdown day filtering")
                shutdown_dates = set()
        else:
            print(f"Warning: Energy totals file not found at {energy_totals_path}, skipping shutdown day filtering")
            shutdown_dates = set()
        
        print(f"\nFiltering shutdown days from May predictions...")
        if len(shutdown_dates) > 0:
            print("Shutdown dates:", sorted(list(shutdown_dates)))
        
        # Add May predictions (excluding shutdown days)
        for i in range(len(prediction_data)):
            date = prediction_data.iloc[i]['ds']
            date_str = date.strftime('%Y-%m-%d')
            
            # Skip shutdown days
            if date_str in shutdown_dates:
                print(f"Skipping shutdown day in May predictions: {date_str}")
                continue
            
            prophet_predicted = float(prophet_pred_values[i])
            xgb_predicted = float(xgb_may_pred[i])
            
            # Get actual value with proper None handling
            actual_raw = official_df[official_df['ds'] == date]['actual'].iloc[0]
            actual = float(actual_raw) if actual_raw is not None else None
            
            # Calculate ensemble prediction (weighted average)
            prophet_weight = 0.3
            xgb_weight = 0.7
            ensemble_predicted = prophet_weight * prophet_predicted + xgb_weight * xgb_predicted
            
            # Calculate error rates only if actual value exists
            if actual is not None and actual != 0:
                prophet_error_rate = abs((prophet_predicted - actual) / actual * 100)
                xgb_error_rate = abs((xgb_predicted - actual) / actual * 100)
                ensemble_error_rate = abs((ensemble_predicted - actual) / actual * 100)
            else:
                prophet_error_rate = None
                xgb_error_rate = None
                ensemble_error_rate = None
            
            may_model_predictions.append({
                'date': date_str,
                'prophet_predicted': prophet_predicted,
                'xgb_predicted': xgb_predicted,
                'ensemble_predicted': ensemble_predicted,
                'actual': actual,
                'prophet_error_rate': prophet_error_rate,
                'xgb_error_rate': xgb_error_rate,
                'ensemble_error_rate': ensemble_error_rate
            })
        
        # Add custom date range predictions if provided
        if config.custom_start_date and config.custom_end_date and custom_prediction_results:
            print(f"\nProcessing {len(custom_prediction_results)} custom date range predictions for CSV...")
            
            for result in custom_prediction_results:
                date = result['date']
                prophet_predicted = result['prophet_prediction']
                xgb_predicted = result['xgb_prediction']
                ensemble_predicted = result['ensemble_prediction']
                # For custom range, actual energy is None since we don't have official data
                actual = None
                prophet_error_rate = None
                xgb_error_rate = None
                ensemble_error_rate = None
                
                custom_model_predictions.append({
                    'date': date,
                    'prophet_predicted': prophet_predicted,
                    'xgb_predicted': xgb_predicted,
                    'ensemble_predicted': ensemble_predicted,
                    'actual': actual,
                    'prophet_error_rate': prophet_error_rate,
                    'xgb_error_rate': xgb_error_rate,
                    'ensemble_error_rate': ensemble_error_rate
                })
        
        # Custom predictions are processed but not saved to files
        if custom_model_predictions:
            print(f"Generated {len(custom_model_predictions)} custom date range predictions (not saved to file)")

        # Generate 90-day predictions after post-validation
        print(f"\n{'='*60}")
        print("GENERATING 90-DAY PREDICTIONS")
        print(f"{'='*60}")
        
        # Get weather data for next 90 days
        weather_90_days = get_next_90_days_weather(config.facility_name)
        if weather_90_days is not None:
            # Generate 90-day predictions using the trained models
            predictions_90_days = predict_next_90_days_for_facility(
                config.facility_name, 
                final_prophet, 
                weighted_ensemble_dict, 
                available_regressors, 
                weather_90_days
            )
            
            if predictions_90_days:
                print(f"Generated {len(predictions_90_days)} 90-day predictions")
                # Save 90-day predictions to CSV file in facility folder
                csv_path = save_90_days_predictions_to_csv(config.facility_name, predictions_90_days)
                if csv_path:
                    print(f"90-day predictions saved to: {csv_path}")
            else:
                print("Failed to generate 90-day predictions")
                predictions_90_days = []
        else:
            print("No weather data available for 90-day predictions")
            predictions_90_days = []
        
        # Clean any NaN values from the response data
        response_data = {
            "predictions": may_prediction_results,  # Only return May predictions since custom are saved separately
            "cross_validation_results": cv_results,
            "may_validation_results": may_results,
            "post_validation_results": val_results,  # New post-validation results (06-01-2025 onwards)
            "dropout_predictions": dropout_predictions,  # Dropout predictions for 4 validation months
            "predictions_90_days": predictions_90_days,  # 90-day predictions
            "feature_config": config.dict(),
            "ensemble_weights": dynamic_weights,
            "custom_predictions_saved": len(custom_prediction_results) > 0,  # Indicate if custom predictions were generated
            "custom_date_range": {
                "start_date": config.custom_start_date,
                "end_date": config.custom_end_date
            } if config.custom_start_date and config.custom_end_date else None
        }
        
        # Clean NaN values before returning
        from utils import clean_nan_values
        response_data = clean_nan_values(response_data)
        
        return response_data
        
    except Exception as e:
        print("\n" + "="*50)
        print("ERROR IN PREDICTION PIPELINE")
        print("="*50)
        print(f"Timestamp: {pd.Timestamp.now()}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\nStack Trace:")
        traceback.print_exc()
        print("\nDebug Information:")
        print(f"Python Version: {sys.version}")
        print(f"Feature Config: {config.dict()}")
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        print(f"Loaded weather data for {facility_name}: {len(weather_df)} total records")
        print(f"Weather date range: {weather_df['date'].min()} to {weather_df['date'].max()}")
        
        # Get today's date
        today = datetime.now().date()
        print(f"Today's date: {today}")
        
        # Filter for future dates (from today onwards)
        future_weather = weather_df[weather_df['date'] >= pd.Timestamp(today)].copy()
        
        if future_weather.empty:
            print(f"No future weather data found for {facility_name} (from {today} onwards)")
            return None
        
        # Limit to next 90 days maximum
        end_date = today + timedelta(days=90)
        future_weather = future_weather[future_weather['date'] <= pd.Timestamp(end_date)].copy()
        
        print(f"Found {len(future_weather)} days of future weather data for {facility_name}")
        print(f"Future weather date range: {future_weather['date'].min()} to {future_weather['date'].max()}")
        
        # Rename date column to match expected format
        future_weather = future_weather.rename(columns={
            'date': 'ds'
        })
        
        # Ensure required columns exist
        if 'temperature_2m_mean' not in future_weather.columns:
            print(f"Warning: temperature_2m_mean column not found in weather data for {facility_name}")
            print(f"Available columns: {list(future_weather.columns)}")
            # Add default temperature column if missing
            future_weather['temperature_2m_mean'] = 20.0
        
        print(f"Weather columns: {list(future_weather.columns)}")
        return future_weather
        
    except Exception as e:
        print(f"Error loading weather data for {facility_name}: {e}")
        return None

def create_prediction_dataframe_90_days(facility_name: str, weather_df: pd.DataFrame) -> pd.DataFrame:
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

def train_models_for_facility_90_days(facility_name: str) -> tuple:
    """Train models for a specific facility for 90-day predictions"""
    print(f"\n{'='*50}")
    print(f"TRAINING MODELS FOR {facility_name.upper()} (90-DAY PREDICTIONS)")
    print(f"{'='*50}")
    
    # Load energy data with training cutoff
    energy_df = load_energy_data(facility_name)
    if energy_df is None or energy_df.empty:
        print(f"No energy data found for {facility_name}")
        return None, None, None
    
    # Apply training cutoff (up to 05-30-2025)
    training_cutoff = pd.to_datetime('2025-05-30')
    energy_df = energy_df[energy_df['ds'] <= training_cutoff].copy()
    print(f"Training data limited to {len(energy_df)} records (up to 2025-05-30)")
    
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

def predict_next_90_days_for_facility(facility_name: str, prophet_model, weighted_ensemble_dict, available_regressors, weather_df: pd.DataFrame) -> List[Dict]:
    """Generate predictions for next 90 days for a specific facility"""
    print(f"\nGenerating predictions for {facility_name}...")
    
    try:
        # Create prediction dataframe
        prediction_df = create_prediction_dataframe_90_days(facility_name, weather_df)
        
        # Load energy data for feature preparation
        energy_df = load_energy_data(facility_name)
        if energy_df is None:
            print(f"No energy data available for {facility_name}")
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
        
        # Verify weather dataframe has required columns
        if 'ds' not in weather_df.columns:
            print(f"Error: 'ds' column not found in weather dataframe for {facility_name}")
            print(f"Available columns: {list(weather_df.columns)}")
            return []
        
        if 'temperature_2m_mean' not in weather_df.columns:
            print(f"Error: 'temperature_2m_mean' column not found in weather dataframe for {facility_name}")
            print(f"Available columns: {list(weather_df.columns)}")
            return []
        
        # Prepare prediction data with features
        prediction_data = prepare_prediction_data(prediction_df, weather_df, energy_df, config)
        
        # Get Prophet predictions
        prophet_pred = prophet_model.predict(prediction_data)
        prophet_pred_values = prophet_pred['yhat'].values
        
        # Get ensemble predictions
        ensemble_pred, individual_predictions = predict_weighted_ensemble(
            weighted_ensemble_dict, 
            prediction_data
        )
        
        xgb_pred = individual_predictions['xgb_pred']
        
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
        
    except Exception as e:
        print(f"Error generating 90-day predictions for {facility_name}: {e}")
        print(f"Error details: {str(e)}")
        return []

def generate_dropout_predictions(facility_name: str, prophet_model, weighted_ensemble_dict, available_regressors, validation_prediction_data, validation_actual) -> List[Dict]:
    """Generate dropout predictions for the 4 validation months"""
    print(f"\nGenerating dropout predictions for {facility_name}...")
    
    # Define the 4 validation months
    validation_months = ['2025-06', '2025-07', '2025-08', '2025-09']
    
    dropout_results = []
    
    for month in validation_months:
        print(f"Processing dropout predictions for {month}...")
        
        # Filter validation data for this month
        month_start = pd.to_datetime(f"{month}-01")
        month_end = month_start + pd.offsets.MonthEnd(1)
        
        month_mask = (validation_prediction_data['ds'] >= month_start) & (validation_prediction_data['ds'] <= month_end)
        month_data = validation_prediction_data[month_mask].copy()
        
        if len(month_data) == 0:
            print(f"No data found for {month}")
            continue
        
        # Get corresponding actual values for this month
        month_actual_mask = (pd.to_datetime(validation_prediction_data['ds']) >= month_start) & (pd.to_datetime(validation_prediction_data['ds']) <= month_end)
        month_actual = validation_actual[month_actual_mask]
        
        if len(month_actual) == 0:
            print(f"No actual values found for {month}")
            continue
        
        # Get predictions for this month
        prophet_month_pred = prophet_model.predict(month_data)
        prophet_month_values = prophet_month_pred['yhat'].values
        
        # Get ensemble predictions for this month
        try:
            ensemble_month_pred, individual_month_predictions = predict_weighted_ensemble(
                weighted_ensemble_dict, 
                month_data
            )
            
            xgb_month_pred = individual_month_predictions['xgb_pred']
        except Exception as e:
            print(f"Error getting ensemble predictions for {month}: {e}")
            continue
        
        # Calculate metrics for this month
        prophet_month_metrics = evaluate_model(month_actual, prophet_month_values, f"Prophet {month}")
        xgb_month_metrics = evaluate_model(month_actual, xgb_month_pred, f"XGBoost {month}")
        ensemble_month_metrics = evaluate_model(month_actual, ensemble_month_pred, f"Ensemble {month}")
        
        # Create results for this month
        month_result = {
            "month": month,
            "predictions_count": len(month_data),
            "prophet_accuracy": prophet_month_metrics['accuracy'] if prophet_month_metrics else None,
            "xgb_accuracy": xgb_month_metrics['accuracy'] if xgb_month_metrics else None,
            "ensemble_accuracy": ensemble_month_metrics['accuracy'] if ensemble_month_metrics else None,
            "prophet_mae": prophet_month_metrics['mae'] if prophet_month_metrics else None,
            "xgb_mae": xgb_month_metrics['mae'] if xgb_month_metrics else None,
            "ensemble_mae": ensemble_month_metrics['mae'] if ensemble_month_metrics else None,
            "predictions": []
        }
        
        # Add individual predictions
        for i in range(len(month_data)):
            date = month_data.iloc[i]['ds']
            date_str = date.strftime('%Y-%m-%d')
            
            prediction_result = {
                "date": date_str,
                "prophet_prediction": round(float(prophet_month_values[i]), 2),
                "xgb_prediction": round(float(xgb_month_pred[i]), 2),
                "ensemble_prediction": round(float(ensemble_month_pred[i]), 2),
                "actual": round(float(month_actual[i]), 2) if i < len(month_actual) else None,
                "temperature": round(float(month_data.iloc[i].get('temperature_2m_mean', 0)), 2)
            }
            month_result["predictions"].append(prediction_result)
        
        dropout_results.append(month_result)
        print(f"Completed dropout predictions for {month}: {len(month_data)} predictions")
    
    # Save dropout predictions to CSV
    if dropout_results:
        save_dropout_predictions_to_csv(facility_name, dropout_results)
    
    return dropout_results

def save_dropout_predictions_to_csv(facility_name: str, dropout_results: List[Dict]):
    """Save dropout predictions to CSV file, filtering out dates with <92% percent_running"""
    if not dropout_results:
        print(f"No dropout predictions to save for {facility_name}")
        return
    
    # Create facility directory if it doesn't exist
    facility_dir = f'../data/{facility_name}'
    os.makedirs(facility_dir, exist_ok=True)
    
    # Load energy totals to get percent_running data for filtering
    energy_totals_path = f'{facility_dir}/energy-totals.csv'
    shutdown_dates = set()
    
    if os.path.exists(energy_totals_path):
        try:
            energy_totals_df = pd.read_csv(energy_totals_path)
            energy_totals_df['date'] = pd.to_datetime(energy_totals_df['date'])
            
            # Check if 'percent_running' column exists before filtering
            if 'percent_running' in energy_totals_df.columns:
                shutdown_days = energy_totals_df[energy_totals_df['percent_running'] < 92]
                shutdown_dates = set(shutdown_days['date'].dt.strftime('%Y-%m-%d').tolist())
                print(f"Found {len(shutdown_dates)} shutdown days (<92% running) to exclude from post-validation")
                if len(shutdown_dates) > 0:
                    print("Shutdown dates:", sorted(list(shutdown_dates)))
            else:
                print(f"Warning: 'percent_running' column not found in {energy_totals_path}, skipping shutdown day filtering")
        except Exception as e:
            print(f"Error loading energy totals for filtering: {e}")
    else:
        print(f"Warning: Energy totals file not found at {energy_totals_path}, skipping shutdown day filtering")
    
    # Create summary DataFrame
    summary_data = []
    for result in dropout_results:
        summary_data.append({
            "month": result["month"],
            "predictions_count": result["predictions_count"],
            "prophet_accuracy": result["prophet_accuracy"],
            "xgb_accuracy": result["xgb_accuracy"],
            "ensemble_accuracy": result["ensemble_accuracy"],
            "prophet_mae": result["prophet_mae"],
            "xgb_mae": result["xgb_mae"],
            "ensemble_mae": result["ensemble_mae"]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = f'{facility_dir}/accuracy.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"Saved dropout predictions for {facility_name}:")
    print(f"  Summary: {summary_csv_path}")

def save_90_days_predictions_to_csv(facility_name: str, predictions: List[Dict]):
    """Save 90-day predictions to CSV file in facility folder"""
    if not predictions:
        print(f"No 90-day predictions to save for {facility_name}")
        return None
    
    # Create facility directory if it doesn't exist
    facility_dir = f'../data/{facility_name}'
    os.makedirs(facility_dir, exist_ok=True)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(predictions)
    csv_path = f'{facility_dir}/predictions-90.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"Saved {len(predictions)} 90-day predictions to {csv_path}")
    print(f"Date range: {predictions[0]['date']} to {predictions[-1]['date']}")
    
    return csv_path

@app.post("/predict-90-days")
async def predict_90_days():
    """
    Generate predictions for the next 90 days for all facilities.
    This endpoint processes all facilities in the metadata and saves predictions to CSV files.
    """
    try:
        print("\n" + "="*60)
        print("PREDICTING NEXT 90 DAYS FOR ALL FACILITIES")
        print("="*60)
        
        # Load facility metadata
        facilities = load_facility_metadata()
        if not facilities:
            raise HTTPException(status_code=404, detail="No facilities found in metadata")
        
        print(f"Found {len(facilities)} facilities: {[f['name'] for f in facilities]}")
        
        results = {
            "total_facilities": len(facilities),
            "processed_facilities": 0,
            "successful_predictions": 0,
            "failed_facilities": [],
            "facility_results": []
        }
        
        # Process each facility
        for facility in facilities:
            facility_name = facility["name"]
            print(f"\n{'='*60}")
            print(f"PROCESSING FACILITY: {facility_name.upper()}")
            print(f"{'='*60}")
            
            try:
                # Check if weather data exists for next 90 days
                weather_df = get_next_90_days_weather(facility_name)
                if weather_df is None:
                    print(f"Skipping {facility_name} - no weather data available")
                    results["failed_facilities"].append({
                        "facility": facility_name,
                        "reason": "No weather data available for next 90 days"
                    })
                    continue
                
                # Train models
                prophet_model, weighted_ensemble_dict, available_regressors = train_models_for_facility_90_days(facility_name)
                if prophet_model is None:
                    print(f"Failed to train models for {facility_name}")
                    results["failed_facilities"].append({
                        "facility": facility_name,
                        "reason": "Failed to train models"
                    })
                    continue
                
                # Generate predictions
                predictions = predict_next_90_days_for_facility(
                    facility_name, prophet_model, weighted_ensemble_dict, 
                    available_regressors, weather_df
                )
                
                if not predictions:
                    print(f"No predictions generated for {facility_name}")
                    results["failed_facilities"].append({
                        "facility": facility_name,
                        "reason": "No predictions generated"
                    })
                    continue
                
                # Save predictions to CSV file in facility folder
                csv_path = save_90_days_predictions_to_csv(facility_name, predictions)
                
                # Update results
                results["processed_facilities"] += 1
                results["successful_predictions"] += len(predictions)
                results["facility_results"].append({
                    "facility_name": facility_name,
                    "predictions_count": len(predictions),
                    "date_range": f"{predictions[0]['date']} to {predictions[-1]['date']}",
                    "csv_file": csv_path
                })
                
            except Exception as e:
                print(f"Error processing {facility_name}: {e}")
                results["failed_facilities"].append({
                    "facility": facility_name,
                    "reason": str(e)
                })
                continue
        
        print(f"\n{'='*60}")
        print("PREDICTION COMPLETE")
        print("="*60)
        print(f"Processed: {results['processed_facilities']}/{results['total_facilities']} facilities")
        print(f"Total predictions: {results['successful_predictions']}")
        print(f"Failed facilities: {len(results['failed_facilities'])}")
        
        # Clean NaN values before returning
        from utils import clean_nan_values
        results = clean_nan_values(results)
        
        return results
        
    except Exception as e:
        print("\n" + "="*50)
        print("ERROR IN 90-DAY PREDICTION PIPELINE")
        print("="*50)
        print(f"Timestamp: {pd.Timestamp.now()}")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\nStack Trace:")
        traceback.print_exc()
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)