import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from config import DEFAULT_RF_PARAMS, DEFAULT_RIDGE_PARAMS, DEFAULT_XGB_PARAMS, DEFAULT_PROPHET_PARAMS, PROPHET_CUSTOM_SEASONALITIES
from feature_engineering import get_prophet_regressors

def create_optimized_prophet_model(config):
    """Create Prophet model with optimized built-in seasonalities and improved parameters"""
    model = Prophet(
        # Use improved parameters from config
        yearly_seasonality=DEFAULT_PROPHET_PARAMS['yearly_seasonality'],
        weekly_seasonality=DEFAULT_PROPHET_PARAMS['weekly_seasonality'],
        daily_seasonality=DEFAULT_PROPHET_PARAMS['daily_seasonality'],
        
        # Improved flexibility and regularization parameters
        changepoint_prior_scale=DEFAULT_PROPHET_PARAMS['changepoint_prior_scale'],
        seasonality_prior_scale=DEFAULT_PROPHET_PARAMS['seasonality_prior_scale'],
        holidays_prior_scale=DEFAULT_PROPHET_PARAMS['holidays_prior_scale'],
        
        # Seasonality mode - additive works better for energy data
        seasonality_mode=DEFAULT_PROPHET_PARAMS['seasonality_mode'],
        
        # Growth trend
        growth=DEFAULT_PROPHET_PARAMS['growth'],
        
        # Improved changepoint optimization
        n_changepoints=DEFAULT_PROPHET_PARAMS['n_changepoints'],
        changepoint_range=DEFAULT_PROPHET_PARAMS['changepoint_range'],
        
        # Uncertainty intervals
        interval_width=DEFAULT_PROPHET_PARAMS['interval_width'],
        
        # No MCMC sampling for speed
        mcmc_samples=DEFAULT_PROPHET_PARAMS['mcmc_samples']
    )
    
    # Add improved quarterly seasonality
    quarterly_config = PROPHET_CUSTOM_SEASONALITIES['quarterly']
    model.add_seasonality(
        name='quarterly',
        period=quarterly_config['period'],
        fourier_order=quarterly_config['fourier_order'],
        prior_scale=quarterly_config['prior_scale']
    )
    
    # Add improved bi-weekly pattern common in energy consumption
    biweekly_config = PROPHET_CUSTOM_SEASONALITIES['biweekly']
    model.add_seasonality(
        name='biweekly',
        period=biweekly_config['period'],
        fourier_order=biweekly_config['fourier_order'],
        prior_scale=biweekly_config['prior_scale']
    )
    
    # Add new monthly seasonality for better monthly patterns
    monthly_config = PROPHET_CUSTOM_SEASONALITIES['monthly']
    model.add_seasonality(
        name='monthly',
        period=monthly_config['period'],
        fourier_order=monthly_config['fourier_order'],
        prior_scale=monthly_config['prior_scale']
    )
    
    # Add new weekly seasonality for better weekly patterns
    weekly_config = PROPHET_CUSTOM_SEASONALITIES['weekly']
    model.add_seasonality(
        name='weekly',
        period=weekly_config['period'],
        fourier_order=weekly_config['fourier_order'],
        prior_scale=weekly_config['prior_scale']
    )
    
    return model

def create_ensemble_model(train_df, config):
    """Create ensemble model combining Prophet with ML models"""
    
    # Prepare features for ML models
    feature_cols = [
        'year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter',
        'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos',
        'is_weekend', 'is_month_start', 'is_month_end',
        'shutdown',  # New shutdown feature
        'days_elapsed'  # New days since commission feature
    ]
    
    # Add weather features if available
    weather_features = [
        'temperature_2m_mean', 'temp_squared', 'temp_cubed',
        'temp_rolling_mean_3d', 'temp_rolling_mean_7d', 'temp_rolling_mean_14d',
        'temp_rolling_std_7d', 'temp_rolling_std_14d',
        'temp_diff_1d', 'temp_diff_7d'
    ]
    
    # Add lag and rolling features
    lag_features = [
        'y_lag1', 'y_lag2', 'y_lag3', 'y_lag7', 'y_lag14', 'y_lag30',
        'y_rolling_mean_3', 'y_rolling_mean_7', 'y_rolling_mean_14', 'y_rolling_mean_30',
        'y_rolling_std_7', 'y_rolling_std_14',
        'y_rolling_max_7', 'y_rolling_min_7',
        'y_diff1', 'y_diff7', 'y_pct_change1', 'y_pct_change7'
    ]
    
    available_features = []
    for feature_group in [feature_cols, weather_features, lag_features]:
        available_features.extend([col for col in feature_group if col in train_df.columns])
    
    # Remove duplicates
    available_features = list(set(available_features))
    
    print(f"Using {len(available_features)} features for ensemble models")
    print(f"Features include shutdown: {'shutdown' in available_features}")
    print(f"Features include days_elapsed: {'days_elapsed' in available_features}")
    
    # Prepare data
    X_train = train_df[available_features].fillna(0)
    y_train = train_df['y']
    
    # Replace infinities with NaN then fill
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())
    
    # Scale features using RobustScaler which is less sensitive to outliers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create models
    rf_model = RandomForestRegressor(**DEFAULT_RF_PARAMS)
    ridge_model = Ridge(**DEFAULT_RIDGE_PARAMS)
    xgb_model = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS)
    
    # Train models
    rf_model.fit(X_train, y_train)
    ridge_model.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train, y_train)
    
    return {
        'rf_model': rf_model,
        'ridge_model': ridge_model,
        'xgb_model': xgb_model,
        'scaler': scaler,
        'features': available_features
    }

def create_weighted_ensemble_model(train_df, config, cv_splits=None):
    """
    Create ensemble model with balanced weights for all models.
    
    Args:
        train_df: Training dataframe
        config: Feature configuration
        cv_splits: Pre-computed cross-validation splits (optional)
    
    Returns:
        Dictionary containing models, scaler, features, and balanced model weights
    """
    from evaluation import evaluate_model
    
    # Prepare features for ML models (same as create_ensemble_model)
    feature_cols = [
        'year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter',
        'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos',
        'is_weekend', 'is_month_start', 'is_month_end',
        'shutdown',  # New shutdown feature
        'days_elapsed'  # New days since commission feature
    ]
    
    # Add weather features if available
    weather_features = [
        'temperature_2m_mean', 'temp_squared', 'temp_cubed',
        'temp_rolling_mean_3d', 'temp_rolling_mean_7d', 'temp_rolling_mean_14d',
        'temp_rolling_std_7d', 'temp_rolling_std_14d',
        'temp_diff_1d', 'temp_diff_7d'
    ]
    
    # Add lag and rolling features
    lag_features = [
        'y_lag1', 'y_lag2', 'y_lag3', 'y_lag7', 'y_lag14', 'y_lag30',
        'y_rolling_mean_3', 'y_rolling_mean_7', 'y_rolling_mean_14', 'y_rolling_mean_30',
        'y_rolling_std_7', 'y_rolling_std_14',
        'y_rolling_max_7', 'y_rolling_min_7',
        'y_diff1', 'y_diff7', 'y_pct_change1', 'y_pct_change7'
    ]
    
    available_features = []
    for feature_group in [feature_cols, weather_features, lag_features]:
        available_features.extend([col for col in feature_group if col in train_df.columns])
    
    # Remove duplicates
    available_features = list(set(available_features))
    
    print(f"Using {len(available_features)} features for balanced ensemble models")
    
    # Create cross-validation splits if not provided
    if cv_splits is None:
        cv_splits = create_time_series_splits(train_df, n_splits=5)
    
    # Initialize lists to store cross-validation scores
    rf_cv_scores, ridge_cv_scores, xgb_cv_scores = [], [], []
    
    print("Performing cross-validation to evaluate model performance...")
    
    # Perform cross-validation for each model
    for i, (cv_train_df, cv_test_df) in enumerate(cv_splits):
        print(f"CV Fold {i+1}/{len(cv_splits)}")
        
        # Prepare features for this fold
        X_cv_train = cv_train_df[available_features].fillna(0)
        X_cv_train = X_cv_train.replace([np.inf, -np.inf], np.nan)
        X_cv_train = X_cv_train.fillna(X_cv_train.median())
        
        X_cv_test = cv_test_df[available_features].fillna(0)
        X_cv_test = X_cv_test.replace([np.inf, -np.inf], np.nan)
        X_cv_test = X_cv_test.fillna(X_cv_test.median())
        
        y_cv_train = cv_train_df['y']
        y_cv_test = cv_test_df['y']  # Keep in original scale for evaluation
        
        # Scale features
        scaler_cv = RobustScaler()
        X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
        X_cv_test_scaled = scaler_cv.transform(X_cv_test)
        
        # Train models on this fold
        rf_model_cv = RandomForestRegressor(**DEFAULT_RF_PARAMS)
        ridge_model_cv = Ridge(**DEFAULT_RIDGE_PARAMS)
        xgb_model_cv = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS)
        
        rf_model_cv.fit(X_cv_train, y_cv_train)
        ridge_model_cv.fit(X_cv_train_scaled, y_cv_train)
        xgb_model_cv.fit(X_cv_train, y_cv_train)
        
        # Make predictions
        rf_pred = rf_model_cv.predict(X_cv_test)
        ridge_pred = ridge_model_cv.predict(X_cv_test_scaled)
        xgb_pred = xgb_model_cv.predict(X_cv_test)
        
        # Evaluate models
        rf_metrics = evaluate_model(y_cv_test, rf_pred, f"RF Fold {i+1}")
        ridge_metrics = evaluate_model(y_cv_test, ridge_pred, f"Ridge Fold {i+1}")
        xgb_metrics = evaluate_model(y_cv_test, xgb_pred, f"XGB Fold {i+1}")
        
        # Store accuracy scores (use 100 - MAPE as accuracy)
        rf_cv_scores.append(100 - rf_metrics.get('mape', 0))
        ridge_cv_scores.append(100 - ridge_metrics.get('mape', 0))
        xgb_cv_scores.append(100 - xgb_metrics.get('mape', 0))
    
    # Calculate average accuracy scores across all folds
    rf_avg_accuracy = np.mean(rf_cv_scores)
    ridge_avg_accuracy = np.mean(ridge_cv_scores)
    xgb_avg_accuracy = np.mean(xgb_cv_scores)
    
    print(f"\nCross-validation accuracy scores:")
    print(f"Random Forest: {rf_avg_accuracy:.2f}%")
    print(f"Ridge Regression: {ridge_avg_accuracy:.2f}%")
    print(f"XGBoost: {xgb_avg_accuracy:.2f}%")
    
    # Calculate accuracy-based weights for ML models
    ml_accuracies = [rf_avg_accuracy, ridge_avg_accuracy, xgb_avg_accuracy]
    total_ml_accuracy = sum(ml_accuracies)
    
    # Normalize weights to sum to 1 for ML models
    rf_weight = rf_avg_accuracy / total_ml_accuracy
    ridge_weight = ridge_avg_accuracy / total_ml_accuracy
    xgb_weight = xgb_avg_accuracy / total_ml_accuracy
    
    print(f"\nAccuracy-based model weights:")
    print(f"Random Forest: {rf_weight:.3f} (accuracy: {rf_avg_accuracy:.2f}%)")
    print(f"Ridge Regression: {ridge_weight:.3f} (accuracy: {ridge_avg_accuracy:.2f}%)")
    print(f"XGBoost: {xgb_weight:.3f} (accuracy: {xgb_avg_accuracy:.2f}%)")
    print(f"Note: Prophet weight will be calculated based on its accuracy when included")
    
    # Train final models on full dataset
    X_train = train_df[available_features].fillna(0)
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(X_train.median())
    
    y_train = train_df['y']
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create and train final models
    rf_model = RandomForestRegressor(**DEFAULT_RF_PARAMS)
    ridge_model = Ridge(**DEFAULT_RIDGE_PARAMS)
    xgb_model = xgb.XGBRegressor(**DEFAULT_XGB_PARAMS)
    
    rf_model.fit(X_train, y_train)
    ridge_model.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train, y_train)
    
    return {
        'rf_model': rf_model,
        'ridge_model': ridge_model,
        'xgb_model': xgb_model,
        'scaler': scaler,
        'features': available_features,
        'weights': {
            'rf_weight': rf_weight,
            'ridge_weight': ridge_weight,
            'xgb_weight': xgb_weight
        },
        'cv_scores': {
            'rf_accuracy': rf_avg_accuracy,
            'ridge_accuracy': ridge_avg_accuracy,
            'xgb_accuracy': xgb_avg_accuracy
        }
    }

def predict_weighted_ensemble(ensemble_dict, features_df, prophet_predictions=None, prophet_accuracy=None):
    """
    Make accuracy-based ensemble predictions using weights proportional to each model's performance.
    
    Args:
        ensemble_dict: Dictionary containing models, scaler, features, and weights
        features_df: DataFrame with features for prediction
        prophet_predictions: Optional Prophet predictions to include in ensemble
        prophet_accuracy: Optional Prophet accuracy score for weighting
    
    Returns:
        Accuracy-based ensemble predictions
    """
    # Prepare features
    features = features_df[ensemble_dict['features']].fillna(0)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(features.median())
    
    # Get individual model predictions
    rf_pred = ensemble_dict['rf_model'].predict(features)
    ridge_pred = ensemble_dict['ridge_model'].predict(ensemble_dict['scaler'].transform(features))
    xgb_pred = ensemble_dict['xgb_model'].predict(features)
    
    # Get accuracy scores from ensemble_dict
    rf_accuracy = ensemble_dict['cv_scores']['rf_accuracy']
    ridge_accuracy = ensemble_dict['cv_scores']['ridge_accuracy']
    xgb_accuracy = ensemble_dict['cv_scores']['xgb_accuracy']
    
    # Calculate accuracy-based ensemble prediction
    if prophet_predictions is not None and prophet_accuracy is not None:
        # All 4 models with accuracy-based weights
        accuracies = [prophet_accuracy, rf_accuracy, ridge_accuracy, xgb_accuracy]
        total_accuracy = sum(accuracies)
        
        # Normalize weights to sum to 1
        prophet_weight = prophet_accuracy / total_accuracy
        rf_weight = rf_accuracy / total_accuracy
        ridge_weight = ridge_accuracy / total_accuracy
        xgb_weight = xgb_accuracy / total_accuracy
        
        print(f"\nAccuracy-based ensemble weights (4 models):")
        print(f"Prophet: {prophet_weight:.3f} (accuracy: {prophet_accuracy:.2f}%)")
        print(f"Random Forest: {rf_weight:.3f} (accuracy: {rf_accuracy:.2f}%)")
        print(f"Ridge Regression: {ridge_weight:.3f} (accuracy: {ridge_accuracy:.2f}%)")
        print(f"XGBoost: {xgb_weight:.3f} (accuracy: {xgb_accuracy:.2f}%)")
        
        ensemble_pred = (prophet_predictions * prophet_weight + 
                        rf_pred * rf_weight + 
                        ridge_pred * ridge_weight + 
                        xgb_pred * xgb_weight)
        
        # Update weights dictionary for return
        accuracy_weights_dict = {
            'prophet_weight': prophet_weight,
            'rf_weight': rf_weight,
            'ridge_weight': ridge_weight,
            'xgb_weight': xgb_weight
        }
        
    else:
        # Only ML models with accuracy-based weights
        ml_accuracies = [rf_accuracy, ridge_accuracy, xgb_accuracy]
        total_ml_accuracy = sum(ml_accuracies)
        
        # Normalize weights to sum to 1
        rf_weight = rf_accuracy / total_ml_accuracy
        ridge_weight = ridge_accuracy / total_ml_accuracy
        xgb_weight = xgb_accuracy / total_ml_accuracy
        
        print(f"\nAccuracy-based ensemble weights (3 ML models):")
        print(f"Random Forest: {rf_weight:.3f} (accuracy: {rf_accuracy:.2f}%)")
        print(f"Ridge Regression: {ridge_weight:.3f} (accuracy: {ridge_accuracy:.2f}%)")
        print(f"XGBoost: {xgb_weight:.3f} (accuracy: {xgb_accuracy:.2f}%)")
        
        ensemble_pred = (rf_pred * rf_weight + 
                        ridge_pred * ridge_weight + 
                        xgb_pred * xgb_weight)
        
        accuracy_weights_dict = {
            'rf_weight': rf_weight,
            'ridge_weight': ridge_weight,
            'xgb_weight': xgb_weight
        }
    
    return ensemble_pred, {
        'rf_pred': rf_pred,
        'ridge_pred': ridge_pred,
        'xgb_pred': xgb_pred,
        'weights': accuracy_weights_dict
    }

def create_time_series_splits(df, n_splits=5):
    """Create time series cross-validation splits"""
    df_sorted = df.sort_values('ds').reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    splits = []
    for train_idx, test_idx in tscv.split(df_sorted):
        train_df = df_sorted.iloc[train_idx].copy()
        test_df = df_sorted.iloc[test_idx].copy()
        splits.append((train_df, test_df))
    
    print(f"Created {n_splits} time series splits")
    return splits