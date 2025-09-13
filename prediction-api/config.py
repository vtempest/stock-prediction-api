import os

# Configuration constants
def get_data_folder(facility_name: str) -> str:
    """Get the data folder path for a specific facility"""
    return f'../data/{facility_name}'


# Model parameters
DEFAULT_RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

DEFAULT_RIDGE_PARAMS = {
    'alpha': 1.0
}

DEFAULT_XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'alpha': 0.05,
    'lambda_': 1.0,
    'tree_method': 'hist',
    'grow_policy': 'depthwise',
    'max_leaves': 0,
    'gamma': 0.1,
    'min_child_weight': 5,
    'random_state': 42,
    'n_jobs': -1,
    'enable_categorical': False,
    'predictor': 'cpu_predictor'
}

DEFAULT_PROPHET_PARAMS = {
    'yearly_seasonality': 20,        # Increased for more complex yearly patterns
    'weekly_seasonality': 6,         # Increased for weekly patterns
    'daily_seasonality': False,
    'changepoint_prior_scale': 0.005,  # Reduced for less flexibility (better for irregular data)
    'seasonality_prior_scale': 5.0,    # Reduced for less overfitting
    'holidays_prior_scale': 5.0,       # Reduced for less overfitting
    'seasonality_mode': 'additive',
    'growth': 'linear',
    'n_changepoints': 15,              # Reduced for less overfitting
    'changepoint_range': 0.7,          # Reduced range for more stable trends
    'interval_width': 0.95,
    'mcmc_samples': 0
}

# Additional Prophet seasonality configurations for custom seasonalities
PROPHET_CUSTOM_SEASONALITIES = {
    'quarterly': {
        'period': 91.25,      # Average days per quarter
        'fourier_order': 6,   # Reduced for less overfitting
        'prior_scale': 1.5    # Reduced for less overfitting
    },
    'biweekly': {
        'period': 14,
        'fourier_order': 3,   # Reduced for less overfitting
        'prior_scale': 1.5    # Reduced for less overfitting
    },
    'monthly': {
        'period': 30.44,      # Average days per month
        'fourier_order': 4,   # Reduced for less overfitting
        'prior_scale': 1.0    # Conservative prior scale
    },
    'weekly': {
        'period': 7,
        'fourier_order': 3,   # New weekly seasonality
        'prior_scale': 1.0    # Conservative prior scale
    }
}