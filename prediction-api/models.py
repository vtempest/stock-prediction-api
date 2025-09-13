from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ProphetParams(BaseModel):
    yearly_seasonality: bool = Field(default=True, description="Enable yearly seasonality")
    weekly_seasonality: bool = Field(default=True, description="Enable weekly seasonality")
    daily_seasonality: bool = Field(default=False, description="Enable daily seasonality")
    changepoint_prior_scale: float = Field(default=0.05, description="Flexibility of the trend")
    seasonality_prior_scale: float = Field(default=10.0, description="Flexibility of the seasonality")
    holidays_prior_scale: float = Field(default=10.0, description="Flexibility of the holidays")
    seasonality_mode: str = Field(default="multiplicative", description="Mode of the seasonality")
    growth: str = Field(default="linear", description="Growth trend type")
    n_changepoints: int = Field(default=25, description="Number of changepoints")
    changepoint_range: float = Field(default=0.8, description="Changepoint range")
    interval_width: float = Field(default=0.95, description="Width of the uncertainty intervals")
    mcmc_samples: int = Field(default=0, description="Number of MCMC samples")

class XGBParams(BaseModel):
    max_depth: int = Field(default=6, description="Maximum tree depth")
    learning_rate: float = Field(default=0.1, description="Learning rate")
    n_estimators: int = Field(default=100, description="Number of boosting rounds")
    min_child_weight: int = Field(default=1, description="Minimum sum of instance weight needed in a child")
    gamma: float = Field(default=0, description="Minimum loss reduction required to make a further partition")
    subsample: float = Field(default=0.8, description="Subsample ratio of the training instances")
    colsample_bytree: float = Field(default=0.8, description="Subsample ratio of columns when constructing each tree")
    reg_alpha: float = Field(default=0, description="L1 regularization term on weights")
    reg_lambda: float = Field(default=1, description="L2 regularization term on weights")
    random_state: int = Field(default=42, description="Random state for reproducibility")
    n_jobs: int = Field(default=-1, description="Number of parallel threads")

class FeatureConfig(BaseModel):
    facility_name: str = Field(description="Name of the facility to process (must be one of the available facilities)")
    time_features: bool = Field(default=True, description="Enable time-based features")
    weather_features: bool = Field(default=True, description="Enable weather-based features")
    rolling_features: bool = Field(default=True, description="Enable rolling statistics features")
    lag_features: bool = Field(default=True, description="Enable lagged features")
    interaction_features: bool = Field(default=True, description="Enable interaction features")
    windows: List[int] = Field(default=[3, 7, 15, 30, 60], description="Rolling windows sizes for statistics")
    lags: List[int] = Field(default=[2, 3, 7, 14, 21, 30, 45, 60], description="Lag periods for features")
    custom_start_date: Optional[str] = Field(default=None, description="Custom start date for predictions (YYYY-MM-DD format)")
    custom_end_date: Optional[str] = Field(default=None, description="Custom end date for predictions (YYYY-MM-DD format)")
    validation_months: List[str] = Field(default=["2024-05", "2024-10", "2024-11", "2025-05"], description="Months to use for post-validation testing")

class PredictionRequest(BaseModel):
    feature_config: FeatureConfig
    data_path: str
    prophet_params: Optional[ProphetParams] = Field(default=None, description="Prophet model parameters")
    xgb_params: Optional[XGBParams] = Field(default=None, description="XGBoost model parameters")