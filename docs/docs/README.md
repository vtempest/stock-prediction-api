
# Energy Prediction System
![logo](https://i.imgur.com/GcTU5Wy.png)

A comprehensive machine learning system for predicting energy consumption using multiple models and advanced time series analysis techniques.

## Overview

This system combines multiple machine learning approaches to predict energy consumption with high accuracy. It uses an ensemble of models including Prophet for time series forecasting, Random Forest, Ridge Regression, and XGBoost for pattern recognition, along with advanced feature engineering and cross-validation techniques.

## Key Features

- **Multi-Model Ensemble**: Combines Prophet, Random Forest Ridge Regression, and XGBoost
- **Advanced Feature Engineering**: Creates sophisticated features from temporal and weather data
- **Time Series Cross-Validation**: Proper evaluation using forward-chaining validation
- **Weather Integration**: Incorporates weather data for improved predictions
- **Confidence Intervals**: Provides prediction uncertainty estimates
- **Real-time API**: FastAPI endpoint for on-demand predictions


## 🏗️ Figures


![fig1](https://i.imgur.com/TFSa30F.png)

![figure2](https://i.imgur.com/I1twzWe.png)


![figure2](https://i.imgur.com/bLV3xu0.png)


![figure2](https://i.imgur.com/IbEcWuW.png)

![figure5](https://i.imgur.com/GZL7BMJ.png)

## System Architecture

### Core Components

1. **Data Pipeline**: Loads and prepares energy consumption and weather data
2. **Feature Engineering**: Creates advanced temporal and weather-based features
3. **Model Training**: Trains multiple ML models with hyperparameter optimization
4. **Cross-Validation**: Evaluates models using time series-aware validation
5. **Ensemble Prediction**: Combines predictions from multiple models
6. **API Interface**: Serves predictions through RESTful API

### Model Types

#### 1. Prophet Model
- **Purpose**: Time series forecasting with seasonality detection
- **Strengths**: Handles trends, seasonality, and holidays automatically
- **Features**: Built-in uncertainty quantification and missing data handling

#### 2. Random Forest
- **Purpose**: Non-linear pattern recognition
- **Strengths**: Robust to outliers, handles feature interactions well
- **Configuration**: Optimized hyperparameters for energy data

#### 3. Ridge Regression
- **Purpose**: Linear modeling with regularization
- **Strengths**: Prevents overfitting, interpretable coefficients
- **Features**: L2 regularization for stable predictions

#### 4. XGBoost
- **Purpose**: Gradient boosting for complex patterns
- **Strengths**: High accuracy, handles missing values
- **Configuration**: Tuned for energy consumption patterns

## Data Sources

### Energy Data
- Historical energy consumption measurements
- Temporal resolution: Daily/hourly readings
- Units: Typically in megawatts (MW) or millions of units

### Weather Data
- Temperature (2m mean)
- Soil moisture (0-7cm depth)
- Precipitation sum
- Relative humidity (2m mean)
- Additional meteorological variables

### Official Data
- Ground truth energy consumption values
- Used for validation and accuracy assessment

## Feature Engineering

The system creates advanced features including:

### Temporal Features
- **Calendar Features**: Day of week, month, quarter, year
- **Cyclical Encoding**: Sin/cos transformations for cyclical patterns
- **Holiday Indicators**: Binary flags for holidays and special events
- **Seasonal Decomposition**: Trend, seasonal, and residual components

### Weather Features
- **Temperature Derivatives**: Heating/cooling degree days
- **Moisture Indicators**: Soil moisture levels and precipitation
- **Comfort Indices**: Heat index, wind chill calculations
- **Lag Features**: Historical weather patterns

### Statistical Features
- **Rolling Statistics**: Moving averages, standard deviations
- **Lag Features**: Previous periods' consumption patterns
- **Difference Features**: Period-over-period changes
- **Fourier Terms**: Frequency domain representations

## Cross-Validation Strategy

### Time Series Cross-Validation
The system uses a forward-chaining approach:

1. **Initial Training**: Start with 60% of historical data
2. **Expanding Window**: Gradually increase training data
3. **Future Testing**: Always test on future periods
4. **Multiple Folds**: 5-fold validation with temporal ordering

### Prophet Cross-Validation
- **Initial Period**: 365 days for initial training
- **Validation Period**: 90-day intervals for rolling validation
- **Forecast Horizon**: 180 days ahead for each validation
- **Parallel Processing**: Utilizes multiple CPU cores

## API Endpoint

### POST /predict

Accepts a `FeatureConfig` object and returns predictions for specified dates.

#### Input Parameters
```json
{
  "feature_config": {
    "include_weather": true,
    "include_temporal": true,
    "include_holidays": true,
    "lag_features": true,
    "rolling_features": true
  }
}
```

#### Response Format
```json
{
  "predictions": [
    {
      "date": "2024-05-01",
      "predicted_energy_millions": 123.45,
      "prophet_prediction": 120.30,
      "rf_prediction": 125.20,
      "ridge_prediction": 124.10,
      "xgb_prediction": 124.30,
      "ensemble_prediction": 123.45,
      "prediction_lower": 118.50,
      "prediction_upper": 128.40,
      "actual_energy_millions": 125.20,
      "error": -1.75,
      "percent_error": 1.40,
      "weather_data": {
        "temperature_2m_mean": 22.5,
        "soil_moisture_0_to_7cm_mean": 0.35,
        "precipitation_sum": 2.3,
        "relative_humidity_2m_mean": 65.2
      }
    }
  ],
  "cross_validation_results": [...],
  "may_validation_results": [...],
  "feature_config": {...}
}
```

## Performance Metrics

The system evaluates models using multiple metrics:

- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: Penalizes larger errors more heavily
- **R² Score**: Coefficient of determination (explained variance)
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error metric
- **Accuracy**: 100% - MAPE (intuitive accuracy percentage)

## Installation and Setup

### Requirements
```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost fbprophet
```

### Environment Setup
1. Clone the repository
2. Install dependencies
3. Prepare data files:
   - Energy consumption data
   - Weather data
   - Official validation data
4. Configure API settings
5. Run the FastAPI server

### Running the API
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Data Preparation

### Expected Data Format

#### Energy Data
- Columns: `ds` (datetime), `y` (energy consumption)
- Format: Daily or hourly timestamps
- Units: Consistent energy units (MW, MWh, etc.)

#### Weather Data
- Columns: Date, temperature, humidity, precipitation, etc.
- Format: Daily weather observations
- Coverage: Same time period as energy data

#### Official Data
- Columns: `ds` (datetime), `actual` (true energy consumption)
- Purpose: Validation and accuracy assessment
- Format: Same as energy data

## Model Training Process

1. **Data Loading**: Load energy, weather, and validation data
2. **Feature Engineering**: Create advanced features based on configuration
3. **Prophet Training**: Train Prophet model with weather regressors
4. **Prophet Cross-Validation**: Evaluate using built-in Prophet CV
5. **Ensemble Training**: Train Random Forest, Ridge, and XGBoost models
6. **Ensemble Cross-Validation**: Time series validation for ML models
7. **Final Training**: Retrain all models on full dataset
8. **Prediction Generation**: Generate predictions for target dates
9. **Validation**: Compare predictions against actual values

## Output Files

The system generates several output files:

- **predicted_energy_prophet.json**: Detailed predictions with all model outputs
- **Cross-validation results**: Performance metrics for each model
- **Feature importance**: Rankings of most important features
- **Model artifacts**: Saved trained models for future use

## Best Practices

### Data Quality
- Ensure consistent data formatting
- Handle missing values appropriately
- Validate data ranges and outliers
- Maintain temporal continuity

### Model Configuration
- Tune hyperparameters for your specific data
- Balance model complexity with interpretability
- Regular retraining with new data
- Monitor model drift over time

### Performance Optimization
- Use parallel processing for cross-validation
- Implement efficient feature engineering
- Cache expensive computations
- Optimize memory usage for large datasets

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install all required packages
2. **Data Format Errors**: Ensure consistent datetime formats
3. **Memory Issues**: Reduce dataset size or optimize features
4. **Convergence Problems**: Adjust model hyperparameters
5. **Cross-Validation Errors**: Check data chronological ordering

### Debugging Tips
- Enable detailed logging
- Validate input data shapes and types
- Check for infinite or NaN values
- Monitor memory usage during training
- Use smaller datasets for testing

## Future Enhancements

### Planned Features
- **Deep Learning Models**: LSTM, Transformer architectures
- **Real-time Streaming**: Online learning capabilities
- **Advanced Ensembling**: Stacking and blending techniques
- **Automated Hyperparameter Tuning**: Bayesian optimization
- **Model Interpretability**: SHAP values and feature analysis
- **A/B Testing Framework**: Model comparison and selection

### Scalability Improvements
- **Distributed Training**: Multi-GPU and multi-node support
- **Cloud Integration**: AWS/GCP deployment options
- **Containerization**: Docker and Kubernetes support
- **Monitoring**: MLOps pipeline integration
