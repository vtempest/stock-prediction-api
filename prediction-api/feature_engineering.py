import pandas as pd
import numpy as np

def create_shutdown_feature(df, energy_csv_path=None):
    """
    Create shutdown feature that identifies days with anomalously low energy output.
    A day is marked as shutdown (1) if its energy output is in the bottom 25% of all days.
    
    Args:
        df: DataFrame with 'y' column (energy output)
        energy_csv_path: Optional path to energy_output.csv for additional context
    
    Returns:
        DataFrame with 'shutdown' column added
    """
    df = df.copy()
    
    # If we have energy output CSV data, use it for better threshold calculation
    if energy_csv_path:
        try:
            energy_csv = pd.read_csv(energy_csv_path)
            energy_csv['date'] = pd.to_datetime(energy_csv['date'])
            
            # Calculate 25th percentile from the full historical dataset
            energy_values = energy_csv['energy_output'].dropna()
            shutdown_threshold = energy_values.quantile(0.25)
            
            print(f"Shutdown threshold (25th percentile): {shutdown_threshold:,.0f} Wh")
            print(f"Total days in energy CSV: {len(energy_values)}")
            print(f"Days below threshold: {len(energy_values[energy_values < shutdown_threshold])}")
            
        except Exception as e:
            print(f"Warning: Could not load energy CSV for shutdown threshold: {e}")
            # Fallback to using current dataframe
            shutdown_threshold = df['y'].quantile(0.25) if 'y' in df.columns else 0
    else:
        # Use current dataframe to calculate threshold
        shutdown_threshold = df['y'].quantile(0.25) if 'y' in df.columns else 0
    
    # Create shutdown feature
    if 'y' in df.columns:
        df['shutdown'] = (df['y'] < shutdown_threshold).astype(int)
        
        # Print statistics
        shutdown_days = df['shutdown'].sum()
        total_days = len(df)
        print(f"Shutdown days identified: {shutdown_days} out of {total_days} ({shutdown_days/total_days*100:.1f}%)")
        
        # Show some examples of shutdown days
        shutdown_examples = df[df['shutdown'] == 1][['ds', 'y']].head(5)
        if len(shutdown_examples) > 0:
            print("Examples of shutdown days:")
            for _, row in shutdown_examples.iterrows():
                print(f"  {row['ds'].strftime('%Y-%m-%d')}: {row['y']:,.0f} Wh")
    else:
        # For future predictions, we'll set shutdown to 0 (no shutdown)
        df['shutdown'] = 0
        print("No 'y' column found - setting shutdown to 0 for all future dates")
    
    return df

def create_advanced_features(df, energy_csv_path=None):
    """Create advanced time-based features with simplified temperature data"""
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Create shutdown feature first
    df = create_shutdown_feature(df, energy_csv_path)
    
    # Create days_elapsed feature
    df = df.sort_values('ds').reset_index(drop=True)
    df['days_elapsed'] = range(1, len(df) + 1)
    
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['quarter'] = df['ds'].dt.quarter
    df['is_weekend'] = (df['ds'].dt.dayofweek >= 5).astype(int)
    df['is_month_start'] = df['ds'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['ds'].dt.is_month_end.astype(int)
    
    # Seasonal features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Temperature features (only if temperature_2m_mean exists)
    if 'temperature_2m_mean' in df.columns:
        # Fill NaN values in temperature data before creating features
        temp_mean = df['temperature_2m_mean'].mean()
        df['temperature_2m_mean'] = df['temperature_2m_mean'].fillna(temp_mean)
        
        df['temp_squared'] = df['temperature_2m_mean'] ** 2
        df['temp_cubed'] = df['temperature_2m_mean'] ** 3
        
        # Temperature rolling statistics
        df['temp_rolling_mean_3d'] = df['temperature_2m_mean'].rolling(3, min_periods=1).mean()
        df['temp_rolling_mean_7d'] = df['temperature_2m_mean'].rolling(7, min_periods=1).mean()
        df['temp_rolling_mean_14d'] = df['temperature_2m_mean'].rolling(14, min_periods=1).mean()
        df['temp_rolling_std_7d'] = df['temperature_2m_mean'].rolling(7, min_periods=1).std()
        df['temp_rolling_std_14d'] = df['temperature_2m_mean'].rolling(14, min_periods=1).std()
        
        # Temperature trend features
        df['temp_diff_1d'] = df['temperature_2m_mean'].diff(1)
        df['temp_diff_7d'] = df['temperature_2m_mean'].diff(7)
        
        # Fill any remaining NaN values in temperature features
        temp_features = ['temp_squared', 'temp_cubed', 'temp_rolling_mean_3d', 'temp_rolling_mean_7d', 
                        'temp_rolling_mean_14d', 'temp_rolling_std_7d', 'temp_rolling_std_14d', 
                        'temp_diff_1d', 'temp_diff_7d']
        for col in temp_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                df[col] = df[col].fillna(0)  # Fill any remaining NaNs with 0
    
    # Lag features for energy output (if y column exists)
    if 'y' in df.columns:
        df['y_lag1'] = df['y'].shift(1)
        df['y_lag2'] = df['y'].shift(2)
        df['y_lag3'] = df['y'].shift(3)
        df['y_lag7'] = df['y'].shift(7)
        df['y_lag14'] = df['y'].shift(14)
        df['y_lag30'] = df['y'].shift(30)
        
        # Rolling statistics
        df['y_rolling_mean_3'] = df['y'].rolling(3, min_periods=1).mean()
        df['y_rolling_mean_7'] = df['y'].rolling(7, min_periods=1).mean()
        df['y_rolling_mean_14'] = df['y'].rolling(14, min_periods=1).mean()
        df['y_rolling_mean_30'] = df['y'].rolling(30, min_periods=1).mean()
        df['y_rolling_std_7'] = df['y'].rolling(7, min_periods=1).std()
        df['y_rolling_std_14'] = df['y'].rolling(14, min_periods=1).std()
        df['y_rolling_max_7'] = df['y'].rolling(7, min_periods=1).max()
        df['y_rolling_min_7'] = df['y'].rolling(7, min_periods=1).min()
        
        # Trend features
        df['y_diff1'] = df['y'].diff(1)
        df['y_diff7'] = df['y'].diff(7)
        df['y_pct_change1'] = df['y'].pct_change(1)
        df['y_pct_change7'] = df['y'].pct_change(7)
        
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['ds']:
            df[col] = df[col].fillna(df[col].median())
            
    # Replace infinities with NaN then fill
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in numeric_cols:
        if col not in ['ds']:
            df[col] = df[col].fillna(df[col].median())
    
    return df

def get_prophet_regressors(config, df=None):
    """Get list of regressors for Prophet model with simplified data structure"""
    base_regressors = [
        'temperature_2m_mean',
        'shutdown',  # New shutdown feature
        'days_elapsed'  # New days since commission feature
    ]
    
    advanced_regressors = [
        # Temperature features
        'temp_squared', 'temp_cubed',
        'temp_rolling_mean_3d', 'temp_rolling_mean_7d', 'temp_rolling_mean_14d',
        'temp_rolling_std_7d', 'temp_rolling_std_14d',
        'temp_diff_1d', 'temp_diff_7d',  # Temperature changes
        
        # Time-based features
        'month_sin', 'month_cos',
        'dayofyear_sin', 'dayofyear_cos',
        
        # Calendar features
        'is_weekend', 'quarter',
        'is_month_start', 'is_month_end'
    ]
    
    
    # Filter features based on config
    if not config.time_features:
        # Remove time-based features
        time_patterns = ['month', 'quarter', 'day', 'weekend']
        advanced_regressors = [f for f in advanced_regressors 
                             if not any(pattern in f for pattern in time_patterns)]
    
    if not config.weather_features:
        # Remove weather-based features
        weather_patterns = ['temp']
        advanced_regressors = [f for f in advanced_regressors 
                             if not any(pattern in f for pattern in weather_patterns)]
        base_regressors = [f for f in base_regressors 
                          if not any(pattern in f for pattern in weather_patterns)]
    
    if not config.rolling_features:
        # Remove rolling statistics and trend features
        rolling_patterns = ['rolling', 'trend', 'diff']
        advanced_regressors = [f for f in advanced_regressors 
                             if not any(pattern in f for pattern in rolling_patterns)]
    
    # If dataframe is provided, filter to only include features that actually exist
    if df is not None:
        all_regressors = base_regressors + advanced_regressors
        available_regressors = [f for f in all_regressors if f in df.columns]
        return available_regressors
    
    return base_regressors + advanced_regressors

def handle_regressors(future_df, historical_df, config):
    """Handle regressors for Prophet model"""
    regressor_cols = get_prophet_regressors(config, historical_df)
    
    # Merge with historical data
    available_cols = [col for col in regressor_cols if col in historical_df.columns]
    
    if available_cols:
        merge_cols = ['ds'] + available_cols
        future_df = future_df.merge(
            historical_df[merge_cols],
            on='ds',
            how='left'
        )
    
    # Fill missing values
    for col in available_cols:
        if col in future_df.columns:
            # Forward fill
            future_df[col] = future_df[col].ffill(limit=10)
            # Backward fill
            future_df[col] = future_df[col].bfill(limit=10)
            # Fill remaining with historical mean
            if col in historical_df.columns:
                fill_value = historical_df[col].mean()
                future_df[col] = future_df[col].fillna(fill_value)
    
    return future_df