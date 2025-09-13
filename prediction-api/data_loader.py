import pandas as pd
import numpy as np
import os
import traceback
from config import get_data_folder
from typing import Optional, List, Dict
from datetime import datetime

def load_and_prepare_data(facility_name, training_cutoff_date='2025-05-30', validation_start_date='2025-06-01'):
    """Load and validate input data from facility-specific energy-totals.csv with training/validation split"""
    try:
        if facility_name:
            # Load facility-specific data
            facility_folder = get_data_folder(facility_name)
            print(f"\nLoading data for facility: {facility_name}")
            print(f"Loading data from {facility_folder}")
            
            # Load main dataset from facility's energy-totals.csv
            print("Loading energy and temperature data...")
            energy_df = pd.read_csv(f'{facility_folder}/energy-totals.csv')
            print(f"Energy data shape: {energy_df.shape}")
            
            # Check if the required columns exist
            required_columns = ['date', 'energy_total', 'temperature']
            if not all(col in energy_df.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Expected: {required_columns}, Found: {list(energy_df.columns)}")
            
            # Only keep the required columns: date, energy_total, temperature
            energy_df = energy_df[required_columns].copy()
            
            # Rename columns to match expected format
            energy_df = energy_df.rename(columns={
                'date': 'ds', 
                'energy_total': 'y',
                'temperature': 'temperature_2m_mean'
            })
            
            energy_df['ds'] = pd.to_datetime(energy_df['ds'])
            print(f"Energy data date range: {energy_df['ds'].min()} to {energy_df['ds'].max()}")
            
            # Split data into training (up to 05-30-2025) and validation (06-01-2025 onwards)
            training_cutoff = pd.to_datetime(training_cutoff_date)
            validation_start = pd.to_datetime(validation_start_date)
            
            # Training data (up to 05-30-2025)
            training_df = energy_df[energy_df['ds'] <= training_cutoff].copy()
            print(f"Training data: {len(training_df)} records (up to {training_cutoff_date})")
            
            # Validation data (06-01-2025 onwards)
            validation_df = energy_df[energy_df['ds'] >= validation_start].copy()
            print(f"Validation data: {len(validation_df)} records (from {validation_start_date})")
            
            # Use training data for model training
            energy_df = training_df
            
            # Create a minimal weather future dataframe with just temperature data
            weather_future_df = energy_df[['ds', 'temperature_2m_mean']].copy()
            
            # For facility-specific data, we might not have official May data
            # Create a placeholder or load from a different source if needed
            print("\nCreating placeholder for official May data...")
            # Create a simple placeholder with some future dates
            future_dates = pd.date_range(start=energy_df['ds'].max() + pd.Timedelta(days=1), periods=30, freq='D')
            official_df = pd.DataFrame({
                'ds': future_dates,
                'actual': [None] * len(future_dates)  # No actual data for future predictions
            })
            print(f"Created placeholder official data for {len(official_df)} future dates")
            
        return energy_df, weather_future_df, official_df
    
    except FileNotFoundError as e:
        print("\n" + "="*50)
        print("ERROR LOADING DATA")
        print("="*50)
        print(f"File not found: {e.filename}")
        print(f"Working directory: {os.getcwd()}")
        print(f"Expected path: {get_data_folder(facility_name)}")
        print("="*50 + "\n")
        raise SystemExit(f"Critical error: Missing file {e.filename}")
    except Exception as e:
        print("\n" + "="*50)
        print("ERROR LOADING DATA")
        print("="*50)
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Message: {str(e)}")
        print("\nStack Trace:")
        traceback.print_exc()
        print("="*50 + "\n")
        raise e

def prepare_prediction_data(official_df, weather_future_df, energy_df, config):
    """Prepare data for official May predictions with simplified temperature data"""
    from feature_engineering import create_advanced_features
    
    # First, get the historical energy data ready
    historical_df = energy_df.copy()
    historical_df = historical_df.sort_values('ds')
    
    # Create prediction dataframe with temperature data
    prediction_df = pd.merge(
        official_df[['ds']],
        weather_future_df[['ds', 'temperature_2m_mean']],
        on='ds',
        how='left'
    )
    
    # Fill missing temperature data with prior day's data
    if 'temperature_2m_mean' in prediction_df.columns:
        # Get the last known temperature from historical data
        last_known_temp = None
        if 'temperature_2m_mean' in energy_df.columns and not energy_df['temperature_2m_mean'].isna().all():
            last_known_temp = energy_df['temperature_2m_mean'].dropna().iloc[-1]
        else:
            last_known_temp = 20.0  # Default temperature if no historical data
        
        # Forward fill (use prior day's data), then backward fill, then fill remaining with last known temp
        prediction_df['temperature_2m_mean'] = prediction_df['temperature_2m_mean'].fillna(method='ffill')
        prediction_df['temperature_2m_mean'] = prediction_df['temperature_2m_mean'].fillna(method='bfill')
        prediction_df['temperature_2m_mean'] = prediction_df['temperature_2m_mean'].fillna(last_known_temp)
        
        print(f"Temperature data prepared - mean: {prediction_df['temperature_2m_mean'].mean():.1f}°C, last known: {last_known_temp:.1f}°C")
    else:
        print("Warning: No temperature data found in weather_future_df")
        # Add default temperature column
        prediction_df['temperature_2m_mean'] = 20.0
    
    # Create a temporary dataframe combining historical and prediction data
    temp_df = pd.concat([
        historical_df[['ds', 'y']],
        prediction_df[['ds']].assign(y=np.nan)
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
    prediction_df = pd.merge(
        prediction_df,
        temp_df[['ds'] + lag_features].tail(len(prediction_df)),
        on='ds',
        how='left'
    )
    
    # Add advanced features with simplified data structure
    # Since we only have temperature data, we'll pass None for energy_csv_path
    prediction_df = create_advanced_features(prediction_df, energy_csv_path=None)
    
    # Fill any remaining NaN values with the last known value
    for col in lag_features:
        if col in prediction_df.columns:
            prediction_df[col] = prediction_df[col].fillna(method='ffill')
            prediction_df[col] = prediction_df[col].fillna(method='bfill')
            prediction_df[col] = prediction_df[col].fillna(0)  # Fill any remaining NaNs with 0
    
    print(f"Prepared prediction data for {len(prediction_df)} dates")
    return prediction_df

def create_train_test_split(df, test_size=0.2, random_state=42):
    """Create train-test split while preserving time series order"""
    df_sorted = df.sort_values('ds').reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df)} samples ({train_df['ds'].min()} to {train_df['ds'].max()})")
    print(f"Test set: {len(test_df)} samples ({test_df['ds'].min()} to {test_df['ds'].max()})")
    
    return train_df, test_df

def load_energy_data(facility_name: str) -> Optional[pd.DataFrame]:
    """Load energy data for a specific facility, filtering out dates with <92% percent_running"""
    try:
        # Load facility-specific data
        facility_folder = get_data_folder(facility_name)
        energy_path = f'{facility_folder}/energy-totals.csv'
        
        if not os.path.exists(energy_path):
            print(f"Energy data file not found: {energy_path}")
            return None
        
        # Load energy data
        energy_df = pd.read_csv(energy_path)
        
        # Check if the required columns exist
        required_columns = ['date', 'energy_total', 'temperature', 'percent_running']
        if not all(col in energy_df.columns for col in required_columns):
            print(f"Missing required columns in {energy_path}. Expected: {required_columns}, Found: {list(energy_df.columns)}")
            return None
        
        # Filter out dates with less than 92% percent_running
        original_count = len(energy_df)
        energy_df = energy_df[energy_df['percent_running'] >= 92].copy()
        filtered_count = len(energy_df)
        
        if filtered_count < original_count:
            print(f"Filtered out {original_count - filtered_count} records with <92% percent_running")
        
        # Only keep the required columns and rename them
        energy_df = energy_df[['date', 'energy_total', 'temperature']].copy()
        energy_df = energy_df.rename(columns={
            'date': 'ds', 
            'energy_total': 'y',
            'temperature': 'temperature_2m_mean'
        })
        
        energy_df['ds'] = pd.to_datetime(energy_df['ds'])
        
        print(f"Loaded energy data for {facility_name}: {len(energy_df)} records (after filtering for percent_running >= 92)")
        print(f"Date range: {energy_df['ds'].min()} to {energy_df['ds'].max()}")
        
        return energy_df
        
    except Exception as e:
        print(f"Error loading energy data for {facility_name}: {e}")
        return None

def load_weather_data(facility_name: str) -> Optional[pd.DataFrame]:
    """Load historical weather data for a specific facility"""
    try:
        # Load facility-specific data
        facility_folder = get_data_folder(facility_name)
        weather_path = f'{facility_folder}/weather.csv'
        
        if not os.path.exists(weather_path):
            print(f"Weather data file not found: {weather_path}")
            return None
        
        # Load weather data
        weather_df = pd.read_csv(weather_path)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        print(f"Loaded weather data for {facility_name}: {len(weather_df)} records")
        print(f"Date range: {weather_df['date'].min()} to {weather_df['date'].max()}")
        
        return weather_df
        
    except Exception as e:
        print(f"Error loading weather data for {facility_name}: {e}")
        return None

def load_future_weather_data(facility_name: str) -> Optional[pd.DataFrame]:
    """Load future weather data for a specific facility"""
    try:
        # Load facility-specific data
        facility_folder = get_data_folder(facility_name)
        weather_path = f'{facility_folder}/weather.csv'
        
        if not os.path.exists(weather_path):
            print(f"Weather data file not found: {weather_path}")
            return None
        
        # Load weather data
        weather_df = pd.read_csv(weather_path)
        weather_df['date'] = pd.to_datetime(weather_df['date'])
        
        # Get today's date
        today = datetime.now().date()
        
        # Filter for future dates
        future_weather = weather_df[weather_df['date'] >= pd.Timestamp(today)].copy()
        
        if future_weather.empty:
            print(f"No future weather data found for {facility_name}")
            return None
        
        print(f"Loaded future weather data for {facility_name}: {len(future_weather)} records")
        print(f"Date range: {future_weather['date'].min()} to {future_weather['date'].max()}")
        
        return future_weather
        
    except Exception as e:
        print(f"Error loading future weather data for {facility_name}: {e}")
        return None

def load_validation_data(facility_name: str, validation_start_date='2025-06-01') -> Optional[List[Dict]]:
    """Load validation data from 06-01-2025 onwards for post-validation scoring, filtering out dates with <92% percent_running"""
    try:
        # Load facility-specific data
        facility_folder = get_data_folder(facility_name)
        energy_path = f'{facility_folder}/energy-totals.csv'
        
        if not os.path.exists(energy_path):
            print(f"Energy data file not found: {energy_path}")
            return None
        
        # Load energy data
        energy_df = pd.read_csv(energy_path)
        
        # Check if the required columns exist
        required_columns = ['date', 'energy_total', 'temperature', 'percent_running']
        if not all(col in energy_df.columns for col in required_columns):
            print(f"Missing required columns in {energy_path}. Expected: {required_columns}, Found: {list(energy_df.columns)}")
            return None
        
        # Filter out dates with less than 92% percent_running
        original_count = len(energy_df)
        energy_df = energy_df[energy_df['percent_running'] >= 92].copy()
        filtered_count = len(energy_df)
        
        if filtered_count < original_count:
            print(f"Filtered out {original_count - filtered_count} records with <92% percent_running")
        
        # Only keep the required columns and rename them
        energy_df = energy_df[['date', 'energy_total', 'temperature']].copy()
        energy_df = energy_df.rename(columns={
            'date': 'ds', 
            'energy_total': 'y',
            'temperature': 'temperature_2m_mean'
        })
        
        energy_df['ds'] = pd.to_datetime(energy_df['ds'])
        
        # Filter for validation period (06-01-2025 onwards)
        validation_start = pd.to_datetime(validation_start_date)
        validation_df = energy_df[energy_df['ds'] >= validation_start].copy()
        
        if validation_df.empty:
            print(f"No validation data found from {validation_start_date} onwards for {facility_name} (after filtering for percent_running >= 92)")
            return None
        
        print(f"Loaded validation data for {facility_name}: {len(validation_df)} records from {validation_start_date} onwards (after filtering for percent_running >= 92)")
        print(f"Validation date range: {validation_df['ds'].min()} to {validation_df['ds'].max()}")
        
        # Convert to list of dictionaries
        validation_data = validation_df.to_dict('records')
        
        return validation_data
        
    except Exception as e:
        print(f"Error loading validation data for {facility_name}: {e}")
        return None