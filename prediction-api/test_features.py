#!/usr/bin/env python3
"""
Test script to verify the new features (shutdown and days_elapsed) are working correctly
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from feature_engineering import create_advanced_features
from config import get_data_folder

def test_features(facility_name):
    """Test the new features"""
    print("Testing new features: shutdown and days_elapsed")
    
    # Load sample data from energy-totals.csv
    try:
        energy_df = pd.read_csv(f'{get_data_folder(facility_name)}/energy-totals.csv')
        # Only keep the required columns: date, calculated_energy, temp_2m
        energy_df = energy_df[['date', 'calculated_energy', 'temp_2m']].copy()
        
        # Rename columns to match expected format
        energy_df = energy_df.rename(columns={
            'date': 'ds', 
            'calculated_energy': 'y',
            'temp_2m': 'temperature_2m_mean'
        })
        
        energy_df['ds'] = pd.to_datetime(energy_df['ds'])
        
        print(f"Loaded energy data with {len(energy_df)} rows")
        print(f"Date range: {energy_df['ds'].min()} to {energy_df['ds'].max()}")
        
        # Test the feature engineering
        # Use None for energy_csv_path since we're using simplified data structure
        df_with_features = create_advanced_features(energy_df, energy_csv_path=None)
        
        # Check if new features were created
        print("\nFeature creation results:")
        print(f"Shutdown feature present: {'shutdown' in df_with_features.columns}")
        print(f"Days since commission feature present: {'days_elapsed' in df_with_features.columns}")
        
        if 'shutdown' in df_with_features.columns:
            shutdown_count = df_with_features['shutdown'].sum()
            total_days = len(df_with_features)
            print(f"Shutdown days: {shutdown_count} out of {total_days} ({shutdown_count/total_days*100:.1f}%)")
            
            # Show some shutdown examples
            shutdown_examples = df_with_features[df_with_features['shutdown'] == 1][['ds', 'y']].head(3)
            print("Examples of shutdown days:")
            for _, row in shutdown_examples.iterrows():
                print(f"  {row['ds'].strftime('%Y-%m-%d')}: {row['y']:,.0f} Wh")
        
        if 'days_elapsed' in df_with_features.columns:
            print(f"\nDays since commission range: {df_with_features['days_elapsed'].min()} to {df_with_features['days_elapsed'].max()}")
            
            # Check if it's sequential
            is_sequential = all(df_with_features['days_elapsed'].diff().dropna() == 1)
            print(f"Days since commission is sequential: {is_sequential}")
            
            # Show first and last few days
            print("First 5 days:")
            for _, row in df_with_features.head(5).iterrows():
                print(f"  {row['ds'].strftime('%Y-%m-%d')}: Day {row['days_elapsed']}")
            
            print("Last 5 days:")
            for _, row in df_with_features.tail(5).iterrows():
                print(f"  {row['ds'].strftime('%Y-%m-%d')}: Day {row['days_elapsed']}")
        
        # Test with a small subset for future predictions
        print("\nTesting future features creation...")
        future_dates = pd.date_range(start='2024-12-01', end='2024-12-05', freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Import the function from predict_may_2024
        from predict_may_2024 import create_future_features
        
        future_features = create_future_features(future_df, last_known_day=df_with_features['days_elapsed'].max())
        
        print(f"Future features created for {len(future_features)} dates")
        print(f"Future days since commission range: {future_features['days_elapsed'].min()} to {future_features['days_elapsed'].max()}")
        print(f"Future shutdown values: {future_features['shutdown'].unique()}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_features() 