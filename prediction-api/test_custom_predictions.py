#!/usr/bin/env python3
"""
Test script to generate predictions for custom date ranges and save them to separate files.
This script demonstrates how to use the prediction API with custom start and end dates.
"""

import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import os

def test_custom_date_range_predictions():
    """Test the prediction API with custom date ranges"""
    
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Test different custom date ranges
    test_ranges = [
        {
            "name": "June_2024",
            "start_date": "2024-06-01",
            "end_date": "2024-06-30"
        },
        {
            "name": "July_2024", 
            "start_date": "2024-07-01",
            "end_date": "2024-07-31"
        },
        {
            "name": "Q3_2024",
            "start_date": "2024-07-01", 
            "end_date": "2024-09-30"
        },
        {
            "name": "Next_3_Months",
            "start_date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            "end_date": (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
        }
    ]
    
    # Feature configuration
    config = {
        "use_weather_features": True,
        "use_temporal_features": True,
        "use_lag_features": True,
        "use_rolling_features": True,
        "use_interaction_features": False,
        "custom_start_date": None,
        "custom_end_date": None
    }
    
    print("Testing custom date range predictions...")
    print("=" * 60)
    
    for test_range in test_ranges:
        print(f"\nTesting {test_range['name']}: {test_range['start_date']} to {test_range['end_date']}")
        print("-" * 50)
        
        # Update config with custom dates
        config["custom_start_date"] = test_range["start_date"]
        config["custom_end_date"] = test_range["end_date"]
        
        try:
            # Make API request
            response = requests.post(url, json=config, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract custom date range predictions
                custom_predictions = []
                for pred in result.get("predictions", []):
                    if pred.get("actual_energy") is None:  # This indicates custom range prediction
                        custom_predictions.append(pred)
                
                print(f"Generated {len(custom_predictions)} predictions for custom date range")
                
                # Save to separate file
                if custom_predictions:
                    save_custom_predictions_to_file(custom_predictions, test_range)
                
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Error testing {test_range['name']}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Custom date range testing completed!")

def save_custom_predictions_to_file(predictions, test_range):
    """Save custom predictions to a separate file"""
    
    # Create output directory
    output_dir = "data/facility_1/custom_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/custom_predictions_{test_range['name']}_{timestamp}.csv"
    
    # Prepare data for CSV
    csv_data = []
    for pred in predictions:
        csv_data.append({
            'date': pred['date'],
            'prophet_prediction': pred['prophet_prediction'],
            'xgb_prediction': pred['xgb_prediction'],
            'ensemble_prediction': pred['ensemble_prediction'],
            'actual_energy': 'N/A',  # No actual data for custom ranges
            'prophet_error_rate': 'N/A',
            'xgb_error_rate': 'N/A',
            'ensemble_error_rate': 'N/A',
            'temperature_2m_mean': pred.get('weather_data', {}).get('temperature_2m_mean', 'N/A')
        })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(filename, index=False)
    
    print(f"Saved {len(csv_data)} predictions to {filename}")
    
    # Also save as JSON for detailed analysis
    json_filename = f"{output_dir}/custom_predictions_{test_range['name']}_{timestamp}.json"
    with open(json_filename, 'w') as f:
        json.dump(predictions, f, indent=2, default=str)
    
    print(f"Saved detailed predictions to {json_filename}")

def create_summary_report():
    """Create a summary report of all custom predictions"""
    
    custom_dir = "data/facility_1/custom_predictions"
    if not os.path.exists(custom_dir):
        print("No custom predictions directory found")
        return
    
    csv_files = [f for f in os.listdir(custom_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print("No custom prediction CSV files found")
        return
    
    print("\n" + "=" * 60)
    print("CUSTOM PREDICTIONS SUMMARY REPORT")
    print("=" * 60)
    
    summary_data = []
    
    for csv_file in csv_files:
        filepath = os.path.join(custom_dir, csv_file)
        df = pd.read_csv(filepath)
        
        # Extract date range from filename
        parts = csv_file.replace('.csv', '').split('_')
        if len(parts) >= 3:
            range_name = parts[2]  # Extract the range name
            
            summary_data.append({
                'range_name': range_name,
                'file': csv_file,
                'predictions_count': len(df),
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'prophet_avg': df['prophet_prediction'].mean(),
                'xgb_avg': df['xgb_prediction'].mean(),
                'ensemble_avg': df['ensemble_prediction'].mean(),
                'temp_avg': df['temperature_2m_mean'].mean() if 'temperature_2m_mean' in df.columns else 'N/A'
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = f"{custom_dir}/custom_predictions_summary_{timestamp}.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"Summary report saved to {summary_file}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    # Test custom date range predictions
    test_custom_date_range_predictions()
    
    # Create summary report
    create_summary_report() 