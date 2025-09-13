import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any
from datetime import datetime, timedelta

def get_month_date_range(year_month: str) -> tuple:
    """Get start and end dates for a given year-month (e.g., '2024-05')"""
    year, month = year_month.split('-')
    start_date = f"{year}-{month}-01"
    
    # Get the last day of the month
    if month == '12':
        next_month = f"{int(year)+1}-01-01"
    else:
        next_month = f"{year}-{int(month)+1:02d}-01"
    
    end_date = (pd.to_datetime(next_month) - timedelta(days=1)).strftime('%Y-%m-%d')
    return start_date, end_date

def create_validation_data_for_month(energy_df: pd.DataFrame, year_month: str) -> pd.DataFrame:
    """Create validation data for a specific month"""
    start_date, end_date = get_month_date_range(year_month)
    
    # Create date range for the month
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create validation dataframe
    validation_df = pd.DataFrame({
        'ds': date_range,
        'actual': [None] * len(date_range)  # No actual data for validation months
    })
    
    return validation_df

def run_validation_for_month(
    month: str,
    energy_df: pd.DataFrame,
    weather_future_df: pd.DataFrame,
    config: Any,
    final_prophet,
    weighted_ensemble_dict,
    available_regressors
) -> Dict[str, Any]:
    """Run validation for a specific month"""
    print(f"\n{'='*50}")
    print(f"VALIDATION FOR {month}")
    print(f"{'='*50}")
    
    # Create validation data for this month
    validation_df = create_validation_data_for_month(energy_df, month)
    
    # Prepare prediction data
    from data_loader import prepare_prediction_data
    prediction_data = prepare_prediction_data(validation_df, weather_future_df, energy_df, config)
    
    # Get predictions from all models
    prophet_pred = final_prophet.predict(prediction_data)
    prophet_pred_values = prophet_pred['yhat'].values
    
    # Get ensemble predictions
    from model_training import predict_weighted_ensemble
    ensemble_pred, individual_predictions = predict_weighted_ensemble(
        weighted_ensemble_dict, 
        prediction_data, 
        available_regressors
    )
    
    rf_pred = individual_predictions['rf_prediction']
    ridge_pred = individual_predictions['ridge_prediction']
    xgb_pred = individual_predictions['xgb_prediction']
    
    # Create results for this month
    month_results = []
    for i in range(len(prediction_data)):
        date = prediction_data.iloc[i]['ds']
        date_str = date.strftime('%Y-%m-%d')
        
        result = {
            "month": month,
            "date": date_str,
            "predicted_energy": round(float(ensemble_pred[i]), 2),
            "prophet_prediction": round(float(prophet_pred_values[i]), 2),
            "rf_prediction": round(float(rf_pred[i]), 2),
            "ridge_prediction": round(float(ridge_pred[i]), 2),
            "xgb_prediction": round(float(xgb_pred[i]), 2),
            "ensemble_prediction": round(float(ensemble_pred[i]), 2),
            "prediction_lower": round(float(prophet_pred.iloc[i]['yhat_lower']), 2),
            "prediction_upper": round(float(prophet_pred.iloc[i]['yhat_upper']), 2),
            "actual_energy": None,  # No actual data for validation months
            "error": None,
            "percent_error": None,
            "weather_data": {
                "temperature_2m_mean": float(prediction_data.iloc[i].get('temperature_2m_mean', 0))
            }
        }
        month_results.append(result)
        
        print(f"{date_str}: Ensemble={ensemble_pred[i]:.2f}, Prophet={prophet_pred_values[i]:.2f}, "
              f"RF={rf_pred[i]:.2f}, Ridge={ridge_pred[i]:.2f}, XGB={xgb_pred[i]:.2f}")
    
    return {
        "month": month,
        "results": month_results,
        "total_predictions": len(month_results)
    }

def save_validation_results_to_csv(all_validation_results: List[Dict], facility_name: str, config: Any):
    """Save all validation results to CSV file"""
    # Create predictions directory if it doesn't exist
    predictions_dir = f'../data/{facility_name}/predictions'
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Prepare data for CSV
    csv_data = []
    for month_result in all_validation_results:
        for result in month_result['results']:
            csv_data.append({
                'month': result['month'],
                'date': result['date'],
                'predicted_energy': result['predicted_energy'],
                'prophet_prediction': result['prophet_prediction'],
                'rf_prediction': result['rf_prediction'],
                'ridge_prediction': result['ridge_prediction'],
                'xgb_prediction': result['xgb_prediction'],
                'ensemble_prediction': result['ensemble_prediction'],
                'prediction_lower': result['prediction_lower'],
                'prediction_upper': result['prediction_upper'],
                'actual_energy': result['actual_energy'],
                'error': result['error'],
                'percent_error': result['percent_error'],
                'temperature': result['weather_data']['temperature_2m_mean']
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(csv_data)
    csv_path = f'{predictions_dir}/post-validation-4-months.csv'
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS SAVED")
    print(f"{'='*60}")
    print(f"File: {csv_path}")
    print(f"Total predictions: {len(df)}")
    print(f"Months covered: {', '.join(df['month'].unique())}")
    
    # Print summary by month
    print(f"\nSummary by month:")
    for month in df['month'].unique():
        month_data = df[df['month'] == month]
        print(f"  {month}: {len(month_data)} predictions")
    
    return csv_path 