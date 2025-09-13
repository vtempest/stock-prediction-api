import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(y_true, y_pred, dataset_name):
    """Calculate and display model performance metrics"""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = mse ** 0.5
    
    non_zero_mask = np.abs(y_true_arr) > 1e-8
    
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true_arr[non_zero_mask] - y_pred_arr[non_zero_mask]) / y_true_arr[non_zero_mask])) * 100
        accuracy_percent = 100 - mape
    else:
        mape = np.nan
        accuracy_percent = np.nan
    
    predicted_avg = y_pred_arr.mean()
    actual_avg = y_true_arr.mean()
    
    if np.abs(actual_avg) > 1e-8:
        avg_percent_error = np.abs(predicted_avg - actual_avg) / actual_avg * 100
    else:
        avg_percent_error = np.nan
    
    r2_score = 1 - (mse / np.var(y_true_arr)) if np.var(y_true_arr) > 0 else np.nan
    
    print(f"\n{dataset_name} Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2_score:.4f}" if not np.isnan(r2_score) else "R² Score: N/A")
    
    if not np.isnan(mape):
        print(f"MAPE: {mape:.2f}%")
        print(f"Accuracy: {accuracy_percent:.2f}%")
    
    print(f"Average Predicted: {predicted_avg:.2f}")
    print(f"Average Actual: {actual_avg:.2f}")
    
    if not np.isnan(avg_percent_error):
        print(f"Average Percent Error: {avg_percent_error:.2f}%")
    
    print(f"Actual Data Range: {y_true_arr.min():.2f} to {y_true_arr.max():.2f}")
    print(f"Predicted Data Range: {y_pred_arr.min():.2f} to {y_pred_arr.max():.2f}")
    print(f"Data points with MAPE calculation: {non_zero_mask.sum()}/{len(y_true_arr)}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'accuracy': accuracy_percent,
        'avg_error': avg_percent_error,
        'r2_score': r2_score,
        'valid_mape_points': non_zero_mask.sum()
    }