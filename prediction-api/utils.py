import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64, int)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64, float)):
            # Handle NaN and infinite values
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            # Handle NaN and infinite values in arrays
            return [None if (np.isnan(x) or np.isinf(x)) else x for x in obj.tolist()]
        return json.JSONEncoder.default(self, obj)

def clean_nan_values(obj):
    """Recursively clean NaN and infinite values from dictionaries and lists"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, (np.float16, np.float32, np.float64, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.intc, np.intp, np.int8,
                         np.int16, np.int32, np.int64, np.uint8,
                         np.uint16, np.uint32, np.uint64, int)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return [None if (np.isnan(x) or np.isinf(x)) else x for x in obj.tolist()]
    # Additional check for any numeric value that might be NaN
    elif hasattr(obj, '__float__'):
        try:
            float_val = float(obj)
            if np.isnan(float_val) or np.isinf(float_val):
                return None
            return float_val
        except (ValueError, TypeError):
            pass
    return obj