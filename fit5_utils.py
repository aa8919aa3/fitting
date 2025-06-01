import numpy as np
import pandas as pd

def load_data(file_path, time_col, value_col, error_col=None):
    try:
        data = pd.read_csv(file_path)
        if time_col not in data.columns:
            print(f"Error: Time column '{time_col}' not found in {file_path}.")
            print(f"Available columns: {data.columns.tolist()}")
            return None, None, None
        if value_col not in data.columns:
            print(f"Error: Value column '{value_col}' not found in {file_path}.")
            print(f"Available columns: {data.columns.tolist()}")
            return None, None, None
        times = data[time_col].values
        values = data[value_col].values
        errors = None
        if error_col:
            if error_col in data.columns:
                errors_series = pd.to_numeric(data[error_col], errors='coerce')
                if errors_series.isnull().any():
                    print(f"Warning: Column '{error_col}' contains non-numeric values or NaNs. Errors will not be used for points where conversion failed.")
                errors_all = errors_series.values 
            else:
                print(f"Warning: Error column '{error_col}' not found. Proceeding without errors.")
                errors_all = None
        else:
            errors_all = None
        valid_mask = ~np.isnan(times) & ~np.isnan(values)
        if errors_all is not None:
            valid_mask &= ~np.isnan(errors_all)
            errors = errors_all[valid_mask]
        else:
            errors = None
        times = times[valid_mask]
        values = values[valid_mask]
        if len(times) == 0:
            print("Error: No valid data points found after removing NaNs.")
            return None, None, None
        print(f"Successfully loaded {len(times)} data points from {file_path}")
        return times, values, errors
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None, None

def detrend_data(times, values, order=1):
    if order is None or not isinstance(order, int) or order < 0:
        print("No detrending performed or invalid order.")
        return values, None 
    if len(times) <= order:
        print(f"Warning: Not enough data points ({len(times)}) to fit a polynomial of order {order}. Skipping detrending.")
        return values, None
    try:
        poly_coeffs = np.polyfit(times, values, order)
        trend = np.polyval(poly_coeffs, times)
        detrended_values = values - trend
        print(f"Data detrended using a polynomial of order {order} for Lomb-Scargle.")
        return detrended_values, poly_coeffs
    except Exception as e:
        print(f"Error during detrending: {e}. Returning original values.")
        return values, None

def calculate_r_squared(y_true, y_pred):
    valid_comparison = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(valid_comparison):
        return np.nan 
    y_true_valid = y_true[valid_comparison]
    y_pred_valid = y_pred[valid_comparison]
    if len(y_true_valid) == 0:
        return np.nan
    ss_res = np.sum((y_true_valid - y_pred_valid)**2)
    ss_tot = np.sum((y_true_valid - np.mean(y_true_valid))**2)
    if ss_tot == 0: 
        return 1.0 if ss_res < 1e-9 else 0.0 
    return 1 - (ss_res / ss_tot)

def custom_model_func(x, A, B, f, d, p, T, r, r2, C):
    """y = A·sin(2πf(x-d)-p) + B·cos(2πf(x-d)-p) + r(x-d) + r₂(x-d)² + C"""
    term1 = 2 * np.pi * f * (x - d) - p
    periodic_part = A * np.sin(term1) + B * np.cos(term1)
    linear_part = r * (x - d)
    quad_part = r2 * (x - d)**2
    return periodic_part + linear_part + quad_part + C
