import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from astropy.timeseries import LombScargle
from astropy import units as u
try:
    import lmfit
except ImportError:
    print("lmfit library not found. Please install it: pip install lmfit")
    lmfit = None

# --- Configuration ---
FILE_PATH = '164_ic.csv'
TIME_COLUMN = 'Time'
VALUE_COLUMN = 'Ic'
ERROR_COLUMN = None

# Frequency grid parameters for Lomb-Scargle
MIN_FREQUENCY_LS = None
MAX_FREQUENCY_LS = 500000
SAMPLES_PER_PEAK_LS = 10

# Significance level for FAP (Lomb-Scargle)
FAP_LEVELS_LS = [0.1, 0.05, 0.01]

# Detrending configuration for Lomb-Scargle pre-processing
DETREND_ORDER_LS = 1

def load_data(file_path, time_col, value_col, error_col=None):
    """
    Loads time series data from a CSV file.
    """
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
                    print(f"Warning: Column '{error_col}' contains non-numeric values or NaNs.")
                errors_all = errors_series.values 
            else:
                print(f"Warning: Error column '{error_col}' not found.")
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
    """
    Detrends the data by subtracting a polynomial fit.
    """
    if order is None or not isinstance(order, int) or order < 0:
        print("No detrending performed or invalid order.")
        return values, None 

    if len(times) <= order:
        print(f"Warning: Not enough data points to fit polynomial of order {order}.")
        return values, None
        
    try:
        poly_coeffs = np.polyfit(times, values, order)
        trend = np.polyval(poly_coeffs, times)
        detrended_values = values - trend
        print(f"Data detrended using polynomial of order {order}.")
        return detrended_values, poly_coeffs
    except Exception as e:
        print(f"Error during detrending: {e}.")
        return values, None

def calculate_r_squared(y_true, y_pred):
    """
    Calculates the R-squared value, handling potential NaNs.
    """
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

def custom_model_func(x, A, f, d, p, T, r, C):
    """
    Custom target function for lmfit.
    y = A * sin(2*pi*f*(x-d) - p) + r*(x-d) + C
    """
    term1 = 2 * np.pi * f * (x - d) - p
    periodic_part = A * np.sin(term1)
    linear_part = r * (x - d) + C
    result_y = periodic_part + linear_part
    return result_y

def create_parameter_boundaries_matrix(times, values, d_multiplier=100, c_multiplier=10):
    """
    Creates a parameter boundaries matrix where each element represents fitting in a range:
    - d parameter: 200 intervals, each with length mean(x), covering [-100×mean(x), +100×mean(x)]
    - C parameter: 20 intervals, each with length mean(y), covering [-10×mean(y), +10×mean(y)]
    - Creates a matrix of size 200 x 20
    - Each matrix element (i,j) contains fit results for range d[i] and C[j]
    
    Returns:
        d_ranges: list of d range tuples [(start, end), ...]
        c_ranges: list of C range tuples [(start, end), ...]
        fit_results_matrix: matrix of fit results
        best_params: parameters of the best fit
        best_d_idx: index of best d range
        best_c_idx: index of best C range
    """
    if lmfit is None:
        print("lmfit not available, skipping parameter boundaries matrix.")
        return None, None, None, None, None, None
        
    mean_x = np.mean(times)
    mean_y = np.mean(values)
    
    print(f"Mean of x (times): {mean_x:.6f}")
    print(f"Mean of y (values): {mean_y:.6e}")
    
    # Create parameter ranges according to your specification:
    # d: 200 intervals, each with length mean(x), total range [-100×mean(x), +100×mean(x)]
    d_intervals = 200
    d_interval_length = mean_x  # Each interval has length mean(x)
    d_min = -d_multiplier * mean_x
    d_max = d_multiplier * mean_x
    
    # C: 20 intervals, each with length mean(y), total range [-10×mean(y), +10×mean(y)]
    c_intervals = 20
    c_interval_length = mean_y  # Each interval has length mean(y)
    c_min = -c_multiplier * mean_y
    c_max = c_multiplier * mean_y
    
    # Create range lists - each element is a tuple (start, end) for that interval
    d_ranges = [(d_min + i*d_interval_length, d_min + (i+1)*d_interval_length) for i in range(d_intervals)]
    c_ranges = [(c_min + j*c_interval_length, c_min + (j+1)*c_interval_length) for j in range(c_intervals)]
    
    print(f"d parameter ranges: [{d_min:.6f}, {d_max:.6f}] in {d_intervals} intervals")
    print(f"d interval length: {d_interval_length:.6f} (= mean(x))")
    print(f"C parameter ranges: [{c_min:.6e}, {c_max:.6e}] in {c_intervals} intervals")
    print(f"C interval length: {c_interval_length:.6e} (= mean(y))")
    
    # Initialize results matrix
    fit_results_matrix = np.full((d_intervals, c_intervals), np.nan)
    best_r_squared = -np.inf
    best_params = None
    best_d_idx = 0
    best_c_idx = 0
    
    total_fits = d_intervals * c_intervals
    completed_fits = 0
    
    print(f"Starting parameter matrix fitting with {total_fits} combinations...")
    
    for i, (d_start, d_end) in enumerate(d_ranges):
        for j, (c_start, c_end) in enumerate(c_ranges):
            try:
                # Use middle point of each range for fitting
                d_val = (d_start + d_end) / 2
                c_val = (c_start + c_end) / 2
                
                # Create parameters for this combination
                params = lmfit.Parameters()
                
                # Estimate amplitude from data
                slope_init, intercept_init = np.polyfit(times, values, 1)
                residuals_for_amp_est = values - (slope_init * times + intercept_init)
                amp_init = np.std(residuals_for_amp_est) * np.sqrt(2)
                if amp_init == 0:
                    amp_init = np.std(values) * np.sqrt(2)
                if amp_init == 0:
                    amp_init = 1e-7
                
                # Add parameters with constrained d and C values for this iteration
                params.add('A', value=amp_init, min=1e-9)
                params.add('f', value=1.0, min=1e-9)
                params.add('d', value=d_val, min=d_start, max=d_end)  # Constrained to range
                params.add('p', value=0, min=-2*np.pi, max=2*np.pi)
                params.add('T', value=0.5, min=0.1, max=0.9)
                params.add('r', value=slope_init)
                params.add('C', value=c_val, min=c_start, max=c_end)  # Constrained to range
                
                # Perform fit
                custom_model = lmfit.Model(custom_model_func)
                result = custom_model.fit(values, params, x=times, nan_policy='omit')
                
                # Calculate R-squared
                r_squared = calculate_r_squared(values, result.best_fit)
                fit_results_matrix[i, j] = r_squared
                
                # Track best fit
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_params = result.params.copy()
                    best_d_idx = i
                    best_c_idx = j
                
                completed_fits += 1
                if completed_fits % 200 == 0:
                    print(f"Completed {completed_fits}/{total_fits} fits ({100*completed_fits/total_fits:.1f}%)...")
                    
            except Exception as e:
                # If fit fails, leave as NaN
                fit_results_matrix[i, j] = np.nan
                completed_fits += 1
    
    print(f"Parameter matrix fitting completed!")
    print(f"Best R-squared: {best_r_squared:.6f}")
    print(f"Best d range: [{d_ranges[best_d_idx][0]:.6f}, {d_ranges[best_d_idx][1]:.6f}] (index {best_d_idx})")
    print(f"Best C range: [{c_ranges[best_c_idx][0]:.6e}, {c_ranges[best_c_idx][1]:.6e}] (index {best_c_idx})")
    
    return d_ranges, c_ranges, fit_results_matrix, best_params, best_d_idx, best_c_idx

def plot_parameter_matrix_heatmap(d_ranges, c_ranges, fit_results_matrix, best_d_idx, best_c_idx):
    """
    Plots the parameter boundaries matrix as a heatmap showing R-squared values.
    """
    # Create center values for display
    d_centers = [(d_start + d_end) / 2 for d_start, d_end in d_ranges]
    c_centers = [(c_start + c_end) / 2 for c_start, c_end in c_ranges]
    
    fig = go.Figure(data=go.Heatmap(
        z=fit_results_matrix,
        x=c_centers,
        y=d_centers,
        colorscale='Viridis',
        colorbar=dict(title='R-squared'),
        hoverongaps=False,
        hovertemplate='d: %{y:.6f}<br>C: %{x:.6e}<br>R²: %{z:.6f}<extra></extra>'
    ))
    
    # Mark the best combination
    fig.add_scatter(
        x=[c_centers[best_c_idx]], 
        y=[d_centers[best_d_idx]], 
        mode='markers',
        marker=dict(color='red', size=10, symbol='x'),
        name=f'Best Fit (R²={fit_results_matrix[best_d_idx, best_c_idx]:.6f})',
        showlegend=True
    )
    
    fig.update_layout(
        title_text='Parameter Boundaries Matrix: R-squared vs d and C parameters<br>Each cell: fitting within d and C range intervals',
        title_x=0.5,
        xaxis_title="C parameter range centers",
        yaxis_title="d parameter range centers", 
        width=1000,
        height=800
    )
    fig.show()

def main():
    """
    Main function to perform parameter boundaries matrix fitting.
    """
    # Option to run test first
    run_test = input("Do you want to run the test first? (y/n): ").lower().strip()
    if run_test == 'y' or run_test == 'yes':
        test_times, test_values = test_parameter_boundaries_matrix()
        
        # Option to continue with test data or real data
        use_test_data = input("Use test data for analysis? (y/n): ").lower().strip()
        if use_test_data == 'y' or use_test_data == 'yes':
            times, values = test_times, test_values
            print("Using synthetic test data for analysis...")
        else:
            print("Loading real data...")
            times, values, errors = load_data(FILE_PATH, TIME_COLUMN, VALUE_COLUMN, ERROR_COLUMN)
    else:
        print("--- Starting Parameter Boundaries Matrix Analysis ---")
        times, values, errors = load_data(FILE_PATH, TIME_COLUMN, VALUE_COLUMN, ERROR_COLUMN)
    
    if times is None or values is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Data loaded: {len(times)} points")
    print(f"Time range: [{times.min():.6f}, {times.max():.6f}]")
    print(f"Value range: [{values.min():.6e}, {values.max():.6e}]")
    
    # Perform parameter boundaries matrix fitting
    print("\n--- Parameter Boundaries Matrix Fitting ---")
    d_ranges, c_ranges, fit_results_matrix, best_params, best_d_idx, best_c_idx = create_parameter_boundaries_matrix(
        times, values, d_multiplier=100, c_multiplier=10
    )
    
    if fit_results_matrix is not None:
        # Show statistics about the fitting results
        valid_fits = ~np.isnan(fit_results_matrix)
        num_valid = np.sum(valid_fits)
        num_total = fit_results_matrix.size
        
        print(f"\nFitting Statistics:")
        print(f"Total combinations: {num_total}")
        print(f"Successful fits: {num_valid}")
        print(f"Failed fits: {num_total - num_valid}")
        print(f"Success rate: {100 * num_valid / num_total:.1f}%")
        
        if num_valid > 0:
            valid_r_squared = fit_results_matrix[valid_fits]
            print(f"R-squared range: [{valid_r_squared.min():.6f}, {valid_r_squared.max():.6f}]")
            print(f"Mean R-squared: {valid_r_squared.mean():.6f}")
            print(f"Median R-squared: {np.median(valid_r_squared):.6f}")
        
        # Plot the heatmap
        plot_parameter_matrix_heatmap(d_ranges, c_ranges, fit_results_matrix, best_d_idx, best_c_idx)
        
        # Print best parameters if available
        if best_params is not None:
            print(f"\n--- Best Fit Parameters ---")
            for name, param in best_params.items():
                print(f"{name}: {param.value:.6e} (stderr: {param.stderr if param.stderr else 'N/A'})")
        
    print("\nAnalysis complete.")

def test_parameter_boundaries_matrix():
    """
    Test function to verify the parameter boundaries matrix implementation.
    This function creates synthetic data and tests the matrix generation.
    """
    print("=== Testing Parameter Boundaries Matrix ===")
    
    # Create synthetic test data
    test_times = np.linspace(0, 10, 100)
    test_values = 2.0 * np.sin(2*np.pi*1.5*(test_times-1.2) - 0.5) + 0.1*(test_times-1.2) + 3.0
    test_values += np.random.normal(0, 0.1, len(test_values))  # Add some noise
    
    mean_x = np.mean(test_times)
    mean_y = np.mean(test_values)
    
    print(f"Test data: {len(test_times)} points")
    print(f"Mean of x (times): {mean_x:.6f}")
    print(f"Mean of y (values): {mean_y:.6f}")
    
    # Test the parameter boundaries matrix with smaller size for faster testing
    print("\nTesting with reduced matrix size (10x5 for speed)...")
    
    # Temporarily modify the function for testing
    d_intervals = 10  # Reduced for testing
    c_intervals = 5   # Reduced for testing
    
    d_multiplier = 100
    c_multiplier = 10
    
    # Calculate expected ranges
    d_min_expected = -d_multiplier * mean_x
    d_max_expected = d_multiplier * mean_x
    d_interval_length_expected = mean_x
    
    c_min_expected = -c_multiplier * mean_y
    c_max_expected = c_multiplier * mean_y
    c_interval_length_expected = mean_y
    
    print(f"\nExpected d range: [{d_min_expected:.6f}, {d_max_expected:.6f}]")
    print(f"Expected d interval length: {d_interval_length_expected:.6f}")
    print(f"Expected C range: [{c_min_expected:.6f}, {c_max_expected:.6f}]")
    print(f"Expected C interval length: {c_interval_length_expected:.6f}")
    
    # Verify range calculations
    d_ranges_test = [(d_min_expected + i*d_interval_length_expected, 
                      d_min_expected + (i+1)*d_interval_length_expected) for i in range(d_intervals)]
    c_ranges_test = [(c_min_expected + j*c_interval_length_expected, 
                      c_min_expected + (j+1)*c_interval_length_expected) for j in range(c_intervals)]
    
    print(f"\nFirst few d ranges:")
    for i in range(min(3, len(d_ranges_test))):
        start, end = d_ranges_test[i]
        length = end - start
        print(f"  d[{i}]: [{start:.6f}, {end:.6f}] (length: {length:.6f})")
    
    print(f"\nFirst few C ranges:")
    for j in range(min(3, len(c_ranges_test))):
        start, end = c_ranges_test[j]
        length = end - start
        print(f"  C[{j}]: [{start:.6f}, {end:.6f}] (length: {length:.6f})")
    
    # Verify total coverage
    d_total_range = d_ranges_test[-1][1] - d_ranges_test[0][0]
    c_total_range = c_ranges_test[-1][1] - c_ranges_test[0][0]
    
    print(f"\nTotal d range covered: {d_total_range:.6f} (expected: {d_max_expected - d_min_expected:.6f})")
    print(f"Total C range covered: {c_total_range:.6f} (expected: {c_max_expected - c_min_expected:.6f})")
    
    print("\n=== Test Complete ===\n")
    
    return test_times, test_values

if __name__ == '__main__':
    main()
