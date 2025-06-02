import pandas as pd
import numpy as np
import plotly.graph_objects as go

try:
    import lmfit
except ImportError:
    print("lmfit library not found. Please install it: pip install lmfit")
    lmfit = None

# --- Configuration ---
FILE_PATH = '164_ic.csv'
TIME_COLUMN = 'Time'
VALUE_COLUMN = 'Ic'

def load_data(file_path, time_col, value_col):
    """
    Loads time series data from a CSV file.
    """
    try:
        data = pd.read_csv(file_path)
        times = data[time_col].values
        values = data[value_col].values
        
        valid_mask = ~np.isnan(times) & ~np.isnan(values)
        times = times[valid_mask]
        values = values[valid_mask]

        print(f"Successfully loaded {len(times)} data points from {file_path}")
        return times, values
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def calculate_r_squared(y_true, y_pred):
    """Calculates R-squared value."""
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
    """Custom target function: y = A * sin(2*pi*f*(x-d) - p) + r*(x-d) + C"""
    term1 = 2 * np.pi * f * (x - d) - p
    periodic_part = A * np.sin(term1)
    linear_part = r * (x - d) + C
    result_y = periodic_part + linear_part
    return result_y

def create_parameter_boundaries_matrix_demo(times, values, d_multiplier=100, c_multiplier=10):
    """
    Demo version with reduced matrix size for faster execution.
    Creates 20×10 matrix instead of 200×20 for demonstration.
    """
    if lmfit is None:
        print("lmfit not available.")
        return None, None, None, None, None, None
        
    mean_x = np.mean(times)
    mean_y = np.mean(values)
    
    print(f"Mean of x (times): {mean_x:.6f}")
    print(f"Mean of y (values): {mean_y:.6e}")
    
    # Create reduced parameter ranges for demo
    # d: 20 intervals (instead of 200), each with length mean(x)
    d_intervals = 20
    d_interval_length = mean_x
    d_min = -d_multiplier * mean_x
    d_max = d_multiplier * mean_x
    
    # C: 10 intervals (instead of 20), each with length mean(y) 
    c_intervals = 10
    c_interval_length = mean_y
    c_min = -c_multiplier * mean_y
    c_max = c_multiplier * mean_y
    
    # Create range lists
    d_ranges = [(d_min + i*d_interval_length, d_min + (i+1)*d_interval_length) for i in range(d_intervals)]
    c_ranges = [(c_min + j*c_interval_length, c_min + (j+1)*c_interval_length) for j in range(c_intervals)]
    
    print(f"DEMO: d parameter {d_intervals} intervals, length={d_interval_length:.6f}")
    print(f"DEMO: C parameter {c_intervals} intervals, length={c_interval_length:.6e}")
    print(f"Total combinations: {d_intervals * c_intervals}")
    
    # Initialize results matrix
    fit_results_matrix = np.full((d_intervals, c_intervals), np.nan)
    best_r_squared = -np.inf
    best_params = None
    best_d_idx = 0
    best_c_idx = 0
    
    total_fits = d_intervals * c_intervals
    completed_fits = 0
    
    print(f"Starting parameter matrix fitting...")
    
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
                params.add('d', value=d_val, min=d_start, max=d_end)
                params.add('p', value=0, min=-2*np.pi, max=2*np.pi)
                params.add('T', value=0.5, min=0.1, max=0.9)
                params.add('r', value=slope_init)
                params.add('C', value=c_val, min=c_start, max=c_end)
                
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
                if completed_fits % 50 == 0:
                    print(f"Completed {completed_fits}/{total_fits} fits ({100*completed_fits/total_fits:.1f}%)...")
                    
            except Exception as e:
                fit_results_matrix[i, j] = np.nan
                completed_fits += 1
    
    print(f"Parameter matrix fitting completed!")
    print(f"Best R-squared: {best_r_squared:.6f}")
    print(f"Best d range: [{d_ranges[best_d_idx][0]:.6f}, {d_ranges[best_d_idx][1]:.6f}] (index {best_d_idx})")
    print(f"Best C range: [{c_ranges[best_c_idx][0]:.6e}, {c_ranges[best_c_idx][1]:.6e}] (index {best_c_idx})")
    
    return d_ranges, c_ranges, fit_results_matrix, best_params, best_d_idx, best_c_idx

def plot_parameter_matrix_heatmap(d_ranges, c_ranges, fit_results_matrix, best_d_idx, best_c_idx):
    """Plots the parameter boundaries matrix as a heatmap."""
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
        title_text='Parameter Boundaries Matrix (DEMO 20×10): R-squared vs d and C parameters<br>Each cell: fitting within d and C range intervals',
        title_x=0.5,
        xaxis_title="C parameter range centers",
        yaxis_title="d parameter range centers", 
        width=1000,
        height=600
    )
    fig.show()

def main():
    """Main function for demonstration."""
    print("=== Parameter Boundaries Matrix Demo ===")
    
    # Load data
    times, values = load_data(FILE_PATH, TIME_COLUMN, VALUE_COLUMN)
    
    if times is None or values is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Data loaded: {len(times)} points")
    print(f"Time range: [{times.min():.6f}, {times.max():.6f}]")
    print(f"Value range: [{values.min():.6e}, {values.max():.6e}]")
    
    # Perform parameter boundaries matrix fitting (reduced size for demo)
    print("\n--- Parameter Boundaries Matrix Fitting (Demo Version) ---")
    d_ranges, c_ranges, fit_results_matrix, best_params, best_d_idx, best_c_idx = create_parameter_boundaries_matrix_demo(
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
        
    print("\nDemo complete.")

if __name__ == '__main__':
    main()
