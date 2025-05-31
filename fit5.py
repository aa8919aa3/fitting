import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff # Added import for plotly.figure_factory
from astropy.timeseries import LombScargle
from astropy import units as u
try:
    import lmfit
except ImportError:
    print("lmfit library not found. Please install it: pip install lmfit")
    lmfit = None


# --- Configuration ---
FILE_PATH = 'Kay2.csv'
# !!! IMPORTANT: Replace these with the actual column names from your CSV file !!!
TIME_COLUMN = 'Time'  # Updated to match provided CSV
VALUE_COLUMN = 'Ic' # Updated to match provided CSV
ERROR_COLUMN = None    # Placeholder: e.g., 'Error', 'Uncertainty'. Set to None if not available.

# Frequency grid parameters for Lomb-Scargle
MIN_FREQUENCY_LS = None  # e.g., 0.1. If None, autopower will determine it.
MAX_FREQUENCY_LS = 500000 # Example: Set a reasonable maximum frequency for LS.
SAMPLES_PER_PEAK_LS = 10 # Higher values give finer frequency resolution for LS.

# Significance level for FAP (Lomb-Scargle)
FAP_LEVELS_LS = [0.1, 0.05, 0.01] # False Alarm Probability levels for LS.

# Detrending configuration for Lomb-Scargle pre-processing
DETREND_ORDER_LS = 1 # Order of polynomial to fit for detrending before LS. 1 for linear.

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
    """
    Detrends the data by subtracting a polynomial fit.
    Returns detrended values and polynomial coefficients.
    """
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

# --- Plotly plotting functions (separated) ---

def plot_lomb_scargle_periodogram(ls_frequency, ls_power, best_frequency_ls, fap_levels_plot_ls, power_thresholds_plot_ls, title_prefix):
    """
    Plots the Lomb-Scargle Periodogram.
    """
    fig = go.Figure()
    if ls_frequency is not None and len(ls_frequency) > 0:
        fig.add_trace(go.Scatter(x=ls_frequency, y=ls_power, mode='lines', name='LS Power', 
                                 line=dict(color='cornflowerblue')))
        fig.add_vline(x=best_frequency_ls, line_dash="dash", line_color="red", 
                      annotation_text=f'Best Freq: {best_frequency_ls:.4f}', 
                      annotation_position="top right")
        if fap_levels_plot_ls is not None and power_thresholds_plot_ls is not None:
            if hasattr(fap_levels_plot_ls, '__iter__') and hasattr(power_thresholds_plot_ls, '__iter__') and len(fap_levels_plot_ls) == len(power_thresholds_plot_ls):
                for level, thresh in zip(fap_levels_plot_ls, power_thresholds_plot_ls):
                    fig.add_hline(y=thresh, line_dash="dot", line_color="grey", 
                                  annotation_text=f'FAP {level*100:.1f}% ({thresh:.2f})', 
                                  annotation_position="bottom right")
    else:
        fig.add_annotation(text="Periodogram not computed.", xref="x domain", yref="y domain",
                           x=0.5, y=0.5, showarrow=False)
    
    fig.update_layout(title_text=f"{title_prefix}: Lomb-Scargle Periodogram", title_x=0.5,
                      xaxis_title=f"Frequency (cycles / time unit)",
                      yaxis_title="Lomb-Scargle Power",
                      showlegend=True)
    fig.show()

def plot_data_with_ls_fit(original_times, original_values, original_errors, t_fit_ls, y_ls_fit_on_original_scale_for_plot, best_period_ls, r_squared_ls, value_column_name, title_prefix):
    """
    Plots the original data with the Lomb-Scargle sinusoidal fit.
    """
    fig = go.Figure()
    if original_errors is not None:
        fig.add_trace(go.Scatter(x=original_times, y=original_values, mode='lines+markers', name='Original Data (with errors)',
                                 marker=dict(color='grey', size=3, opacity=0.5),
                                 error_y=dict(type='data', array=original_errors, visible=True, color='grey', thickness=0.5)))
    else:
        fig.add_trace(go.Scatter(x=original_times, y=original_values, mode='lines+markers', name='Original Data',
                                 marker=dict(color='grey', size=3, opacity=0.7)))
    
    ls_label = 'Lomb-Scargle Fit'
    if r_squared_ls is not None and not np.isnan(r_squared_ls):
        ls_label += f' (R²={r_squared_ls:.4f})'
    fig.add_trace(go.Scatter(x=t_fit_ls, y=y_ls_fit_on_original_scale_for_plot, mode='lines', name=ls_label,
                             line=dict(color='dodgerblue', width=1.5)))
    
    fig.update_layout(title_text=f"{title_prefix}: Data with Lomb-Scargle Sinusoidal Fit (Period = {best_period_ls:.4f})", title_x=0.5,
                      xaxis_title="Time",
                      yaxis_title=value_column_name.capitalize(),
                      showlegend=True)
    fig.show()

def plot_phase_folded_data(ls_input_times, ls_input_values_for_phase_plot, best_period_ls, ls_input_errors_for_phase_plot, value_column_name, title_prefix):
    """
    Plots the phase-folded data.
    """
    fig = go.Figure()
    if best_period_ls != 0 and not np.isinf(best_period_ls) and not np.isnan(best_period_ls) and len(ls_input_times) > 0:
        phase_individual = (ls_input_times / best_period_ls) % 1.0
        plot_phase_individual = np.concatenate((phase_individual - 1, phase_individual, phase_individual + 1))
        plot_values_individual = np.concatenate((ls_input_values_for_phase_plot, ls_input_values_for_phase_plot, ls_input_values_for_phase_plot))
        
        if ls_input_errors_for_phase_plot is not None:
            plot_errors_individual = np.concatenate((ls_input_errors_for_phase_plot, ls_input_errors_for_phase_plot, ls_input_errors_for_phase_plot))
            fig.add_trace(go.Scatter(x=plot_phase_individual, y=plot_values_individual, mode='markers', name='Data points',
                                     marker=dict(color='grey', size=3, opacity=0.5),
                                     error_y=dict(type='data', array=plot_errors_individual, visible=True, color='grey', thickness=0.5)))
        else:
            fig.add_trace(go.Scatter(x=plot_phase_individual, y=plot_values_individual, mode='markers', name='Data points',
                                     marker=dict(color='grey', size=3, opacity=0.7)))

        num_bins = 20 
        phase_bins = np.linspace(0, 1, num_bins + 1)
        bin_centers = (phase_bins[:-1] + phase_bins[1:]) / 2
        
        single_cycle_phase_for_binning = (ls_input_times / best_period_ls) % 1.0
        
        mean_binned_values = np.full(num_bins, np.nan)
        if len(ls_input_values_for_phase_plot) > 0:
            binned_statistic_result = np.histogram(single_cycle_phase_for_binning, bins=phase_bins, weights=ls_input_values_for_phase_plot)
            value_sums_in_bin = binned_statistic_result[0]
            counts_in_bin = np.histogram(single_cycle_phase_for_binning, bins=phase_bins)[0]
            
            valid_bins_mask = counts_in_bin > 0
            mean_binned_values[valid_bins_mask] = value_sums_in_bin[valid_bins_mask] / counts_in_bin[valid_bins_mask]

        if not np.all(np.isnan(mean_binned_values)):
            fig.add_trace(go.Scatter(x=np.concatenate((bin_centers, bin_centers + 1)), 
                                     y=np.concatenate((mean_binned_values, mean_binned_values)), 
                                     mode='lines', name='Mean Phase-Folded Profile', # Corrected mode from '' to 'lines'
                                     line=dict(color='blue', width=1.5, dash='dash')))
        
        fig.update_xaxes(title_text=f"Phase (Period = {best_period_ls:.6f} time units)", range=[0, 2])
        fig.update_yaxes(title_text=f"{value_column_name.capitalize()} (Detrended for LS if applicable)")
    else:
        fig.add_annotation(text="Phase-folded plot skipped<br>due to invalid period or no data.", xref="x domain", yref="y domain",
                           x=0.5, y=0.5, showarrow=False)

    fig.update_layout(title_text=f"{title_prefix}: Phase-Folded Data", title_x=0.5, showlegend=True)
    fig.show()

def plot_ls_fit_parameters_text(best_frequency_ls, best_period_ls, r_squared_ls, ls_amplitude, ls_phase_rad, ls_offset_val, title_prefix):
    """
    Displays Lomb-Scargle fit parameters as text in a Plotly figure.
    """
    fig = go.Figure()
    param_text = (
        f"<b>Lomb-Scargle Fit Parameters:</b><br><br>"
        f"  Best Frequency: {best_frequency_ls:.4f}<br>"
        f"  Best Period: {best_period_ls:.6f}<br>"
        f"  R-squared: {r_squared_ls:.4f}<br>"
        f"  Amplitude (sinusoid): {ls_amplitude:.3e}<br>"
        f"  Phase (sinusoid, radians): {ls_phase_rad:.3f}<br>"
        f"  Offset (mean term): {ls_offset_val:.3e}<br><br>"
        f"Note: Detecting quantitative phase shift<br>"
        f"over time requires more advanced analysis."
    )
    fig.add_annotation(text=param_text, xref="paper", yref="paper",
                       x=0.05, y=0.95, showarrow=False, align="left",
                       bgcolor="wheat", opacity=0.5, bordercolor="black", borderwidth=1,
                       font=dict(size=12))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(title_text=f"{title_prefix}: Lomb-Scargle Fit Parameters", title_x=0.5,)
    fig.show()

def custom_model_func(x, A, f, d, p, T, r, C):
    """
    Simplified custom target function for lmfit.
    y = A * sin(2*pi*f*(x-d) - p) + r*(x-d) + C
    """
    term1 = 2 * np.pi * f * (x - d) - p
    periodic_part = A * np.sin(term1)
    linear_part = r * (x - d) + C
    result_y = periodic_part + linear_part

    return result_y

# Adjust lmfit parameters and boundaries for better fit
lmfit_params = lmfit.Parameters()
lmfit_params.add('A', value=1e-6, min=1e-8, max=1e-4)
lmfit_params.add('f', value=5e4, min=1e-2, max=1e6)
lmfit_params.add('d', value=0, min=-0.5, max=0.5)
lmfit_params.add('p', value=0, min=-np.pi, max=np.pi)
lmfit_params.add('T', value=0.5, min=0.1, max=0.9)
lmfit_params.add('r', value=0, min=-1e-4, max=1e-4)
lmfit_params.add('C', value=0, min=-1e-5, max=1e-5)

def plot_custom_model_fit_separate(times, original_data, lmfit_result, errors=None, 
                                   title=r"$y=\frac{A\sin(2 \pi f(x-d)-p)}{\sqrt{1-T \sin^2\left(\frac{2 \pi f(x-d)-p}{2}\right)}}+r(x-d)+C$", 
                                   value_column_name='Value', t_ls_fit=None, y_ls_model_on_original_scale=None, r_squared_ls=None, r_squared_custom=None,
                                   best_frequency_ls=None, best_period_ls=None, ls_amplitude=None, ls_phase_rad=None, ls_offset_val=None):
    """
    Plots the original data with the best fit from the custom lmfit model,
    and optionally the Lomb-Scargle fit, including R-squared values in legends, using Plotly.
    """
    fig = go.Figure()

    if errors is not None:
        fig.add_trace(go.Scatter(x=times, y=original_data, mode='lines+markers', name='Original Data (with errors)',
                                 marker=dict(color='grey', size=3, opacity=0.5),
                                 error_y=dict(type='data', array=errors, visible=True, color='grey', thickness=0.5)))
    else:
        fig.add_trace(go.Scatter(x=times, y=original_data, mode='lines+markers', name='Original Data',
                                 marker=dict(color='grey', size=3, opacity=0.7)))

    if t_ls_fit is not None and y_ls_model_on_original_scale is not None:
        ls_plot_label = 'Lomb-Scargle Fit'
        if r_squared_ls is not None and not np.isnan(r_squared_ls):
            ls_plot_label += f' (R²={r_squared_ls:.4f})'
        fig.add_trace(go.Scatter(x=t_ls_fit, y=y_ls_model_on_original_scale, mode='lines', name=ls_plot_label,
                                 line=dict(color='lightcoral', width=1.5, dash='dash')))
    param_text = (
        f"<b>Lomb-Scargle Fit Parameters:</b><br><br>"
        f"  Best Frequency: {best_frequency_ls:.4f}<br>"
        f"  Best Period: {best_period_ls:.6f}<br>"
        f"  R-squared: {r_squared_ls:.4f}<br>"
        f"  Amplitude (sinusoid): {ls_amplitude:.3e}<br>"
        f"  Phase (sinusoid, radians): {ls_phase_rad:.3f}<br>"
        f"  Offset (mean term): {ls_offset_val:.3e}<br><br>"
        f"Note: Detecting quantitative phase shift<br>"
        f"over time requires more advanced analysis."
    )
    fig.add_annotation(text=param_text, xref="paper", yref="paper",
                       xanchor="left", yanchor="top",   
                       x=1, y=0.75, showarrow=False, align="left",
                       bgcolor="wheat", opacity=0.5, bordercolor="black", borderwidth=1,
                       font=dict(size=12))
    
    sort_indices_times = np.argsort(times)
    times_sorted = times[sort_indices_times]
    custom_fit_line = lmfit_result.model.eval(params=lmfit_result.params, x=times_sorted)
    
    custom_plot_label = 'Best Custom Fit'
    if r_squared_custom is not None and not np.isnan(r_squared_custom):
        custom_plot_label += f' (R²={r_squared_custom:.4f})'
    fig.add_trace(go.Scatter(x=times_sorted, y=custom_fit_line, mode='lines', name=custom_plot_label,
                             line=dict(color='orangered', width=1.5)))
    
    fig.update_layout(title_text=f"Custom Model Fit: {title}", title_x=0.5,
                      xaxis_title="Time",
                      yaxis_title=value_column_name.capitalize(),
                      showlegend=True)
    fig.show()

def plot_residuals_vs_time(times, residuals, label, integral, title="Residuals vs. Time"):
    """
    Plots residuals from a model against time using Plotly.
    """
    fig = go.Figure()
    res_plot_label = f'{label} Residuals'
    if integral is not None and not np.isnan(integral):
        res_plot_label += f' (Integral={integral:.2e})'
    fig.add_trace(go.Scatter(x=times, y=residuals, mode='lines+markers', name=res_plot_label,
                             marker=dict(size=3, opacity=0.7)))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig.update_layout(title_text=title, title_x=0.5,
                      xaxis_title="Time",
                      yaxis_title="Residual (Data - Model)",
                      showlegend=True)
    fig.show()

def plot_combined_residuals_vs_time(times1, residuals1, label1, integral1, 
                                    times2, residuals2, label2, integral2, 
                                    title="Combined Residuals vs. Time"):
    """
    Plots residuals from two different models against time in the same figure for comparison.
    """
    fig = go.Figure()

    res1_plot_label = f'{label1} Residuals'
    if integral1 is not None and not np.isnan(integral1):
        res1_plot_label += f' (Integral={integral1:.2e})'
    fig.add_trace(go.Scatter(x=times1, y=residuals1, mode='lines+markers', name=res1_plot_label,
                             marker=dict(size=3, opacity=0.7, color='dodgerblue')))
    
    res2_plot_label = f'{label2} Residuals'
    if integral2 is not None and not np.isnan(integral2):
         res2_plot_label += f' (Integral={integral2:.2e})'
    if times2 is not None and residuals2 is not None and not np.all(np.isnan(residuals2)):
        fig.add_trace(go.Scatter(x=times2, y=residuals2, mode='lines+markers', name=res2_plot_label,
                                 marker=dict(size=3, opacity=0.7, symbol='x', color='orangered')))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig.update_layout(title_text=title, title_x=0.5,
                      xaxis_title="Time",
                      yaxis_title="Residual (Data - Model)",
                      showlegend=True)
    fig.show()


def plot_residuals_histogram(residuals_list, labels_list, title="Histogram of Residuals"):
    """
    Plots histograms of residuals from one or more models using Plotly's figure_factory.create_distplot.
    """
    valid_data = []
    valid_labels = []
    for i, res_data in enumerate(residuals_list):
        res_array = np.array(res_data)
        res_valid = res_array[~np.isnan(res_array)]
        if len(res_valid) > 0:
            valid_data.append(res_valid)
            valid_labels.append(labels_list[i])
        else:
            print(f"Warning: Label '{labels_list[i]}' has no valid data after NaN removal. Skipping.")

    if not valid_data:
        print("Error: No valid data available to create the distribution plot.")
        return
    def _bin_size(valid_data):
        bin_edges_fd = np.histogram_bin_edges(valid_data, bins='rice')
        bin_size = float(f"{float(bin_edges_fd[1] - bin_edges_fd[0]):.2e}")
        if len(bin_edges_fd) > 1:
            bin_size = float(f"{float(bin_edges_fd[1] - bin_edges_fd[0]):.2e}")
        else:
            bin_size = None #讓 Plotly 自動決定
        return bin_size

    fig = ff.create_distplot(valid_data, valid_labels, show_hist=True, show_rug=True, bin_size=_bin_size(valid_data))

    fig.update_layout(title_text=title, title_x=0.5,
                      xaxis_title="Residual Value",
                      yaxis_title="Density",
                      showlegend=True)
    fig.show()

def plot_correlation_heatmap(lmfit_result):
    """
    Plots the parameter correlation matrix from an lmfit result as a heatmap using Plotly.
    """
    if lmfit is None:
        print("lmfit not available, skipping correlation heatmap.")
        return

    if lmfit_result is None or not hasattr(lmfit_result, 'params'):
        print("No lmfit result or parameters found to plot correlation heatmap.")
        return

    vary_params = [name for name, param in lmfit_result.params.items() if param.vary]
    
    if not vary_params:
        print("No varying parameters in the fit, skipping correlation heatmap.")
        return

    correl_matrix = np.zeros((len(vary_params), len(vary_params)))
    for i, p1_name in enumerate(vary_params):
        correl_matrix[i, i] = 1.0
        for j, p2_name in enumerate(vary_params[i+1:], start=i+1):
            if lmfit_result.params[p1_name].correl is not None and \
               p2_name in lmfit_result.params[p1_name].correl:
                correlation_value = lmfit_result.params[p1_name].correl[p2_name]
                correl_matrix[i, j] = correlation_value
                correl_matrix[j, i] = correlation_value
            else:
                correl_matrix[i,j] = 0 
                correl_matrix[j,i] = 0

    correl_df = pd.DataFrame(correl_matrix, index=vary_params, columns=vary_params)

    if correl_df.empty:
        print("Correlation matrix is empty, skipping heatmap.")
        return

    fig = go.Figure(data=go.Heatmap(
        z=correl_df.values,
        x=correl_df.columns.tolist(),
        y=correl_df.index.tolist(),
        colorscale='RdBu',
        zmin=-1, zmax=1,
        colorbar=dict(title='Correlation'),
        texttemplate="%{z:.2f}",
        textfont={"size":10}
    ))

    fig.update_layout(
        title_text='Parameter Correlation Heatmap (lmfit Custom Model)',
        title_x=0.5,
        xaxis_title="Parameter",
        yaxis_title="Parameter",
    )
    fig.show()


def main():
    """
    Main function to perform Lomb-Scargle analysis and custom model fitting.
    """
    print("--- Starting Lomb-Scargle Analysis ---")
    original_times, original_values, original_errors = load_data(FILE_PATH, TIME_COLUMN, VALUE_COLUMN, ERROR_COLUMN)
    
    if original_times is None or original_values is None:
        print("Failed to load data. Exiting.")
        return

    ls_input_times = original_times.copy()
    ls_input_values = original_values.copy()
    ls_input_errors = original_errors.copy() if original_errors is not None else None
    ls_detrend_coeffs = None


    if DETREND_ORDER_LS is not None:
        ls_input_values, ls_detrend_coeffs = detrend_data(ls_input_times, ls_input_values, order=DETREND_ORDER_LS)
    
    dy_ls = ls_input_errors if ls_input_errors is not None else None
    ls = LombScargle(ls_input_times, ls_input_values, dy=dy_ls, fit_mean=True, center_data=True)

    current_min_freq_ls = MIN_FREQUENCY_LS
    if current_min_freq_ls is None and len(ls_input_times) > 1:
        time_span_ls = ls_input_times.max() - ls_input_times.min()
        if time_span_ls > 0:
            current_min_freq_ls = 0.5 / time_span_ls 
            print(f"Auto-setting MIN_FREQUENCY_LS to ~{current_min_freq_ls:.2f}")
        else:
            print("Warning: Cannot auto-determine MIN_FREQUENCY_LS.")

    ls_frequency, ls_power = ls.autopower(minimum_frequency=current_min_freq_ls, 
                                          maximum_frequency=MAX_FREQUENCY_LS,
                                          samples_per_peak=SAMPLES_PER_PEAK_LS)
    
    best_frequency_ls = 0 
    best_period_ls = float('inf')
    highest_ls_peak_power = 0
    y_ls_pred_on_original_scale_at_orig_times = np.full_like(original_values, np.nan) 
    ls_residuals = np.full_like(original_values, np.nan)
    r_squared_ls = np.nan 
    integral_ls_residuals = np.nan
    ls_amplitude_fit = np.nan
    ls_phase_fit_rad = np.nan
    ls_offset_fit = np.nan


    if ls_frequency is not None and len(ls_frequency) > 0:
        best_power_index_ls = np.argmax(ls_power)
        best_frequency_ls = ls_frequency[best_power_index_ls]
        highest_ls_peak_power = ls_power[best_power_index_ls]
        if best_frequency_ls != 0: 
            best_period_ls = 1.0 / best_frequency_ls
            
            ls_offset_fit = ls.offset()
            params_ls_sinusoid = ls.model_parameters(best_frequency_ls)
            ls_amplitude_fit = np.sqrt(params_ls_sinusoid[0]**2 + params_ls_sinusoid[1]**2)
            ls_phase_fit_rad = np.arctan2(params_ls_sinusoid[1], params_ls_sinusoid[0])


    else:
        print("Failed to compute Lomb-Scargle periodogram.")
            
    print(f"\n--- Lomb-Scargle Results ---")
    if ls_frequency is not None and len(ls_frequency) > 0 :
      print(f"Highest LS peak power: {highest_ls_peak_power:.4f}")
    print(f"Best LS frequency: {best_frequency_ls:.6f} cycles / time unit")
    print(f"Corresponding LS period: {best_period_ls:.6f} time units")
    print(f"Fitted LS Sinusoid Amplitude: {ls_amplitude_fit:.3e}")
    print(f"Fitted LS Sinusoid Phase (radians): {ls_phase_fit_rad:.3f}")
    print(f"Fitted LS Offset: {ls_offset_fit:.3e}")


    if best_frequency_ls != 0 and not np.isinf(best_period_ls) :
        ls_model_pred_on_detrended_at_orig_times = ls.model(original_times, best_frequency_ls)
        y_ls_pred_on_original_scale_at_orig_times = ls_model_pred_on_detrended_at_orig_times 
        if ls_detrend_coeffs is not None:
            original_trend_at_orig_times = np.polyval(ls_detrend_coeffs, original_times)
            y_ls_pred_on_original_scale_at_orig_times = original_trend_at_orig_times + ls_model_pred_on_detrended_at_orig_times
        
        r_squared_ls = calculate_r_squared(original_values, y_ls_pred_on_original_scale_at_orig_times)
        print(f"R-squared for Lomb-Scargle sinusoidal fit (on original scale): {r_squared_ls:.4f}")
        
        ls_residuals = original_values - y_ls_pred_on_original_scale_at_orig_times
        
        if len(original_times) > 1 and not np.all(np.isnan(ls_residuals)):
            sort_idx_ls = np.argsort(original_times)
            if hasattr(np, 'trapezoid'):
                integral_ls_residuals = np.trapezoid(ls_residuals[sort_idx_ls], original_times[sort_idx_ls])
            else:
                integral_ls_residuals = np.trapz(ls_residuals[sort_idx_ls], original_times[sort_idx_ls])
            print(f"Integral of Lomb-Scargle residuals: {integral_ls_residuals:.4e}")
        else:
            print("Could not calculate integral of Lomb-Scargle residuals (insufficient data or all NaNs).")


    fap_levels_plot_ls = None
    power_thresholds_plot_ls = None
    if ls_frequency is not None and len(ls_frequency) > 0 and highest_ls_peak_power > 0: 
        try:
            fap_value_peak_ls = ls.false_alarm_probability(highest_ls_peak_power, method='baluev')
            print(f"FAP for highest LS peak (Baluev): {fap_value_peak_ls:.2e}")
            power_thresholds_ls = ls.false_alarm_level(FAP_LEVELS_LS, method='baluev')
            for level, thresh in zip(FAP_LEVELS_LS, power_thresholds_ls):
                print(f"Power threshold for {level*100:.1f}% FAP (LS): {thresh:.4f}")
            fap_levels_plot_ls = FAP_LEVELS_LS
            power_thresholds_plot_ls = power_thresholds_ls
        except Exception as e:
            print(f"Could not calculate FAP for LS: {e}")
    elif ls_frequency is not None and len(ls_frequency) > 0 :
             print("Skipping FAP calculation for LS as best power is not positive.")

    # Define t_fit_ls for plotting the LS model smoothly
    time_span_plot_ls = original_times.max() - original_times.min()
    if time_span_plot_ls > 0:
        t_fit_ls_min = original_times.min() - 0.05 * time_span_plot_ls 
        t_fit_ls_max = original_times.max() + 0.05 * time_span_plot_ls
        t_fit_ls = np.linspace(t_fit_ls_min, t_fit_ls_max, num=max(1000, 2*len(original_times)))
    else: 
        t_fit_ls = np.sort(np.unique(original_times)) 
        if len(t_fit_ls) == 1: 
             t_fit_ls = np.array([t_fit_ls[0]-0.5, t_fit_ls[0], t_fit_ls[0]+0.5])

    # LS model evaluated on t_fit_ls (for the detrended data scale)
    y_ls_model_fit_on_detrended = ls.model(t_fit_ls, best_frequency_ls) if best_frequency_ls !=0 else np.zeros_like(t_fit_ls)
    
    # Convert LS model to original scale for plotting
    y_ls_fit_on_original_scale_for_plot = y_ls_model_fit_on_detrended
    if ls_detrend_coeffs is not None:
        trend_for_t_fit_ls = np.polyval(ls_detrend_coeffs, t_fit_ls)
        y_ls_fit_on_original_scale_for_plot = trend_for_t_fit_ls + y_ls_model_fit_on_detrended
    
    # --- Plotting Lomb-Scargle results separately ---
    if ls_frequency is not None and len(ls_frequency) > 0 :
        plot_lomb_scargle_periodogram(ls_frequency, ls_power, best_frequency_ls, fap_levels_plot_ls, power_thresholds_plot_ls, "Lomb-Scargle Analysis")
        # plot_data_with_ls_fit(original_times, original_values, original_errors, t_fit_ls, y_ls_fit_on_original_scale_for_plot, best_period_ls, r_squared_ls, VALUE_COLUMN, "Lomb-Scargle Analysis")
        plot_phase_folded_data(ls_input_times, ls_input_values, best_period_ls, ls_input_errors, VALUE_COLUMN, "Lomb-Scargle Analysis")
        # plot_ls_fit_parameters_text(best_frequency_ls, best_period_ls, r_squared_ls, ls_amplitude_fit, ls_phase_fit_rad, ls_offset_fit, "Lomb-Scargle Analysis")

    
    print("\n--- Starting Custom Model Fit (lmfit) ---")
    if lmfit is None:
        print("lmfit library is not installed. Skipping custom model fit.")
        if not np.all(np.isnan(ls_residuals)): 
             plot_residuals_vs_time(original_times, ls_residuals, "Lomb-Scargle", integral_ls_residuals, title="Lomb-Scargle Residuals vs. Time")
             plot_residuals_histogram([ls_residuals], ["Lomb-Scargle"], title="Lomb-Scargle Residuals Histogram")
        print("\nAnalysis complete (Lomb-Scargle part only).")
        return

    custom_model = lmfit.Model(custom_model_func)
    params = lmfit.Parameters()

    slope_init, intercept_init = np.polyfit(original_times, original_values, 1)
    residuals_for_amp_est = original_values - (slope_init * original_times + intercept_init)
    amp_init = np.std(residuals_for_amp_est) * np.sqrt(2) 
    if amp_init == 0 : amp_init = np.std(original_values) * np.sqrt(2) 
    if amp_init == 0 : amp_init = 1e-7 

    params.add('A', value=amp_init if amp_init > 0 else 1e-7, min=1e-9) 
    params.add('f', value=best_frequency_ls if best_frequency_ls > 1e-9 else 1.0, min=1e-9) 
    params.add('d', value=original_times.min()) 
    params.add('p', value=0, min=-2*np.pi, max=2*np.pi) 
    params.add('T', value=0.5, min=0.1, max=0.9)  
    params.add('r', value=slope_init) 
    params.add('C', value = np.polyval([slope_init, intercept_init], params['d'].value))

    print("\nInitial parameters for custom fit:")
    params.pretty_print()

    fit_weights = None
    times_for_fit = original_times 
    data_for_fit = original_values
    custom_residuals = np.full_like(data_for_fit, np.nan) 
    r_squared_custom = np.nan 
    integral_custom_residuals = np.nan
    lmfit_result_obj = None 


    if original_errors is not None and len(original_errors) == len(original_values):
        valid_error_mask = original_errors > 1e-9 
        if np.all(valid_error_mask):
            fit_weights = 1.0 / original_errors 
            print(f"Using measurement errors as weights for lmfit for all {len(data_for_fit)} points.")
        elif np.any(valid_error_mask):
            times_for_fit = original_times[valid_error_mask]
            data_for_fit = original_values[valid_error_mask]
            errors_for_fit = original_errors[valid_error_mask]
            fit_weights = 1.0 / errors_for_fit
            print(f"Warning: Some measurement errors are not positive. Using weights only for {len(data_for_fit)} points with valid errors.")
        else:
            print("Warning: All measurement errors are non-positive or zero. Not using weights for lmfit.")
    else:
        print("No valid measurement errors provided. Performing unweighted fit for lmfit.")
        
    try:
        lmfit_result_obj = custom_model.fit(data_for_fit, params, x=times_for_fit, weights=fit_weights, nan_policy='omit')
        print("\n--- Custom Model Fit Report (lmfit) ---")
        print(lmfit_result_obj.fit_report())

        r_squared_custom = calculate_r_squared(data_for_fit, lmfit_result_obj.best_fit)
        print(f"R-squared for custom model fit (lmfit): {r_squared_custom:.4f}")

        custom_model_predictions_for_residuals = lmfit_result_obj.model.eval(params=lmfit_result_obj.params, x=times_for_fit)
        custom_residuals = data_for_fit - custom_model_predictions_for_residuals
        
        if len(times_for_fit) > 1 and not np.all(np.isnan(custom_residuals)):
            sort_idx_custom = np.argsort(times_for_fit)
            if hasattr(np, 'trapezoid'):
                integral_custom_residuals = np.trapezoid(custom_residuals[sort_idx_custom], times_for_fit[sort_idx_custom])
            else:
                integral_custom_residuals = np.trapz(custom_residuals[sort_idx_custom], times_for_fit[sort_idx_custom])
            print(f"Integral of custom model residuals: {integral_custom_residuals:.4e}")
        else:
            print("Could not calculate integral of custom model residuals (insufficient data or all NaNs).")

        # --- Plotting custom model results separately ---
        plot_custom_model_fit_separate(original_times, original_values, lmfit_result_obj, errors=original_errors,
                                      value_column_name=VALUE_COLUMN,
                                      t_ls_fit=t_fit_ls, y_ls_model_on_original_scale=y_ls_fit_on_original_scale_for_plot,
                                      r_squared_ls=r_squared_ls, r_squared_custom=r_squared_custom,
                                      best_frequency_ls=best_frequency_ls, best_period_ls=best_period_ls, ls_amplitude=ls_amplitude_fit, ls_phase_rad=ls_phase_fit_rad, ls_offset_val=ls_offset_fit) 
        
        # plot_correlation_heatmap(lmfit_result_obj)

    except Exception as e:
        print(f"\nAn error occurred during custom model fitting with lmfit: {e}")
        import traceback
        traceback.print_exc()
    
    # --- Plotting residuals separately for comparison ---
    plot_combined_residuals_vs_time(original_times, ls_residuals, "Lomb-Scargle", integral_ls_residuals,
                                    times_for_fit, custom_residuals, "Custom Model", integral_custom_residuals,
                                    title="Comparison of Residuals vs. Time")
    plot_residuals_histogram([ls_residuals, custom_residuals], ["Lomb-Scargle", "Custom Model"], title="Comparison of Residuals Histograms")


    print("\nAnalysis complete.")

if __name__ == '__main__':
    main()






