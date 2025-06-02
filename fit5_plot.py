# 匯入必要套件
import plotly.graph_objects as go
import numpy as np
import os

def save_plotly_svg(fig, basename, dataid):
    """儲存 Plotly 圖表為 SVG 格式，並添加 dataid 標註"""
    fig.add_annotation(
        text=f"dataid: {dataid}", 
        xref="paper", yref="paper", 
        x=0.99, y=0.99, 
        showarrow=False, 
        font=dict(size=12, color="crimson"), 
        align="right", 
        xanchor="right", yanchor="top", 
        bgcolor="rgba(255,255,255,0.7)"
    )
    svg_path = os.path.join("Plot", f"{basename}.svg")
    fig.update_layout(width=1920, height=1080)
    fig.write_image(svg_path, format="svg")
    print(f"圖表已儲存: {svg_path}")

def add_fit_info_annotation(fig, info_text, pos=(0.01, 0.99)):
    """在圖表右下角添加擬合資訊註解"""
    fig.add_annotation(
        text=info_text,
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        align="right",
        xanchor="right", yanchor="bottom",
        bgcolor="rgba(255,255,200,0.85)",
        bordercolor="black", borderwidth=1,
        font=dict(size=13, color="black")
    )

def plot_custom_model_fit_separate(times, values, fit_values, params_dict, title, out_svg_path):
    """
    繪製自定義模型擬合結果的獨立圖表
    
    times: 時間序列 (1D array)
    values: 原始資料 (1D array)
    fit_values: 擬合結果 (1D array)
    params_dict: 參數字典，會自動補齊註解
    title: 主標題（此處應為目標函數內容）
    out_svg_path: 輸出 SVG 路徑
    """
    # 參數註解字串
    param_lines = []
    for k, v in params_dict.items():
        param_lines.append(f"{k} = {v:.5g}" if isinstance(v, (float, int, np.floating, np.integer)) else f"{k} = {v}")
    param_text = "\n".join(param_lines)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=values, mode='markers', name='原始資料', marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=times, y=fit_values, mode='lines', name='擬合曲線', line=dict(width=3)))
    fig.update_layout(
        title=title,
        width=1920,
        height=1080,
        xaxis_title="Time",
        yaxis_title="Ic",
        font=dict(size=24),
        margin=dict(l=80, r=80, t=100, b=80)
    )
    # 右下角參數註解
    fig.add_annotation(
        text=param_text,
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        font=dict(size=20),
        align="right",
        xanchor="right", yanchor="bottom",
        bordercolor="#888", borderwidth=1, borderpad=8,
        bgcolor="rgba(255,255,255,0.8)"
    )
    # 儲存為 SVG
    fig.write_image(out_svg_path, width=1920, height=1080)

def plot_lomb_scargle_periodogram(ls_frequency, ls_power, best_frequency_ls, fap_levels_plot_ls, power_thresholds_plot_ls, title_prefix):
    """繪製 Lomb-Scargle 週期圖"""
    fig = go.Figure()
    if ls_frequency is not None and len(ls_frequency) > 0:
        fig.add_trace(go.Scatter(x=ls_frequency, y=ls_power, mode='lines', name='LS Power', line=dict(color='cornflowerblue')))
        fig.add_vline(x=best_frequency_ls, line_dash="dash", line_color="red", annotation_text=f'Best Freq: {best_frequency_ls:.4f}', annotation_position="top right")
        if fap_levels_plot_ls is not None and power_thresholds_plot_ls is not None:
            if hasattr(fap_levels_plot_ls, '__iter__') and hasattr(power_thresholds_plot_ls, '__iter__') and len(fap_levels_plot_ls) == len(power_thresholds_plot_ls):
                for level, thresh in zip(fap_levels_plot_ls, power_thresholds_plot_ls):
                    fig.add_hline(y=thresh, line_dash="dot", line_color="grey", annotation_text=f'FAP {level*100:.1f}% ({thresh:.2f})', annotation_position="bottom right")
    else:
        fig.add_annotation(text="Periodogram not computed.", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        title_text=f"{title_prefix}: Lomb-Scargle Periodogram", 
        title_x=0.5, 
        xaxis_title="Frequency (cycles / time unit)", 
        yaxis_title="Lomb-Scargle Power", 
        showlegend=True,
        width=1920,
        height=1080
    )
    return fig

def plot_data_with_ls_fit(original_times, original_values, original_errors, t_fit_ls, y_ls_fit_on_original_scale_for_plot, best_period_ls, r_squared_ls, value_column_name, title_prefix):
    """繪製原始資料與 Lomb-Scargle 擬合結果"""
    fig = go.Figure()
    if original_errors is not None:
        fig.add_trace(go.Scatter(x=original_times, y=original_values, mode='lines+markers', name='Original Data (with errors)', marker=dict(color='grey', size=3, opacity=0.5), error_y=dict(type='data', array=original_errors, visible=True, color='grey', thickness=0.5)))
    else:
        fig.add_trace(go.Scatter(x=original_times, y=original_values, mode='lines+markers', name='Original Data', marker=dict(color='grey', size=3, opacity=0.7)))
    ls_label = 'Lomb-Scargle Fit'
    if r_squared_ls is not None and not (r_squared_ls != r_squared_ls):
        ls_label += f' (R²={r_squared_ls:.4f})'
    fig.add_trace(go.Scatter(x=t_fit_ls, y=y_ls_fit_on_original_scale_for_plot, mode='lines', name=ls_label, line=dict(color='dodgerblue', width=1.5)))
    fig.update_layout(
        title_text=f"{title_prefix}: Data with Lomb-Scargle Sinusoidal Fit (Period = {best_period_ls:.4f})", 
        title_x=0.5, 
        xaxis_title="Time", 
        yaxis_title=value_column_name.capitalize(), 
        showlegend=True,
        width=1920,
        height=1080
    )
    return fig

def plot_phase_folded_data(ls_input_times, ls_input_values_for_phase_plot, best_period_ls, ls_input_errors_for_phase_plot, value_column_name, title_prefix):
    """繪製相位摺疊資料"""
    fig = go.Figure()
    if best_period_ls != 0 and not (best_period_ls != best_period_ls) and not (best_period_ls == float('inf')) and len(ls_input_times) > 0:
        phase_individual = (ls_input_times / best_period_ls) % 1.0
        plot_phase_individual = np.concatenate((phase_individual - 1, phase_individual, phase_individual + 1))
        plot_values_individual = np.concatenate((ls_input_values_for_phase_plot, ls_input_values_for_phase_plot, ls_input_values_for_phase_plot))
        if ls_input_errors_for_phase_plot is not None:
            plot_errors_individual = np.concatenate((ls_input_errors_for_phase_plot, ls_input_errors_for_phase_plot, ls_input_errors_for_phase_plot))
            fig.add_trace(go.Scatter(x=plot_phase_individual, y=plot_values_individual, mode='markers', name='Data points', marker=dict(color='grey', size=3, opacity=0.5), error_y=dict(type='data', array=plot_errors_individual, visible=True, color='grey', thickness=0.5)))
        else:
            fig.add_trace(go.Scatter(x=plot_phase_individual, y=plot_values_individual, mode='markers', name='Data points', marker=dict(color='grey', size=3, opacity=0.7)))
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
            fig.add_trace(go.Scatter(x=np.concatenate((bin_centers, bin_centers + 1)), y=np.concatenate((mean_binned_values, mean_binned_values)), mode='lines', name='Mean Phase-Folded Profile', line=dict(color='blue', width=1.5, dash='dash')))
        fig.update_xaxes(title_text=f"Phase (Period = {best_period_ls:.6f} time units)", range=[0, 2])
        fig.update_yaxes(title_text=f"{value_column_name.capitalize()} (Detrended for LS if applicable)")
    else:
        fig.add_annotation(text="Phase-folded plot skipped<br>due to invalid period or no data.", xref="x domain", yref="y domain", x=0.5, y=0.5, showarrow=False)
    fig.update_layout(
        title_text=f"{title_prefix}: Phase-Folded Data", 
        title_x=0.5, 
        showlegend=True,
        width=1920,
        height=1080
    )
    return fig

def plot_residuals_vs_time(times, residuals, label, integral, title="Residuals vs. Time"):
    """繪製殘差隨時間變化的圖表"""
    fig = go.Figure()
    res_plot_label = f'{label} Residuals'
    if integral is not None and not np.isnan(integral):
        res_plot_label += f' (Integral={integral:.2e})'
    fig.add_trace(go.Scatter(x=times, y=residuals, mode='lines+markers', name=res_plot_label,
                             marker=dict(size=3, opacity=0.7)))
    
    fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=0.8)
    fig.update_layout(
        title_text=title, 
        title_x=0.5,
        xaxis_title="Time",
        yaxis_title="Residual (Data - Model)",
        showlegend=True,
        width=1920,
        height=1080
    )
    return fig

def plot_combined_residuals_vs_time(times1, residuals1, label1, integral1, 
                                    times2, residuals2, label2, integral2, 
                                    title="Combined Residuals vs. Time"):
    """繪製兩個模型殘差比較圖"""
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
    fig.update_layout(
        title_text=title, 
        title_x=0.5,
        xaxis_title="Time",
        yaxis_title="Residual (Data - Model)",
        showlegend=True,
        width=1920,
        height=1080
    )
    return fig

def plot_residuals_histogram(residuals_list, labels_list, title="Histogram of Residuals"):
    """繪製殘差直方圖"""
    import plotly.figure_factory as ff
    
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
        if len(bin_edges_fd) > 1:
            bin_size = float(f"{float(bin_edges_fd[1] - bin_edges_fd[0]):.2e}")
        else:
            bin_size = None  # 讓 Plotly 自動決定
        return bin_size

    fig = ff.create_distplot(valid_data, valid_labels, show_hist=True, show_rug=True, bin_size=_bin_size(valid_data))

    fig.update_layout(
        title_text=title, 
        title_x=0.5,
        xaxis_title="Residual Value",
        yaxis_title="Density",
        showlegend=True,
        width=1920,
        height=1080
    )
    return fig
