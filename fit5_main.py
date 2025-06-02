import os
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from fit5_utils import load_data, detrend_data, calculate_r_squared, custom_model_func
try:
    import lmfit
except ImportError:
    print("lmfit library not found. Please install it: pip install lmfit")
    lmfit = None


# 全域參數（可依需求調整）
TIME_COLUMN = "Time"
VALUE_COLUMN = "Ic"
ERROR_COLUMN = None  # 自動修正：無誤差欄位時設為 None
MIN_FREQUENCY_LS = 1e-3
MAX_FREQUENCY_LS = 1e3
SAMPLES_PER_PEAK_LS = 10
DETREND_ORDER_LS = 1


def analyze_file_and_get_results(file_path, param_bounds=None):
    result = {
        'file': file_path,
        'best_frequency_ls': None,
        'best_period_ls': None,
        'r_squared_ls': None,
        'ls_amplitude_fit': None,
        'ls_phase_fit_rad': None,
        'ls_offset_fit': None,
        'r_squared_custom': None,
        'custom_A': None,
        'custom_B': None,
        'custom_f': None,
        'custom_d': None,
        'custom_p': None,
        'custom_T': None,
        'custom_r': None,
        'custom_r2': None,
        'custom_C': None,
        'fit_status': None,
        'error_msg': None
    }
    original_times, original_values, original_errors = load_data(file_path, TIME_COLUMN, VALUE_COLUMN, ERROR_COLUMN)
    if original_times is None or original_values is None:
        result['fit_status'] = 'fail'
        result['error_msg'] = 'data_load_fail'
        return result
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
    try:
        ls_frequency, ls_power = ls.autopower(minimum_frequency=current_min_freq_ls, 
                                              maximum_frequency=MAX_FREQUENCY_LS,
                                              samples_per_peak=SAMPLES_PER_PEAK_LS)
        best_frequency_ls = 0
        best_period_ls = float('inf')
        ls_amplitude_fit = np.nan
        ls_phase_fit_rad = np.nan
        ls_offset_fit = np.nan
        r_squared_ls = np.nan
        if ls_frequency is not None and len(ls_frequency) > 0:
            best_power_index_ls = np.argmax(ls_power)
            best_frequency_ls = ls_frequency[best_power_index_ls]
            if best_frequency_ls != 0:
                best_period_ls = 1.0 / best_frequency_ls
                ls_offset_fit = ls.offset()
                params_ls_sinusoid = ls.model_parameters(best_frequency_ls)
                ls_amplitude_fit = np.sqrt(params_ls_sinusoid[0]**2 + params_ls_sinusoid[1]**2)
                ls_phase_fit_rad = np.arctan2(params_ls_sinusoid[1], params_ls_sinusoid[0])
        y_ls_pred_on_original_scale_at_orig_times = np.full_like(original_values, np.nan)
        if best_frequency_ls != 0 and not np.isinf(best_period_ls):
            ls_model_pred_on_detrended_at_orig_times = ls.model(original_times, best_frequency_ls)
            y_ls_pred_on_original_scale_at_orig_times = ls_model_pred_on_detrended_at_orig_times
            if ls_detrend_coeffs is not None:
                original_trend_at_orig_times = np.polyval(ls_detrend_coeffs, original_times)
                y_ls_pred_on_original_scale_at_orig_times = original_trend_at_orig_times + ls_model_pred_on_detrended_at_orig_times
            r_squared_ls = calculate_r_squared(original_values, y_ls_pred_on_original_scale_at_orig_times)
        result['best_frequency_ls'] = best_frequency_ls
        result['best_period_ls'] = best_period_ls
        result['r_squared_ls'] = r_squared_ls
        result['ls_amplitude_fit'] = ls_amplitude_fit
        result['ls_phase_fit_rad'] = ls_phase_fit_rad
        result['ls_offset_fit'] = ls_offset_fit
    except Exception as e:
        import traceback
        if 'LinAlgError' in str(type(e)) or 'Singular matrix' in str(e):
            result['fit_status'] = 'fail'
            result['error_msg'] = f'LombScargle LinAlgError: {e}'
            return result
        else:
            result['fit_status'] = 'fail'
            result['error_msg'] = f'LombScargle error: {e}\n{traceback.format_exc()}'
            return result
    # custom model fit
    if lmfit is not None and best_frequency_ls > 0:
        custom_model = lmfit.Model(custom_model_func)
        params = lmfit.Parameters()
        slope_init, intercept_init = np.polyfit(original_times, original_values, 1)
        residuals_for_amp_est = original_values - (slope_init * original_times + intercept_init)
        amp_init = np.std(residuals_for_amp_est) * np.sqrt(2)
        if amp_init == 0:
            amp_init = np.std(original_values) * np.sqrt(2)
        if amp_init == 0:
            amp_init = 1e-7
        x_min, x_max = np.min(original_times), np.max(original_times)
        y_min, y_max = np.min(original_values), np.max(original_values)
        x_range = x_max - x_min
        y_range = y_max - y_min
        default_bounds = {
            'A': (1e-9, 1e2),
            'B': (1e-9, 1e2),
            'f': (1e-9, 1e7),
            'd': (x_min-100*x_range, x_max+100*x_range),
            'p': (-4*np.pi, 4*np.pi),
            'T': (0.001, 0.999),
            'r': (-100*y_range/x_range, 100*y_range/x_range),
            'r2': (-100*y_range/(x_range**2+1e-12), 100*y_range/(x_range**2+1e-12)),
            'C': (y_min-100*y_range, y_max+100*y_range)
        }
        bounds = default_bounds.copy()
        if param_bounds is not None:
            for k in param_bounds:
                if k in bounds and param_bounds[k][0] is not None and param_bounds[k][1] is not None:
                    bounds[k] = param_bounds[k]
        params.add('A', value=amp_init if amp_init > 0 else 1e-7, min=bounds['A'][0], max=bounds['A'][1])
        params.add('B', value=amp_init/2 if amp_init > 0 else 1e-7, min=bounds['B'][0], max=bounds['B'][1])
        params.add('f', value=best_frequency_ls if best_frequency_ls > 1e-9 else 1.0, min=bounds['f'][0], max=bounds['f'][1])
        params.add('d', value=x_min, min=bounds['d'][0], max=bounds['d'][1])
        params.add('p', value=0, min=bounds['p'][0], max=bounds['p'][1])
        params.add('T', value=0.5, min=bounds['T'][0], max=bounds['T'][1])
        params.add('r', value=slope_init, min=bounds['r'][0], max=bounds['r'][1])
        try:
            quad_coeffs = np.polyfit(original_times, original_values, 2)
            r2_init = quad_coeffs[0]
        except Exception:
            r2_init = 0
        params.add('r2', value=r2_init, min=bounds['r2'][0], max=bounds['r2'][1])
        params.add('C', value=np.polyval([slope_init, intercept_init], x_min), min=bounds['C'][0], max=bounds['C'][1])
        fit_weights = None
        times_for_fit = original_times
        data_for_fit = original_values
        try:
            lmfit_result_obj = custom_model.fit(data_for_fit, params, x=times_for_fit, weights=fit_weights, nan_policy='omit')
            r_squared_custom = calculate_r_squared(data_for_fit, lmfit_result_obj.best_fit)
            result['r_squared_custom'] = r_squared_custom
            for pname in ['A','B','f','d','p','T','r','r2','C']:
                result['custom_'+pname] = lmfit_result_obj.params[pname].value
            result['fit_status'] = 'success'
            result['error_msg'] = ''

            # --- custom fit 圖繪製 ---
            from fit5_plot import plot_custom_model_fit_separate
            # 產生 fit 曲線
            fit_values = lmfit_result_obj.model.eval(params=lmfit_result_obj.params, x=original_times)
            params_dict = {k: lmfit_result_obj.params[k].value for k in lmfit_result_obj.params}
            # 主標題為目標函數內容
            if hasattr(custom_model_func, '__doc__') and custom_model_func.__doc__:
                title = custom_model_func.__doc__.strip()
            else:
                title = "y = A·sin(2πf(x-d)-p) + B·cos(2πf(x-d)-p) + r(x-d) + r₂(x-d)² + C"
            basename = os.path.splitext(os.path.basename(file_path))[0]
            out_svg_path = os.path.join("Plot", f"{basename}_custom_fit.svg")
            plot_custom_model_fit_separate(
                original_times, original_values, fit_values, params_dict, title, out_svg_path
            )
        except Exception as e:
            result['fit_status'] = 'fail'
            result['error_msg'] = str(e)
    else:
        result['fit_status'] = 'fail'
        result['error_msg'] = 'lmfit_not_available_or_invalid_ls_freq'
    return result

def batch_analyze_and_save(csv_list, output_csv, param_bounds=None):
    all_results = []
    for csvfile in csv_list:
        res = analyze_file_and_get_results(csvfile, param_bounds=param_bounds)
        all_results.append(res)
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"所有分析結果已儲存到 {output_csv}")

def auto_batch_fit_until_r2(csv_list, output_csv, r2_target=0.95, max_iter=10):
    bound_matrix = [
        {'d': '1x', 'C': '1y', 'r': '1r', 'r2': '1r2'},
        {'d': '10x', 'C': '10y', 'r': '10r', 'r2': '10r2'},
        {'d': '100x', 'C': '100y', 'r': '100r', 'r2': '100r2'},
        {'d': (None, None), 'C': (None, None), 'r': (None, None), 'r2': (None, None)}
    ]
    for i, bound_case in enumerate(bound_matrix):
        print(f"\n=== Auto batch fit: 邊界組合 {i+1}/{len(bound_matrix)} ===")
        def make_bounds(file_path):
            times, values, _ = load_data(file_path, TIME_COLUMN, VALUE_COLUMN, ERROR_COLUMN)
            x_min, x_max = np.min(times), np.max(times)
            y_min, y_max = np.min(values), np.max(values)
            x_range = x_max - x_min
            y_range = y_max - y_min
            base = {
                'd': (x_min-100*x_range, x_max+100*x_range),
                'C': (y_min-100*y_range, y_max+100*y_range),
                'r': (-100*y_range/x_range, 100*y_range/x_range),
                'r2': (-100*y_range/(x_range**2+1e-12), 100*y_range/(x_range**2+1e-12))
            }
            if bound_case['d'] == '100x':
                d = (x_min-100*x_range, x_max+100*x_range)
            elif bound_case['d'] == '10x':
                d = (x_min-10*x_range, x_max+10*x_range)
            elif bound_case['d'] == '1x':
                d = (x_min-x_range, x_max+x_range)
            else:
                d = base['d'] if bound_case['d'] is None else bound_case['d']
            if bound_case['C'] == '100y':
                C = (y_min-100*y_range, y_max+100*y_range)
            elif bound_case['C'] == '10y':
                C = (y_min-10*y_range, y_max+10*y_range)
            elif bound_case['C'] == '1y':
                C = (y_min-y_range, y_max+y_range)
            else:
                C = base['C'] if bound_case['C'] is None else bound_case['C']
            if bound_case['r'] == '100r':
                r = (-100*y_range/x_range, 100*y_range/x_range)
            elif bound_case['r'] == '10r':
                r = (-10*y_range/x_range, 10*y_range/x_range)
            elif bound_case['r'] == '1r':
                r = (-1*y_range/x_range, 1*y_range/x_range)
            else:
                r = base['r'] if bound_case['r'] is None else bound_case['r']
            if bound_case['r2'] == '100r2':
                r2 = (-100*y_range/(x_range**2+1e-12), 100*y_range/(x_range**2+1e-12))
            elif bound_case['r2'] == '10r2':
                r2 = (-10*y_range/(x_range**2+1e-12), 10*y_range/(x_range**2+1e-12))
            elif bound_case['r2'] == '1r2':
                r2 = (-1*y_range/(x_range**2+1e-12), 1*y_range/(x_range**2+1e-12))
            else:
                r2 = base['r2'] if bound_case['r2'] is None else bound_case['r2']
            return {'d': d, 'C': C, 'r': r, 'r2': r2}
        param_bounds_dict = {}
        for f in csv_list:
            param_bounds_dict[f] = make_bounds(f)
        def analyze_with_case(file_path):
            return analyze_file_and_get_results(file_path, param_bounds=param_bounds_dict[file_path])
        all_results = [analyze_with_case(f) for f in csv_list]
        df = pd.DataFrame(all_results)
        df.to_csv(output_csv, index=False)
        print(f"所有分析結果已儲存到 {output_csv}")
        if df['r_squared_custom'].dropna().ge(r2_target).all():
            print(f"所有檔案 r_squared_custom >= {r2_target}，自動化結束。")
            return
        print("部分檔案 r_squared_custom 未達標，嘗試下一組邊界。")
    print("所有邊界組合皆嘗試完畢，仍有部分檔案未達標。")

if __name__ == "__main__":
    import glob
    csv_list = sorted(glob.glob("*_ic.csv"))
    if not csv_list:
        print("找不到 *_ic.csv 檔案，請確認資料檔案存在於工作目錄。")
    else:
        print(f"將分析以下檔案：{csv_list}")
        auto_batch_fit_until_r2(csv_list, "all_results.csv", r2_target=0.95)
        df = pd.read_csv("all_results.csv")
        print("\n分析結果摘要：")
        print(df[[c for c in df.columns if c.startswith('file') or c.startswith('r_squared') or c.startswith('fit_status') or c.startswith('error_msg')]].head())

# 其餘主程式流程、analyze_file_and_get_results、auto_batch_fit_until_r2等請從原 fit5.py 複製進來
