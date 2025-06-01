#!/usr/bin/env python3

import sys
import os

# 添加當前目錄到路徑
sys.path.insert(0, os.getcwd())

from fit5 import analyze_file_and_get_results

def test_single_file():
    """測試單個文件的分析"""
    print("開始測試單個文件分析...")
    
    # 測試一個較小的文件
    test_file = "335_ic.csv"
    
    try:
        result = analyze_file_and_get_results(test_file)
        if result:
            print(f"分析成功完成：{test_file}")
            print(f"Lomb-Scargle R²: {result.get('r_squared_ls', 'N/A')}")
            print(f"自定義模型 R²: {result.get('r_squared_custom', 'N/A')}")
        else:
            print(f"分析失敗：{test_file}")
    except Exception as e:
        print(f"分析過程中發生錯誤：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_file()
