# fitting

## 需求安裝

```bash
pip install -r requirements.txt
```

## 執行說明

本專案主程式為 `fit5.py`，會自動讀取 `Kay2.csv`，進行時間序列的Lomb-Scargle分析與自訂模型擬合。

### 執行方式

```bash
python3 fit5.py
```

- 預設不會顯示繪圖（如需開啟，請將 `fig.show()` 的註解移除）。
- 輸出包含Lomb-Scargle與自訂模型的統計結果、R²、殘差積分等。

### 主要依賴
- pandas
- numpy
- plotly
- astropy
- lmfit

### 資料格式
- 輸入CSV需包含 `Time` 與 `Ic` 欄位（可於 `fit5.py` 內調整欄位名稱）。

---
如需進一步協助，請聯絡專案維護者。