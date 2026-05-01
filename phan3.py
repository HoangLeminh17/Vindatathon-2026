"""
Phan 3 - V6: Cai thien V1 goc + Ensemble nhe voi Prophet
==========================================================
V1 goc (recursive dual-branch GBR) da tot tren Kaggle.
V6 chi cai thien nhe:
- Giu nguyen cau truc V1
- Them Fourier features, end-of-month
- Tang so luong estimators
- Ensemble V1_improved + Prophet (weight nho cho Prophet)
- Tao nhieu version de thu tren Kaggle
"""

from pathlib import Path
import warnings
import calendar as cal
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = Path('dataset')
sales = pd.read_csv(DATA_DIR / 'sales.csv', parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
sales['cogs_ratio'] = sales['COGS'] / sales['Revenue']
sample_submission = pd.read_csv(DATA_DIR / 'sample_submission.csv', parse_dates=['Date'])

print(f"Train: {sales['Date'].min().date()} -> {sales['Date'].max().date()} | {len(sales):,}")
print(f"Test:  {sample_submission['Date'].min().date()} -> {sample_submission['Date'].max().date()} | {len(sample_submission):,}")

# =====================================================================
# V1 IMPROVED: Recursive dual-branch GBR (giu nguyen logic V1 goc)
# =====================================================================

CONFIGS = {
    'short': {
        'lags': [7, 14, 28, 56, 364],
        'rolls': [7, 28, 91, 364],
        'rev_params': dict(random_state=42, n_estimators=350, learning_rate=0.05, max_depth=4, loss='huber', subsample=0.9),
        'ratio_params': dict(random_state=42, n_estimators=250, learning_rate=0.05, max_depth=2, loss='huber', subsample=0.9),
    },
    'long': {
        'lags': [364, 728],
        'rolls': [],
        'rev_params': dict(random_state=42, n_estimators=400, learning_rate=0.05, max_depth=4, loss='huber', subsample=0.9),
        'ratio_params': dict(random_state=42, n_estimators=250, learning_rate=0.05, max_depth=2, loss='huber', subsample=0.9),
    },
}

def make_row(history, date, lags, rolls):
    doy = date.timetuple().tm_yday
    dow = date.dayofweek
    dom = date.day
    month = date.month
    year = date.year
    week = int(date.isocalendar().week)
    _, dim = cal.monthrange(year, month)
    row = {
        'trend': len(history), 'dow': dow, 'dom': dom,
        'month': month, 'quarter': (month-1)//3+1, 'year': year,
        'dayofyear': doy, 'weekofyear': week,
        'is_weekend': int(dow >= 5),
        'is_month_start': int(date.is_month_start),
        'is_month_end': int(date.is_month_end),
        'days_to_month_end': dim - dom,
        'sin_doy': np.sin(2*np.pi*doy/365.25),
        'cos_doy': np.cos(2*np.pi*doy/365.25),
        'sin_doy2': np.sin(4*np.pi*doy/365.25),
        'cos_doy2': np.cos(4*np.pi*doy/365.25),
        'sin_week': np.sin(2*np.pi*dow/7),
        'cos_week': np.cos(2*np.pi*dow/7),
    }
    for lag in lags:
        row[f'revenue_lag_{lag}'] = history['Revenue'].iloc[-lag] if lag <= len(history) else np.nan
        row[f'ratio_lag_{lag}'] = history['cogs_ratio'].iloc[-lag] if lag <= len(history) else np.nan
    for w in rolls:
        row[f'revenue_roll_mean_{w}'] = history['Revenue'].iloc[-w:].mean() if w <= len(history) else np.nan
        row[f'ratio_roll_mean_{w}'] = history['cogs_ratio'].iloc[-w:].mean() if w <= len(history) else np.nan
    if 364 in lags and 728 in lags and 728 <= len(history):
        row['revenue_yoy_ratio'] = history['Revenue'].iloc[-364] / max(history['Revenue'].iloc[-728], 1)
    return row

def build_training_frame(history, lags, rolls):
    start = max(max(lags), max(rolls) if rolls else 0)
    rows = []
    for i in range(start, len(history)):
        rows.append(make_row(history.iloc[:i], history.iloc[i]['Date'], lags, rolls))
    X = pd.DataFrame(rows)
    yr = history['Revenue'].iloc[start:].reset_index(drop=True)
    yc = history['cogs_ratio'].iloc[start:].reset_index(drop=True)
    return X, yr, yc

def fit_branch(history, config):
    X, yr, yc = build_training_frame(history, config['lags'], config['rolls'])
    rm = GradientBoostingRegressor(**config['rev_params']).fit(X, yr)
    cm = GradientBoostingRegressor(**config['ratio_params']).fit(X, yc)
    return rm, cm, X.columns.tolist()

def recursive_ensemble_forecast(train_history, future_dates):
    branch_models = {}
    branch_features = {}
    for name, config in CONFIGS.items():
        rm, cm, feat = fit_branch(train_history, config)
        branch_models[name] = (config, rm, cm)
        branch_features[name] = feat

    ratio_low, ratio_high = train_history['cogs_ratio'].quantile([0.02, 0.98])
    histories = {name: train_history.copy().reset_index(drop=True) for name in branch_models}
    predictions = []

    for date in future_dates:
        b_revs, b_rats = [], []
        for name, (config, rm, cm) in branch_models.items():
            row = pd.DataFrame([make_row(histories[name], date, config['lags'], config['rolls'])])
            rv = max(float(rm.predict(row)[0]), 0)
            rt = float(np.clip(cm.predict(row)[0], ratio_low, ratio_high))
            b_revs.append(rv)
            b_rats.append(rt)

        revenue = float(np.mean(b_revs))
        ratio = float(np.mean(b_rats))
        cogs = float(revenue * ratio)

        predictions.append({'Date': date, 'Revenue': revenue, 'COGS': cogs, 'cogs_ratio': ratio})
        nr = pd.DataFrame([{'Date': date, 'Revenue': revenue, 'COGS': cogs, 'cogs_ratio': ratio}])
        for name in histories:
            histories[name] = pd.concat([histories[name], nr], ignore_index=True)

    return pd.DataFrame(predictions), branch_models, branch_features



# =====================================================================
# EVALUATION
# =====================================================================

def evaluate(actual, pred, label=""):
    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    r2 = r2_score(actual, pred)
    if label:
        print(f"  {label}: MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.4f}")
    return mae, rmse, r2

def save_submission(dates, rev, cogs, filename):
    s = pd.DataFrame({'Date': dates})
    s['Revenue'] = np.maximum(rev, 0)
    s['COGS'] = np.maximum(cogs, 0)
    s.to_csv(filename, index=False)
    print(f"  Saved: {filename}")
    return s

# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':

    # --- Shadow validation ---
    print("\n" + "="*70)
    print("SHADOW VALIDATION")
    print("="*70)

    sh = len(sample_submission)
    s_train = sales.iloc[:-sh].copy().reset_index(drop=True)
    s_valid = sales.iloc[-sh:].copy().reset_index(drop=True)
    print(f"Train -> {s_train['Date'].max().date()} ({len(s_train)})")
    print(f"Valid: {s_valid['Date'].min().date()} -> {s_valid['Date'].max().date()} ({len(s_valid)})")

    td = s_valid['Date'].tolist()
    a_rev = s_valid['Revenue'].values
    a_cogs = s_valid['COGS'].values

    # V1 improved
    print("\n--- V1 Improved (Recursive GBR) ---")
    v1_df, _, _ = recursive_ensemble_forecast(s_train, td)
    v1_rev = v1_df['Revenue'].values
    v1_cogs = v1_df['COGS'].values
    evaluate(a_rev, v1_rev, "V1 Revenue")
    evaluate(a_cogs, v1_cogs, "V1 COGS")


    # =====================================================================
    # FINAL FORECAST & EXPLAINABILITY (Đáp ứng yêu cầu cuộc thi)
    # =====================================================================
    print("\n" + "="*70)
    print("FINAL FORECAST & EXPLAINABILITY")
    print("="*70)

    test_dates = sample_submission['Date'].tolist()
    dates_col = sample_submission['Date']

    # Chạy mô hình V1 Improved cho tập test
    f_v1, f_models, f_feats = recursive_ensemble_forecast(sales.copy().reset_index(drop=True), test_dates)
    
    # Lưu file nộp bài chuẩn format Kaggle
    save_submission(dates_col, f_v1['Revenue'].values, f_v1['COGS'].values, 'submission.csv')
    print("\n-> Đã tạo file submission.csv")

    # Xuất Feature Importances để viết Report (Yêu cầu Khả năng giải thích)
    print("\n--- FEATURE IMPORTANCES (Dùng cho Report) ---")
    
    for branch_name, (config, rm, cm) in f_models.items():
        feats = f_feats[branch_name]
        
        # Revenue Importances
        rev_imp = pd.DataFrame({
            'Feature': feats,
            'Importance': rm.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 đặc trưng quan trọng nhất dự báo Doanh thu (Nhánh {branch_name}):")
        print(rev_imp.head(10).to_string(index=False))
        
        # COGS Importances
        cogs_imp = pd.DataFrame({
            'Feature': feats,
            'Importance': cm.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 đặc trưng quan trọng nhất dự báo COGS (Nhánh {branch_name}):")
        print(cogs_imp.head(10).to_string(index=False))

    print("\n" + "="*70)
    print("HOÀN THÀNH!")
    print("="*70)
