# train_model.py

import pandas as pd
import numpy as np
import holidays
import joblib
import json
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error

def load_and_prepare(path_train):
    df = pd.read_csv(path_train, parse_dates=['Datum'])
    df.rename(columns={
        'Datum': 'date',
        'Warengruppe': 'product_group',
        'Umsatz': 'sales'
    }, inplace=True)
    daily = df.groupby(['date','product_group'])['sales'].sum().reset_index()
    # date features
    daily['month']       = daily['date'].dt.month
    daily['day_of_week'] = daily['date'].dt.dayofweek
    daily['is_weekend']  = (daily['day_of_week'] >= 5).astype(int)
    # lags & rolling means
    for lag in [7, 14, 28, 365]:
        daily[f'lag_{lag}'] = daily.groupby('product_group')['sales'].shift(lag)
    daily['rmean_7']  = (daily.groupby('product_group')['sales']
                             .shift(1).rolling(7).mean()
                             .reset_index(level=0, drop=True))
    daily['rmean_30'] = (daily.groupby('product_group')['sales']
                              .shift(1).rolling(30).mean()
                              .reset_index(level=0, drop=True))
    # holiday flag
    ger = holidays.CountryHoliday('DE',
           years=range(daily['date'].dt.year.min(),
                       daily['date'].dt.year.max()+1))
    daily['is_holiday'] = daily['date'].isin(ger).astype(int)
    daily = daily.dropna().reset_index(drop=True)
    return daily

def train_and_evaluate(daily):
    feats = [c for c in daily.columns if c not in ('date','sales')]
    X = pd.get_dummies(
        daily[feats],
        columns=['product_group','month','day_of_week','is_weekend','is_holiday'],
        drop_first=True
    )
    y = daily['sales']
    cutoff = daily['date'].max() - pd.Timedelta(days=300)
    mask = daily['date'] < cutoff
    X_train, X_val = X[mask], X[~mask]
    y_train, y_val = y[mask], y[~mask]
    print(f"Train: {len(X_train)} rows – Val: {len(X_val)} rows")
    model = LGBMRegressor(
        objective='regression', n_estimators=200,
        num_leaves=50, max_depth=5,
        learning_rate=0.05, min_child_samples=20,
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, preds)
    print(f"Validation MAPE: {mape:.4f} ({mape*100:.2f}%)")
    return model, X_train.columns.tolist()

def save_artifacts(model, feature_cols):
    joblib.dump(model, 'best_model.pkl')
    with open('feature_cols.json','w') as f:
        json.dump(feature_cols, f)
    print("Saved: best_model.pkl, feature_cols.json")

if __name__ == '__main__':
    daily = load_and_prepare('data/train.csv')
    model, feats = train_and_evaluate(daily)
    save_artifacts(model, feats)
