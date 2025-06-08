# app.py

import io
import json
import joblib
import holidays
import pandas as pd
from flask import Flask, request, jsonify, send_file

# Load model artifacts
model        = joblib.load('best_model.pkl')
feature_cols = json.load(open('feature_cols.json'))

# Holiday calendar
ger_hols = holidays.CountryHoliday('DE')

app = Flask(__name__)

def make_features(dates, groups):
    """
    Build a DataFrame of features for (dates × groups).
    """
    df = pd.DataFrame({
        'date':        pd.to_datetime(dates).repeat(len(groups)),
        'product_group': groups * len(dates)
    })
    # Date features
    df['month']       = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['is_holiday']  = df['date'].isin(ger_hols).astype(int)

    # One-hot encode
    X = pd.get_dummies(
        df[['product_group','month','day_of_week','is_weekend','is_holiday']],
        columns=['product_group','month','day_of_week','is_weekend','is_holiday'],
        drop_first=True
    )
    # Align to training columns
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_cols]
    return X, df

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single-horizon or multi-horizon forecasting.
    Request JSON:
      {
        "start_date": "2025-06-20",
        "product_group": [1,2,3],
        "horizon": 7     # optional, default 1
      }
    Response JSON:
      {
        "dates": ["2025-06-20", ..., "2025-06-26"],
        "predictions": {
          "1": [90.66, ..., 95.12],
          "2": [...]
          "3": [...]
        }
      }
    """
    payload = request.get_json(force=True)
    start_date = pd.to_datetime(payload['start_date'])
    groups     = payload['product_group']
    horizon    = int(payload.get('horizon', 1))

    # build list of forecast dates
    dates = [start_date + pd.Timedelta(days=i) for i in range(horizon)]
    X, df_map = make_features(dates, groups)

    # predict
    raw_preds = model.predict(X).reshape(horizon, len(groups))
    # assemble by group
    preds = {str(g): list(raw_preds[:, i]) for i, g in enumerate(groups)}
    return jsonify({
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'predictions': preds
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Upload a CSV with columns "date" and "product_group" and get back a CSV
    with an extra "forecast" column.
    """
    file = request.files.get('file')
    if not file:
        return "No file uploaded", 400

    df = pd.read_csv(file, parse_dates=['date'])
    X, _ = make_features(df['date'].dt.strftime('%Y-%m-%d').tolist(),
                         df['product_group'].tolist())

    df['forecast'] = model.predict(X)
    # send CSV back
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        attachment_filename='batch_forecasts.csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 