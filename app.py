# app.py

import json
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load artifacts
model        = joblib.load('best_model.pkl')
feature_cols = json.load(open('feature_cols.json'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON payload:
      {
        "date": "2025-06-15",
        "product_group": [1, 3, 5]
      }
    Returns:
      {
        "date": "2025-06-15",
        "predictions": {
          "1": 123.45,
          "3": 67.89,
          "5": 210.11
        }
      }
    """
    data = request.get_json(force=True)
    date = pd.to_datetime(data['date'])
    groups = data['product_group']

    # Build input DataFrame
    df = pd.DataFrame({
        'date':           [date] * len(groups),
        'product_group':  groups
    })

    # Feature engineering
    df['month']       = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    # (Add any holiday flag here if you used it)

    # One‐hot encode
    X = pd.get_dummies(
        df[['product_group','month','day_of_week','is_weekend']],
        columns=['product_group','month','day_of_week','is_weekend'],
        drop_first=True
    )

    # Align columns exactly with training
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    # Predict
    preds = model.predict(X)
    # Round or format as you like
    preds = [float(p) for p in preds]

    return jsonify({
        'date':        date.strftime('%Y-%m-%d'),
        'predictions': dict(zip(map(str, groups), preds))
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 