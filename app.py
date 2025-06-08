# app.py

import sqlite3, io, json, joblib, holidays, pandas as pd
from flask import (
    Flask, request, jsonify,
    render_template, redirect, url_for, flash, send_file
)
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

# --- App & Login Setup ---
app = Flask(__name__)
app.secret_key = "replace-with-a-strong-random-secret"
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- Database Helpers ---
DB = 'users.db'
def init_db():
    """Create the users table if it doesn't exist."""
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
      CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        branches TEXT NOT NULL
      )
    ''')
    conn.commit(); conn.close()

class User(UserMixin):
    def __init__(self, id, username, pwd_hash, branches):
        self.id = id
        self.username = username
        self.pwd_hash = pwd_hash
        # branches stored as "1,2,3"
        self.branches = [int(b) for b in branches.split(',')]

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute("SELECT id,username,password,branches FROM users WHERE id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    return User(*row) if row else None

# --- User Routes ---
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        u = request.form['username']
        p = generate_password_hash(request.form['password'])
        # allow user to pick their branches from checkboxes
        bs = request.form.getlist('branches')  # e.g. ['1','3']
        br_str = ','.join(bs)
        conn = sqlite3.connect(DB); c = conn.cursor()
        try:
            c.execute(
              "INSERT INTO users (username,password,branches) VALUES (?,?,?)",
              (u,p,br_str)
            )
            conn.commit()
            flash('Registered successfully—please log in','success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already taken','danger')
        finally:
            conn.close()
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        u = request.form['username']
        p = request.form['password']
        conn = sqlite3.connect(DB); c = conn.cursor()
        c.execute("SELECT id,username,password,branches FROM users WHERE username=?", (u,))
        row = c.fetchone()
        conn.close()
        if row and check_password_hash(row[2], p):
            user = User(*row)
            login_user(user)
            return redirect(url_for('portal'))
        flash('Invalid credentials','danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out','info')
    return redirect(url_for('login'))

# --- Load Model & Holidays ---
model        = joblib.load('best_model.pkl')
feature_cols = json.load(open('feature_cols.json'))
ger_hols     = holidays.CountryHoliday('DE')

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

def load_historical_data():
    df = pd.read_csv('data/train.csv', parse_dates=['Datum'])
    df.rename(columns={
        'Datum': 'date',
        'Warengruppe': 'product_group',
        'Umsatz': 'sales'
    }, inplace=True)
    return df

# --- Multi-Tenant Portal ---
@app.route('/portal', methods=['GET','POST'])
@login_required
def portal():
    # user.branches is list of allowed product_group IDs
    allowed = current_user.branches
    forecasts = None
    fig_ts_json = None
    fig_trend_json = None
    fig_seas_json = None
    fig_resid_json = None
    fig_heat_json = None

    # Load historical data for EDA
    df_hist = load_historical_data()

    # Default EDA filters
    eda_groups_default = allowed
    min_date_hist = df_hist['date'].min().date()
    max_date_hist = df_hist['date'].max().date()
    start_eda_default = min_date_hist
    end_eda_default = max_date_hist

    if request.method=='POST':
        # Forecasting logic (existing)
        if 'date' in request.form: # Check if it's a forecast request
            date    = request.form['date']
            horizon = int(request.form.get('horizon',1))
            dates   = [pd.to_datetime(date) + pd.Timedelta(days=i) for i in range(horizon)]
            X, dfm = make_features(dates, allowed)
            raw  = model.predict(X).reshape(horizon, len(allowed))
            # build a table of date × branch
            records = []
            for i,d in enumerate(dates):
                for j,grp in enumerate(allowed):
                    records.append({
                        'date': d.date(), 'branch': grp,
                        'forecast': round(float(raw[i,j]),1)
                    })
            forecasts = pd.DataFrame(records)

        # EDA logic
        if 'eda_groups' in request.form: # Check if it's an EDA filter submission
            eda_groups = [int(g) for g in request.form.getlist('eda_groups')]
            start_eda = pd.to_datetime(request.form['start_eda']).date()
            end_eda = pd.to_datetime(request.form['end_eda']).date()
            period = int(request.form.get('period', 30))
        else:
            eda_groups = eda_groups_default
            start_eda = start_eda_default
            end_eda = end_eda_default
            period = 30 # Default period for seasonal decomposition

        mask = (
            df_hist['product_group'].isin(eda_groups) &
            (df_hist['date'].dt.date >= start_eda) &
            (df_hist['date'].dt.date <= end_eda)
        )
        df_sel = df_hist[mask]

        # Aggregate daily total
        df_ts = df_sel.groupby('date')['sales'].sum().reset_index()

        # 2.1 Time-Series Plot with pan/zoom
        fig_ts = px.line(
            df_ts, x='date', y='sales',
            title='Daily Sales (Selected Groups)',
            labels={'date':'Date','sales':'Sales'}
        )
        fig_ts.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        fig_ts_json = fig_ts.to_dict()

        # 2.2 Seasonal Decomposition
        if not df_ts.empty and len(df_ts) >= period * 2: # Ensure enough data for decomposition
            decomp = seasonal_decompose(
                df_ts.set_index('date')['sales'],
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )
            # Trend
            fig_trend = px.line(
                x=decomp.trend.index, y=decomp.trend,
                title='Trend Component',
                labels={'x':'Date','y':'Trend'}
            )
            fig_trend_json = fig_trend.to_dict()

            # Seasonal
            fig_seas = px.line(
                x=decomp.seasonal.index, y=decomp.seasonal,
                title='Seasonal Component',
                labels={'x':'Date','y':'Seasonality'}
            )
            fig_seas_json = fig_seas.to_dict()

            # Residual
            fig_resid = px.line(
                x=decomp.resid.index, y=decomp.resid,
                title='Residual Component',
                labels={'x':'Date','y':'Residual'}
            )
            fig_resid_json = fig_resid.to_dict()

        # 2.3 Heatmap of Average Sales
        df_heat = df_sel.copy()
        if not df_heat.empty: # Only proceed if df_heat is not empty
            df_heat['Month'] = df_heat['date'].dt.month_name().str[:3]
            df_heat['Day']   = df_heat['date'].dt.day_name().str[:3]
            pivot = (
                df_heat
                .groupby(['Day','Month'])['sales']
                .mean()
                .reset_index()
                .pivot(index='Day', columns='Month', values='sales')
            )

            # Ensure ordering
            days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
            pivot = pivot.reindex(index=days, columns=months)

            fig_heat = px.imshow(
                pivot,
                labels={'x':'Month','y':'Day','color':'Avg Sales'},
                aspect='auto'
            )
            fig_heat_json = fig_heat.to_dict()

    return render_template(
      'portal.html',
      username=current_user.username,
      branches=allowed,
      forecasts=forecasts.to_dict(orient='records') if forecasts is not None else None,
      fig_ts_json=fig_ts_json,
      fig_trend_json=fig_trend_json,
      fig_seas_json=fig_seas_json,
      fig_resid_json=fig_resid_json,
      fig_heat_json=fig_heat_json,
      eda_groups_default=eda_groups_default,
      min_date_hist=min_date_hist,
      max_date_hist=max_date_hist,
      start_eda_default=start_eda_default,
      end_eda_default=end_eda_default
    )

# --- Initialize & Run ---
if __name__=='__main__':
    init_db()
    app.run(debug=True) 