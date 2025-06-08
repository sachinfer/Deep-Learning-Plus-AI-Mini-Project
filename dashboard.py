# dashboard.py

import io
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose

API_URL = "http://localhost:5000"

st.title("📦 Bakery Sales Forecast & EDA")

# --- Forecasting Section (existing) ---
st.header("1️⃣ Forecast")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", value=pd.to_datetime("2025-06-20"))
    horizon    = st.slider("Horizon (days)", 1, 30, 7)
with col2:
    groups = st.multiselect(
        "Product groups", [1,2,3,4,5,6], default=[1,2,3]
    )

if st.button("Get Forecast"):
    payload = {
        "start_date": start_date.isoformat(),
        "product_group": groups,
        "horizon": horizon
    }
    res = requests.post(f"{API_URL}/predict", json=payload).json()
    dates = res['dates']; preds = res['predictions']

    df_fc = pd.DataFrame(preds, index=dates)
    df_fc.index.name = 'Date'
    df_fc = df_fc.reset_index().melt(
        id_vars='Date', var_name='Group', value_name='Forecast'
    )
    df_fc['Forecast'] = df_fc['Forecast'].round(1)

    st.subheader("Forecast Table")
    st.dataframe(df_fc)

    st.subheader("Forecast Plot")
    chart = df_fc.pivot(index='Date', columns='Group', values='Forecast')
    st.line_chart(chart)

# --- EDA Section ---
st.header("2️⃣ Interactive EDA")

@st.cache_data
def load_data():
    df = pd.read_csv('data/train.csv', parse_dates=['Datum'])
    df.rename(columns={
        'Datum': 'date',
        'Warengruppe': 'product_group',
        'Umsatz': 'sales'
    }, inplace=True)
    return df

df_hist = load_data()

# Filters
st.subheader("Filters")
eda_groups = st.multiselect(
    "Select product groups for EDA",
    [1,2,3,4,5,6],
    default=[1,2,3]
)
min_date = df_hist['date'].min().date()
max_date = df_hist['date'].max().date()
start_eda, end_eda = st.date_input(
    "Date range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

mask = (
    df_hist['product_group'].isin(eda_groups) &
    (df_hist['date'].dt.date >= start_eda) &
    (df_hist['date'].dt.date <= end_eda)
)
df_sel = df_hist[mask]

# Aggregate daily total
df_ts = df_sel.groupby('date')['sales'].sum().reset_index()

# 2.1 Time-Series Plot with pan/zoom
st.subheader("2.1 Daily Sales Time Series")
fig_ts = px.line(
    df_ts, x='date', y='sales',
    title='Daily Sales (Selected Groups)',
    labels={'date':'Date','sales':'Sales'}
)
fig_ts.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
st.plotly_chart(fig_ts, use_container_width=True)

# 2.2 Seasonal Decomposition
st.subheader("2.2 Seasonal Decomposition")
period = st.slider("Choose seasonal period (days)", 7, 365, 30)
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
st.plotly_chart(fig_trend, use_container_width=True)

# Seasonal
fig_seas = px.line(
    x=decomp.seasonal.index, y=decomp.seasonal,
    title='Seasonal Component',
    labels={'x':'Date','y':'Seasonality'}
)
st.plotly_chart(fig_seas, use_container_width=True)

# Residual
fig_resid = px.line(
    x=decomp.resid.index, y=decomp.resid,
    title='Residual Component',
    labels={'x':'Date','y':'Residual'}
)
st.plotly_chart(fig_resid, use_container_width=True)

# 2.3 Heatmap of Average Sales
st.subheader("2.3 Heatmap: Avg Sales by Day of Week & Month")
df_heat = df_sel.copy()
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
st.plotly_chart(fig_heat, use_container_width=True)

# --- Batch forecasting via CSV upload ---
st.header("Batch Forecast from CSV")
st.markdown("Upload a CSV with **`date`** and **`product_group`** columns.")
csv_file = st.file_uploader("CSV file", type=["csv"])
if csv_file is not None:
    files = {"file": csv_file}
    resp = requests.post(f"{API_URL}/batch_predict", files=files)
    if resp.status_code == 200:
        st.success("Batch forecasting complete!")
        st.download_button(
            "Download forecasts CSV",
            data=resp.content,
            file_name="batch_forecasts.csv",
            mime="text/csv"
        )
    else:
        st.error(f"Error: {resp.text}") 