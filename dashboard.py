# dashboard.py

import streamlit as st
import requests
import pandas as pd

st.title("Bakery Sales Forecast")

# 1) User inputs
date = st.date_input("Select a date")
groups = st.multiselect(
    "Select product groups to forecast",
    options=[1,2,3,4,5,6],
    default=[1,2,3]
)

# 2) When the user clicks Forecast
if st.button("Forecast"):
    payload = {
        "date": date.isoformat(),
        "product_group": groups
    }
    # Call your local API
    resp = requests.post("http://localhost:5000/predict", json=payload)
    data = resp.json()

    # 3) Display results in a table
    df = pd.DataFrame.from_dict(
        data["predictions"], orient="index", columns=["Forecast"]
    )
    df.index.name = "Product Group"
    st.table(df)

    # 4) Optionally, plot a bar chart
    st.bar_chart(df) 