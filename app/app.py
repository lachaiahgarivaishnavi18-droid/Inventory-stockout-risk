import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_stockout

st.title("Inventory Stockout Risk Prediction")
st.caption("Beginner-friendly demo for classification-based inventory risk monitoring.")

with st.form("inventory_form"):
    col1, col2 = st.columns(2)
    with col1:
        current_stock = st.number_input("Current Stock", min_value=0.0, value=25.0)
        average_daily_sales = st.number_input("Average Daily Sales", min_value=0.0, value=8.0)
        lead_time = st.number_input("Lead Time (days)", min_value=1, value=5)
        reorder_level = st.number_input("Reorder Level", min_value=0.0, value=50.0)
    with col2:
        promotion = st.selectbox("Promotion", [0, 1], index=0)
        category = st.selectbox("Category", ["Electronics", "Home", "Fashion", "Grocery", "Health", "Office"])
        price = st.number_input("Price", min_value=1.0, value=120.0)
        incoming_stock = st.number_input("Incoming Stock", min_value=0.0, value=20.0)

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    row = {
        "current_stock": current_stock,
        "daily_sales": average_daily_sales,
        "units_sold_7d": average_daily_sales * 7,
        "units_sold_30d": average_daily_sales * 30,
        "reorder_level": reorder_level,
        "lead_time_days": lead_time,
        "incoming_stock": incoming_stock,
        "promotion": promotion,
        "price": price,
        "holiday": 0,
        "returns": 0,
        "stockout": 0,
        "stockout_risk": 0,
        "category": category,
        "supplier_id": "SUP1",
        "date": pd.Timestamp("2024-01-01"),
    }
    result = predict_stockout(row)
    st.subheader("Prediction")
    st.metric("Risk Probability", f"{result['risk_probability']:.3f}")
    st.metric("Risk Level", result["risk_level"])

    if result["risk_level"] == "High Risk":
        st.warning("Consider replenishment or supplier follow-up.")
    elif result["risk_level"] == "Medium Risk":
        st.info("Monitor sales and reorder timing closely.")
    else:
        st.success("Inventory looks healthy for the next 7 days.")
