from pathlib import Path

import joblib
import pandas as pd

try:
    from src.feature_engineering import create_inventory_features
except ModuleNotFoundError:  # pragma: no cover
    from feature_engineering import create_inventory_features

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "stockout_model.pkl"


def risk_level(probability):
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.30:
        return "Medium Risk"
    return "Low Risk"


def predict_stockout(row):
    """Predict stockout risk for a single product record."""
    model = joblib.load(MODEL_PATH)

    record = pd.DataFrame([row])
    if "date" not in record.columns:
        record["date"] = pd.Timestamp("2024-01-01")
    if "category" not in record.columns:
        record["category"] = "Electronics"
    if "supplier_id" not in record.columns:
        record["supplier_id"] = "SUP1"
    if "promotion" not in record.columns:
        record["promotion"] = 0
    if "holiday" not in record.columns:
        record["holiday"] = 0
    if "returns" not in record.columns:
        record["returns"] = 0

    record["date"] = pd.to_datetime(record["date"])
    record["promotion"] = pd.to_numeric(record["promotion"], errors="coerce").fillna(0)
    record["holiday"] = pd.to_numeric(record["holiday"], errors="coerce").fillna(0)
    record["returns"] = pd.to_numeric(record["returns"], errors="coerce").fillna(0)

    record = create_inventory_features(record)
    feature_order = [
        "current_stock", "daily_sales", "units_sold_7d", "units_sold_30d",
        "reorder_level", "lead_time_days", "incoming_stock", "promotion",
        "price", "holiday", "returns", "average_daily_sales", "sales_7d_avg",
        "sales_30d_avg", "sales_growth", "days_of_inventory", "lead_time_demand",
        "stock_gap", "reorder_gap", "incoming_stock_ratio", "promotion_flag",
        "month", "day_of_week", "weekend_flag", "category", "supplier_id"
    ]

    for col in feature_order:
        if col not in record.columns:
            record[col] = 0 if col not in ["category", "supplier_id"] else "Unknown"

    probability = model.predict_proba(record[feature_order])[0, 1]
    return {"risk_probability": round(float(probability), 3), "risk_level": risk_level(float(probability))}
