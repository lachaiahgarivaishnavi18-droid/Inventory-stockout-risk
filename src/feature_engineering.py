import numpy as np
import pandas as pd


def create_inventory_features(df):
    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Basic business metrics
    df["average_daily_sales"] = df["daily_sales"].replace(0, np.nan)
    df["sales_7d_avg"] = df["units_sold_7d"] / 7
    df["sales_30d_avg"] = df["units_sold_30d"] / 30
    df["sales_growth"] = (df["sales_7d_avg"] - df["sales_30d_avg"]) / df["sales_30d_avg"].replace(0, np.nan)

    # Inventory coverage
    df["days_of_inventory"] = df["current_stock"] / df["average_daily_sales"].replace(0, np.nan)
    df["lead_time_demand"] = df["average_daily_sales"].replace(0, np.nan) * df["lead_time_days"]
    df["stock_gap"] = df["current_stock"] - df["lead_time_demand"]
    df["reorder_gap"] = df["reorder_level"] - df["current_stock"]
    df["incoming_stock_ratio"] = df["incoming_stock"] / df["reorder_level"].replace(0, np.nan)

    # Promotions and calendar features
    df["promotion_flag"] = df["promotion"].astype(int)
    if "date" in df.columns:
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
        df["weekend_flag"] = (df["day_of_week"] >= 5).astype(int)

    # Replace infinities created from division
    for col in ["sales_growth", "days_of_inventory", "stock_gap", "reorder_gap", "incoming_stock_ratio"]:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN values for model compatibility
    numeric_cols = [
        "average_daily_sales", "sales_7d_avg", "sales_30d_avg", "sales_growth",
        "days_of_inventory", "lead_time_demand", "stock_gap", "reorder_gap",
        "incoming_stock_ratio", "promotion_flag", "month", "day_of_week", "weekend_flag"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def create_target(df):
    """Create a binary target using future inventory logic. It uses only current feature values and the next 7-day demand estimate."""
    df = df.copy()
    df["future_demand_7d"] = df["average_daily_sales"].replace(0, np.nan) * 7 * (1 + 0.35 * df["promotion_flag"] + 0.2 * df["holiday"])
    df["stockout_risk"] = (
        (df["current_stock"] + df["incoming_stock"] - df["future_demand_7d"] <= 0)
        | (df["current_stock"] < df["lead_time_demand"] * 0.8)
    ).astype(int)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/inventory_stockout_clean.csv")
    df = create_inventory_features(df)
    df = create_target(df)
    df.to_csv("data/processed/inventory_stockout_features.csv", index=False)
    print(df[["current_stock", "average_daily_sales", "lead_time_demand", "stockout_risk"]].head())
