import numpy as np
import pandas as pd


def load_inventory_data(path):
    df = pd.read_csv(path)
    return df


def missing_value_report(df):
    missing = df.isna().sum().sort_values(ascending=False)
    percent = (missing / len(df)) * 100
    return pd.DataFrame({"missing_count": missing, "missing_percentage": percent})


def clean_inventory_data(df):
    df = df.copy()

    # Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # Standardize text columns
    for col in ["category", "supplier_id"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title().replace({"Nan": np.nan, "None": np.nan})

    # Convert dates and ensure type consistency
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Correct numeric columns
    numeric_cols = [
        "current_stock", "daily_sales", "units_sold_7d", "units_sold_30d",
        "reorder_level", "lead_time_days", "incoming_stock", "price",
        "returns", "stockout", "stockout_risk", "promotion", "holiday"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Detect invalid values
    if "current_stock" in df.columns:
        df.loc[df["current_stock"] < 0, "current_stock"] = np.nan
    if "daily_sales" in df.columns:
        df.loc[df["daily_sales"] < 0, "daily_sales"] = np.nan
    if "units_sold_7d" in df.columns:
        df.loc[df["units_sold_7d"] < 0, "units_sold_7d"] = np.nan
    if "units_sold_30d" in df.columns:
        df.loc[df["units_sold_30d"] < 0, "units_sold_30d"] = np.nan
    if "incoming_stock" in df.columns:
        df.loc[df["incoming_stock"] < 0, "incoming_stock"] = np.nan
    if "returns" in df.columns:
        df.loc[df["returns"] < 0, "returns"] = np.nan

    # Fill missing numeric values with median for robustness
    for col in numeric_cols:
        if col in df.columns:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)

    # Fill missing categorical values with mode
    for col in ["category", "supplier_id"]:
        if col in df.columns:
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value.iloc[0])

    # Flag outliers using IQR rule
    for col in ["current_stock", "daily_sales", "lead_time_days", "price"]:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


if __name__ == "__main__":
    raw_df = load_inventory_data("data/raw/inventory_stockout_raw.csv")
    clean_df = clean_inventory_data(raw_df)
    clean_df.to_csv("data/processed/inventory_stockout_clean.csv", index=False)
    print(clean_df.head())
    print(missing_value_report(clean_df))
