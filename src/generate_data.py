import numpy as np
import pandas as pd
from pathlib import Path


rng = np.random.default_rng(42)


def generate_inventory_data(output_path="data/raw/inventory_stockout_raw.csv", n_products=200, days_per_product=60):
    """Generate a realistic synthetic inventory dataset for stockout risk prediction."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = ["Electronics", "Home", "Fashion", "Grocery", "Health", "Office"]
    category_params = {
        "Electronics": {"base_demand": 35, "price": (120, 500), "lead_time": (4, 12)},
        "Home": {"base_demand": 28, "price": (60, 220), "lead_time": (5, 14)},
        "Fashion": {"base_demand": 40, "price": (30, 180), "lead_time": (3, 10)},
        "Grocery": {"base_demand": 55, "price": (10, 80), "lead_time": (2, 7)},
        "Health": {"base_demand": 22, "price": (25, 170), "lead_time": (4, 11)},
        "Office": {"base_demand": 30, "price": (50, 250), "lead_time": (5, 13)},
    }

    start_date = pd.Timestamp("2023-01-01")
    rows = []

    for product_id in range(1, n_products + 1):
        category = rng.choice(categories)
        params = category_params[category]
        base_daily_sales = rng.uniform(params["base_demand"] * 0.7, params["base_demand"] * 1.5)
        supplier_id = f"SUP{rng.integers(1, 18)}"
        reorder_level = int(rng.uniform(base_daily_sales * 8, base_daily_sales * 18))
        lead_time_days = int(rng.integers(params["lead_time"][0], params["lead_time"][1] + 1))
        avg_price = rng.uniform(params["price"][0], params["price"][1])

        for day_offset in range(days_per_product):
            record_date = start_date + pd.Timedelta(days=day_offset + (product_id % 30))
            month = record_date.month
            seasonal_factor = 1 + 0.30 * np.sin((month / 12) * 2 * np.pi)
            weekend_factor = 1.25 if record_date.dayofweek >= 5 else 1.0
            promotion = int(rng.random() < 0.22)
            holiday = int(rng.random() < 0.12)
            price = avg_price * (1 + rng.uniform(-0.2, 0.18))

            expected_demand = base_daily_sales * seasonal_factor * weekend_factor
            expected_demand *= 1 + 0.65 * promotion + 0.45 * holiday
            daily_sales = max(0, int(np.round(rng.normal(expected_demand, expected_demand * 0.45))))
            units_sold_7d = max(0, int(daily_sales * 7 * (0.9 + rng.random() * 0.8)))
            units_sold_30d = max(0, int(daily_sales * 30 * (0.85 + rng.random() * 0.9)))

            reorder_buffer = max(5, int(reorder_level * rng.uniform(0.8, 1.15)))
            current_stock = max(0, int(reorder_buffer + rng.normal(0, reorder_buffer * 0.35)))
            incoming_stock = max(0, int(rng.normal(reorder_level * 0.55, reorder_level * 0.25)))

            if promotion:
                daily_sales = max(daily_sales, int(daily_sales * 1.7))
            if holiday:
                daily_sales = max(daily_sales, int(daily_sales * 1.45))

            current_stock = min(current_stock, reorder_buffer * 2)
            current_stock = max(0, current_stock)

            future_demand_7d = daily_sales * 7 * (1 + 0.4 * promotion + 0.25 * holiday)
            future_stock_after_replenishment = current_stock + incoming_stock - future_demand_7d

            stockout_risk = int(
                (future_stock_after_replenishment <= 0)
                or (current_stock < daily_sales * lead_time_days * 0.75)
                or (reorder_level > 0 and current_stock < reorder_level * 0.50)
            )

            stockout = int(current_stock + incoming_stock <= daily_sales * 1.5)
            returns = max(0, int(rng.normal(1.5, 2.0))) if rng.random() < 0.6 else 0
            if rng.random() < 0.04:
                daily_sales = np.nan
            if rng.random() < 0.02:
                current_stock = np.nan
            if rng.random() < 0.015:
                supplier_id = np.nan
            if rng.random() < 0.03:
                category = np.nan
            if rng.random() < 0.02:
                rows.append(rows[-1].copy() if rows else {})

            row = {
                "product_id": product_id,
                "date": record_date,
                "category": category,
                "supplier_id": supplier_id,
                "current_stock": current_stock,
                "daily_sales": daily_sales,
                "units_sold_7d": units_sold_7d,
                "units_sold_30d": units_sold_30d,
                "reorder_level": reorder_level,
                "lead_time_days": lead_time_days,
                "incoming_stock": incoming_stock,
                "promotion": promotion,
                "price": round(price, 2),
                "holiday": holiday,
                "returns": returns,
                "stockout": stockout,
                "stockout_risk": stockout_risk,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["product_id", "date"], keep="first")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    generate_inventory_data()
