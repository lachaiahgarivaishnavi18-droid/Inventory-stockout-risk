import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.feature_engineering import create_inventory_features, create_target
    from src.preprocessing import clean_inventory_data, load_inventory_data
except ModuleNotFoundError:  # pragma: no cover
    from feature_engineering import create_inventory_features, create_target
    from preprocessing import clean_inventory_data, load_inventory_data


FEATURE_COLUMNS = [
    "current_stock", "daily_sales", "units_sold_7d", "units_sold_30d",
    "reorder_level", "lead_time_days", "incoming_stock", "promotion",
    "price", "holiday", "returns", "average_daily_sales", "sales_7d_avg",
    "sales_30d_avg", "sales_growth", "days_of_inventory", "lead_time_demand",
    "stock_gap", "reorder_gap", "incoming_stock_ratio", "promotion_flag",
    "month", "day_of_week", "weekend_flag", "category", "supplier_id"
]

TARGET_COLUMN = "stockout_risk"


def build_model_pipeline(model_name="logistic_regression"):
    categorical_features = ["category", "supplier_id"]
    numerical_features = [
        col for col in FEATURE_COLUMNS if col not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced"
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    return pipeline


def train_and_evaluate():
    raw_df = load_inventory_data("data/raw/inventory_stockout_raw.csv")
    clean_df = clean_inventory_data(raw_df)
    processed_df = create_inventory_features(clean_df)
    processed_df = create_target(processed_df)

    processed_df = processed_df.sort_values("date").reset_index(drop=True)
    X = processed_df[FEATURE_COLUMNS]
    y = processed_df[TARGET_COLUMN]

    train_idx = processed_df["date"] < processed_df["date"].quantile(0.7)
    valid_idx = processed_df["date"] >= processed_df["date"].quantile(0.7)

    X_train = X.loc[train_idx]
    X_valid = X.loc[valid_idx]
    y_train = y.loc[train_idx]
    y_valid = y.loc[valid_idx]

    pipeline = build_model_pipeline("logistic_regression")
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_valid)
    proba = pipeline.predict_proba(X_valid)[:, 1]

    print("Classification Report:\n", classification_report(y_valid, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_valid, preds))
    print("ROC AUC:", roc_auc_score(y_valid, proba))

    joblib.dump(pipeline, "models/stockout_model.pkl")
    print("Model saved to models/stockout_model.pkl")
    return pipeline


if __name__ == "__main__":
    train_and_evaluate()
