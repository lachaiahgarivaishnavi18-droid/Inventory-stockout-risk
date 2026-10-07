# Inventory Stockout Risk Prediction

## Project title
Inventory Stockout Risk Prediction

## Problem statement
Inventory stockout happens when a product is unavailable for sale because stock falls below demand before a replenishment order arrives. This creates lost sales, unhappy customers, lower service levels, and higher operational cost. This project builds a beginner-friendly machine learning system to identify products at risk of running out within the next 7 days.

## Objective
The goal is to predict whether a product will have a high risk of stockout in the next 7 days using historical inventory, sales, supplier, promotion, and demand-related features. The system focuses on classifying risk as a binary outcome:

- 0 = Not expected to stock out in the next 7 days
- 1 = Expected to stock out in the next 7 days

## Dataset
The project uses a synthetic inventory dataset designed to reflect realistic business behavior. It includes different product categories, lead times, delivery patterns, promotions, seasonal demand, duplicates, missing values, and outlier behavior.

Key fields include:

- product_id: unique product identifier
- date: record date
- category: product category
- supplier_id: supplier source
- current_stock: stock available at the time of record
- daily_sales: average sales per day
- units_sold_7d: total units sold in 7 days
- units_sold_30d: total units sold in 30 days
- reorder_level: threshold for ordering more stock
- lead_time_days: supplier lead time in days
- incoming_stock: expected inventory arriving soon
- promotion: promotion flag
- price: product price
- holiday: holiday flag
- returns: returned units
- stockout: historical stockout indicator
- stockout_risk: target variable

## EDA and business understanding
Exploratory Data Analysis is used to understand the data before modeling. It helps answer questions such as:

- Which categories have higher stockout risk?
- Do long supplier lead times increase risk?
- Do promotions create sudden demand spikes?
- Are low-stock items more likely to run out?

## Feature engineering
The model uses useful inventory features such as:

- average_daily_sales: product demand averaged over time
- sales_7d_avg: average demand over the last 7 days
- sales_30d_avg: average demand over the last 30 days
- sales_growth: recent demand compared with longer-term demand
- days_of_inventory: current_stock / average_daily_sales
- lead_time_demand: average_daily_sales * lead_time_days
- stock_gap: current_stock - lead_time_demand
- reorder_gap: reorder_level - current_stock
- incoming_stock_ratio: incoming_stock / reorder_level
- promotion_flag: whether a promotion is active
- weekend_flag: whether the date is a weekend
- month: month number
- day_of_week: day of week

These features help the model capture the relationship between demand, available stock, and time to restock.

## Models
This project uses a simple and defensible model stack:

- Logistic Regression: a strong baseline for binary classification
- Random Forest Classifier: a more flexible tree-based model

The project also includes a preprocessing pipeline using ColumnTransformer, OneHotEncoder, and StandardScaler.

## Evaluation
Model quality is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC

Recall is especially important in stockout prediction because missing a product that is actually at risk can cause lost sales and service issues.

## Dashboard
A basic Streamlit dashboard is included for quick product-level risk prediction and category/supplier summaries.

## Installation
```bash
cd Inventory-stockout-risk
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## How to run
Dataset generation:
```bash
python src/generate_data.py
```

Data cleaning and preprocessing:
```bash
python src/preprocessing.py
```

Feature engineering:
```bash
python src/feature_engineering.py
```

Model training:
```bash
python src/train.py
```

Dashboard:
```bash
streamlit run app/app.py
```

## Project structure
```text
inventory-stockout-risk/
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   └── 06_model_evaluation.ipynb
├── src/
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── generate_data.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── __init__.py
├── .gitignore
├── README.md
├── requirements.txt
└──
```

## Limitations
- Synthetic data may not perfectly reflect real supply-chain conditions.
- Unexpected demand shocks can be difficult to model.
- Supplier reliability is not always captured in the data.
- The prediction is risk-based and not a guarantee of stockout.

## Future improvements
- Add more detailed demand forecasting features.
- Incorporate supplier reliability and delays.
- Use live inventory feeds or ERP data.
- Retrain the model with more recent time windows.
- Explore time-series forecasting methods with a real business context.

## Business value
This project demonstrates a realistic data science workflow for a common supply-chain problem. It is suitable for a fresher-level portfolio because it focuses on data cleaning, EDA, feature engineering, modeling, business insight, and model evaluation without being overly complex.
