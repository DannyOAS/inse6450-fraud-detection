from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(RAW / "fraudTrain.csv")
test = pd.read_csv(RAW / "fraudTest.csv")
for df in (train, test):
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

def summary(df, name):
    fraud = int(df["is_fraud"].sum())
    return {
        "split": name,
        "rows": len(df),
        "columns": len(df.columns),
        "fraud": fraud,
        "legitimate": len(df) - fraud,
        "fraud_rate_pct": fraud / len(df) * 100,
        "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_trans_num": int(df["trans_num"].duplicated().sum()),
        "start": df["trans_date_trans_time"].min(),
        "end": df["trans_date_trans_time"].max(),
    }

pd.DataFrame([summary(train, "Training"), summary(test, "Test")]).to_csv(
    TAB / "dataset_summary.csv", index=False
)

missing = pd.DataFrame({
    "column": train.columns,
    "train_missing": [train[c].isna().sum() for c in train.columns],
    "train_missing_pct": [train[c].isna().mean() * 100 for c in train.columns],
    "test_missing": [test[c].isna().sum() for c in train.columns],
    "test_missing_pct": [test[c].isna().mean() * 100 for c in train.columns],
})
missing.to_csv(TAB / "missingness.csv", index=False)

category = train.groupby("category")["is_fraud"].agg(
    transactions="size", frauds="sum", fraud_rate="mean"
)
category["fraud_rate_pct"] = category["fraud_rate"] * 100
category.sort_values("fraud_rate", ascending=False).to_csv(
    TAB / "fraud_by_category_train.csv"
)

train["hour"] = train["trans_date_trans_time"].dt.hour
hour = train.groupby("hour")["is_fraud"].agg(
    transactions="size", frauds="sum", fraud_rate="mean"
)
hour["fraud_rate_pct"] = hour["fraud_rate"] * 100
hour.to_csv(TAB / "fraud_by_hour_train.csv")

plt.figure(figsize=(6.5, 4.2))
counts = train["is_fraud"].value_counts().sort_index()
plt.bar(["Legitimate", "Fraud"], counts.values)
plt.ylabel("Transactions")
plt.title("Training Set Class Distribution")
plt.tight_layout()
plt.savefig(FIG / "01_class_distribution.png", dpi=180)
plt.close()

p99 = train["amt"].quantile(.99)
plt.figure(figsize=(6.5, 4.2))
plt.hist(train.loc[train["amt"] <= p99, "amt"], bins=60)
plt.xlabel("Transaction amount ($)")
plt.ylabel("Transactions")
plt.title("Transaction Amount Distribution (up to 99th percentile)")
plt.tight_layout()
plt.savefig(FIG / "02_amount_distribution.png", dpi=180)
plt.close()

plt.figure(figsize=(7.2, 4.8))
ordered = category.sort_values("fraud_rate_pct")
plt.barh(ordered.index, ordered["fraud_rate_pct"])
plt.xlabel("Fraud rate (%)")
plt.ylabel("Merchant category")
plt.title("Fraud Rate by Merchant Category - Training Set")
plt.tight_layout()
plt.savefig(FIG / "03_fraud_rate_by_category.png", dpi=180)
plt.close()

plt.figure(figsize=(6.8, 4.2))
plt.bar(hour.index.astype(str), hour["fraud_rate_pct"])
plt.xlabel("Hour of day")
plt.ylabel("Fraud rate (%)")
plt.title("Fraud Rate by Transaction Hour - Training Set")
plt.tight_layout()
plt.savefig(FIG / "04_fraud_rate_by_hour.png", dpi=180)
plt.close()

print("EDA complete. Outputs saved under outputs/.")
