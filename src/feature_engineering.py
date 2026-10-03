from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

DROP_IDENTIFIER_COLUMNS = [
    "Unnamed: 0", "cc_num", "first", "last", "street", "trans_num", "unix_time"
]

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (
        np.sin((lat2-lat1)/2)**2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2)**2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))

def engineer(df):
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
    df["is_night"] = df["hour"].isin([22, 23, 0, 1, 2, 3]).astype("int8")
    df["customer_age"] = (
        df["trans_date_trans_time"].dt.year - df["dob"].dt.year
        - (
            (df["trans_date_trans_time"].dt.month < df["dob"].dt.month)
            | (
                (df["trans_date_trans_time"].dt.month == df["dob"].dt.month)
                & (df["trans_date_trans_time"].dt.day < df["dob"].dt.day)
            )
        ).astype(int)
    )
    df["distance_km"] = haversine_km(
        df["lat"].to_numpy(),
        df["long"].to_numpy(),
        df["merch_lat"].to_numpy(),
        df["merch_long"].to_numpy(),
    )
    df["log_amount"] = np.log1p(df["amt"])

    # Merchant values in this synthetic dataset all include a literal "fraud_" prefix.
    # The prefix is constant and therefore carries no target information; strip it for cleaner labels.
    df["merchant"] = df["merchant"].str.replace(r"^fraud_", "", regex=True)

    # Keep transaction time only long enough to derive time features.
    drop_cols = [c for c in DROP_IDENTIFIER_COLUMNS if c in df.columns]
    drop_cols += ["dob", "trans_date_trans_time"]
    return df.drop(columns=drop_cols)

if __name__ == "__main__":
    for source, target in [
        ("fraudTrain.csv", "train_features.csv"),
        ("fraudTest.csv", "test_features.csv"),
    ]:
        raw = pd.read_csv(RAW / source)
        engineered = engineer(raw)
        engineered.to_csv(PROCESSED / target, index=False)
        print(f"{source}: {raw.shape} -> {engineered.shape}")
