import numpy as np
import pandas as pd
import torch
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import PROCESSED_DATA_DIR

# parametry sekwencji
INPUT_LEN = 72
OUTPUT_LEN = 48

FEATURES = [
    "traffic",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

# przedzial czasowy
TRAIN_START = "2025-11-09"
TRAIN_END   = "2025-12-04"

TEST_START  = "2025-12-05"
TEST_END    = "2025-12-11"

FORECAST_START = "2025-12-12"
FORECAST_END   = "2025-12-15"



# budowanie sekwencji
def build_sequences(df_road: pd.DataFrame):
    X, y = [], []

    values = df_road[FEATURES].values
    target = df_road["traffic"].values

    if len(df_road) < INPUT_LEN + OUTPUT_LEN:
        return X, y

    for i in range(len(df_road) - INPUT_LEN - OUTPUT_LEN + 1):
        X.append(values[i : i + INPUT_LEN])
        y.append(target[i + INPUT_LEN : i + INPUT_LEN + OUTPUT_LEN])

    return X, y


# glowna funkcja
def make_sequences(input_file="traffic_series.csv"):
    input_path = PROCESSED_DATA_DIR / input_file

    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("traffic_series.csv is empty")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["road", "timestamp"]).reset_index(drop=True)

    # podzial po dacie
    train_df = df.query(
        "@TRAIN_START <= timestamp <= @TRAIN_END"
    )
    test_df = df.query(
        "@TEST_START <= timestamp <= @TEST_END"
    )
    forecast_df = df.query(
        "@FORECAST_START <= timestamp <= @FORECAST_END"
    )

    splits = {
        "train": train_df,
        "test": test_df,
        "forecast": forecast_df,
    }

    results = {}

    # sekwencje
    for split_name, split_df in splits.items():
        X_all, y_all = [], []

        for road in split_df["road"].unique():
            df_road = split_df[split_df["road"] == road]

            X_road, y_road = build_sequences(df_road)
            X_all.extend(X_road)
            y_all.extend(y_road)

        X = torch.tensor(np.array(X_all), dtype=torch.float32)
        y = torch.tensor(np.array(y_all), dtype=torch.float32)

        results[split_name] = (X, y)

        print(
            f"{split_name.upper():8s} | "
            f"X: {X.shape} | y: {y.shape}"
        )

    # zapis do pliku
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    for split_name, (X, y) in results.items():
        torch.save(
            (X, y),
            PROCESSED_DATA_DIR / f"sequences_{split_name}.pt"
        )

    return results


if __name__ == "__main__":
    make_sequences()
