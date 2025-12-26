import pandas as pd
import os
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def preprocess_tfl(
    input_file="traffic_history.csv",
    output_file="traffic_series.csv"
):
    input_path = RAW_DATA_DIR / input_file
    output_path = PROCESSED_DATA_DIR / output_file
    unknown_path = PROCESSED_DATA_DIR / "unknown_statuses.csv"

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # wczytanie danych
    df = pd.read_csv(input_path)

    # mapowanie statusu
    mapping = {
        "Good": 0.0,
        "Serious": 1.0,
        "Severe": 2.0,
    }

    unknown = df[~df["status"].isin(mapping)]
    if not unknown.empty:
        file_exists = unknown_path.exists()
        unknown.to_csv(
            unknown_path,
            mode="a",
            header=not file_exists,
            index=False
        )

    df["traffic"] = df["status"].map(mapping)
    df = df.dropna(subset=["traffic"])

    # datetime i sortowanie
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df = (
        df
        .sort_values(["road", "timestamp"])
        .reset_index(drop=True)
    )

    # cechy czasowe
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    # zestawienie kolumn
    df = df[
        [
            "road",
            "timestamp",
            "traffic",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
    ]

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    preprocess_tfl()
