import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import PROCESSED_DATA_DIR

SERIES_FILE = PROCESSED_DATA_DIR / "traffic_series.csv"
FORECAST_FILE = PROCESSED_DATA_DIR / "predictions" / "traffic_forecast_12_15_dec.csv"

START_DATE = pd.Timestamp("2025-12-12 00:00:00")
END_DATE   = pd.Timestamp("2025-12-13 23:59:59")

# wczytanie danych
series = pd.read_csv(
    SERIES_FILE,
    names=["road", "timestamp", "traffic", "hs", "hc", "ds", "dc"],
    low_memory=False
)

series["timestamp"] = pd.to_datetime(series["timestamp"], errors="coerce")
series = series.dropna(subset=["timestamp"])
series["traffic"] = pd.to_numeric(series["traffic"], errors="coerce")
series = series.dropna(subset=["traffic"])

series = series[
    (series["timestamp"] >= START_DATE) &
    (series["timestamp"] <= END_DATE)
]

series["hour"] = series["timestamp"].dt.floor("H")

series_hourly = (
    series
    .sort_values("timestamp")
    .groupby(["road", "hour"], as_index=False)
    .first()
)[["road", "hour", "traffic"]]

forecast = pd.read_csv(FORECAST_FILE, parse_dates=["timestamp"])
forecast["hour"] = forecast["timestamp"].dt.floor("H")

forecast = forecast[
    (forecast["timestamp"] >= START_DATE) &
    (forecast["timestamp"] <= END_DATE)
]

# testowanie progow
results = []

for threshold in np.arange(0.30, 0.81, 0.01):
    forecast["cls"] = (forecast["predicted_traffic_cont"] >= threshold).astype(int)

    merged = series_hourly.merge(
        forecast[["road", "hour", "cls"]],
        on=["road", "hour"],
        how="inner"
    )

    diff = (merged["traffic"] != merged["cls"]).sum()

    results.append((threshold, diff))

# wynik
results = sorted(results, key=lambda x: x[1])

print("Najlepsze progi:")
for t, d in results[:10]:
    print(f"próg = {t:.2f} | rozbieżności = {d}")
