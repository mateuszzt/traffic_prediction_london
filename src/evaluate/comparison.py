import pandas as pd
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import PROCESSED_DATA_DIR

# pliki
TRAFFIC_SERIES_FILE = PROCESSED_DATA_DIR / "traffic_series.csv"
FORECAST_FILE = PROCESSED_DATA_DIR / "predictions" / "traffic_forecast_12_13_dec.csv"

# zakres czasowy
START_DATE = pd.Timestamp("2025-12-12 00:00:00")
END_DATE   = pd.Timestamp("2025-12-13 23:59:59")

# wczytanie traffic_series
print("Wczytywanie traffic_series.csv")

series_cols = [
    "road",
    "timestamp",
    "traffic",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

series = pd.read_csv(
    TRAFFIC_SERIES_FILE,
    names=series_cols,
    low_memory=False
)

# jawna konwersja timestamp
series["timestamp"] = pd.to_datetime(
    series["timestamp"],
    errors="coerce"
)

# usunięcie blednych rekordow
series = series.dropna(subset=["timestamp"])

# konwersja traffic na int (0/1/2)
series["traffic"] = pd.to_numeric(series["traffic"], errors="coerce").astype("Int64")

# filtr zakresu dat
series = series[
    (series["timestamp"] >= START_DATE) &
    (series["timestamp"] <= END_DATE)
]

# godzina jako pelna godzina
series["hour"] = series["timestamp"].dt.floor("H")

# pierwszy rekord z każdej godziny
series_hourly = (
    series
    .sort_values("timestamp")
    .groupby(["road", "hour"], as_index=False)
    .first()
)

series_hourly = series_hourly[["road", "hour", "traffic"]]

print(f"Rekordy godzinowe (rzeczywiste): {len(series_hourly)}")

# wczytanie predykcji
print("Wczytywanie traffic_forecast_12_13_dec.csv")

forecast = pd.read_csv(
    FORECAST_FILE,
    parse_dates=["timestamp"]
)

forecast["hour"] = forecast["timestamp"].dt.floor("H")

forecast = forecast[
    (forecast["timestamp"] >= START_DATE) &
    (forecast["timestamp"] <= END_DATE)
]

forecast = forecast[[
    "road",
    "hour",
    "predicted_traffic_class"
]]

forecast["predicted_traffic_class"] = forecast["predicted_traffic_class"].astype(int)

print(f"Rekordy godzinowe (predykcja): {len(forecast)}")

# mergowanie i porownanie
merged = series_hourly.merge(
    forecast,
    on=["road", "hour"],
    how="inner"
)

diff = merged[
    merged["traffic"] != merged["predicted_traffic_class"]
]

# wynik
print("\nREKORDY Z ROZBIEŻNOŚCIAMI:")
print(diff.to_string(index=False))

print(f"\nLiczba rozbieżności: {len(diff)}")
