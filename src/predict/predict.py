import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import timedelta
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.model.model import TrafficModel

# konfiguracja
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURES = [
    "traffic",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

INPUT_LEN = 72          # 72 godziny historii
OUTPUT_LEN = 48         # 48 godzin predykcji

FORECAST_START = pd.Timestamp("2025-12-12 00:00:00")
FORECAST_END   = pd.Timestamp("2025-12-13 23:00:00")

# progi klasowe
CLASS_BINS = [0.7, 1.5]  


def predict_48h(input_file="traffic_series.csv"):
    # wczytanie modelu
    model_path = MODELS_DIR / "traffic_model.pt"
    if not model_path.exists():
        raise FileNotFoundError("traffic_model.pt not found")

    model = TrafficModel(input_size=len(FEATURES)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # wczytanie danych
    df = pd.read_csv(PROCESSED_DATA_DIR / input_file)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["road", "timestamp"]).reset_index(drop=True)

    predictions = []

    # predykcja dla kazdej drogi
    for road in df["road"].unique():
        df_road = df[df["road"] == road]

        # historia wyłącznie sprzed okresu prognozy
        history = df_road[df_road["timestamp"] < FORECAST_START]

        if len(history) < INPUT_LEN:
            continue

        last_window = history.iloc[-INPUT_LEN:]

        X = torch.tensor(
            last_window[FEATURES].values,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_cont = model(X).cpu().numpy().flatten()

        # ograniczenie zakresu
        pred_cont = np.clip(pred_cont, 0.0, 2.0)

        # mapowanie na klasy
        pred_cls = np.digitize(pred_cont, bins=CLASS_BINS)

        # zapis
        for i in range(len(pred_cont)):
            ts = FORECAST_START + timedelta(hours=i)
            if ts > FORECAST_END:
                break

            predictions.append({
                "road": road,
                "timestamp": ts,
                "hour_ahead": i + 1,
                "predicted_traffic_cont": float(pred_cont[i]),
                "predicted_traffic_class": int(pred_cls[i]),
            })

    # zapis do csv
    out_dir = PROCESSED_DATA_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "traffic_forecast_12_13_dec.csv"
    pd.DataFrame(predictions).to_csv(out_file, index=False)

    return out_file


if __name__ == "__main__":
    predict_48h()
