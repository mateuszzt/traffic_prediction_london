import numpy as np
import pandas as pd
import torch
import os
from utils.config import PROCESSED_DATA_DIR  # korzystamy z centralnej konfiguracji

def make_sequences(input_file="traffic_series.csv", seq_length=10):
    """
    Tworzy sekwencje czasowe z danych o ruchu (status_code)
    i zapisuje je jako tensory PyTorch w data/processed/.
    """
    input_path = PROCESSED_DATA_DIR / input_file
    output_path = PROCESSED_DATA_DIR / "traffic_tensors.pt"

    # Upewnij się, że katalog istnieje
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print(f"📂 Wczytywanie danych z {input_path}")
    df = pd.read_csv(input_path)
    
    # Wybierz jedną drogę (np. A2) na początek
    df_road = df[df["road"] == "A2"]

    if df_road.empty:
        raise ValueError("❌ Brak danych dla wybranej drogi (A2). Sprawdź kolumnę 'road' w pliku CSV.")

    series = df_road["status_code"].values
    X, y = [], []
    
    for i in range(len(series) - seq_length):
        X.append(series[i:i+seq_length])
        y.append(series[i+seq_length])

    X = np.array(X)
    y = np.array(y)

    print(f"✅ Utworzono {len(X)} sekwencji (długość = {seq_length})")

    # Zapisz jako tensory do treningu
    X_tensor = torch.tensor(X).float().unsqueeze(2)
    y_tensor = torch.tensor(y).float().unsqueeze(1)

    torch.save((X_tensor, y_tensor), str(output_path))
    print(f"💾 Dane zapisane w {output_path}")

    return X_tensor, y_tensor


if __name__ == "__main__":
    make_sequences()
