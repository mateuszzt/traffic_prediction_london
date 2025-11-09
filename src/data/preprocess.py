import pandas as pd
import os
from utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR  # korzystamy z centralnych ścieżek

def preprocess_tfl(input_file="traffic_history.csv", output_file="traffic_series.csv"):
    """
    Wczytuje dane surowe z katalogu data/raw/, przetwarza je
    i zapisuje wynik do katalogu data/processed/.
    """
    input_path = RAW_DATA_DIR / input_file
    output_path = PROCESSED_DATA_DIR / output_file

    # Upewnij się, że katalogi istnieją
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    print(f"📂 Wczytywanie danych z {input_path}")
    df = pd.read_csv(input_path, names=["road", "status", "description", "timestamp"])

    # Zamień statusy tekstowe na liczby (dla modelu)
    mapping = {"Good": 0, "Moderate": 1, "Severe": 2, "Closure": 3}
    df["status_code"] = df["status"].map(mapping)

    # Usuń błędne lub puste rekordy
    df.dropna(subset=["status_code"], inplace=True)

    # Konwersja timestampu i sortowanie
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(by=["road", "timestamp"])

    # Zapisz do pliku CSV
    df.to_csv(str(output_path), index=False)
    print(f"✅ Dane przetworzone i zapisane w {output_path}")
    return df


if __name__ == "__main__":
    preprocess_tfl()
