import sys
from pathlib import Path
import os
import time
from datetime import datetime
import pandas as pd

# 🔧 Automatyczne ustawienie głównego katalogu projektu (np. /Users/mateusztruszkowski/Desktop/lon)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Importy po dodaniu ścieżki projektu
from utils.config import RAW_DATA_DIR
from src.data.data_loader import get_tfl_roads

# === Ścieżka do pliku wyjściowego ===
OUTPUT_FILE = RAW_DATA_DIR / "traffic_history.csv"

# === Upewnij się, że katalog istnieje ===
os.makedirs(RAW_DATA_DIR, exist_ok=True)


def collect_traffic_data(interval: int = 60):
    """
    Pętla zbierająca dane z API TfL co `interval` sekund.
    Dane dopisywane są do pliku `traffic_history.csv`.
    """
    print("🚦 Uruchamianie zbierania danych z TfL API (co 1 minutę)...")
    print(f"💾 Dane będą zapisywane do: {OUTPUT_FILE}")

    while True:
        try:
            # Pobierz aktualne dane z API
            df = get_tfl_roads()

            if df is not None and not df.empty:
                df["timestamp"] = datetime.utcnow()

                # Dopisz dane do pliku (nagłówki tylko przy pierwszym zapisie)
                file_exists = OUTPUT_FILE.exists()
                df.to_csv(str(OUTPUT_FILE), mode="a", header=not file_exists, index=False)

                print(f"✅ {datetime.now().strftime('%H:%M:%S')} - Zapisano {len(df)} rekordów")
            else:
                print("⚠️ Brak danych w odpowiedzi z API")

        except Exception as e:
            print(f"❌ Błąd podczas pobierania danych: {e}")

        # Czekaj określony czas przed kolejnym zapytaniem
        time.sleep(interval)


if __name__ == "__main__":
    collect_traffic_data()
