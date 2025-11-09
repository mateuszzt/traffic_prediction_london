"""
config.py
==================
Centralna konfiguracja ścieżek projektu Smart City Traffic Prediction.
"""

from pathlib import Path

# Główny katalog projektu (tam, gdzie jest folder 'data' i 'src')
BASE_DIR = Path(__file__).resolve().parents[1]

# Ścieżki do danych
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ścieżki do modeli i wizualizacji
MODELS_DIR = BASE_DIR / "src" / "model"
VISUALIZATION_DIR = BASE_DIR / "src" / "visualization"

# Możesz dodać kolejne ścieżki:
# REPORTS_DIR = BASE_DIR / "reports"
# NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Test poprawności — tylko jeśli uruchamiasz ten plik samodzielnie
if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("PROCESSED_DATA_DIR:", PROCESSED_DATA_DIR)
    print("MODELS_DIR:", MODELS_DIR)
    print("VISUALIZATION_DIR:", VISUALIZATION_DIR)
