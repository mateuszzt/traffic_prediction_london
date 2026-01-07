"""
config.py
==================
Konfiguracja sciezek
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "src" / "model"
VISUALIZATION_DIR = BASE_DIR / "src" / "visualization"

if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("PROCESSED_DATA_DIR:", PROCESSED_DATA_DIR)
    print("MODELS_DIR:", MODELS_DIR)
    print("VISUALIZATION_DIR:", VISUALIZATION_DIR)
