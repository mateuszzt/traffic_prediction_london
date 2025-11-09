import requests
import pandas as pd
from datetime import datetime
import os
from utils.config import RAW_DATA_DIR  # używamy ścieżki z config.py

APP_KEY = "26660da0e4da49109d60f163b460307e"  # <-- wklej swój klucz tutaj

def get_tfl_roads():
    url = "https://api.tfl.gov.uk/Road/"
    params = {"app_key": APP_KEY}
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print("❌ Błąd:", response.status_code, response.text)
        return None

    data = response.json()

    # Konwersja do DataFrame
    df = pd.DataFrame([{
        "road": r["displayName"],
        "status": r["statusSeverity"],
        "description": r["statusSeverityDescription"],
        "timestamp": datetime.utcnow()
    } for r in data])

    # 🔧 Zapisz do pliku we właściwym katalogu
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    output_path = RAW_DATA_DIR / "tfl_roads.csv"
    df.to_csv(str(output_path), index=False)

    print(f"✅ Zapisano {len(df)} rekordów do {output_path}")
    return df

if __name__ == "__main__":
    df = get_tfl_roads()
    print(df.head(100))
