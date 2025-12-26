import requests
import pandas as pd
from datetime import datetime
import os
from utils.config import RAW_DATA_DIR  

APP_KEY = "26660da0e4da49109d60f163b460307e"  

def get_tfl_roads():
    url = "https://api.tfl.gov.uk/Road/"
    params = {"app_key": APP_KEY}
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print("Błąd:", response.status_code, response.text)
        return None

    data = response.json()

    df = pd.DataFrame([{
        "road": r["displayName"],
        "status": r["statusSeverity"],
        "description": r["statusSeverityDescription"],
        "timestamp": datetime.utcnow()
    } for r in data])

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    output_path = RAW_DATA_DIR / "tfl_roads.csv"
    df.to_csv(str(output_path), index=False)

    print(f"Zapisano {len(df)} rekordów do {output_path}")
    return df

if __name__ == "__main__":
    df = get_tfl_roads()
    print(df.head(100))
