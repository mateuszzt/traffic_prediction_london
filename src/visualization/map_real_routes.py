import osmnx as ox
import folium
import pandas as pd
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import PROCESSED_DATA_DIR, VISUALIZATION_DIR  # ścieżki z config.py


# === 1️⃣ Wczytaj dane o natężeniu ruchu ===
print("📊 Wczytywanie danych o ruchu...")
input_path = PROCESSED_DATA_DIR / "traffic_series.csv"
df = pd.read_csv(input_path)

# Średni poziom natężenia dla każdej drogi
traffic_mean = df.groupby("road")["status_code"].mean().reset_index()

# === 2️⃣ Pobranie danych drogowych z OpenStreetMap ===
print("📡 Pobieram rzeczywiste drogi z OpenStreetMap (może potrwać chwilę)...")
G = ox.graph_from_place("Greater London, UK", network_type="drive")

# Konwersja do GeoDataFrame
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True, fill_edge_geometry=True)

# === 3️⃣ Filtr interesujących nas dróg (A1, A2, A3, A40, itd.) ===
pattern = r"\bA1\b|\bA2\b|\bA3\b|\bA4\b|\bA10\b|\bA12\b|\bA13\b|\bA20\b|\bA23\b|\bA40\b|\bA406\b"

# Jeśli kolumna 'ref' istnieje – użyj jej, inaczej fallback do 'name'
if "ref" in gdf_edges.columns:
    search_column = "ref"
#else:
#    search_column = "name"

# Spłaszcz wartości listowe w kolumnie 'ref' do stringów
def flatten_ref(val):
    if isinstance(val, list):
        return ", ".join(map(str, val))
    return str(val)

gdf_edges[search_column] = gdf_edges[search_column].apply(flatten_ref)

# Wyszukiwanie dróg po ref (np. A1, A40, A406)
roads_gdf = gdf_edges[gdf_edges[search_column].fillna("").astype(str).str.contains(pattern, regex=True, case=False)]

print(f"✅ Znaleziono {len(roads_gdf)} odcinków dróg w Londynie (kolumna: {search_column})")
print("📋 Przykładowe wartości:", roads_gdf[search_column].dropna().astype(str).unique()[:10])

# === 4️⃣ Funkcja kolorująca drogi wg natężenia ===
def get_color(road_name: str):
    row = traffic_mean[traffic_mean["road"] == road_name]
    if row.empty:
        return "gray"
    score = row["status_code"].values[0]
    if score < 0.3:
        return "green"      # płynny ruch
    elif score < 0.6:
        return "orange"     # umiarkowane natężenie
    else:
        return "red"        # duże zatłoczenie

# === 5️⃣ Tworzenie mapy ===
print("🗺️ Generuję mapę Londynu z zaznaczonymi drogami...")
m = folium.Map(location=[51.5074, -0.1278], zoom_start=11, tiles="CartoDB dark_matter")

# === 6️⃣ Rysowanie rzeczywistych dróg ===
for _, row in roads_gdf.iterrows():
    road_name = str(row["name"])
    ref_value = str(row[search_column])
    
    if "," in ref_value:
        ref_value = ref_value.split(",")[0].strip()
    elif ";" in ref_value:
        ref_value = ref_value.split(";")[0].strip()
    
    color = get_color(ref_value)
    geom = row["geometry"]
    if geom is None:
        continue
    if geom.geom_type == "LineString":
        coords = [(lat, lon) for lon, lat in geom.coords]
        folium.PolyLine(coords, color=color, weight=5, opacity=0.8, popup=road_name).add_to(m)
    elif geom.geom_type == "MultiLineString":
        for line in geom:
            coords = [(lat, lon) for lon, lat in line.coords]
            folium.PolyLine(coords, color=color, weight=5, opacity=0.8, popup=road_name).add_to(m)

# === 7️⃣ Dodanie legendy ===
legend_html = """
<div style="
     position: fixed; 
     bottom: 40px; left: 40px; width: 200px; height: 130px; 
     background-color: rgba(25, 25, 25, 0.8);
     border:2px solid lightgray; 
     z-index:9999; font-size:14px;
     color: white;
     padding: 10px;
     border-radius: 8px;">
<b>Legenda natężenia ruchu</b><br>
<svg width="20" height="10"><rect width="20" height="10" style="fill:green;"/></svg>  Płynny ruch<br>
<svg width="20" height="10"><rect width="20" height="10" style="fill:orange;"/></svg>  Umiarkowany ruch<br>
<svg width="20" height="10"><rect width="20" height="10" style="fill:red;"/></svg>  Duże zatłoczenie<br>
<svg width="20" height="10"><rect width="20" height="10" style="fill:gray;"/></svg>  Brak danych
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# === 8️⃣ Zapis do pliku ===
os.makedirs(VISUALIZATION_DIR, exist_ok=True)
output_path = VISUALIZATION_DIR / "london_real_traffic_map.html"
m.save(str(output_path))
print(f"✅ Mapa gotowa! Zapisano jako: {output_path}")
print("👉 Otwórz plik w przeglądarce, żeby zobaczyć kolorowe drogi Londynu.")
