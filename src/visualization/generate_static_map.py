import osmnx as ox
import folium
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import VISUALIZATION_DIR


print("📡 Pobieram drogi z OSM tylko raz...")
G = ox.graph_from_place("Greater London, UK", network_type="drive")
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True, fill_edge_geometry=True)

pattern = r"\bA1\b|\bA2\b|\bA3\b|\bA4\b|\bA10\b|\bA12\b|\bA13\b|\bA20\b|\bA23\b|\bA40\b|\bA406\b|\bNorth Circular\b"



if "ref" in gdf_edges.columns:
    search_column = "ref"
else:
    search_column = "name"

def flatten_ref(val):
    if isinstance(val, list):
        return ", ".join(map(str, val))
    return str(val)

gdf_edges[search_column] = gdf_edges[search_column].apply(flatten_ref)

roads_gdf = gdf_edges[
    gdf_edges["ref"].fillna("").str.contains(pattern, case=False, regex=True) |
    gdf_edges["name"].fillna("").str.contains(pattern, case=False, regex=True)
]

print(f"🔍 Znalazłem {len(roads_gdf)} odcinków dróg.")

m = folium.Map(location=[51.5074, -0.1278], zoom_start=11, tiles="CartoDB dark_matter")

for _, row in roads_gdf.iterrows():
    ref_value = str(row[search_column])

    if "," in ref_value:
        ref_value = ref_value.split(",")[0].strip()
    elif ";" in ref_value:
        ref_value = ref_value.split(";")[0].strip()

    geom = row["geometry"]
    if geom is None:
        continue

    color = "white"   # KLUCZ – statyczna mapa bez trafficu

    if geom.geom_type == "LineString":
        coords = [(lat, lon) for lon, lat in geom.coords]
        folium.PolyLine(coords,
                        color=color,
                        weight=5,
                        opacity=0.8,
                        popup=ref_value).add_to(m)

    elif geom.geom_type == "MultiLineString":
        for line in geom:
            coords = [(lat, lon) for lon, lat in line.coords]
            folium.PolyLine(coords,
                            color=color,
                            weight=5,
                            opacity=0.8,
                            popup=ref_value).add_to(m)

output = VISUALIZATION_DIR / "base_london_map.html"
m.save(str(output))

print("✅ Mapa statyczna gotowa:", output)
