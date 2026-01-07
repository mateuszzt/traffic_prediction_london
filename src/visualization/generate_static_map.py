import osmnx as ox
import folium
import sys
import pandas as pd
from pathlib import Path
import json
import re

# sciezki projektu
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import VISUALIZATION_DIR

DATA_DIR = ROOT / "data" / "processed"

# kolory drog
COLORS = {
    0: "green",
    1: "orange",
    2: "red"
}

# pobieranie dróg z OSM
print("Pobieranie drogi z OSM")
G = ox.graph_from_place("Greater London, UK", network_type="drive")
gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True, fill_edge_geometry=True)

pattern = (
    r"\bA1\b|\bA2\b|\bA3\b|\bA4\b|\bA10\b|\bA12\b|\bA13\b|"
    r"\bA20\b|\bA21\b|\bA23\b|\bA24\b|\bA40\b|\bA41\b|\bA316\b|"
    r"\bA205\b|\bA406\b|"
    r"Inner Ring|"
    r"North Circular|South Circular|"
    r"Bishopsgate Cross Route|"
    r"City Route|"
    r"Farringdon Cross Route|"
    r"Western Cross Route|"
    r"Southern River Route|"
    r"Blackwall Tunnel|"
    r"Silvertown Tunnel"
)

search_column = "ref" if "ref" in gdf_edges.columns else "name"

def flatten_ref(val):
    if isinstance(val, list):
        return ", ".join(map(str, val))
    return str(val)

gdf_edges[search_column] = gdf_edges[search_column].apply(flatten_ref)

roads_gdf = gdf_edges[
    gdf_edges["ref"].fillna("").str.contains(pattern, case=False, regex=True)
    |
    gdf_edges["name"].fillna("").str.contains(pattern, case=False, regex=True)
].copy()


print(f"Znaleziono {len(roads_gdf)} odcinków dróg.")

def normalize_road_name(name: str) -> str:
    if not name:
        return ""

    name = name.lower()

    replacements = [
        " road", " street", " route", " tunnel",
        "(", ")", "-", ","
    ]

    for r in replacements:
        name = name.replace(r, "")

    return name.strip()


def extract_road_codes(name: str):

    return set(re.findall(r"\bA\d+\b", name.upper()))

# generowanie map
def generate_maps_timeline(csv_path, class_column, prefix):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    timeline = (
        df[["date", "hour"]]
        .drop_duplicates()
        .sort_values(["date", "hour"])
        .reset_index(drop=True)
    )

    print("Oś czasu (data + godzina):")
    print(timeline)

    timeline_meta = []

    # petla czasu
    for idx, row_time in timeline.iterrows():
        date = row_time["date"]
        hour = row_time["hour"]

        subset = df[
            (df["date"] == date) &
            (df["hour"] == hour)
        ]

        # wybór rekordu
        if class_column == "predicted_traffic_class":
            subset = (
                subset
                .sort_values("hour_ahead")
                .groupby("road")
                .head(1)
            )
        else:
            subset = (
                subset
                .sort_values("timestamp")
                .groupby("road")
                .tail(1)
            )

        # mapa
        m = folium.Map(
            location=[51.5074, -0.1278],
            zoom_start=11,
            tiles="CartoDB dark_matter"
        )

        # agregacja
        road_to_class = (
            subset
            .groupby("road")[class_column]
            .max()
            .to_dict()
        )

        # rysowanie drog
        for road, traffic_class in road_to_class.items():
            road_norm = normalize_road_name(road)
            color = COLORS.get(int(traffic_class), "gray")

            for _, edge in roads_gdf.iterrows():
                ref_norm = normalize_road_name(str(edge[search_column]))

                road_codes = extract_road_codes(road)
                ref_codes = extract_road_codes(str(edge[search_column]))

                if not road_codes or not ref_codes:
                    continue

                if road_codes.isdisjoint(ref_codes):
                    continue


                geom = edge["geometry"]
                if geom is None:
                    continue

                if geom.geom_type == "LineString":
                    coords = [(lat, lon) for lon, lat in geom.coords]
                    folium.PolyLine(
                        coords,
                        color=color,
                        weight=5,
                        tooltip=folium.Tooltip(str(road), sticky=True)
                    ).add_to(m)

                elif geom.geom_type == "MultiLineString":
                    for line in geom:
                        coords = [(lat, lon) for lon, lat in line.coords]
                        folium.PolyLine(
                            coords,
                            color=color,
                            weight=5,
                            tooltip=folium.Tooltip(str(road), sticky=True)
                        ).add_to(m)


        # legenda
        legend_html = f"""
        <div style="
        position: fixed;
        bottom: 40px; left: 40px;
        width: 260px;
        background-color: rgba(20,20,20,0.85);
        border:2px solid lightgray;
        z-index:9999;
        font-size:14px;
        color:white;
        padding:10px;
        border-radius:8px;">
        <b>Natężenie ruchu</b><br>
        <span style="color:green;">■</span> Płynny (0)<br>
        <span style="color:orange;">■</span> Umiarkowany (1)<br>
        <span style="color:red;">■</span> Duże (2)<br><br>
        <b>Czas:</b><br>
        {date} {hour:02d}:00
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        output_file = VISUALIZATION_DIR / f"{prefix}_{idx:03d}.html"
        m.save(output_file)

        timeline_meta.append({
            "index": idx,
            "date": str(date),
            "hour": int(hour),
            "file": output_file.name
        })

        print(f"✔ {output_file.name} → {date} {hour:02d}:00")

    # zapis osi czasu
    timeline_file = VISUALIZATION_DIR / f"{prefix}_timeline.json"
    with open(timeline_file, "w", encoding="utf-8") as f:
        json.dump(timeline_meta, f, indent=2)

    print("Zapisano oś czasu:", timeline_file)

# uruchomienie forecast
generate_maps_timeline(
    DATA_DIR / "traffic_series.csv",
    "traffic",
    "map_history"
)

generate_maps_timeline(
    DATA_DIR / "predictions" / "traffic_forecast_12_13_dec.csv",
    "predicted_traffic_class",
    "map_forecast"
)
