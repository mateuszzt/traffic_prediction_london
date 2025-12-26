import re
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import PROCESSED_DATA_DIR, VISUALIZATION_DIR

# === 1. Wczytaj HTML mapy bazowej ===
html_path = VISUALIZATION_DIR / "base_london_map.html"
html = html_path.read_text(encoding="utf-8")

# === 2. Wczytaj dane traffic ===
df = pd.read_csv(PROCESSED_DATA_DIR / "traffic_series.csv")
traffic_mean = df.groupby("road")["status_code"].mean().to_dict()

def get_color(score):
    if score < 0.3: return "green"
    if score < 0.6: return "orange"
    return "red"

updated = html

# === 3. REGEXY DO WYSZUKIWANIA POWIĄZAŃ polyline <-> popup <-> html ===

poly_pattern = re.compile(
    r'(var\s+(poly_line_[0-9a-f]+)\s*=\s*L\.polyline\([\s\S]*?\)\.addTo\([^)]*\);)',
    flags=re.IGNORECASE
)

bind_pattern = re.compile(
    r'(poly_line_[0-9a-f]+)\.bindPopup\((popup_[0-9a-f]+)\)',
    flags=re.IGNORECASE
)

popup_set_pattern = re.compile(
    r'(popup_[0-9a-f]+)\.setContent\((html_[0-9a-f]+)\)',
    flags=re.IGNORECASE
)

popup_html_pattern = re.compile(
    r'var\s+(html_[0-9a-f]+)\s*=\s*\$\(`?<div[^>]*>(.*?)<\/div>`?\)',
    flags=re.IGNORECASE | re.DOTALL
)

# === 4. MAPOWANIE polyline → popup → html → tekst drogi ===

poly_to_popup = {}
for m in bind_pattern.finditer(html):
    poly_to_popup[m.group(1)] = m.group(2)

popup_to_html = {}
for m in popup_set_pattern.finditer(html):
    popup_to_html[m.group(1)] = m.group(2)

html_to_text = {}
for m in popup_html_pattern.finditer(html):
    html_to_text[m.group(1)] = m.group(2).strip()

# === 5. PRZETWARZANIE BLOKÓW polyline ===

def replace_colors_in_block(block, color):
    """
    Zamienia WSZYSTKIE warianty kolorów w definicji PolyLine:
    - color
    - fillColor
    - strokeColor
    - stroke
    - style="stroke:white"
    """

    # 1) Podmień "color": "white"
    block = re.sub(
        r'"color"\s*:\s*"[^"]+"',
        f'"color":"{color}"',
        block
    )

    # 2) Podmień "fillColor": "white"
    block = re.sub(
        r'"fillColor"\s*:\s*"[^"]+"',
        f'"fillColor":"{color}"',
        block
    )

    # 3) Podmień strokeColor
    block = re.sub(
        r'"strokeColor"\s*:\s*"[^"]+"',
        f'"strokeColor":"{color}"',
        block
    )

    # 4) Podmień w atrybucie style stroke:white
    block = re.sub(
        r'style="([^"]*?)stroke\s*:\s*[^;"]+',
        lambda m: m.group(0).split("stroke")[0] + f'stroke:{color}',
        block
    )

    return block


# === 6. GŁÓWNA PĘTLA — dla każdego polyline ===

for poly_block, poly_name in poly_pattern.findall(html):

    if poly_name not in poly_to_popup:
        continue

    popup_name = poly_to_popup[poly_name]

    if popup_name not in popup_to_html:
        continue

    html_name = popup_to_html[popup_name]

    if html_name not in html_to_text:
        continue

    road_name = html_to_text[html_name]  # np. A406

    # jeśli brak danych → kolor szary
    if road_name not in traffic_mean:
        color = "gray"
    else:
        color = get_color(traffic_mean[road_name])

    # ZASTOSUJ KOLOR
    new_block = replace_colors_in_block(poly_block, color)

    # Podmień blok w HTML
    updated = updated.replace(poly_block, new_block, 1)

# === 7. Zapis wynikowej mapy ===
output_path = VISUALIZATION_DIR / "london_real_traffic_map.html"
output_path.write_text(updated, encoding="utf-8")

print("✅ SUCCESS! Traffic poprawnie naniesiony na mapę (łącznie z A406).")
