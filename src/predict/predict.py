import torch
import pandas as pd
from src.model.model import TrafficRNN
from src.data.make_sequences import make_sequences
from utils.config import MODELS_DIR  # poprawna ścieżka do katalogu modeli

# Wczytaj dane sekwencyjne
X, y = make_sequences()

# Załaduj model
model = TrafficRNN()
model.load_state_dict(torch.load(str(MODELS_DIR / "traffic_rnn.pth")))
model.eval()

# 🔮 Predykcja
with torch.no_grad():
    last_seq = X[-1].unsqueeze(0)
    pred = model(last_seq).item()

print("🔮 Przewidywany status (0=Good, 1=Moderate, 2=Severe, 3=Closure):", round(pred))
