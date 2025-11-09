import torch
import torch.nn as nn
import os
from utils.config import MODELS_DIR, PROCESSED_DATA_DIR  # ścieżki z config.py
from src.data.make_sequences import make_sequences  # poprawny import z projektu


class TrafficRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train_model(epochs=50):
    X, y = make_sequences()

    model = TrafficRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

    # 🔧 Upewnij się, że katalog models/ istnieje
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = MODELS_DIR / "traffic_rnn.pth"
    torch.save(model.state_dict(), str(model_path))
    print(f"✅ Model zapisany jako {model_path}")


if __name__ == "__main__":
    train_model()
