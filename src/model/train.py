import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import MODELS_DIR
from src.model.model import TrafficModel
from src.data.make_sequences import make_sequences

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 6
LR = 1e-3


def train_model():
    # wczytanie sekwencji
    data = make_sequences()

    X_train, y_train = data["train"]
    X_val, y_val = data["test"]   

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=False   
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # model
    model = TrafficModel(
        input_size=X_train.shape[2]
    ).to(DEVICE)


    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    patience = 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "traffic_model.pt"

    # trening
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            preds = model(Xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # walidacja czasowa
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(Xb)
                val_loss += loss_fn(preds, yb).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"train={train_loss:.4f} | val={val_loss:.4f}"
        )

        # early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping")
                break


if __name__ == "__main__":
    train_model()
