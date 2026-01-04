import torch
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error

from torch.utils.data import DataLoader, TensorDataset
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config import MODELS_DIR, PROCESSED_DATA_DIR
from src.model.model import TrafficModel

# konfiguracja
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURES_COUNT = 5          # traffic + hour_sin + hour_cos + dow_sin + dow_cos
BATCH_SIZE = 256            
CLASS_BINS = [0.7, 1.5]     
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = ["good", "serious", "severe"]


def to_classes(y: np.ndarray) -> np.ndarray:
    """
    Zamienia wartości ciągłe na klasy 0 / 1 / 2
    """
    return np.digitize(y, CLASS_BINS)


def evaluate():
    print("[1/7] Start ewaluacji")

    # wczytywanie danych testowych
    test_path = PROCESSED_DATA_DIR / "sequences_test.pt"
    if not test_path.exists():
        raise FileNotFoundError("sequences_test.pt not found")

    print("[2/7] Wczytywanie danych testowych")
    X_test, y_test = torch.load(test_path, map_location="cpu")

    print(f"    X_test shape: {tuple(X_test.shape)}")
    print(f"    y_test shape: {tuple(y_test.shape)}")

    # wczytywanie modelu
    model_path = MODELS_DIR / "traffic_model.pt"
    if not model_path.exists():
        raise FileNotFoundError("traffic_model.pt not found")

    print("[3/7] Wczytywanie modelu")
    model = TrafficModel(input_size=FEATURES_COUNT).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    print(f"    Model uruchomiony na: {DEVICE}")

    # data loader
    dataset = TensorDataset(X_test, y_test)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # predykcja na zbiorze testowym
    print("[4/7] Predykcja na zbiorze testowym (batchami)")

    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for idx, (Xb, yb) in enumerate(loader, start=1):
            Xb = Xb.to(DEVICE)
            preds = model(Xb).cpu()

            y_pred_all.append(preds)
            y_true_all.append(yb)

            if idx % 20 == 0:
                print(f"    Przetworzono batch {idx}/{len(loader)}")

    y_pred = torch.cat(y_pred_all).numpy()
    y_true = torch.cat(y_true_all).numpy()

    print("[5/7] Przetwarzanie wyników")

    # flatten
    y_pred_flat = y_pred.reshape(-1)
    y_true_flat = y_true.reshape(-1)

    # metryki regresyjne
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_flat, y_pred_flat)

    # dyskretyzacja
    y_pred_cls = to_classes(y_pred_flat)
    y_true_cls = to_classes(y_true_flat)

    


    print("[6/7] Liczenie metryk")

    # metryki
    cm = confusion_matrix(
        y_true_cls,
        y_pred_cls,
        labels=CLASS_LABELS
    )

    report = classification_report(
        y_true_cls,
        y_pred_cls,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        zero_division=0
    )

    # wyniki
    print("\nCONFUSION MATRIX:")
    print(cm)

    print("\nCLASSIFICATION REPORT:")
    print(report)

    print("\nREGRESSION METRICS:")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")


    print("[7/7] Ewaluacja zakończona")

    return cm, report


if __name__ == "__main__":
    evaluate()
