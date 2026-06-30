import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_processed_data(processed_dir: Path):
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_val = pd.read_csv(processed_dir / "X_val.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze("columns")
    y_val = pd.read_csv(processed_dir / "y_val.csv").squeeze("columns")
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")
    return X_train, X_val, X_test, y_train, y_val, y_test
