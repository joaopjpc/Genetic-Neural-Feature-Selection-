import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config
from src.utils import ensure_dir


def load_raw_data(path=config.RAW_FILE) -> pd.DataFrame:
    return pd.read_excel(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how="all")
    return df


def split_and_scale(df: pd.DataFrame, target_column: str = config.TARGET_COLUMN):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    val_relative_size = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=config.RANDOM_STATE,
        stratify=y_train_val if y_train_val.nunique() > 1 else None,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


def run_preprocessing() -> None:
    ensure_dir(config.INTERIM_DIR)
    ensure_dir(config.PROCESSED_DIR)

    df = load_raw_data()
    clean_df = clean_data(df)
    clean_df.to_csv(config.INTERIM_FILE, index=False)

    X_train, X_val, X_test, y_train, y_val, y_test = split_and_scale(clean_df)
    X_train.to_csv(config.PROCESSED_DIR / "X_train.csv", index=False)
    X_val.to_csv(config.PROCESSED_DIR / "X_val.csv", index=False)
    X_test.to_csv(config.PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(config.PROCESSED_DIR / "y_train.csv", index=False)
    y_val.to_csv(config.PROCESSED_DIR / "y_val.csv", index=False)
    y_test.to_csv(config.PROCESSED_DIR / "y_test.csv", index=False)
