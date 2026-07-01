import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, MinMaxScaler, OneHotEncoder


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

# Keep this easy to change. If this path does not exist, the script uses the
# first .xlsx found in data/raw, which avoids filename accent/encoding issues.
INPUT_PATH = RAW_DIR / "Base Slim Morte cancer de utero.xlsx"

TARGET = "label_cid"
RANDOM_STATE = 42
VALID_TARGET_CLASSES = ["C53", "C54", "C55"]

DROP_COLS = [
    # constantes ou praticamente nao informativas
    "TIPOBITO",
    "SEXO",
    # vazamento direto do alvo / causa / CID
    "LINHAA",
    "LINHAB",
    "LINHAC",
    "LINHAD",
    "LINHAII",
    "CAUSABAS",
    "causabas_categoria",
    "causabas_subcategoria",
    "CB_PRE",
    "COMUNSVOIM",
    "CAUSABAS_O",
    "CAUSAMAT",
    # nulos demais
    "NUDIASINF",
    "ALTCAUSA",
    "ESTABDESCR",
    # escolaridade redundante
    "ESC2010",
    "ESCFALAGR1",
    "SERIESCFAL",
    # redundante com ocor_SIGLA_UF
    "ocor_CODIGO_UF",
]

NUMERIC_COLS = [
    "idade_obito_anos",
    "ano_obito",
    "ano_nascimento",
]

BINARY_SN_COLS = [
    "res_AMAZONIA",
    "res_FRONTEIRA",
    "res_CAPITAL",
    "ocor_AMAZONIA",
    "ocor_FRONTEIRA",
    "ocor_CAPITAL",
]

CATEGORICAL_COLS = [
    "mes_obito",
    "mes_nascimento",
    "RACACOR",
    "ESTCIV",
    "ESC",
    "LOCOCOR",
    "ASSISTMED",
    "EXAME",
    "CIRURGIA",
    "NECROPSIA",
    "res_SIGLA_UF",
    "res_REGIAO",
    "ocor_SIGLA_UF",
    "ocor_REGIAO",
    "OCUP_GRUPO",
]

FREQUENCY_ENCODING_COLS = [
    "NATURAL",
    "CODMUNNATU",
    "CODMUNRES",
    "CODMUNOCOR",
    "CODESTAB",
]

IGNORED_TOKENS = {"", "9", "9.0", "99", "99.0", "999", "999.0", "9999", "9999.0", "NAN", "NONE", "<NA>"}
MONTH_COLS = {"mes_obito", "mes_nascimento"}
MONTH_IGNORED_TOKENS = IGNORED_TOKENS - {"9", "9.0"}


def _resolve_input_path(path: Path | str | None) -> Path:
    if path is not None:
        path = Path(path)
        if path.exists():
            return path

    if INPUT_PATH.exists():
        return INPUT_PATH

    xlsx_files = sorted(RAW_DIR.glob("*.xlsx"))
    if not xlsx_files:
        raise FileNotFoundError(f"Nenhum arquivo .xlsx encontrado em {RAW_DIR}")
    return xlsx_files[0]


def load_data(path: Path | str | None = INPUT_PATH) -> pd.DataFrame:
    path = _resolve_input_path(path)

    if path.suffix.lower() in {".xlsx", ".xls"}:
        excel_file = pd.ExcelFile(path)
        for sheet_name in excel_file.sheet_names:
            preview = pd.read_excel(path, sheet_name=sheet_name, nrows=5)
            if preview.shape[0] > 0 and preview.shape[1] > 0:
                return pd.read_excel(path, sheet_name=sheet_name)
        raise ValueError(f"Nenhuma aba com dados encontrada em {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Formato de arquivo nao suportado: {path.suffix}")


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    required_cols = [TARGET, "idade_obito_anos"]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise KeyError(f"Colunas obrigatorias ausentes: {missing_required}")

    df = df.drop_duplicates()
    df = df.dropna(subset=["idade_obito_anos", TARGET])
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET].isin(VALID_TARGET_CLASSES)]

    return df.reset_index(drop=True)


def _parse_compact_br_date(series: pd.Series) -> pd.Series:
    date_text = (
        series
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.zfill(8)
    )
    date_text = date_text.mask(date_text.str.len() != 8)
    return pd.to_datetime(date_text, format="%d%m%Y", errors="coerce")


def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "DTOBITO" not in df.columns:
        raise KeyError("Coluna obrigatoria ausente: DTOBITO")

    death_dates = _parse_compact_br_date(df["DTOBITO"])
    df["ano_obito"] = death_dates.dt.year
    df["mes_obito"] = death_dates.dt.month.astype("Int64").astype("string")

    if "DTNASC" in df.columns:
        birth_dates = _parse_compact_br_date(df["DTNASC"])
        df["ano_nascimento"] = birth_dates.dt.year
        df["mes_nascimento"] = birth_dates.dt.month.astype("Int64").astype("string")
    else:
        df["ano_nascimento"] = np.nan
        df["mes_nascimento"] = pd.Series(pd.NA, index=df.index, dtype="string")

    df = df.drop(columns=[col for col in ["DTOBITO", "DTNASC"] if col in df.columns])

    return df.reset_index(drop=True)


def create_occupation_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "OCUP" not in df.columns:
        df["OCUP_GRUPO"] = "IGNORADO"
        return df

    ocup_text = (
        df["OCUP"]
        .astype("string")
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"\D", "", regex=True)
        .str.strip()
    )
    ignored_mask = ocup_text.isna() | ocup_text.str.upper().isin(IGNORED_TOKENS)
    df["OCUP_GRUPO"] = ocup_text.str[:2]
    df.loc[ignored_mask | (df["OCUP_GRUPO"].astype("string").str.len() == 0), "OCUP_GRUPO"] = "IGNORADO"
    df = df.drop(columns=["OCUP"])

    return df


def drop_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [col for col in DROP_COLS if col in df.columns]
    return df.drop(columns=cols_to_drop)


def split_data(df: pd.DataFrame):
    feature_cols = NUMERIC_COLS + BINARY_SN_COLS + CATEGORICAL_COLS + FREQUENCY_ENCODING_COLS
    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def _binary_sn_to_numeric(values):
    frame = pd.DataFrame(values).copy()
    frame = frame.apply(lambda col: col.astype("string").str.strip().str.upper())
    frame = frame.replace({"S": 1, "N": 0})
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.fillna(0)
    return frame.to_numpy(dtype=float)


def _normalize_category_frame(values):
    frame = pd.DataFrame(values).copy()
    frame = frame.astype("string").fillna("IGNORADO")
    frame = frame.apply(lambda col: col.str.strip())
    frame = frame.apply(lambda col: col.str.replace(r"\.0$", "", regex=True))

    for col in frame.columns:
        ignored_tokens = MONTH_IGNORED_TOKENS if str(col) in MONTH_COLS else IGNORED_TOKENS
        frame[col] = frame[col].mask(frame[col].str.upper().isin(ignored_tokens), "IGNORADO")

    return frame


def _to_string_with_ignored(values):
    return _normalize_category_frame(values).to_numpy(dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.frequency_maps_ = []

    def fit(self, X, y=None):
        frame = _normalize_category_frame(X)
        self.feature_names_in_ = [str(col) for col in frame.columns]
        self.frequency_maps_ = []

        for col in frame.columns:
            frequencies = frame[col].value_counts(normalize=True, dropna=False).to_dict()
            self.frequency_maps_.append(frequencies)

        return self

    def transform(self, X):
        frame = _normalize_category_frame(X)
        encoded_cols = []

        for idx, col in enumerate(frame.columns):
            frequencies = self.frequency_maps_[idx]
            encoded = frame[col].map(frequencies).fillna(0).astype(float)
            encoded_cols.append(encoded.to_numpy())

        return np.column_stack(encoded_cols)


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler(clip=True)),
        ]
    )

    # Unknown binary values are filled explicitly with 0 after S/N conversion.
    binary_pipeline = Pipeline(
        steps=[
            ("sn_to_numeric", FunctionTransformer(_binary_sn_to_numeric, validate=False)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("to_string", FunctionTransformer(_to_string_with_ignored, validate=False)),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    frequency_pipeline = Pipeline(
        steps=[
            ("frequency", FrequencyEncoder()),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLS),
            ("binary", binary_pipeline, BINARY_SN_COLS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLS),
            ("frequency", frequency_pipeline, FREQUENCY_ENCODING_COLS),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    categorical_encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categorical_features = categorical_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    frequency_features = [f"{col}_freq" for col in FREQUENCY_ENCODING_COLS]
    return NUMERIC_COLS + BINARY_SN_COLS + categorical_features + frequency_features


def get_feature_groups(preprocessor: ColumnTransformer) -> dict[str, list[str]]:
    categorical_encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categorical_features = categorical_encoder.get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_groups = {}

    for col in NUMERIC_COLS:
        feature_groups[col] = [col]

    for col in BINARY_SN_COLS:
        feature_groups[col] = [col]

    start = 0
    for col, categories in zip(CATEGORICAL_COLS, categorical_encoder.categories_):
        end = start + len(categories)
        feature_groups[col] = categorical_features[start:end]
        start = end

    for col in FREQUENCY_ENCODING_COLS:
        feature_groups[col] = [f"{col}_freq"]

    return feature_groups


def _target_distribution(y) -> dict[str, int]:
    return pd.Series(y).value_counts().sort_index().astype(int).to_dict()


def _save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=True)


def preprocess_and_save(input_path: Path | str | None = INPUT_PATH, output_dir: Path | str = OUTPUT_DIR) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_original = load_data(input_path)
    n_linhas_original = len(df_original)
    n_colunas_original = df_original.shape[1]

    df = remove_invalid_rows(df_original)
    df = create_date_features(df)
    df = create_occupation_group(df)
    df = drop_unwanted_columns(df)

    selected_cols = NUMERIC_COLS + BINARY_SN_COLS + CATEGORICAL_COLS + FREQUENCY_ENCODING_COLS + [TARGET]
    missing_cols = [col for col in selected_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Colunas esperadas ausentes apos limpeza: {missing_cols}")

    df = df[selected_cols].copy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df[TARGET])
    class_mapping = {
        class_name: int(class_id)
        for class_id, class_name in enumerate(label_encoder.classes_)
    }
    df[TARGET] = y_encoded

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    y_train_array = y_train.to_numpy(dtype=int)
    y_val_array = y_val.to_numpy(dtype=int)
    y_test_array = y_test.to_numpy(dtype=int)

    feature_names = get_feature_names(preprocessor)
    feature_groups = get_feature_groups(preprocessor)
    n_features_numericas = len(NUMERIC_COLS)
    n_features_binarias = len(BINARY_SN_COLS)
    n_features_onehot = sum(len(features) for col, features in feature_groups.items() if col in CATEGORICAL_COLS)
    n_features_frequency = len(FREQUENCY_ENCODING_COLS)

    np.save(output_dir / "X_train.npy", X_train_processed)
    np.save(output_dir / "X_val.npy", X_val_processed)
    np.save(output_dir / "X_test.npy", X_test_processed)
    np.save(output_dir / "y_train.npy", y_train_array)
    np.save(output_dir / "y_val.npy", y_val_array)
    np.save(output_dir / "y_test.npy", y_test_array)

    _save_json({"feature_names": feature_names}, output_dir / "feature_names.json")
    _save_json(feature_groups, output_dir / "feature_groups.json")
    _save_json(class_mapping, output_dir / "class_mapping.json")
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    joblib.dump(label_encoder, output_dir / "label_encoder.joblib")

    inverse_mapping = {value: key for key, value in class_mapping.items()}
    report = {
        "n_linhas_original": int(n_linhas_original),
        "n_linhas_final": int(len(df)),
        "n_colunas_original": int(n_colunas_original),
        "n_features_finais": int(len(feature_names)),
        "classes_do_alvo": list(label_encoder.classes_),
        "distribuicao_do_alvo_total": _target_distribution(df[TARGET].map(inverse_mapping)),
        "distribuicao_do_alvo_treino": _target_distribution(pd.Series(y_train_array).map(inverse_mapping)),
        "distribuicao_do_alvo_validacao": _target_distribution(pd.Series(y_val_array).map(inverse_mapping)),
        "distribuicao_do_alvo_teste": _target_distribution(pd.Series(y_test_array).map(inverse_mapping)),
        "colunas_removidas": [col for col in DROP_COLS if col in df_original.columns],
        "colunas_transformadas": {
            "DTOBITO": ["ano_obito", "mes_obito"] if "DTOBITO" in df_original.columns else [],
            "DTNASC": ["ano_nascimento", "mes_nascimento"] if "DTNASC" in df_original.columns else [],
            "OCUP": ["OCUP_GRUPO"] if "OCUP" in df_original.columns else [],
        },
        "colunas_numericas": NUMERIC_COLS,
        "colunas_binarias": BINARY_SN_COLS,
        "colunas_categoricas": CATEGORICAL_COLS,
        "colunas_frequency_encoding": FREQUENCY_ENCODING_COLS,
        "n_features_onehot": int(n_features_onehot),
        "n_features_frequency": int(n_features_frequency),
        "n_features_numericas": int(n_features_numericas),
        "n_features_binarias": int(n_features_binarias),
        "feature_groups_path": str(output_dir / "feature_groups.json"),
        "colunas_candidatas_ag": feature_names,
    }
    _save_json(report, output_dir / "preprocessing_report.json")

    print(f"Shape X_train: {X_train_processed.shape}")
    print(f"Shape X_val: {X_val_processed.shape}")
    print(f"Shape X_test: {X_test_processed.shape}")
    print("Distribuicao das classes em treino:")
    print(report["distribuicao_do_alvo_treino"])
    print("Distribuicao das classes em validacao:")
    print(report["distribuicao_do_alvo_validacao"])
    print("Distribuicao das classes em teste:")
    print(report["distribuicao_do_alvo_teste"])
    print(f"Numero final de features apos o pre-processamento: {len(feature_names)}")
    print(f"Numero de features numericas: {n_features_numericas}")
    print(f"Numero de features binarias: {n_features_binarias}")
    print(f"Numero de features one-hot: {n_features_onehot}")
    print(f"Numero de features frequency encoding: {n_features_frequency}")

    return report


def main() -> None:
    preprocess_and_save(INPUT_PATH, OUTPUT_DIR)


def run_preprocessing() -> None:
    main()


if __name__ == "__main__":
    main()
