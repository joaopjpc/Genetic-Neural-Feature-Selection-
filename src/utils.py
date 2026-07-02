"""Funcoes auxiliares compartilhadas pelos demais modulos.

Reune utilidades genericas de entrada/saida:

* criacao de diretorios (`ensure_dir`);
* leitura/escrita de JSON (`load_json`, `save_json`);
* carregamento dos artefatos gerados pelo preprocessing (`load_processed_data`).

Manter essas funcoes isoladas evita duplicacao de codigo entre o algoritmo
genetico, os experimentos e a agregacao de resultados.
"""

import json
from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> None:
    """Cria o diretorio (e os pais) se ainda nao existir. Idempotente."""
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    """Salva um dicionario como JSON indentado (UTF-8), criando a pasta destino.

    Usa ``ensure_ascii=False`` para preservar acentos nos nomes das features.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json(path: Path):
    """Le e retorna o conteudo de um arquivo JSON."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_processed_data(processed_dir: Path) -> dict:
    """Carrega os artefatos gerados por :mod:`src.preprocessing`.

    Espera encontrar em ``processed_dir`` os arquivos produzidos pelo
    preprocessing: as matrizes ``X_*.npy`` (ja normalizadas por Min-Max), os
    rotulos ``y_*.npy`` (inteiros), ``feature_names.json`` (nomes das colunas
    finais), ``feature_groups.json`` (atributo original -> colunas geradas) e
    ``class_mapping.json`` (rotulo -> inteiro).

    Ponto-chave: o **cromossomo do AG e indexado por ATRIBUTO ORIGINAL**, nao
    pela coluna final. Uma variavel categorica vira varias colunas one-hot, mas
    conta como um unico gene: se o gene esta ligado, todas as suas colunas
    entram; se desligado, nenhuma entra. Por isso expomos aqui o mapeamento
    grupo -> indices de coluna em ``X``.

    Args:
        processed_dir: pasta ``data/processed`` com os artefatos.

    Returns:
        Dicionario com:

        * ``X_train``/``X_val``/``X_test``: matrizes numpy de colunas finais;
        * ``y_train``/``y_val``/``y_test``: vetores numpy de rotulos;
        * ``feature_names``: nome de cada coluna final (len == nº de colunas);
        * ``feature_groups``: {atributo_original: [colunas geradas]};
        * ``group_names``: lista ordenada dos atributos originais (len == L);
        * ``group_column_indices``: lista (len == L) de arrays com os indices de
          coluna em ``X`` de cada grupo;
        * ``n_groups``: L, o comprimento do cromossomo (nº de atributos originais);
        * ``class_mapping``: mapa {rotulo_original: inteiro};
        * ``n_classes``: nº de classes do alvo (neuronios da saida softmax);
        * ``n_columns``: nº de colunas finais de ``X`` (apos one-hot).
    """
    processed_dir = Path(processed_dir)

    X_train = np.load(processed_dir / "X_train.npy")
    X_val = np.load(processed_dir / "X_val.npy")
    X_test = np.load(processed_dir / "X_test.npy")
    y_train = np.load(processed_dir / "y_train.npy")
    y_val = np.load(processed_dir / "y_val.npy")
    y_test = np.load(processed_dir / "y_test.npy")

    feature_names = load_json(processed_dir / "feature_names.json")["feature_names"]
    feature_groups = load_json(processed_dir / "feature_groups.json")
    class_mapping = load_json(processed_dir / "class_mapping.json")
    n_classes = len(class_mapping)

    # Mapeia cada atributo original (grupo) -> indices das suas colunas em X.
    name_to_index = {name: idx for idx, name in enumerate(feature_names)}
    group_names = list(feature_groups.keys())
    group_column_indices = [
        np.array([name_to_index[col] for col in feature_groups[group]], dtype=int)
        for group in group_names
    ]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "feature_names": feature_names,
        "feature_groups": feature_groups,
        "group_names": group_names,
        "group_column_indices": group_column_indices,
        "n_groups": len(group_names),
        "class_mapping": class_mapping,
        "n_classes": n_classes,
        "n_columns": X_train.shape[1],
    }
