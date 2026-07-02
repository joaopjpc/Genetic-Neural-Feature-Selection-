"""Ponto de entrada para gerar os dados processados a partir da base bruta.

Executa o pipeline de :mod:`src.preprocessing` (limpeza, codificacao,
normalizacao Min-Max e split 70/15/15) e salva os artefatos ``.npy``/``.json``
em ``data/processed/``, que sao a entrada do Algoritmo Genetico.

Uso:
    python scripts/run_preprocessing.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import run_preprocessing


if __name__ == "__main__":
    run_preprocessing()
