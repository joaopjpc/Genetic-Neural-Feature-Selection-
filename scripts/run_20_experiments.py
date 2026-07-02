"""Ponto de entrada para rodar os N experimentos completos e agregar tudo.

Executa ``config.N_EXPERIMENTS`` (20) execucoes do AG com sementes distintas e,
ao final, gera a curva media de convergencia, a frequencia de selecao das
features e o resumo (ver :mod:`src.aggregate`).

Uso:
    python scripts/run_20_experiments.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.aggregate import aggregate_results
from src.experiment import run_experiment
from src.utils import load_processed_data


def main() -> None:
    """Roda os 20 experimentos (reusando os dados carregados) e agrega os resultados."""
    # Carrega os dados uma unica vez e reaproveita em todos os experimentos.
    data = load_processed_data(config.PROCESSED_DIR)
    print(
        f"Dados: {data['n_groups']} atributos (L), {data['n_columns']} colunas apos one-hot, "
        f"{data['n_classes']} classes, "
        f"treino={len(data['y_train'])}, val={len(data['y_val'])}, teste={len(data['y_test'])}"
    )

    start = time.time()
    for experiment_number in range(1, config.N_EXPERIMENTS + 1):
        run_experiment(experiment_number=experiment_number, data=data)

    aggregate_results()
    print(f"Concluido em {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
