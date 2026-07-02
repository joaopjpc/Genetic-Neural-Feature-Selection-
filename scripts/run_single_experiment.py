"""Ponto de entrada para rodar UM experimento do AG.

Uso:
    python scripts/run_single_experiment.py            # experimento 1
    python scripts/run_single_experiment.py -n 3       # experimento 3 (semente derivada)
    python scripts/run_single_experiment.py -n 3 -s 7  # experimento 3 com semente 7

Insere a raiz do repositorio no ``sys.path`` para permitir ``from src ...`` ao
executar o script diretamente.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import run_experiment


def main() -> None:
    """Le os argumentos de linha de comando e dispara ``run_experiment``."""
    parser = argparse.ArgumentParser(description="Roda um experimento do AG de selecao de atributos.")
    parser.add_argument("-n", "--number", type=int, default=1, help="Numero do experimento (define a pasta e a semente).")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Semente explicita (opcional).")
    args = parser.parse_args()

    run_experiment(experiment_number=args.number, seed=args.seed)


if __name__ == "__main__":
    main()
