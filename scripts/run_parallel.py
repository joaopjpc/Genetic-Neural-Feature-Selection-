"""Roda os N experimentos em paralelo (1 por nucleo) e agrega os resultados.

Cada experimento e independente (semente e pasta proprias), entao a
paralelizacao e trivial. Cada worker carrega os dados processados por conta
(evita picklar arrays grandes) e usa 1 thread de BLAS, para que os processos
nao disputem os mesmos nucleos.
"""
import os

# Desliga o progresso por geracao ANTES de importar o AG: com varios experimentos
# em paralelo, a saida intercalada seria ilegivel. Cada [exp NN] final ainda e
# impresso por experiment.py.
os.environ["GA_VERBOSE"] = "0"

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from joblib import Parallel, delayed, parallel_config

from src import config
from src.aggregate import aggregate_results
from src.experiment import run_experiment


def _run(experiment_number: int):
    # data=None -> cada worker le os .npy de data/processed (rapido, cache do SO).
    return run_experiment(experiment_number=experiment_number)


def main() -> None:
    # Numa maquina dedicada, usa quase todos os nucleos. Como cada experimento e
    # single-thread, o limite util e o proprio nº de experimentos: com >= 20
    # nucleos, os 20 rodam de uma vez (tempo total ~ tempo de 1 experimento).
    n_jobs = max(1, min(config.N_EXPERIMENTS, (os.cpu_count() or 2) - 1))
    print(f"Rodando {config.N_EXPERIMENTS} experimentos com n_jobs={n_jobs} "
          f"(nucleos disponiveis: {os.cpu_count()})...", flush=True)
    start = time.time()
    with parallel_config(backend="loky", n_jobs=n_jobs, inner_max_num_threads=1):
        Parallel()(delayed(_run)(i) for i in range(1, config.N_EXPERIMENTS + 1))
    aggregate_results()
    print(f"Concluido em {time.time() - start:.1f}s (n_jobs={n_jobs})")


if __name__ == "__main__":
    main()
