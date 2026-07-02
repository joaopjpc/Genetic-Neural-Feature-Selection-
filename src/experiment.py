"""Orquestracao de um experimento completo de selecao de atributos.

Um experimento consiste em: carregar os dados processados, rodar o Algoritmo
Genetico (que usa a RNA como avaliadora), reavaliar o melhor cromossomo no
conjunto de **teste** (nunca visto pelo AG) e salvar todos os artefatos em
``results/experiments/exp_XX/``.

Rodar varios experimentos com sementes distintas permite depois agregar a curva
media de convergencia e a frequencia de selecao das features (ver
:mod:`src.aggregate`).
"""

import time

import numpy as np
from sklearn.model_selection import train_test_split

from src import config
from src.fitness import selected_columns, selected_groups
from src.genetic_algorithm import run_genetic_algorithm
from src.neural_network import train_and_evaluate
from src.utils import ensure_dir, load_processed_data, save_json


def _fitness_subsample(X_train, y_train, seed: int):
    """Retorna um subconjunto estratificado do treino para acelerar a fitness.

    Cada avaliacao do AG treina a rede so nesse subconjunto (mesmo sinal de F1
    para comparar cromossomos, muito mais rapido). Se ``config.FITNESS_SUBSAMPLE``
    for None ou >= ao tamanho do treino, usa a base cheia. O ``seed`` varia o
    subconjunto entre experimentos.
    """
    n = config.FITNESS_SUBSAMPLE
    if not n or n >= len(y_train):
        return X_train, y_train
    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train, train_size=n, random_state=seed, stratify=y_train
    )
    return X_sub, y_sub


def run_experiment(experiment_number: int = 1, seed: int | None = None, data: dict | None = None):
    """Executa um experimento do AG e avalia o melhor cromossomo no teste.

    Args:
        experiment_number: numero do experimento; define a pasta de saida
            (``exp_XX``) e, se ``seed`` for None, a semente
            (``RANDOM_STATE + experiment_number - 1``).
        seed: semente explicita (opcional). Sobrescreve a derivada do numero.
        data: dicionario de dados ja carregado por ``load_processed_data``
            (opcional). Passa-lo evita reler os ``.npy`` a cada experimento
            quando se roda os 20 em sequencia.

    Efeitos colaterais:
        Cria ``results/experiments/exp_XX/`` com:

        * ``convergence.csv`` - curva de convergencia da execucao;
        * ``best_chromosome.json`` - cromossomo, contagem e nomes das features;
        * ``metrics.json`` - fitness, metricas de validacao e de teste.

    Returns:
        Dicionario com o resultado do AG acrescido de ``test_metrics`` e
        ``selected_features``.
    """
    if seed is None:
        seed = config.RANDOM_STATE + (experiment_number - 1)

    if data is None:
        data = load_processed_data(config.PROCESSED_DIR)

    group_column_indices = data["group_column_indices"]

    start = time.perf_counter()

    # Subamostra o treino apenas para a fitness (acelera cada avaliacao do AG).
    X_fit, y_fit = _fitness_subsample(data["X_train"], data["y_train"], seed)

    # Busca do melhor subconjunto de atributos (fitness = treino subamostrado + validacao).
    result = run_genetic_algorithm(
        X_fit, y_fit, data["X_val"], data["y_val"],
        group_column_indices, seed=seed,
    )

    best = result["best_chromosome"]                        # cromossomo por atributo (len L)
    selected = selected_groups(best, data["group_names"])   # atributos originais escolhidos
    cols = selected_columns(best, group_column_indices)     # colunas expandidas em X

    # Avaliacao final no conjunto de TESTE (nunca usado pelo AG): retreina a RNA
    # com os atributos escolhidos e mede o desempenho de generalizacao.
    _, test_metrics = train_and_evaluate(
        data["X_train"][:, cols],
        data["y_train"],
        data["X_test"][:, cols],
        data["y_test"],
        random_state=seed,
    )

    elapsed = time.perf_counter() - start

    exp_dir = config.EXPERIMENTS_DIR / f"exp_{experiment_number:02d}"
    ensure_dir(exp_dir)

    result["convergence"].to_csv(exp_dir / "convergence.csv", index=False)

    save_json(
        {
            "seed": seed,
            "chromosome": best,
            "selected_features_count": len(selected),
            "total_features": len(best),
            "selected_columns_count": int(cols.size),
            "total_columns": data["n_columns"],
            "selected_features": selected,
        },
        exp_dir / "best_chromosome.json",
    )

    val_metrics = result["best_metrics"]
    save_json(
        {
            "seed": seed,
            "fitness": result["best_fitness"],
            "generations_run": result["generations_run"],
            "n_evaluations": result["n_evaluations"],
            "n_selected": len(selected),
            "n_columns_used": int(cols.size),
            "fitness_train_rows": int(len(y_fit)),
            "elapsed_seconds": elapsed,
            "val": {k: v for k, v in val_metrics.items() if k != "error"},
            "test": test_metrics,
        },
        exp_dir / "metrics.json",
    )

    print(
        f"[exp {experiment_number:02d}] seed={seed} "
        f"fitness={result['best_fitness']:.4f} "
        f"F1_val={val_metrics.get('f1', float('nan')):.4f} "
        f"F1_test={test_metrics['f1']:.4f} "
        f"atributos={len(selected)}/{len(best)} "
        f"(colunas={cols.size}/{data['n_columns']}) "
        f"geracoes={result['generations_run']} "
        f"tempo={elapsed:.1f}s ({elapsed / 60:.1f} min)"
    )

    return {**result, "test_metrics": test_metrics, "selected_features": selected}
