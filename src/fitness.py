"""Funcao de aptidao (fitness) que conecta o AG a Rede Neural.

O cromossomo e indexado por **atributo original** (grupo), nao por coluna final.
Cada gene liga/desliga um atributo inteiro: se ligado, TODAS as colunas one-hot
geradas por ele entram na rede; se desligado, nenhuma entra. Assim ``Ns/Nt`` mede
atributos de verdade (Nt = nº de atributos originais, ex.: 29), e nao dummies
isoladas.

A aptidao combina desempenho preditivo e parcimonia:

    Fitness = 0.9 * F1-macro(validacao) + 0.1 * (1 - Ns/Nt)

onde ``Ns`` = nº de atributos selecionados e ``Nt`` = nº total de atributos.
"""

import numpy as np

from src import config
from src.neural_network import train_and_evaluate


def selected_columns(chromosome, group_column_indices) -> np.ndarray:
    """Expande o cromossomo (por grupo) para os indices de coluna ativos em X.

    Faz a uniao dos indices de coluna de todos os grupos cujo gene esta ligado.

    Args:
        chromosome: vetor binario de tamanho L (um gene por atributo original).
        group_column_indices: lista (len L) com os indices de coluna de cada grupo.

    Returns:
        Array 1D (possivelmente vazio) com os indices das colunas selecionadas.
    """
    selected = [
        group_column_indices[gene_idx]
        for gene_idx, active in enumerate(chromosome)
        if active
    ]
    if not selected:
        return np.empty(0, dtype=int)
    return np.concatenate(selected)


def selected_groups(chromosome, group_names) -> list[str]:
    """Retorna os nomes dos atributos originais cujos genes estao ligados (== 1)."""
    return [name for name, active in zip(group_names, chromosome) if active]


def combined_fitness(f1: float, n_selected: int, n_total: int) -> float:
    """Aplica a formula da aptidao: ``0.9 * F1 + 0.1 * (1 - Ns/Nt)``.

    Args:
        f1: F1-Score (macro) obtido na validacao.
        n_selected: Ns, nº de atributos (grupos) selecionados.
        n_total: Nt, nº total de atributos (grupos).

    Returns:
        Valor de aptidao. Quando todos os atributos sao usados (Ns == Nt), o
        termo de parcimonia zera e a aptidao vira ``0.9 * F1``.
    """
    size_term = 1.0 - (n_selected / n_total) if n_total > 0 else 0.0
    return config.FITNESS_F1_WEIGHT * f1 + config.FITNESS_SIZE_WEIGHT * size_term


def evaluate_chromosome(
    chromosome,
    X_train,
    y_train,
    X_val,
    y_val,
    group_column_indices,
    random_state: int | None = None,
    cache: dict | None = None,
):
    """Avalia um cromossomo: treina a RNA nos atributos ativos e calcula a aptidao.

    Passos: (1) conta atributos (genes) ativos; (2) se nenhum, aptidao 0;
    (3) expande os genes ligados para as colunas correspondentes de X;
    (4) treina a RNA apenas nessas colunas; (5) mede F1-macro na validacao;
    (6) combina com o termo de parcimonia (em nº de atributos).

    Args:
        chromosome: vetor binario por atributo original (tamanho L).
        X_train, y_train, X_val, y_val: dados (matrizes numpy completas; o
            fatiamento pelas colunas ativas e feito internamente).
        group_column_indices: mapeamento grupo -> indices de coluna em X.
        random_state: semente da RNA (torna o treino deterministico).
        cache: dicionario opcional {tupla_do_cromossomo: (fitness, metrics)}.
            Como o treino e deterministico dado ``random_state``, reaproveitar
            avaliacoes de cromossomos repetidos e exato e acelera muito o AG.

    Returns:
        Tupla ``(fitness, metrics)``. ``metrics`` traz F1/acuracia/precisao/
        recall da validacao mais ``n_selected`` (atributos), ``n_columns`` (colunas
        usadas) e o proprio ``fitness``.
    """
    key = tuple(int(g) for g in chromosome)
    if cache is not None and key in cache:
        return cache[key]

    n_total = len(chromosome)              # Nt = nº de atributos originais
    n_selected = int(sum(key))             # Ns = nº de atributos ligados

    # Cromossomo sem atributos nao pode treinar rede -> aptidao minima.
    if n_selected == 0:
        result = (0.0, {"error": "no_features_selected", "n_selected": 0, "n_columns": 0})
        if cache is not None:
            cache[key] = result
        return result

    cols = selected_columns(chromosome, group_column_indices)
    _, metrics = train_and_evaluate(
        X_train[:, cols],
        y_train,
        X_val[:, cols],
        y_val,
        random_state=random_state,
    )

    fitness = combined_fitness(metrics["f1"], n_selected, n_total)
    metrics = {**metrics, "n_selected": n_selected, "n_columns": int(cols.size), "fitness": fitness}
    result = (fitness, metrics)

    if cache is not None:
        cache[key] = result
    return result
