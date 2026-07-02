"""Agregacao e visualizacao dos resultados dos multiplos experimentos.

Le os artefatos salvos em ``results/experiments/exp_XX/`` e produz os
entregaveis consolidados:

* **curva de convergencia** = media (+/- desvio padrao) do melhor fitness por
  geracao entre todos os experimentos (grafico exigido pela especificacao);
* **frequencia de selecao** de cada feature entre os melhores cromossomos;
* **resumo** com media/dp da F1 de teste, do nº de features e o melhor
  cromossomo global.

Saidas em ``results/summary/`` (csv/json) e ``results/figures/`` (png).
"""

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # backend sem interface grafica (salva em arquivo)
import matplotlib.pyplot as plt

from src import config
from src.utils import ensure_dir, load_json, save_json


def _experiment_dirs():
    """Lista as pastas de experimento que ja possuem ``metrics.json``, ordenadas."""
    if not config.EXPERIMENTS_DIR.exists():
        return []
    dirs = sorted(
        d for d in config.EXPERIMENTS_DIR.glob("exp_*") if (d / "metrics.json").exists()
    )
    return dirs


def build_convergence_curve(exp_dirs) -> pd.DataFrame:
    """Constroi a curva media de convergencia a partir dos ``convergence.csv``.

    Como cada experimento pode parar em uma geracao diferente (criterio de
    estagnacao), as series sao alinhadas ate a maior geracao observada e o
    ultimo valor de cada uma e repetido para frente (``ffill``). Isso e correto
    porque ``best_fitness`` e o melhor-ate-agora, monotonico nao-decrescente.

    Args:
        exp_dirs: lista de pastas de experimento.

    Returns:
        DataFrame com ``generation``, ``mean_best_fitness``,
        ``std_best_fitness`` e ``n_experiments``.
    """
    series = []
    for d in exp_dirs:
        conv = pd.read_csv(d / "convergence.csv")
        series.append(conv.set_index("generation")["best_fitness"])

    if not series:
        return pd.DataFrame(columns=["generation", "mean_best_fitness", "std_best_fitness", "n_experiments"])

    max_gen = max(int(s.index.max()) for s in series)
    full_index = range(0, max_gen + 1)
    aligned = [s.reindex(full_index).ffill() for s in series]
    matrix = np.vstack([s.to_numpy() for s in aligned])

    return pd.DataFrame(
        {
            "generation": list(full_index),
            "mean_best_fitness": matrix.mean(axis=0),
            "std_best_fitness": matrix.std(axis=0),
            "n_experiments": matrix.shape[0],
        }
    )


def build_feature_frequency(exp_dirs) -> pd.DataFrame:
    """Conta, entre os melhores cromossomos, quantas vezes cada feature foi escolhida.

    Args:
        exp_dirs: lista de pastas de experimento.

    Returns:
        DataFrame ordenado por ``times_selected`` (desc), com colunas
        ``feature``, ``times_selected`` e ``selection_rate`` (fracao dos
        experimentos que selecionaram a feature).
    """
    counter: dict[str, int] = {}
    n = 0
    for d in exp_dirs:
        best = load_json(d / "best_chromosome.json")
        n += 1
        for feature in best["selected_features"]:
            counter[feature] = counter.get(feature, 0) + 1

    rows = [
        {"feature": feature, "times_selected": count, "selection_rate": count / n}
        for feature, count in counter.items()
    ]
    df = pd.DataFrame(rows).sort_values("times_selected", ascending=False).reset_index(drop=True)
    return df


def build_summary(exp_dirs) -> dict:
    """Resume as metricas dos experimentos e identifica o melhor cromossomo global.

    Args:
        exp_dirs: lista de pastas de experimento.

    Returns:
        Dicionario com contagem de experimentos, estatisticas (media/dp) de
        fitness, F1 de validacao, F1 de teste e nº de features, alem do
        ``best_overall`` (experimento de maior fitness e seu cromossomo).
    """
    test_f1, val_f1, n_selected, fitness = [], [], [], []
    best_overall = None
    for d in exp_dirs:
        m = load_json(d / "metrics.json")
        test_f1.append(m["test"]["f1"])
        val_f1.append(m["val"].get("f1", float("nan")))
        n_selected.append(m["n_selected"])
        fitness.append(m["fitness"])
        if best_overall is None or m["fitness"] > best_overall["fitness"]:
            best_overall = {
                "experiment_dir": d.name,
                "fitness": m["fitness"],
                "seed": m["seed"],
                **load_json(d / "best_chromosome.json"),
            }

    def stats(values):
        arr = np.asarray(values, dtype=float)
        return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}

    return {
        "n_experiments": len(exp_dirs),
        "fitness": stats(fitness),
        "f1_val": stats(val_f1),
        "f1_test": stats(test_f1),
        "n_selected_features": stats(n_selected),
        "best_overall": best_overall,
    }


def plot_convergence(curve: pd.DataFrame, path) -> None:
    """Salva o grafico da curva media de convergencia (media +/- 1 dp)."""
    ensure_dir(path.parent)
    gen = curve["generation"]
    mean = curve["mean_best_fitness"]
    std = curve["std_best_fitness"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(gen, mean, color="#1f77b4", label="Media do melhor fitness")
    ax.fill_between(gen, mean - std, mean + std, color="#1f77b4", alpha=0.2, label="+/- 1 desvio padrao")
    ax.set_xlabel("Geracao")
    ax.set_ylabel("Melhor fitness")
    ax.set_title(f"Curva de convergencia (media de {int(curve['n_experiments'].iloc[0])} experimentos)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_frequency(freq: pd.DataFrame, path, top: int = 30) -> None:
    """Salva o grafico de barras com as ``top`` features mais selecionadas."""
    ensure_dir(path.parent)
    data = freq.head(top).iloc[::-1]  # inverte p/ a maior ficar no topo do barh

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(data))))
    ax.barh(data["feature"], data["selection_rate"], color="#2ca02c")
    ax.set_xlabel("Taxa de selecao")
    ax.set_title(f"Frequencia de selecao das features (top {min(top, len(freq))})")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aggregate_results() -> dict:
    """Le todos os experimentos, salva csv/json/figuras e retorna o resumo.

    Raises:
        FileNotFoundError: se nao houver experimentos com ``metrics.json``.

    Returns:
        O dicionario de resumo (ver ``build_summary``).
    """
    exp_dirs = _experiment_dirs()
    if not exp_dirs:
        raise FileNotFoundError(
            f"Nenhum experimento encontrado em {config.EXPERIMENTS_DIR}. Rode os experimentos primeiro."
        )

    ensure_dir(config.SUMMARY_DIR)
    ensure_dir(config.FIGURES_DIR)

    curve = build_convergence_curve(exp_dirs)
    freq = build_feature_frequency(exp_dirs)
    summary = build_summary(exp_dirs)

    curve.to_csv(config.SUMMARY_DIR / "convergence_curve.csv", index=False)
    freq.to_csv(config.SUMMARY_DIR / "feature_frequency.csv", index=False)
    save_json(summary, config.SUMMARY_DIR / "summary.json")

    plot_convergence(curve, config.FIGURES_DIR / "convergence.png")
    plot_feature_frequency(freq, config.FIGURES_DIR / "feature_frequency.png")

    print(f"Agregacao de {len(exp_dirs)} experimentos:")
    print(f"  F1 teste: {summary['f1_test']['mean']:.4f} +/- {summary['f1_test']['std']:.4f}")
    print(f"  Nº features: {summary['n_selected_features']['mean']:.1f} +/- {summary['n_selected_features']['std']:.1f}")
    print(f"  Figuras salvas em {config.FIGURES_DIR}")

    return summary
