import pandas as pd

from src import config
from src.fitness import selected_columns
from src.genetic_algorithm import run_genetic_algorithm
from src.utils import ensure_dir, load_processed_data, save_json


def run_experiment(experiment_number: int = 1):
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(config.PROCESSED_DIR)
    result = run_genetic_algorithm(X_train, y_train, X_val, y_val)

    exp_dir = config.EXPERIMENTS_DIR / f"exp_{experiment_number:02d}"
    ensure_dir(exp_dir)

    selected = selected_columns(result["best_chromosome"], X_train.columns)
    pd.DataFrame({"feature": selected}).to_csv(exp_dir / "selected_features.csv", index=False)
    result["convergence"].to_csv(exp_dir / "convergence.csv", index=False)

    save_json(
        {
            "chromosome": result["best_chromosome"],
            "selected_features_count": len(selected),
            "selected_features": selected,
        },
        exp_dir / "best_chromosome.json",
    )
    save_json(
        {
            "fitness": result["best_fitness"],
            **result["best_metrics"],
        },
        exp_dir / "metrics.json",
    )

    return result
