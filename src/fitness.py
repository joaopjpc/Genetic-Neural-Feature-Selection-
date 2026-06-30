import numpy as np

from src.neural_network import train_and_evaluate


def selected_columns(chromosome, feature_names):
    mask = np.array(chromosome, dtype=bool)
    return list(np.array(feature_names)[mask])


def evaluate_chromosome(chromosome, X_train, y_train, X_val, y_val):
    features = selected_columns(chromosome, X_train.columns)
    if not features:
        return 0.0, {"error": "no_features_selected"}

    _, metrics = train_and_evaluate(
        X_train[features],
        y_train,
        X_val[features],
        y_val,
    )
    fitness = metrics["f1"]
    return fitness, metrics
