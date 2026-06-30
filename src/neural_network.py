import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from src import config


def build_model(input_dim: int) -> Sequential:
    model = Sequential(
        [
            Dense(32, activation="relu", input_shape=(input_dim,)),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(X_train, y_train, X_val, y_val, epochs=config.NN_EPOCHS):
    model = build_model(X_train.shape[1])
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=config.NN_BATCH_SIZE,
        verbose=0,
    )

    probabilities = model.predict(X_val, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "best_val_loss": float(np.min(history.history["val_loss"])),
    }
    return model, metrics
