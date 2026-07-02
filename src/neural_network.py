"""Rede Neural Artificial (RNA) usada como funcao avaliadora do AG.

A cada cromossomo avaliado, uma MLP e treinada usando *exclusivamente* as
features selecionadas e o seu desempenho (F1-Score) alimenta a aptidao.

Escolha de implementacao: ``sklearn.neural_network.MLPClassifier``. Ele e uma
MLP treinada por backpropagation com o otimizador Adam e, em problemas
multiclasse, usa softmax na saida com perda de entropia cruzada (log-loss) -
exatamente a arquitetura pedida na especificacao. Foi preferido ao
TensorFlow/Keras por ser leve e rapido: o AG treina milhares de redes minusculas
(150 individuos + 200 geracoes x 20 experimentos), cenario em que o overhead por
chamada do Keras seria proibitivo.

Arquitetura resultante: entrada (nº de features ativas) -> 32 ReLU -> 16 ReLU ->
softmax (nº de classes). Entrada e saida sao inferidas de ``X`` e ``y`` no fit().
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

from src import config


def build_model(random_state: int | None = None) -> MLPClassifier:
    """Constroi (sem treinar) a MLP avaliadora conforme os parametros da spec.

    Camadas ocultas ``(32, 16)`` com ReLU; solver Adam com
    ``learning_rate_init=0.001``. ``early_stopping=True`` separa internamente um
    pequeno conjunto de validacao e mantem a configuracao de menor erro,
    atendendo ao criterio "melhor configuracao = menor erro de validacao".

    Args:
        random_state: semente do modelo. Fixa-la torna o treino deterministico,
            o que valida o cache de aptidao no algoritmo genetico.

    Returns:
        Um ``MLPClassifier`` ainda nao treinado.
    """
    return MLPClassifier(
        hidden_layer_sizes=config.NN_HIDDEN,
        activation="relu",
        solver="adam",
        learning_rate_init=config.NN_LEARNING_RATE,
        batch_size=config.NN_BATCH_SIZE,
        max_iter=config.NN_MAX_ITER,
        early_stopping=True,
        n_iter_no_change=config.NN_N_ITER_NO_CHANGE,
        random_state=random_state,
    )


def _metrics(y_true, y_pred) -> dict:
    """Calcula F1 (media configurada), acuracia, precisao e recall.

    ``zero_division=0`` evita warnings/NaN quando uma classe nao e predita.
    """
    average = config.F1_AVERAGE
    return {
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }


def train_and_evaluate(X_train, y_train, X_eval, y_eval, random_state: int | None = None):
    """Treina a rede em ``(X_train, y_train)`` e a avalia em ``(X_eval, y_eval)``.

    O conjunto de avaliacao muda conforme o momento: e a **validacao** durante o
    AG (a F1 vira aptidao) e o **teste** na etapa final do experimento.

    Args:
        X_train, y_train: dados de treino (ja restritos as features ativas).
        X_eval, y_eval: dados de avaliacao (mesmas colunas de X_train).
        random_state: semente do modelo (ver ``build_model``).

    Returns:
        Tupla ``(modelo_treinado, metricas)``, em que ``metricas`` e o dict de
        ``_metrics`` (F1 macro, acuracia, precisao, recall) sobre ``X_eval``.
    """
    model = build_model(random_state=random_state)
    model.fit(X_train, np.asarray(y_train).ravel())

    y_pred = model.predict(X_eval)
    return model, _metrics(np.asarray(y_eval).ravel(), y_pred)
