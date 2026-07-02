"""Configuracao central do projeto.

Concentra em um so lugar:

* os caminhos de todos os diretorios e arquivos usados no pipeline;
* os hiperparametros do Algoritmo Genetico (Steady-State GA);
* os pesos da funcao de aptidao;
* os hiperparametros da Rede Neural avaliadora;
* o numero de experimentos.

Todos os valores seguem a especificacao do trabalho (CEFET-RJ - AG + RNA).
Alterar um parametro aqui reflete automaticamente em todos os modulos que o
importam (`from src import config`), evitando "numeros magicos" espalhados pelo
codigo.
"""

from pathlib import Path

# Raiz do repositorio (dois niveis acima deste arquivo: src/ -> raiz).
ROOT_DIR = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------
# Diretorios e arquivos
# ----------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"                 # base bruta (.xlsx)
INTERIM_DIR = DATA_DIR / "interim"         # etapas intermediarias (opcional)
PROCESSED_DIR = DATA_DIR / "processed"     # saidas do preprocessing (.npy/.json)

RESULTS_DIR = ROOT_DIR / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"  # results/experiments/exp_XX/
SUMMARY_DIR = RESULTS_DIR / "summary"          # agregacoes (csv/json)
FIGURES_DIR = RESULTS_DIR / "figures"          # graficos (.png)

REPORTS_DIR = ROOT_DIR / "reports"

RAW_FILE = RAW_DIR / "Base Slim Morte cancer de utero.xlsx"

# Semente base da reprodutibilidade. O experimento i (1-indexado) usa a semente
# RANDOM_STATE + (i - 1), de modo que as 20 execucoes sejam distintas porem
# reproduziveis.
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# Algoritmo genetico (Steady-State GA)
# ----------------------------------------------------------------------------
POPULATION_SIZE = 150         # nº de cromossomos na populacao
MIN_FEATURES = 3              # minimo de atributos ligados por cromossomo (reparo)
MAX_GENERATIONS = 200         # criterio de parada 1: geracoes maximas (enunciado)
CROSSOVER_RATE = 0.85         # Pc: probabilidade de aplicar o crossover
ELITISM_SIZE = 10             # os 10 melhores nunca sao descartados
STEADY_STATE_GAP = 2          # filhos gerados/inseridos por PASSO steady-state (gap)
NO_IMPROVEMENT_LIMIT = 20     # criterio de parada 2: geracoes sem melhoria (enunciado)
MIN_IMPROVEMENT = 1e-3        # ganho minimo de fitness para contar como "melhoria"
                              # (definicao operacional de "sem melhora"; sem isto,
                              # ruido de ponto flutuante impede a parada por estagnacao)

# Definicao de "geracao" no steady-state: uma geracao = uma renovacao completa
# da populacao = POPULATION_SIZE / STEADY_STATE_GAP passos (cada passo cria `gap`
# filhos). Ex.: 150/2 = 75 passos por geracao (150 filhos). Assim os numeros da
# spec (200 geracoes / 20 sem melhoria) ficam com escala real de busca, em vez de
# apenas 2 filhos por "geracao".

# Pm = 1 / L, onde L e o comprimento do cromossomo (= nº de features). Como L
# so e conhecido apos o preprocessing, o valor e calculado em runtime pela
# funcao mutation_rate(n_features) abaixo, e nao fixado como constante.

# Multiplicador C do escalonamento linear de Goldberg: o maior fitness
# escalonado fica igual a C * (media). Mantem a pressao seletiva controlada e
# evita o "super-individuo" que dominaria a selecao por roleta.
LINEAR_SCALING_C = 2.0

# ----------------------------------------------------------------------------
# Funcao de aptidao: Fitness = 0.9 * F1 + 0.1 * (1 - Ns/Nt)
# ----------------------------------------------------------------------------
FITNESS_F1_WEIGHT = 0.9       # peso do desempenho preditivo (F1)
FITNESS_SIZE_WEIGHT = 0.1     # peso da parcimonia (menos features)
F1_AVERAGE = "macro"          # media do F1 em problema multiclasse

# ----------------------------------------------------------------------------
# Rede neural avaliadora (sklearn MLPClassifier)
# Arquitetura: entrada (nº features selecionadas) -> 32 ReLU -> 16 ReLU -> softmax
# ----------------------------------------------------------------------------
NN_HIDDEN = (32, 16)          # neuronios das camadas ocultas
NN_LEARNING_RATE = 0.001      # taxa de aprendizado do Adam
NN_MAX_ITER = 200             # nº maximo de epocas de treino
NN_N_ITER_NO_CHANGE = 10      # paciencia do early stopping (menor erro de validacao)
NN_BATCH_SIZE = 512           # tamanho do mini-batch (lote maior = treino bem mais rapido)

# ----------------------------------------------------------------------------
# Aceleracao da fitness
# ----------------------------------------------------------------------------
# Nº de linhas de treino usadas para TREINAR a rede-avaliadora em cada fitness.
# Um subconjunto estratificado da o mesmo sinal de F1 para comparar cromossomos e
# corta muito o tempo (o AG dispara milhares de treinos). A avaliacao final do
# melhor cromossomo (e o teste) usa a base de treino cheia. None = base cheia.
FITNESS_SUBSAMPLE = 4000

# ----------------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------------
N_EXPERIMENTS = 20            # execucoes completas para a curva media de convergencia


def mutation_rate(n_features: int) -> float:
    """Retorna a probabilidade de mutacao por gene: ``Pm = 1 / L``.

    Args:
        n_features: comprimento do cromossomo (L = nº total de features).

    Returns:
        1 / n_features, ou 0.0 quando n_features <= 0 (guarda contra divisao
        por zero).
    """
    return 1.0 / n_features if n_features > 0 else 0.0
