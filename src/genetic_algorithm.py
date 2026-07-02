"""Algoritmo Genetico (Steady-State GA) para selecao de atributos.

Cada individuo/cromossomo e um vetor binario de comprimento ``L`` = nº de
**atributos originais** (grupos), e nao o nº de colunas apos o one-hot. Gene 1
significa "usar o atributo" (todas as colunas one-hot dele entram na rede);
gene 0 significa "descartar" (nenhuma entra). O AG busca o subconjunto de
atributos que maximiza a aptidao definida em :mod:`src.fitness` (desempenho da
RNA + parcimonia). A expansao gene -> colunas usa ``group_column_indices``.

Resumo da configuracao (conforme a especificacao):

* populacao de 150 cromossomos;
* **crossover uniforme** com Pc = 0.85;
* **mutacao** bit-flip com Pm = 1 / L;
* **elitismo**: os 10 melhores nunca sao descartados;
* abordagem **Steady-State** com Gap = 2 (a cada geracao geram-se 2 filhos que
  substituem os 2 piores individuos);
* **selecao por roleta** sobre a aptidao apos **escalonamento linear** de
  Goldberg (evita o dominio de um super-individuo);
* parada em 200 geracoes OU 20 geracoes consecutivas sem melhoria do melhor.

Convencao de "geracao": **1 geracao = 1 renovacao completa da populacao** =
``POPULATION_SIZE / gap`` passos steady-state (cada passo gera ``gap`` filhos).
Ex.: 150/2 = 75 passos = 150 filhos por geracao. Isso da escala real a busca -
"200 geracoes" e "20 sem melhoria" passam a significar milhares de filhos, e nao
apenas 2 por "geracao". A convergencia e registrada uma vez por geracao.
"""

import numpy as np
import pandas as pd

from src import config
from src.fitness import evaluate_chromosome


# ---------------------------------------------------------------------------
# Operadores geneticos
# ---------------------------------------------------------------------------
def repair(individual, rng: np.random.Generator, min_features: int = config.MIN_FEATURES) -> np.ndarray:
    """Garante que o cromossomo tenha ao menos ``min_features`` atributos ligados.

    Se houver genes ativos de menos, liga genes atualmente em 0 escolhidos ao
    acaso ate atingir o piso. Aplicado apos criacao/mutacao para que nenhum
    individuo caia abaixo do minimo (indivduos com pouquissimos atributos sao
    excluidos por construcao). ``min_features`` e limitado ao tamanho do cromossomo.

    Args:
        individual: cromossomo (modificado in-place e retornado).
        rng: gerador aleatorio.
        min_features: piso de atributos ligados.
    """
    min_features = min(min_features, len(individual))
    deficit = min_features - int(individual.sum())
    if deficit > 0:
        inactive = np.flatnonzero(individual == 0)
        chosen = rng.choice(inactive, size=deficit, replace=False)
        individual[chosen] = 1
    return individual


def create_individual(
    n_features: int, rng: np.random.Generator, min_features: int = config.MIN_FEATURES
) -> np.ndarray:
    """Cria um cromossomo binario aleatorio respeitando o piso de atributos.

    Evita individuos "vazios" ou pequenos demais (que teriam aptidao ruim ou nem
    treinariam a rede) via ``repair``.

    Args:
        n_features: comprimento do cromossomo (L).
        rng: gerador de numeros aleatorios (reprodutibilidade por experimento).
        min_features: minimo de atributos ligados.
    """
    individual = rng.integers(0, 2, size=n_features)
    return repair(individual, rng, min_features)


def create_population(
    population_size: int, n_features: int, rng: np.random.Generator,
    min_features: int = config.MIN_FEATURES,
):
    """Gera a populacao inicial como uma lista de cromossomos aleatorios."""
    return [create_individual(n_features, rng, min_features) for _ in range(population_size)]


def uniform_crossover(parent_a, parent_b, rng: np.random.Generator, pc: float):
    """Crossover uniforme entre dois pais, produzindo dois filhos.

    Com probabilidade ``pc`` aplica o cruzamento: para cada gene, sorteia-se de
    qual pai o filho A herda (o filho B recebe o gene do outro pai). Se o
    crossover nao ocorrer, os filhos sao copias dos pais.

    Args:
        parent_a, parent_b: cromossomos dos pais (arrays de 0/1).
        rng: gerador aleatorio.
        pc: probabilidade de crossover (Pc).

    Returns:
        Tupla ``(child_a, child_b)`` (novos arrays; os pais nao sao mutados).
    """
    if rng.random() >= pc:
        return parent_a.copy(), parent_b.copy()
    # mask[i] = True -> filho A herda o gene i do pai A; False -> do pai B.
    mask = rng.random(len(parent_a)) < 0.5
    child_a = np.where(mask, parent_a, parent_b)
    child_b = np.where(mask, parent_b, parent_a)
    return child_a, child_b


def mutate(
    individual, rng: np.random.Generator, pm: float,
    min_features: int = config.MIN_FEATURES,
) -> np.ndarray:
    """Mutacao bit-flip: cada gene inverte com probabilidade ``pm`` (= 1/L).

    Apos a inversao, aplica ``repair`` para manter o piso de ``min_features``
    atributos ligados.

    Args:
        individual: cromossomo de entrada (nao e modificado in-place).
        rng: gerador aleatorio.
        pm: probabilidade de mutacao por gene.
        min_features: minimo de atributos ligados.

    Returns:
        Novo cromossomo mutado.
    """
    flips = rng.random(len(individual)) < pm
    mutated = np.where(flips, 1 - individual, individual)
    return repair(mutated, rng, min_features)


def linear_scaling(fitness_values, c: float = config.LINEAR_SCALING_C) -> np.ndarray:
    """Escalonamento linear de Goldberg da aptidao (``f' = a*f + b``).

    Objetivo: preparar as aptidoes para a selecao por roleta preservando a media
    (``media(f') = media(f)``) e fixando o maior valor escalonado em
    ``c * media``. Isso controla a pressao seletiva:

    * no inicio, impede que um "super-individuo" abocanhe quase toda a roleta;
    * no fim, quando as aptidoes ficam parecidas, amplia diferencas pequenas.

    Trata dois casos padrao do metodo:

    1. **normal**: o mapeamento acima nao gera valores negativos;
    2. **corrigido**: se geraria negativos, ajusta-se para mapear o menor
       fitness em 0 (mantendo a media).

    Populacao homogenea (todos iguais) recebe pesos iguais (vetor de 1s). Por
    seguranca numerica, valores negativos residuais sao zerados.

    Args:
        fitness_values: aptidoes brutas da populacao.
        c: multiplicador C (maximo escalonado = c * media).

    Returns:
        Array de aptidoes escalonadas, todas >= 0.
    """
    f = np.asarray(fitness_values, dtype=float)
    f_avg = f.mean()
    f_max = f.max()
    f_min = f.min()

    if f_max == f_min:
        return np.ones_like(f)

    if f_min > (c * f_avg - f_max) / (c - 1.0):
        # Caso normal: maximo -> c*media, media preservada.
        denom = f_max - f_avg
        a = (c - 1.0) * f_avg / denom
        b = f_avg * (f_max - c * f_avg) / denom
    else:
        # Caso corrigido: minimo -> 0, media preservada.
        denom = f_avg - f_min
        a = f_avg / denom
        b = -f_min * f_avg / denom

    scaled = a * f + b
    return np.clip(scaled, 0.0, None)


def roulette_indices(scaled_fitness, rng: np.random.Generator, k: int = 2):
    """Selecao por roleta: sorteia ``k`` indices proporcionalmente a aptidao escalonada.

    Se a soma escalonada for 0 (degenerado), cai para sorteio uniforme. A
    amostragem e com reposicao (um mesmo individuo pode ser escolhido para os
    dois papeis de pai).

    Args:
        scaled_fitness: aptidoes ja escalonadas (ver ``linear_scaling``).
        rng: gerador aleatorio.
        k: quantidade de indices a sortear.

    Returns:
        Array com ``k`` indices da populacao.
    """
    total = scaled_fitness.sum()
    if total <= 0:
        probabilities = None  # uniforme
    else:
        probabilities = scaled_fitness / total
    return rng.choice(len(scaled_fitness), size=k, replace=True, p=probabilities)


# ---------------------------------------------------------------------------
# Loop principal (Steady-State GA)
# ---------------------------------------------------------------------------
def run_genetic_algorithm(
    X_train,
    y_train,
    X_val,
    y_val,
    group_column_indices,
    seed: int = config.RANDOM_STATE,
    population_size: int = config.POPULATION_SIZE,
    max_generations: int = config.MAX_GENERATIONS,
    crossover_rate: float = config.CROSSOVER_RATE,
    elitism_size: int = config.ELITISM_SIZE,
    gap: int = config.STEADY_STATE_GAP,
    no_improvement_limit: int = config.NO_IMPROVEMENT_LIMIT,
    min_features: int = config.MIN_FEATURES,
    steps_per_generation: int | None = None,
):
    """Executa o Steady-State GA de selecao de atributos.

    Fluxo:

    1. cria e avalia a populacao inicial (registrada como geracao 0);
    2. cada geracao = ``steps_per_generation`` passos steady-state (default
       ``population_size / gap``, i.e. uma renovacao completa). Cada passo:
       escalona as aptidoes, seleciona pais por roleta, aplica crossover uniforme
       e mutacao para gerar ``gap`` filhos, avalia-os e faz a **substituicao
       steady-state** (mantem os melhores ``population_size`` da uniao populacao +
       filhos, descartando os ``gap`` piores). Essa truncagem preserva
       integralmente os melhores individuos (elitismo dos 10);
    3. ao fim de cada geracao, registra a curva de convergencia (melhor-ate-agora
       e media);
    4. para em ``max_generations`` OU apos ``no_improvement_limit`` geracoes sem
       melhoria do melhor.

    Os parametros com valores em ``config`` podem ser sobrescritos (util para um
    smoke test com populacao/geracoes reduzidas).

    Args:
        X_train, y_train, X_val, y_val: dados de treino e validacao (numpy).
        group_column_indices: mapeamento atributo (gene) -> indices de coluna em
            X. Seu comprimento define L (o tamanho do cromossomo).
        seed: semente do experimento; controla o RNG do AG e o ``random_state``
            da RNA (treino deterministico -> cache exato).
        population_size, max_generations, crossover_rate, elitism_size, gap,
        no_improvement_limit: hiperparametros do AG (default = ``config``).

    Returns:
        Dicionario com:

        * ``best_chromosome``: melhor cromossomo (lista de 0/1, tamanho L);
        * ``best_fitness``: aptidao do melhor;
        * ``best_metrics``: metricas de validacao do melhor;
        * ``convergence``: DataFrame com colunas
          ``generation``/``best_fitness``/``mean_fitness``;
        * ``n_evaluations``: nº de cromossomos distintos avaliados (tamanho do cache);
        * ``generations_run``: ultima geracao executada.
    """
    # L = nº de atributos originais (genes), NAO o nº de colunas de X.
    n_features = len(group_column_indices)
    pm = config.mutation_rate(n_features)          # Pm = 1 / L
    rng = np.random.default_rng(seed)
    cache: dict = {}                                # evita reavaliar cromossomos repetidos

    def evaluate(chromosome):
        # random_state fixo por experimento -> treino da RNA deterministico,
        # o que torna o cache exato.
        return evaluate_chromosome(
            chromosome, X_train, y_train, X_val, y_val, group_column_indices,
            random_state=seed, cache=cache,
        )

    # --- Geracao 0: populacao inicial avaliada ---
    population = create_population(population_size, n_features, rng, min_features)
    evaluated = [evaluate(ind) for ind in population]
    fitness = np.array([item[0] for item in evaluated])

    best_idx = int(np.argmax(fitness))
    best_fitness = float(fitness[best_idx])
    best_chromosome = population[best_idx].copy()
    best_metrics = evaluated[best_idx][1]

    convergence = [
        {"generation": 0, "best_fitness": best_fitness, "mean_fitness": float(fitness.mean())}
    ]

    generations_without_improvement = 0
    if steps_per_generation is None:
        steps_per_generation = max(1, population_size // gap)

    def steady_state_step(population, evaluated, fitness):
        """Um passo steady-state: gera `gap` filhos e substitui os `gap` piores."""
        scaled = linear_scaling(fitness)

        children = []
        while len(children) < gap:
            i, j = roulette_indices(scaled, rng, k=2)
            child_a, child_b = uniform_crossover(
                population[i], population[j], rng, crossover_rate
            )
            children.append(mutate(child_a, rng, pm, min_features))
            if len(children) < gap:
                children.append(mutate(child_b, rng, pm, min_features))

        child_eval = [evaluate(child) for child in children]

        # Mantem os melhores `population_size` da uniao (populacao + filhos):
        # preserva os melhores individuos (elitismo dos 10) e descarta os piores.
        pool = population + children
        pool_fitness = np.concatenate([fitness, [e[0] for e in child_eval]])
        pool_eval = evaluated + child_eval

        keep = np.argsort(pool_fitness)[::-1][:population_size]
        return (
            [pool[k].copy() for k in keep],
            [pool_eval[k] for k in keep],
            pool_fitness[keep],
        )

    # --- Geracoes 1..max_generations (cada uma = uma renovacao da populacao) ---
    for generation in range(1, max_generations + 1):
        for _ in range(steps_per_generation):
            population, evaluated, fitness = steady_state_step(population, evaluated, fitness)

        gen_best_idx = int(np.argmax(fitness))
        gen_best_fitness = float(fitness[gen_best_idx])

        # Atualiza o melhor global e o contador de estagnacao.
        if gen_best_fitness > best_fitness + 1e-9:
            best_fitness = gen_best_fitness
            best_chromosome = population[gen_best_idx].copy()
            best_metrics = evaluated[gen_best_idx][1]
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        convergence.append(
            {
                "generation": generation,
                "best_fitness": best_fitness,
                "mean_fitness": float(fitness.mean()),
            }
        )

        # Criterio de parada 2: estagnacao do melhor.
        if generations_without_improvement >= no_improvement_limit:
            break

    return {
        "best_chromosome": best_chromosome.tolist(),
        "best_fitness": best_fitness,
        "best_metrics": best_metrics,
        "convergence": pd.DataFrame(convergence),
        "n_evaluations": len(cache),
        "generations_run": convergence[-1]["generation"],
    }
