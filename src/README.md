# Codigo e pipelines em `src`

Este diretorio concentra o codigo principal do projeto. Cada etapa fica separada:
pre-processamento, rede neural, funcao de fitness, algoritmo genetico, execucao
de experimentos e agregacao de resultados.

## Pipeline principal

```text
data/raw/*.xlsx
        |
        v
src/preprocessing.py
        |
        v
data/processed/*.npy + *.json + *.joblib
        |
        v
src/experiment.py
        |
   +----+---------------------------+
   |                                |
   v                                v
src/genetic_algorithm.py --> src/fitness.py --> src/neural_network.py
        |
        v
results/experiments/exp_XX/
        |
        v
src/aggregate.py --> results/summary/ + results/figures/
```

## Conceito central: cromossomo por atributo original

O cromossomo **nao** tem um gene por coluna final. Ele tem um gene por
**atributo original** candidato (L = 29): 3 numericos + 6 binarios + 15
categoricos + 5 frequency. Uma variavel categorica vira varias colunas one-hot
(ex.: `res_SIGLA_UF` -> 27 colunas), mas conta como **um** gene. Gene ligado ->
todas as colunas daquele atributo entram na rede; desligado -> nenhuma entra.
O mapeamento gene -> colunas vem de `feature_groups.json`.

## `preprocessing.py`

Prepara a base final para a rede neural e para o algoritmo genetico.

Funcoes principais:

- `load_data(path)`: carrega `.xlsx`, `.xls` ou `.csv`. Para Excel, usa a primeira aba com dados.
- `remove_invalid_rows(df)`: remove duplicatas, linhas sem `idade_obito_anos`/`label_cid` e mantem apenas `C53`, `C54`, `C55`.
- `create_date_features(df)`: interpreta `DTOBITO`/`DTNASC` como `DDMMAAAA` e cria ano/mes de obito e nascimento.
- `create_occupation_group(df)`: cria `OCUP_GRUPO` a partir de `OCUP`.
- `drop_unwanted_columns(df)`: remove colunas constantes, administrativas, muito nulas, redundantes e com vazamento de alvo/CID.
- `split_data(df)`: split estratificado `70%` treino, `15%` validacao, `15%` teste.
- `build_preprocessor()`: `ColumnTransformer` com imputacao + codificacao + normalizacao Min-Max.
- `get_feature_names` / `get_feature_groups`: nomes das colunas finais e o mapa atributo -> colunas.
- `preprocess_and_save(...)`: executa o fluxo completo e salva os artefatos.

Saidas em `data/processed/`: `X_train/val/test.npy`, `y_train/val/test.npy`,
`feature_names.json`, `feature_groups.json`, `class_mapping.json`,
`preprocessor.joblib`, `label_encoder.joblib`, `preprocessing_report.json`.

Cuidados: transformadores sao ajustados apenas no treino; validacao e teste usam
so `.transform()`; nenhuma coluna de causa/CID entra como feature.

## `utils.py`

Utilidades de I/O e o carregador dos dados processados.

- `ensure_dir`, `save_json`, `load_json`.
- `load_processed_data(dir)`: le os `.npy`/`.json` e retorna um dicionario com as
  matrizes, os rotulos, `feature_names`, `feature_groups`, `group_names`,
  `group_column_indices` (indices de coluna de cada atributo), `n_groups` (L),
  `n_classes` e `n_columns`.

## `neural_network.py`

Rede Neural avaliadora, implementada com `sklearn.neural_network.MLPClassifier`
(MLP treinada por backpropagation + Adam; softmax + log-loss em multiclasse).
Escolhida no lugar do TensorFlow/Keras por ser leve e rapida para os milhares de
treinos pequenos que o AG dispara.

- `build_model(random_state)`: arquitetura 32 ReLU -> 16 ReLU -> softmax, Adam
  `learning_rate_init=0.001`, `early_stopping=True` (menor erro de validacao).
- `train_and_evaluate(X_train, y_train, X_eval, y_eval, random_state)`: treina e
  retorna `(modelo, metricas)` com F1 macro, acuracia, precisao e recall.

## `fitness.py`

Conecta cromossomos (por atributo) a rede neural.

- `selected_columns(chromosome, group_column_indices)`: expande os genes ligados
  para os indices de coluna correspondentes em `X`.
- `selected_groups(chromosome, group_names)`: nomes dos atributos ligados.
- `combined_fitness(f1, ns, nt)`: `0.9 * F1 + 0.1 * (1 - Ns/Nt)`.
- `evaluate_chromosome(...)`: treina a rede nas colunas ativas, mede F1-macro na
  validacao e retorna `(fitness, metrics)`. Aceita um `cache` por cromossomo
  (o treino e deterministico dado o `random_state`, entao o cache e exato).

## `genetic_algorithm.py`

Steady-State GA. Componentes:

- `repair`: garante o piso de `config.MIN_FEATURES` atributos ligados;
- `create_individual` / `create_population`: populacao inicial (respeitando o piso);
- `uniform_crossover`: crossover uniforme (Pc), dois filhos;
- `mutate`: bit-flip com Pm = 1/L + reparo;
- `linear_scaling`: escalonamento linear de Goldberg da aptidao;
- `roulette_indices`: selecao proporcional sobre a aptidao escalonada;
- `run_genetic_algorithm(...)`: loop principal (elitismo 10, Gap 2, parada por
  200 geracoes ou 20 sem melhoria). 1 geracao = 1 renovacao da populacao
  (`pop/gap` passos steady-state). Recebe `group_column_indices` (define L) e
  retorna melhor cromossomo, melhor fitness, metricas, convergencia, nº de
  avaliacoes e geracoes executadas.

## `experiment.py`

Orquestra um experimento completo:

- carrega os dados processados (uma vez pode ser reusado via argumento `data`);
- subamostra o treino so para a fitness (`FITNESS_SUBSAMPLE`);
- roda o AG (treino subamostrado + validacao);
- reavalia o melhor cromossomo no conjunto de **teste** (treino cheio);
- salva `best_chromosome.json`, `metrics.json` (inclui tempo em segundos) e
  `convergence.csv` em `results/experiments/exp_XX/`.

## `aggregate.py`

Consolida os multiplos experimentos:

- `build_convergence_curve`: media (+/- dp) do melhor fitness por geracao;
- `build_feature_frequency`: frequencia de selecao de cada atributo;
- `build_summary`: media/dp de F1 de teste, nº de atributos e melhor cromossomo global;
- `plot_convergence` / `plot_feature_frequency`: figuras `.png`;
- `aggregate_results()`: le tudo, salva csv/json em `results/summary/` e as
  figuras em `results/figures/`.

## `config.py`

Centraliza caminhos e todos os hiperparametros: parametros do AG
(`POPULATION_SIZE`, `MIN_FEATURES`, `MAX_GENERATIONS`, `CROSSOVER_RATE`,
`ELITISM_SIZE`, `STEADY_STATE_GAP`, `NO_IMPROVEMENT_LIMIT`, `LINEAR_SCALING_C`),
pesos da fitness, arquitetura/treino da rede, `N_EXPERIMENTS` e `RANDOM_STATE`.
`Pm = 1/L` e calculado em runtime por `mutation_rate(n_features)`.
