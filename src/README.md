# Codigo e pipelines em `src`

Este diretorio concentra o codigo principal do projeto. A ideia e manter cada etapa separada: pre-processamento, rede neural, funcao de fitness, algoritmo genetico e execucao de experimentos.

## Pipeline principal

Fluxo esperado:

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
        v
src/genetic_algorithm.py -> src/fitness.py -> src/neural_network.py
        |
        v
results/experiments/
```

## `preprocessing.py`

Responsavel por preparar a base final para a rede neural e para o algoritmo genetico.

Funcoes principais:

- `load_data(path)`: carrega `.xlsx`, `.xls` ou `.csv`. Para Excel, usa a primeira aba com dados.
- `remove_invalid_rows(df)`: remove duplicatas, linhas sem `idade_obito_anos`, linhas sem `label_cid` e mantem apenas `C53`, `C54`, `C55`.
- `create_date_features(df)`: interpreta `DTOBITO` e `DTNASC` como `DDMMAAAA`, cria `ano_obito`, `mes_obito`, `ano_nascimento` e `mes_nascimento`, depois remove as datas brutas.
- `create_occupation_group(df)`: cria `OCUP_GRUPO` a partir de `OCUP`, depois remove `OCUP`.
- `drop_unwanted_columns(df)`: remove colunas constantes, administrativas, muito nulas, redundantes e colunas com vazamento de alvo/CID.
- `split_data(df)`: faz split estratificado em `70%` treino, `15%` validacao e `15%` teste.
- `build_preprocessor()`: cria o `ColumnTransformer` com os pipelines de features.
- `preprocess_and_save(input_path, output_dir)`: executa o fluxo completo e salva os artefatos finais.
- `main()`: ponto de entrada usado por `python preprocessing.py` e `python scripts/run_preprocessing.py`.

Features finais antes da codificacao:

- Numericas: `idade_obito_anos`, `ano_obito`, `ano_nascimento`
- Binarias `S/N`: `res_AMAZONIA`, `res_FRONTEIRA`, `res_CAPITAL`, `ocor_AMAZONIA`, `ocor_FRONTEIRA`, `ocor_CAPITAL`
- Categoricas: `mes_obito`, `mes_nascimento`, `RACACOR`, `ESTCIV`, `ESC`, `LOCOCOR`, `ASSISTMED`, `EXAME`, `CIRURGIA`, `NECROPSIA`, `res_SIGLA_UF`, `res_REGIAO`, `ocor_SIGLA_UF`, `ocor_REGIAO`, `OCUP_GRUPO`
- Frequency encoding: `NATURAL`, `CODMUNNATU`, `CODMUNRES`, `CODMUNOCOR`, `CODESTAB`

Transformacoes:

- numericas: `SimpleImputer(strategy="median")` + `MinMaxScaler(clip=True)`;
- binarias: `"S" -> 1`, `"N" -> 0`, desconhecidos -> `0`;
- categoricas: string + preenchimento com `"IGNORADO"` + `OneHotEncoder(handle_unknown="ignore")`;
- frequency encoding: frequencia relativa aprendida apenas no treino, desconhecidos -> `0`;
- alvo `label_cid`: `LabelEncoder`.

Saidas salvas em `data/processed/`:

- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- `feature_names.json`
- `feature_groups.json`
- `class_mapping.json`
- `preprocessor.joblib`
- `label_encoder.joblib`
- `preprocessing_report.json`

## `neural_network.py`

Define a MLP usada para avaliar subconjuntos de atributos.

Funcoes principais:

- `build_model(input_dim)`: cria uma rede neural sequencial.
- `train_and_evaluate(X_train, y_train, X_val, y_val, epochs)`: treina a rede e retorna metricas de validacao.

Observacao: este modulo nao faz selecao de atributos sozinho. Ele recebe uma matriz ja filtrada pelo cromossomo avaliado.

## `fitness.py`

Conecta cromossomos do algoritmo genetico com a rede neural.

Funcoes principais:

- `selected_columns(chromosome, feature_names)`: converte o cromossomo binario em lista de features selecionadas.
- `evaluate_chromosome(chromosome, X_train, y_train, X_val, y_val)`: treina/avalia a rede usando apenas as features selecionadas e retorna o fitness.

## `genetic_algorithm.py`

Implementa a busca por subconjuntos de features.

Componentes principais:

- criacao de populacao inicial;
- selecao por torneio;
- crossover;
- mutacao;
- elitismo;
- registro da convergencia por geracao.

A saida principal inclui:

- melhor cromossomo;
- melhor fitness;
- metricas do melhor individuo;
- historico de convergencia.

## `experiment.py`

Orquestra um experimento completo.

Funcoes principais:

- carrega os dados processados;
- executa o algoritmo genetico;
- salva cromossomo, features selecionadas, metricas e convergencia em `results/experiments/exp_XX/`.

## `config.py`

Centraliza caminhos e parametros gerais do projeto.

Use esse arquivo para alterar valores como:

- diretorios de dados e resultados;
- seed (`RANDOM_STATE`);
- parametros do algoritmo genetico;
- parametros basicos da rede neural.

## `utils.py`

Funcoes auxiliares usadas por outros modulos, como:

- criacao de diretorios;
- leitura dos dados processados;
- escrita de JSON.

## Cuidados importantes

- Nao usar colunas de causa/CID como entrada quando o alvo for `label_cid`.
- Ajustar transformadores apenas no treino.
- Usar validacao e teste apenas com `.transform()`.
- Manter `feature_names.json` alinhado com as colunas de `X`.
- Usar `class_mapping.json` para interpretar os inteiros do alvo.
