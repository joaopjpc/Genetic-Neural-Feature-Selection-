# Cancer AG RNA

Projeto para selecao de atributos em dados de cancer de utero usando algoritmo genetico e uma rede neural artificial como modelo avaliador.

O pipeline principal prepara a base bruta, transforma as variaveis em uma matriz numerica pronta para MLP/Softmax e permite que o Algoritmo Genetico selecione subconjuntos de features por cromossomos binarios.

Ponto importante: o cromossomo e indexado por **atributo original** (ex.: `RACACOR`, `res_SIGLA_UF`), nao pela coluna final apos o one-hot. Sao **29 atributos** (3 numericos + 6 binarios + 15 categoricos + 5 frequency), que se expandem para **181 colunas**. Cada gene liga/desliga um atributo inteiro: se ligado, todas as colunas one-hot daquele atributo entram na rede.

## Estrutura

- `data/raw/`: base original. Coloque aqui o arquivo `Base Slim Morte cancer de utero.xlsx`.
- `data/interim/`: dados limpos intermediarios, se forem necessarios em etapas futuras.
- `data/processed/`: matrizes finais de treino, validacao e teste geradas pelo pre-processamento.
- `src/`: codigo principal do projeto.
- `scripts/`: pontos de entrada para executar etapas do trabalho.
- `results/`: metricas, cromossomos, convergencia e figuras geradas pelos experimentos.
- `reports/`: relatorio e parametros do experimento.

Arquivos de dados e resultados gerados, como `.xlsx`, `.csv`, `.npy`, `.json`, `.joblib` e `.png`, nao devem ser versionados por padrao.

## Base e alvo

A base esperada fica em `data/raw/`. O script procura primeiro por:

```text
data/raw/Base Slim Morte cancer de utero.xlsx
```

Se esse nome nao existir, ele usa o primeiro arquivo `.xlsx` encontrado em `data/raw/`. Isso evita problemas com acentos no nome do arquivo.

O alvo do problema e:

```python
label_cid
```

Ele representa uma classificacao multiclasse entre `C53`, `C54` e `C55`. Como o alvo e derivado de informacoes CID/causa, o pre-processamento remove colunas de causa basica, linhas da declaracao de obito e outras colunas que poderiam vazar a resposta.

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Pre-processamento

O pre-processamento esta em `src/preprocessing.py` e tambem pode ser chamado pelo wrapper da raiz:

1. Coloque a base original em `data/raw/`.
2. Rode o pre-processamento:

```bash
python preprocessing.py
```

ou:

```bash
python scripts/run_preprocessing.py
```

Esse comando salva em `data/processed/`:

- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- `feature_names.json`
- `feature_groups.json`
- `class_mapping.json`
- `preprocessor.joblib`
- `label_encoder.joblib`
- `preprocessing_report.json`

Resumo do que o pipeline faz:

- remove duplicatas e linhas sem `idade_obito_anos` ou `label_cid`;
- cria `ano_obito` e `mes_obito` a partir de `DTOBITO`;
- cria `ano_nascimento` e `mes_nascimento` a partir de `DTNASC`;
- cria `OCUP_GRUPO` a partir de `OCUP`;
- remove colunas com vazamento de alvo e colunas administrativas descartadas;
- separa os dados em `70%` treino, `15%` validacao e `15%` teste com `stratify`;
- aplica `MinMaxScaler(clip=True)` nas features numericas;
- converte colunas `S/N` para `1/0`;
- aplica `OneHotEncoder(handle_unknown="ignore")` nas categoricas;
- aplica frequency encoding em colunas administrativas de alta cardinalidade, mantendo desconhecidos como `0`;
- codifica o alvo com `LabelEncoder`.

Os transformadores sao ajustados apenas no treino. Validacao e teste recebem apenas `.transform()`.

## Algoritmo genetico e rede neural

O AG (`src/genetic_algorithm.py`) e um **Steady-State GA**:

- cromossomo binario por atributo original (`L = 29`), com **minimo de 3 atributos** ligados (reparo);
- **crossover uniforme** (Pc = 0.85) e **mutacao** bit-flip (Pm = 1/L);
- **elitismo** dos 10 melhores e **Steady-State com Gap = 2** (2 filhos por geracao, substituindo os 2 piores);
- **selecao por roleta** sobre a aptidao apos **escalonamento linear** (Goldberg), evitando super-individuo;
- 1 geracao = 1 renovacao completa da populacao (`pop/gap` = 75 passos, 150 filhos), para dar escala real a busca;
- parada em 200 geracoes ou 20 geracoes sem melhoria.

Para viabilizar os milhares de treinos, cada rede-avaliadora e treinada num subconjunto estratificado do treino (`FITNESS_SUBSAMPLE`, ex.: 20000 linhas); a avaliacao final do melhor cromossomo e o teste usam a base cheia.

A aptidao de cada cromossomo (`src/fitness.py`) treina uma rede neural (`src/neural_network.py`, `sklearn` MLPClassifier: 32 ReLU -> 16 ReLU -> softmax, Adam lr 0.001) apenas nas colunas dos atributos ligados e calcula:

```text
Fitness = 0.9 * F1-macro(validacao) + 0.1 * (1 - Ns/Nt)
```

com `Ns` = atributos selecionados e `Nt` = 29. Os parametros ficam todos em `src/config.py`.

## Experimentos

Depois de gerar `data/processed/`, rode um experimento (imprime fitness, F1 de validacao/teste, nº de atributos/colunas, geracoes e tempo):

```bash
python scripts/run_single_experiment.py            # experimento 1
python scripts/run_single_experiment.py -n 3       # experimento 3
```

Cada experimento salva em `results/experiments/exp_XX/`: `best_chromosome.json`, `metrics.json` e `convergence.csv`.

Para rodar os 20 experimentos e agregar tudo:

```bash
python scripts/run_20_experiments.py
```

Isso gera, alem das pastas por experimento:

- `results/summary/summary.json` (media/dp de F1 de teste, nº de atributos, melhor cromossomo global);
- `results/summary/convergence_curve.csv` e `results/summary/feature_frequency.csv`;
- `results/figures/convergence.png` (curva media de convergencia dos 20 experimentos);
- `results/figures/feature_frequency.png` (frequencia de selecao de cada atributo).

## Documentacao do codigo

Veja `src/README.md` para uma descricao focada nos modulos, funcoes e pipelines implementados em `src/`.
