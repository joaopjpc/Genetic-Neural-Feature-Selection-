# Cancer AG RNA

Projeto para selecao de atributos em dados de cancer de utero usando algoritmo genetico e uma rede neural artificial como modelo avaliador.

O pipeline principal prepara a base bruta, transforma as variaveis em uma matriz numerica pronta para MLP/Softmax e permite que o Algoritmo Genetico selecione subconjuntos de features por cromossomos binarios.

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

## Experimentos

Depois de gerar `data/processed/`, rode um experimento:

```bash
python scripts/run_single_experiment.py
```

Para rodar os 20 experimentos:

```bash
python scripts/run_20_experiments.py
```

## Documentacao do codigo

Veja `src/README.md` para uma descricao focada nos modulos, funcoes e pipelines implementados em `src/`.
