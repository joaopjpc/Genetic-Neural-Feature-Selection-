# Cancer AG RNA

Projeto para selecao de atributos em dados de cancer de utero usando algoritmo genetico e uma rede neural artificial como modelo avaliador.

## Estrutura

- `data/raw/`: base original. Coloque aqui o arquivo `Base Slim Morte cancer de utero.xlsx`.
- `data/interim/`: dados limpos intermediarios gerados pelo pre-processamento.
- `data/processed/`: divisao final em treino, validacao e teste.
- `src/`: codigo principal do projeto.
- `scripts/`: pontos de entrada para executar etapas do trabalho.
- `results/`: metricas, cromossomos, convergencia e figuras geradas pelos experimentos.
- `reports/`: relatorio e parametros do experimento.

Arquivos de dados e resultados gerados, como `.xlsx`, `.csv`, `.json` e `.png`, nao sao versionados por padrao.

## Instalar dependencias

```bash
pip install -r requirements.txt
```

## Executar

1. Coloque a base original em `data/raw/`.
2. Rode o pre-processamento:

```bash
python scripts/run_preprocessing.py
```

3. Rode um experimento:

```bash
python scripts/run_single_experiment.py
```

4. Rode os 20 experimentos:

```bash
python scripts/run_20_experiments.py
```
