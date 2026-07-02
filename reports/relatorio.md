# Relatorio

## Tema

Selecao de atributos com Algoritmo Genetico (AG) e avaliacao por Rede Neural Artificial (RNA)
para a base de mortalidade por cancer do colo do utero.

## Objetivo

Identificar subconjuntos de atributos que maximizem a capacidade preditiva da RNA e, ao mesmo
tempo, minimizem o numero de atributos utilizados, investigando o impacto da reducao de
dimensionalidade no desempenho de classificacao.

## Metodologia

1. **Pre-processamento** (`src/preprocessing.py`): limpeza, remocao de vazamento de alvo/CID,
   tratamento de ausentes, codificacao de categoricas (One-Hot / Frequency), normalizacao Min-Max
   e split estratificado 70/15/15. Alvo multiclasse `C53/C54/C55`.
2. **Representacao**: cada cromossomo e um vetor binario de comprimento `L` (nº de features);
   gene 1 = feature usada.
3. **Avaliacao (fitness)**: para cada cromossomo, treina-se uma RNA (32 ReLU -> 16 ReLU -> softmax,
   Adam lr=0.001) usando apenas as features ativas; a aptidao e
   `0.9 x F1-macro(validacao) + 0.1 x (1 - Ns/Nt)`.
4. **Busca (AG)**: Steady-State GA (populacao 150, gap 2), crossover uniforme (Pc=0.85),
   mutacao Pm=1/L, elitismo de 10, selecao por roleta com escalonamento linear de fitness.
   Parada em 200 geracoes ou 20 sem melhoria.
5. **Experimentos**: 20 execucoes completas com sementes distintas; agregacao da curva de
   convergencia (media dos melhores por geracao) e da frequencia de selecao de cada feature.
6. **Teste**: o melhor cromossomo de cada experimento e reavaliado no conjunto de teste.

## Parametros

Ver `reports/parametros.md`.

## Resultados

> Preencher apos rodar `python scripts/run_20_experiments.py`.

- Melhor cromossomo encontrado: _(ver `results/summary/summary.json`)_
- Nº de atributos selecionados (media +/- dp): _..._
- F1-Score de teste (media +/- dp): _..._
- Curva de convergencia: `results/figures/convergence.png`
- Frequencia de selecao das features: `results/figures/feature_frequency.png`

## Analise critica

> Preencher: relacao entre reducao de dimensionalidade e desempenho, features mais recorrentes,
> estabilidade entre execucoes e limitacoes.
