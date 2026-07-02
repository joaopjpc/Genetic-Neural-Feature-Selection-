# Parametros

## Pre-processamento

- Divisao treino / validacao / teste: `70% / 15% / 15%` (estratificada)
- Tratamento de ausentes: imputacao (mediana p/ numericas, `IGNORADO` p/ categoricas)
- Normalizacao: linear Min-Max `x' = (x - xmin) / (xmax - xmin)` -> intervalo `[0, 1]`
- Codificacao categorica: One-Hot; alta cardinalidade: Frequency Encoding
- Alvo: `label_cid` (`C53`, `C54`, `C55`) -> 3 classes via LabelEncoder

## Algoritmo genetico (Steady-State GA)

- Representacao: cromossomo binario (gene 1 = atributo usado, 0 = nao usado); comprimento `L` = nº de **atributos originais** (3 numericos + 6 binarios + 15 categoricos + 5 frequency = 29), e nao o nº de colunas apos o one-hot (181). Gene ligado -> todas as colunas one-hot daquele atributo entram na rede.
- Tamanho da populacao: `150`
- Minimo de atributos por cromossomo: `3` (reparo apos criacao/mutacao)
- Crossover: uniforme, `Pc = 0.85`
- Mutacao: bit-flip, `Pm = 1 / L`
- Elitismo: `10` melhores preservados a cada geracao
- Abordagem: Steady-State, `Gap = 2` (2 filhos por passo, substituindo os 2 piores)
- Definicao de geracao: 1 geracao = 1 renovacao completa da populacao = `pop/gap` = 75 passos (150 filhos). Logo "200 geracoes" e "20 sem melhoria" tem escala real de busca.
- Aceleracao da fitness: cada rede-avaliadora treina em um subconjunto estratificado de `20000` linhas de treino (`FITNESS_SUBSAMPLE`); a avaliacao final e o teste usam a base de treino cheia.
- Selecao: roleta sobre a aptidao escalonada
- Escalonamento: linear (Goldberg), `C = 2.0` (evita super-individuo)
- Parada: `200` geracoes OU `20` geracoes consecutivas sem melhoria do melhor
- Experimentos: `20` execucoes completas (sementes `42..61`)

## Funcao de aptidao

- `Fitness = 0.9 x F1-Score + 0.1 x (1 - Ns/Nt)`
- `Ns` = nº de atributos selecionados, `Nt` = nº total de atributos
- F1-Score: media `macro` (problema multiclasse)

## Rede neural (avaliadora)

- Biblioteca: scikit-learn `MLPClassifier` (MLP, backpropagation + Adam, softmax multiclasse)
- Camada de entrada: nº de neuronios = nº de features selecionadas
- Camadas ocultas: `32` (ReLU) -> `16` (ReLU)
- Camada de saida: nº de neuronios = nº de classes, ativacao `softmax`
- Otimizador: `adam`, taxa de aprendizado `0.001`
- Selecao do melhor modelo: menor erro de validacao (early stopping)
