import random

import numpy as np
import pandas as pd

from src import config
from src.fitness import evaluate_chromosome


def create_individual(n_features: int) -> list[int]:
    individual = np.random.randint(0, 2, size=n_features).tolist()
    if sum(individual) == 0:
        individual[random.randrange(n_features)] = 1
    return individual


def create_population(population_size: int, n_features: int) -> list[list[int]]:
    return [create_individual(n_features) for _ in range(population_size)]


def tournament_selection(population, fitness_values, k=3):
    candidates = random.sample(range(len(population)), k)
    best_idx = max(candidates, key=lambda idx: fitness_values[idx])
    return population[best_idx].copy()


def crossover(parent_a, parent_b):
    if random.random() > config.CROSSOVER_RATE or len(parent_a) < 2:
        return parent_a.copy(), parent_b.copy()
    point = random.randint(1, len(parent_a) - 1)
    child_a = parent_a[:point] + parent_b[point:]
    child_b = parent_b[:point] + parent_a[point:]
    return child_a, child_b


def mutate(individual):
    mutated = individual.copy()
    for idx in range(len(mutated)):
        if random.random() < config.MUTATION_RATE:
            mutated[idx] = 1 - mutated[idx]
    if sum(mutated) == 0:
        mutated[random.randrange(len(mutated))] = 1
    return mutated


def run_genetic_algorithm(X_train, y_train, X_val, y_val):
    n_features = X_train.shape[1]
    population = create_population(config.POPULATION_SIZE, n_features)
    best_chromosome = None
    best_fitness = -1.0
    best_metrics = {}
    convergence = []

    for generation in range(1, config.N_GENERATIONS + 1):
        evaluated = [
            evaluate_chromosome(individual, X_train, y_train, X_val, y_val)
            for individual in population
        ]
        fitness_values = [item[0] for item in evaluated]

        generation_best_idx = int(np.argmax(fitness_values))
        generation_best_fitness = float(fitness_values[generation_best_idx])
        generation_mean_fitness = float(np.mean(fitness_values))

        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_chromosome = population[generation_best_idx].copy()
            best_metrics = evaluated[generation_best_idx][1]

        convergence.append(
            {
                "generation": generation,
                "best_fitness": generation_best_fitness,
                "mean_fitness": generation_mean_fitness,
            }
        )

        elite_indices = np.argsort(fitness_values)[-config.ELITISM_SIZE :]
        next_population = [population[idx].copy() for idx in elite_indices]

        while len(next_population) < config.POPULATION_SIZE:
            parent_a = tournament_selection(population, fitness_values)
            parent_b = tournament_selection(population, fitness_values)
            child_a, child_b = crossover(parent_a, parent_b)
            next_population.extend([mutate(child_a), mutate(child_b)])

        population = next_population[: config.POPULATION_SIZE]

    return {
        "best_chromosome": best_chromosome,
        "best_fitness": best_fitness,
        "best_metrics": best_metrics,
        "convergence": pd.DataFrame(convergence),
    }
