import torch
import random
from copy import deepcopy

class GA:
    def __init__(self, population_size, mutation_rate, F, n_k, w, h, max_iter, tournament_size, fitness, std):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.F = F
        self.n_k = n_k
        self.w = w
        self.h = h
        self.max_iter = max_iter
        self.tournament_size = tournament_size
        self.fitness = fitness  # function
        self.std = std

    def tournament_selection(self, pool_fitness):
        selected_indices = []
        for j in range(0, len(pool_fitness), self.tournament_size):
            group = pool_fitness[j: j + self.tournament_size]
            if len(group) == 0:
                continue
            best_in_group = j + torch.argmin(group).item()
            selected_indices.append(best_in_group)
        return selected_indices

    def solve(self):
        population = torch.rand(self.population_size, self.n_k, 3, self.w, self.h).cuda() * self.std
        fitness = self.fitness(population)

        history = []

        for iter in range(self.max_iter):
            r1, r2, r3 = [], [], []
            for i in range(self.population_size):
                choices = [idx for idx in range(self.population_size) if idx != i]
                selected = random.sample(choices, 3)
                r1.append(selected[0])
                r2.append(selected[1])
                r3.append(selected[2])

            x1 = population[r1]
            x2 = population[r2]
            x3 = population[r3]

            v = x1 + self.F * (x2 - x3)
            v = torch.clamp(v, -self.std, self.std)

            mask = torch.rand(self.population_size, self.n_k, 3, self.w, self.h) < self.mutation_rate
            mask[:, :, 0] = False  # không mutate kênh đầu tiên
            u = torch.where(mask, v, population)

            current_fitness = self.fitness(u)

            # Combine current and mutated
            pool = torch.cat([population.unsqueeze(1), u.unsqueeze(1)], dim=1)  # (population_size, 2, ...)
            pool_fitness = torch.stack([fitness, current_fitness], dim=1)  # (population_size, 2)

            # Flatten pool and fitness for tournament selection
            pool_flat = pool.view(-1, self.n_k, 3, self.w, self.h)
            fitness_flat = pool_fitness.view(-1)

            indices = list(range(len(fitness_flat)))
            random.shuffle(indices)
            fitness_shuffled = fitness_flat[indices]

            selected_indices = self.tournament_selection(fitness_shuffled)
            selected_indices = torch.tensor([indices[i] for i in selected_indices])

            population = pool_flat[selected_indices]
            fitness = self.fitness(population)

            history.append(fitness)

        return population, fitness, history
