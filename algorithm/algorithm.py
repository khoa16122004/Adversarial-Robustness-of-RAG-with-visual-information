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
            # crossover - mutation
            r1, r2, r3 = [], [], []
            for i in range(self.population_size):
                choices = [idx for idx in range(self.population_size) if idx != i]
                selected = random.sample(choices, 3)
                r1.append(selected[0])
                r2.append(selected[1])
                r3.append(selected[2])

            r1 = torch.tensor(r1, dtype=torch.long, device="cuda")
            r2 = torch.tensor(r2, dtype=torch.long, device="cuda")
            r3 = torch.tensor(r3, dtype=torch.long, device="cuda")
            
            x1 = deepcopy(population[r1])
            x2 = deepcopy(population[r2])
            x3 = deepcopy(population[r3])

            v = x1 + self.F * (x2 - x3)
            v = torch.clamp(v, -self.std, self.std)
            mask = torch.rand(self.population_size, self.n_k, 3, self.w, self.h) < self.mutation_rate
            mask[:, :, 0] = False  # không mutate kênh đầu tiên
            u = torch.where(mask.cuda(), v, population)

            # calculate new fitness
            current_fitness = self.fitness(u)
            
            # pool
            pool = torch.cat([population, u], dim=0)  # (population_size, 2, ...)
            pool_fitness = torch.cat([fitness, current_fitness], dim=0)  # (population_size, 2)

            print(len(pool_fitness), len(pool_fitness))
            
            # tournament selection
            selected_indices = []
            for _ in range(self.tournament_size // 2):
                indices = torch.randperm(len(pool))
                fitness_shuffled = pool_fitness[indices]
                tournament_selected_ids = self.tournament_selection(fitness_shuffled)
                selected_indices.extend(indices[tournament_selected_ids])
            
            population = pool[selected_indices]
            fitness = pool_fitness[selected_indices]
            print(f"Iteration {iter + 1}/{self.max_iter}, Best fitness: {fitness.min().item()}")
            history.append(fitness)

        return population, fitness, history
