from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn.functional as F
import random
from typing import List, Tuple, Any
from config import Config
from individual import Individual
from PIL import Image
import pickle
import os
from fitness import RetrievalObjective
from tqdm import tqdm
class GABase(ABC):
    def __init__(self, clean_idv: Individual, config: Config):
        self.clean_idv = clean_idv
        self.config = config

    @abstractmethod
    def attack(self, *args, **kwargs):
        pass
    @abstractmethod
    def initialize_population(self, *args, **kwargs):
        pass
    @abstractmethod
    def evaluate_fitness(self, *args, **kwargs):
        pass
    @abstractmethod
    def crossover(self, *args, **kwargs):
        pass
    @abstractmethod
    def mutate(self, *args, **kwargs):
        pass
    @abstractmethod
    def selection(self, *args, **kwargs):
        pass

class POPOP(GABase):
    def __init__(self, 
                 clean_idv: Individual, 
                 retrieval_objective: RetrievalObjective,
                config: Config):   
        super().__init__(clean_idv, config)
        
        self.population_size = self.config.get('population_size')
        self.max_generations = self.config.get('max_generations')
        self.crossover_rate = self.config.get('crossover_rate')
        self.mutation_rate = self.config.get('mutation_rate')
        self.mean = self.config.get('mean')
        self.std = self.config.get('std')
        self.retrieval_objective = retrieval_objective
        self.alpha = self.config.get('alpha')
        self.device = self.config.get('device')
        self.output_path = self.config.get('output_path')

    def save_history(self, data):
        history_path = os.path.join(self.output_path, "history.pkl")
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                history = pickle.load(f)
        else:
            history = []

        history.append(data)

        with open(history_path, "wb") as f:
            pickle.dump(history, f)

        if data.get("attack_success"):
            for i, img in enumerate(data["best_individual"]["images"]):
                img.save(os.path.join(self.output_path, f"{data['gen']}_{i}.png"))

    def attack(self) -> Individual:
        population = self.initialize_population(self.clean_idv)
        
        best_fitness = [[0, float('-inf')]]
        best_individual = None
        fitness_scores = [idv.min_retrieval_score for idv in population]
       
        for generation in tqdm(range(self.max_generations)):
            if len(set(population)) == 1:
                print(f"\nPopulation converged at generation {generation+1}! All individuals are identical.")
                break
            attack_success = None
            childs = []
            # Variation
            indices = np.arange(len(population))
            np.random.shuffle(indices)
            for i in range(0, len(population), 2):
                parent1 = population[indices[i]]
                parent2 = population[indices[i + 1]]

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                childs.append(child1)
                childs.append(child2)
            pool = population.copy()
            pool.extend(childs)

            fitness_scores = [idv.min_retrieval_score for idv in pool]
            new_pop = self.selection(population=pool, 
                                     fitness_scores=fitness_scores, 
                                     num_individual=self.population_size)
            population = new_pop
            fitness_scores = [idv.min_retrieval_score for idv in population]

            gen_best_fitness = 0
            for idv in population:
                if idv.min_retrieval_score > best_fitness[-1][1] and idv.min_retrieval_score > gen_best_fitness:
                    gen_best_fitness = idv.min_retrieval_score
                    best_individual = idv
            if gen_best_fitness:
                best_fitness.append([generation+1, gen_best_fitness])
            if best_fitness[-1][1] > self.clean_idv.max_retrieval_score:
                attack_success = True
            else:
                attack_success = False
            self.save_history({
                'gen': generation,
                "pop": population,
                'best_fitness': best_fitness[-1][1],
                'attack_success': attack_success,
                'best_individual': {
                    "retrieval_scores": best_individual.retrieval_scores,
                    "min_retrieval_score": best_individual.min_retrieval_score,
                    "max_retrieval_score": best_individual.max_retrieval_score,
                    "images": best_individual.images,
                }
            })
        
        return population, fitness_scores, best_fitness, best_individual

    def gaussian_noise(self, idv: Individual):
        noisy_imgs = []
        for i in range(len(idv.images)):
            arr = np.array(idv.images[i]).astype(np.float32)
            noise = np.random.normal(loc=self.mean, scale=self.std, size=arr.shape).astype(np.float32)
            noisy_arr = arr + noise * 255
            noisy_arr = np.clip(noisy_arr, 0, 255)

            noisy_img = Image.fromarray(noisy_arr.astype(np.uint8))
            noisy_imgs.append(noisy_img)
        return noisy_imgs
    def initialize_population(self, clean_idv: Individual) -> List[Individual]:
        population = []

        for _ in range(self.population_size):
            noisy_imgs = self.gaussian_noise(clean_idv)
            retrieval_scores = self.evaluate_fitness(noisy_imgs)
            population.append(Individual(images=noisy_imgs, 
                                         retrieval_scores=retrieval_scores))
        return population
    
    def evaluate_fitness(self, noise_images: List[Image.Image]) -> List[float]:
        fitness = []
        for image in noise_images:
            fitness.append(self.retrieval_objective.calculate_fitness(image))
        return fitness
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Uniform crossover 
        """
        if self.crossover_rate > random.random():
        # if self.crossover_rate > 0.5:
            c1_imgs, c2_imgs = [], []
            c1_scores, c2_scores = [], []
            num_img = len(parent1.images)
            mask = np.random.rand(num_img) < 0.5 # if idx > 0.5 -> crossover
            for i, m in enumerate(mask):
                if m:
                    c1_imgs.append(parent1.images[i])
                    c2_imgs.append(parent2.images[i])
                    c1_scores.append(parent1.retrieval_scores[i])
                    c2_scores.append(parent2.retrieval_scores[i])
                else:
                    c1_imgs.append(parent2.images[i])
                    c2_imgs.append(parent1.images[i])
                    c1_scores.append(parent2.retrieval_scores[i])
                    c2_scores.append(parent1.retrieval_scores[i])
            child1 = Individual(images=c1_imgs, 
                                retrieval_scores=c1_scores)
            child2 = Individual(images=c2_imgs, 
                                retrieval_scores=c2_scores)
        else:
            child1 = parent1 
            child2 = parent2
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        if self.mutation_rate > random.random():
            mutated_imgs = self.gaussian_noise(individual)
            mutated_scores = self.evaluate_fitness(mutated_imgs)
            mutated = Individual(images=mutated_imgs, 
                                 retrieval_scores=mutated_scores)
        else:
            mutated = individual
        return mutated
    
    def selection(self, population: List[Individual], fitness_scores: np.ndarray, num_individual: int) -> List[Individual]:
        def tournament_selection(population: List[Individual], fitness_scores: np.ndarray) -> List[Individual]:
            tournament_size = self.config.get('tournament_size')
            n = len(population)
            

            parents: List[Individual] = []
            while len(parents) < num_individual:
                indices = list(range(n))
                random.shuffle(indices)
                for i in range(0, n, tournament_size):
                    group = indices[i:i + tournament_size]
                    best_idx = max(group, key=lambda idx: fitness_scores[idx])
                    parents.append(population[best_idx])
            return parents
        
        if self.config.get('selection_method', 'tournament') == 'tournament':
            return tournament_selection(population, fitness_scores)

if __name__ == "__main__":
    from vl_models.clip import CLIPModel

    config = Config("config.json")
    model = CLIPModel()

    clean_idv = Individual(images=[Image.open("data/100/images/0b460614-9297-4c8d-b3e2-bab37b0f3df8.jpg"),
                           Image.open("data/100/images/0e6df0da-f5e3-42ce-bb0a-837ebd055ad8.jpg")], 
                           retrieval_scores=[0.02, 0.01])
    retrieval_objective = RetrievalObjective(model=model,
                                             query=config.get('query'))
    popop = POPOP(clean_idv=clean_idv, 
                  retrieval_objective=retrieval_objective,
                  config=config)
    child1, child2 = popop.crossover(
        Individual(images=[Image.open("data/100/images/0b460614-9297-4c8d-b3e2-bab37b0f3df8.jpg"),
                           Image.open("data/100/images/0e6df0da-f5e3-42ce-bb0a-837ebd055ad8.jpg")], 
                   retrieval_scores=[0.02, 0.01]),
        Individual(images=[Image.open("data/100/images/3fa50358-1f3d-4eda-9e33-4947cb845fea.jpg"),
                           Image.open("data/100/images/4a9730e1-dd63-4a5a-9423-ab53e22f7181.jpg")], 
                   retrieval_scores=[0.04, 0.03])
    )
    print(child1.images[0])
    print(child2.images[0])
    print(child1.retrieval_scores[0])
    print(child1.retrieval_scores[1])
    print(child2.retrieval_scores[0])
    print(child2.retrieval_scores[1])