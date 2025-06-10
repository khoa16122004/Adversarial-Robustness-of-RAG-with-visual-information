from config import Config
import os
import json
from typing import List
from torch import Tensor
from vl_models.clip import CLIPModel
from PIL import Image
from torchvision import transforms
from fitness import RetrievalObjective
import numpy as np


class OriginalTop:
    def __init__(self, config: Config = None, 
                 transform: transforms.Compose = None, retrieval_obj: RetrievalObjective = None):
        self.top_k = config.get('top_k')
        self.paths = config.get('dataset_path')
        self.query = config.get('query')    
        self.transform = transform
        self.org_paths = [
            os.path.join(self.paths, 'images', f.split('/')[-1]) for f in config.annot['image_paths'][:self.top_k]
        ]
        self.retrieval_objective = retrieval_obj
        self.fitness = self._calculate_fitness()

    def _calculate_fitness(self):
        fitness = []
        for path in self.org_paths:
            img = self.transform(Image.open(path))
            fitness.append(self.retrieval_objective.calculate_fitness(img))
        sorted_idx = np.argsort(fitness)[::-1]
        sorted_fitness = [fitness[i] for i in sorted_idx]
        sorted_org_paths = [self.org_paths[i] for i in sorted_idx]
        self.org_paths = sorted_org_paths
        return sorted_fitness
    
    def get_fitness(self, idx: int) -> float:
        return self.fitness[idx]
    def get_image(self, idx: int) -> Image.Image:
        return self.transform(Image.open(self.org_paths[idx]))
    def __str__(self):
        return f"OriginalTop(query={self.query}, top_k={self.top_k}, images=[{', '.join(self.org_paths)}], fitness={self.fitness})"

class Individual:
    def __init__(self, images: List[Image.Image], retrieval_scores: List[float]):
        self.images = images
        self.retrieval_scores = retrieval_scores
        self.min_retrieval_score = min(self.retrieval_scores)
        self.max_retrieval_score = max(self.retrieval_scores)
    
    def __call__(self, *args, **kwargs):
        pass
    def __hash__(self):
        return hash((tuple(self.retrieval_scores), self.min_retrieval_score, self.max_retrieval_score))

    def __str__(self):
        return f"Individual(images={self.images}, retrieval_scores={self.retrieval_scores}, min_retrieval_score={self.min_retrieval_score})"
    
if __name__ == "__main__":
    config = Config('config.json')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    clip_model = CLIPModel()
    retrieval_obj = RetrievalObjective(clip_model, config.get('query'))
    org_top = OriginalTop(config=config, 
                          transform=transform, 
                          retrieval_obj=retrieval_obj)
    print(org_top)