from abc import ABC, abstractmethod
from typing import List
from torch import Tensor
from vl_models.clip import CLIPModel
from PIL import Image

class RetrievalObjective:
    def __init__(self, model: CLIPModel, query: str):
        self.model = model
        self.query = query
    def sim_score(self, img_feature: Tensor, query_feature: Tensor):
        img_feature /= img_feature.norm(dim=-1, keepdim=True)
        query_feature /= query_feature.norm(dim=-1, keepdim=True)
        score = (img_feature @ query_feature.T).item()
        return score
    def calculate_fitness(self, img: Image.Image):
        img_feature = self.model.extract_visual_features(img)
        query_feature = self.model.extract_textual_features(self.query)
        s = self.sim_score(img_feature, query_feature)
        return s

