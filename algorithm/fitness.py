import sys
import os
sys.path.append(os.path.abpatch('..'))

from vl_models import CLIPModel


class RetrieverFitness:
    def __init__(self, vl_models, imgs, query):
        self.vl_models = vl_models
        self.imgs = imgs
        self.query = query
        self.clean_embeddings = self.vl_models.extract_visual_features(self.imgs)
        self.query_embedding = self.vl_models.extract_textual_features([self.query])[0]
        self.clean_scores = self.clean_embeddings.mean(dim=1)  # popsize x dim
        
        
    def __call__(self, pertubations): # popsize x 2 x 3 x 224 x 224
        adv_imgs = self.imgs + pertubations
        adv_embeddings = self.vl_models.extract_visual_features(adv_imgs) # popsize x 2 x dim
        adv_scores = adv_embeddings.mean(dim=1)  # popsize x dim
        fitness_scores = self.clean_scores / adv_scores
        
        return fitness_scores
            
            
        