import os
import sys
import torch
from PIL import Image
import pickle as pkl
from vl_models import CLIPModel
sys.path.append('..')
from utils import DataLoader

class Reader(torch.nn.Module):
    def __init__(self, model_name="llava"):
        super().__init__()
        self.clip_model =  CLIPModel()
        
        if model_name == "llava":
            from lvlm_models.llava_ import LLava
            self.instruction = "Answer the given question based only on the visual content of the images. Do not guess or use outside knowledge. Keep the answer short."
            self.model = LLava(
                pretrained="llava-next-interleave-qwen-7b",
                model_name="llava_qwen",
            )
            
    def init_data(self, golden_answer):
        self.gt_embedding = self.clip_model.extract_textual_features([golden_answer])[0]
    
    def compute_similarity(self, preds):
        pred_embeddings = self.clip_model.extract_textual_features(preds)
        sim = pred_embeddings @ self.gt_embedding.T
        return sim
        
          
    @torch.no_grad()
    def forward(self, qs, img_files):
        prompt = f"{self.instruction}\n question: {qs}\n images: <image>"
        all_outputs = []

        for topk_imgs in img_files:
            text_output = self.model(prompt, topk_imgs)[0]  # string output
            all_outputs.append(text_output)

        scores = self.compute_similarity(all_outputs)
        return torch.tensor(scores).cuda(), all_outputs
    

            
            

        