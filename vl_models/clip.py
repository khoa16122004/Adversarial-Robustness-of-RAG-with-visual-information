import torch
import os
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
from PIL import Image
from typing import List
from .model import VisionModel

from dotenv import load_dotenv
load_dotenv()  
class CLIPModel(VisionModel):
    def __init__(self, model_name: str = "clip", pretrained: str = "openai/clip-vit-large-patch14-336"):
        super().__init__(model_name, pretrained)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        print(os.getenv("HF_TOKEN"))
        self.model = HFCLIPModel.from_pretrained(self.pretrained,
                                                 token=os.getenv("HF_TOKEN")).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.pretrained)

    def extract_visual_features(self, imgs: List[Image.Image]):
        inputs = self.processor(images=imgs, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features

    def extract_textual_features(self, texts: List[str]):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)
        return features
