import torch
import os
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
from PIL import Image
from typing import List
from .model import VisionModel

from dotenv import load_dotenv
load_dotenv()  
os.environ['CURL_CA_BUNDLE'] = ''

class CLIPModel(VisionModel):
    def __init__(self, model_name: str = "clip", pretrained: str = "openai/clip-vit-large-patch14-336"):
        super().__init__(model_name, pretrained)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()

    def load_model(self):
        self.model = HFCLIPModel.from_pretrained(self.pretrained,
                                                 use_safetensors=True,
                                                 ).to(self.device)
        # self.model = HFCLIPModel.from_pretrained("/home/elo/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/58a93f4112bab95a07748c37c004849e6acbdc0f").to(self.device)
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
