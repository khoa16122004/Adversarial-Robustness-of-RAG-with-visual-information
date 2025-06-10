import torch
import json
import os

class Config:
    def __init__(self, config_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = json.load(open(config_path))
        with open(os.path.join(self.config['dataset_path'], 'annot.json'), 'r', encoding='utf-8') as f:
            self.annot = json.load(f)
        self.config['query'] = self.annot['question']

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value