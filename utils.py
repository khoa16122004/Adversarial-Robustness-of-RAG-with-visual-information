import json
import os
import torch.nn.functional as F
class DataLoader:    
    def __init__(self, path, img_dir):
        self.img_dir = img_dir
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]    

    def take_data(self, idx):
        question = self.data[idx]['question']
        answer = self.data[idx]['answer']
        image_items = self.data[idx]['images']
        paths = []
        gt_paths = []
        for path, label in image_items.items():
            img_path = os.path.join(self.img_dir, path)
            paths.append(img_path)
            if label == 1:
                gt_paths.append(img_path)
                
        return question, answer, paths, gt_paths


def s(x, y):
    return F.cosine_similarity(x, y, dim=-1).mean().item()
    
    
if __name__ == "__main__":
    loader = DataLoader(path=r'v1_anno.jsonl',
                        img_dir=r'../extracted/train')
    question, answer, paths, gt_path = loader.take_data(183)
    
    print(gt_path)    
