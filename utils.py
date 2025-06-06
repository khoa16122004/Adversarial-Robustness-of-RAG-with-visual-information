import json
import os
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
            paths.append(path)
            if label == 1:
                gt_paths.append(path)
                
        return question, answer, paths, gt_path
            
if __name__ == "__main__":
    loader = DataLoader(path=r'visual-rag\v1_anno.jsonl',
                        img_dir=r'visual-rag\train')
    question, answer, paths, gt_path = loader.take_data(0)
