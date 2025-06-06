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
            paths.append(img_path)
            if label == 1:
                gt_paths.append(img_path)
                
        return question, answer, paths, gt_paths
    
    
if __name__ == "__main__":
    loader = DataLoader(path=r'v1_anno.jsonl',
                        img_dir=r'../extracted/train')
    question, answer, paths, gt_path = loader.take_data(100)
    
    test = "../extracted/train/03115_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Accipiter_striatus/97992ba6-b82f-4004-a87f-a9934fea0c3b.jpg"
    if test in paths:
        print("True")
    # print(paths)
    
    
