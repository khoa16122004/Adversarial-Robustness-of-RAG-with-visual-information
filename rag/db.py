import os
import csv
import numpy as np
from PIL import Image
import faiss
from tqdm import tqdm
import json
from vl_models import VisionModel, CLIPModel
from utils import *
class Database:
    def __init__(self, data_loader, database_dir):
        self.loader = data_loader
        self.database_dir = database_dir
        
    def extract_db(self, qs_id: int, 
                   vs_model: 'VisionModel', batch_size: int = 128):

        question, answer, paths, gt_path = self.loader.take_data(qs_id)
        
        # question id dir
        feature_dir = os.path.join(self.database_dir, str(qs_id), f'features_{vs_model.model_name}')
        os.makedirs(feature_dir, exist_ok=True)

        # file_paths = list(self.qs_data[qs_id]['images'].keys()) # take qs sample
        
        map_path = os.path.join(self.database_dir, str(qs_id), f'mapfile_{vs_model.model_name}.csv')
        os.makedirs(os.path.dirname(map_path), exist_ok=True)

        fail_f = open("fail_read.txt", "a+")
        
        with open(map_path, 'w', newline='') as map_file:
            writer = csv.writer(map_file)
            writer.writerow(['index', 'image_path'])

            global_index = 0
            print("Extracting features...")
            for i in range(0, len(paths), batch_size):
                batch_files = paths[i:i + batch_size]
                batch_imgs = []
                for j, file_path in enumerate(batch_files):
                    try:
                        img = Image.open(file_path).convert("RGB")
                        batch_imgs.append(img)
                    except:
                        fail_f.write(file_path + "\n")
                        continue
                    
                    writer.writerow([global_index, file_path])
                    global_index += 1
                # print("Imgs batch: ", batch_imgs)
                batch_features = vs_model.extract_visual_features(batch_imgs)
                batch_features = batch_features.cpu().numpy()

                np.save(os.path.join(feature_dir, f'{i}.npy'), batch_features)



    def create_db(self, qs_id: int, dim: int, vs_model: 'VisionModel'):
        index_path = os.path.join(self.database_dir, str(qs_id), f"faiss_{vs_model.model_name}.index")
        feature_dir = os.path.join(self.database_dir, str(qs_id), f'features_{vs_model.model_name}')

        index = faiss.IndexFlatIP(dim)
        for file_name in sorted(os.listdir(feature_dir)):
            if not file_name.endswith('.npy'):
                continue
            feature_path = os.path.join(feature_dir, file_name)
            features = np.load(feature_path)
            index.add(features)

        faiss.write_index(index, index_path)

    def read_db(self, qs_id: int, vs_model: 'VisionModel'):
        self.vl_model = vs_model
        
        index_path = os.path.join(self.database_dir, str(qs_id), f"faiss_{vs_model.model_name}.index")
        self.index = faiss.read_index(index_path)

        map_path = os.path.join(self.database_dir, str(qs_id), f'mapfile_{vs_model.model_name}.csv')
        with open(map_path, 'r') as f:
            reader = csv.DictReader(f)
            self.map = [row['image_path'] for row in reader]

    def search_index(self, queries: list['str'], k: int = 5):
        queries = self.vl_model.extract_textual_features(queries).cpu()
        D, I = self.index.search(queries, k)
        return D, I

    def get_image_paths(self, indices):
        final_paths = []
        for indxs in indices:
            paths = [self.map[i] for i in indxs]
            final_paths.append(paths)
        return final_paths
    
# if __name__ == '__main__':

    
#     dataset_dir = "/data/elo/khoatn/Visual-RAG/vs_rag_dataset/images"
#     annotaion_path = "/data/elo/khoatn/Visual-RAG/vs_rag_dataset/v1_anno.jsonl"
#     db = Database(dataset_dir=dataset_dir, annotation_path=annotaion_path)
    
#     qs_id = 0
#     database_dir = "/data/elo/khoatn/Visual-RAG/vs_rag_dataset/database"
    
#     vlm = CLIPModel()
#     # db.extract_db(qs_id=qs_id, database_dir=database_dir, vs_model=vlm)
    
#     db.create_db(qs_id=qs_id, database_dir=database_dir, dim=768, vs_model=vlm)
    
    
