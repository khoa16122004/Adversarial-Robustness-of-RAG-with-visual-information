from utils import DataLoader
from tqdm import tqdm
import json

k = 5
model_name = "clip"

annotation_path = "v1_anno.jsonl"
dataset_dir = "../extracted/train"
loader = DataLoader(path=annotation_path,
                    img_dir=dataset_dir)

start_index = 0
end_index = 200

for i in tqdm(range(start_index, end_index)): 
    question, answer, paths, gt_paths = loader.take_data(i)
    
    # load retri_paths
    with open(f"{model_name}_retrieval_result/{i}.json") as f:
        retrieved_data = json.load(f)
    retrieved_data_retri = retrieved_data['image_paths'][:k]
    correct = sum([1 for p in retrieved_data_retri if p in gt_paths])
    recall_at_k = correct / len(gt_paths) if len(gt_paths) > 0 else 0
    print(f"Sample ID: {i}, Recall@{k}: {recall_at_k}")
    break