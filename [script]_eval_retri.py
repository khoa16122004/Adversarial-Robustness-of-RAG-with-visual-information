from utils import DataLoader
from tqdm import tqdm
import json
import os

k = 5
model_name = "clip"

annotation_path = "v1_anno.jsonl"
dataset_dir = "../extracted/train"
loader = DataLoader(path=annotation_path,
                    img_dir=dataset_dir)

start_index = 0
end_index = 200

output_dir = f"eval_grouped_{model_name}"
os.makedirs(output_dir, exist_ok=True)

group_files = {}  # cache file handles

def write_to_group_file(correct_count, info):
    # Tạo tên file dựa trên số ảnh đúng
    if correct_count >= k:
        file_key = f"{k}_plus_gt"
    else:
        file_key = f"{correct_count}_gt"

    file_path = os.path.join(output_dir, f"{file_key}.jsonl")
    if file_key not in group_files:
        group_files[file_key] = open(file_path, "w", encoding="utf-8")
    json.dump(info, group_files[file_key])
    group_files[file_key].write("\n")

for i in tqdm(range(start_index, end_index)): 
    question, answer, paths, gt_paths = loader.take_data(i)

    with open(f"{model_name}_retrieval_result/{i}.json") as f:
        retrieved_data = json.load(f)
    retrieved_data_retri = retrieved_data['image_paths'][:k]

    # Recall@k
    correct = sum([1 for p in retrieved_data_retri if p in gt_paths])
    
    # Ghi thông tin vào file phù hợp
    sample_info = {
        "index": i,
        "question": question,
        "gt_paths": gt_paths,
        "retrieved_paths": retrieved_data_retri,
        "num_correct": correct
    }
    write_to_group_file(correct, sample_info)

# Đóng file sau khi ghi xong
for f in group_files.values():
    f.close()
