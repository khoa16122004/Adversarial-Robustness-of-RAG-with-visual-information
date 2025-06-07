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

# Stats
hit_at = [0] * k  # hit_at[0] là Hit@1, hit_at[4] là Hit@5
count = 0

# For grouping by correct gt count
output_dir = f"eval_grouped_{model_name}"
os.makedirs(output_dir, exist_ok=True)
group_files = {}

def write_index(correct_count, idx):
    if correct_count >= k:
        file_key = f"{k}_plus_gt"
    else:
        file_key = f"{correct_count}_gt"

    file_path = os.path.join(output_dir, f"{file_key}.txt")
    if file_key not in group_files:
        group_files[file_key] = open(file_path, "w")
    group_files[file_key].write(f"{idx}\n")

for i in tqdm(range(start_index, end_index)):
    _, _, _, gt_paths = loader.take_data(i)

    with open(f"{model_name}_retrieval_result/{i}.json") as f:
        retrieved_data = json.load(f)
    retrieved_data_retri = retrieved_data['image_paths'][:k]

    correct = sum([1 for p in retrieved_data_retri if p in gt_paths])
    write_index(correct, i)

    # Hit@1 đến Hit@k
    for j in range(k):
        top_j_paths = retrieved_data_retri[:j + 1]
        if any(p in gt_paths for p in top_j_paths):
            hit_at[j] += 1

    count += 1

# Đóng file ghi
for f in group_files.values():
    f.close()

# In kết quả Hit@1 → Hit@5
for j in range(k):
    hit_rate = hit_at[j] / count if count > 0 else 0
    print(f"Hit@{j + 1}: {hit_rate:.4f}")
