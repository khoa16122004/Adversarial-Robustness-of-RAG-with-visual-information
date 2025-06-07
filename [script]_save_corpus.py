import json
import os
import shutil

output_dir = "split_corpus"
sample_id = 180
result_dir = f"clip_retrieval_result/{sample_id}.json"

with open(result_dir, "r") as f:
    data = json.load(f)

output_sample_dir = os.path.join(output_dir, str(sample_id))
os.makedirs(output_sample_dir, exist_ok=True)

annot_path = os.path.join(output_sample_dir, "annot.json")
with open(annot_path, "w") as f:
    json.dump(data, f, indent=2)

images_dir = os.path.join(output_sample_dir, "images")
os.makedirs(images_dir, exist_ok=True)

image_paths = data["image_paths"]
for img_path in image_paths:
    img_basename = os.path.basename(img_path)
    dest_path = os.path.join(images_dir, img_basename)
    shutil.copy(img_path, dest_path)

print(f"Đã lưu annot.json và {len(image_paths)} ảnh vào thư mục {output_sample_dir}")
