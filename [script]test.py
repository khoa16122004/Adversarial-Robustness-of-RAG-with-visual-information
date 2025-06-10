from algorithm.algorithm import GA
from algorithm.fitness import RetrieverFitness
import torch
import json
import os
from PIL import Image
from vl_models import CLIPModel

# data
data_sample_id = 100
json_path = os.path.join(r"/kaggle/input/sample/split_corpus",
                         str(data_sample_id),
                         'annot.json')

with open(json_path, "r") as f:
    data = json.load(f)
    question = data['question']
    img_names = [os.path.basename(paths) for paths in data['image_paths']]
    img_paths = [os.path.join(f"/kaggle/input/sample/split_corpus/{data_sample_id}/images",
                              img_name) for img_name in img_names]
    
    imgs = [Image.open(img_path).resize((224, 224)) for img_path in img_paths]  
    vl_model = CLIPModel()
    img_embeddings = vl_model.extract_visual_features(imgs)
    text_embedding = vl_model.extract_textual_features([question])
    sim = img_embeddings @ text_embedding.T
    values, indices = torch.topk(sim, k=5, dim=0, largest=True)
    img_topk = [imgs[id] for id in indices]
    
# repair for attack
n_k = 2
pop_size = 4
mutation_rate = 0.5
F = 0.8
max_iter = 100
tournament_size = 50
std = 0.05
img_to_attack = [img_topk[i] for i in range(-n_k, 0)]
w, h = img_to_attack[0].size


fitness = RetrieverFitness(
    vl_models=vl_model,
    imgs=img_to_attack,  # list of PIL Images
    query=question  # text
)
print(f"Clean score: {fitness.clean_score}")

algo = GA(
    population_size=pop_size,
    mutation_rate=mutation_rate,
    F=F,
    n_k=n_k,
    w=w,
    h=h,
    max_iter=100,
    tournament_size=tournament_size,
    fitness=fitness,
    std=std,
)

algo.solve()