import os
import sys
import torch
from PIL import Image
sys.path.append('..')
from utils import DataLoader
from fitness import MultiScore
from algorithm import NSGAII



#argsd
--annotation_path # default
--dataset_dir # default
-- pop_size # default 50
-- F # defalut 0.9
-- n_k # default 1
-- w # default 224
-- h # default 224
-- max_iter # default 100
-- std # not defauiltm, require


annotation_path = "../v1_anno.jsonl"
dataset_dir = "../../extracted/train"
loader = DataLoader(path=annotation_path,
                    img_dir=dataset_dir)   

pop_size = 50
mutation_rate = 0.1
F = 0.9
n_k = 1
w = 224
h = 224
max_iter = 100
std = 0.1

for sample_id in range(len(data)):
    question, answer, paths, gt_paths = loader.take_data(arsg.ample_id)
    img_files = [Image.open(path).convert('RGB').resize((w, h)) for path in paths]
    
    
    fitness = MultiScore(reader_name="llava", 
                        retriever_name="clip", 
                        question=question, 
                        corpus=img_files, 
                        )

    algorithm = NSGAII(
        population_size=pop_size,
        mutation_rate=mutation_rate,
        F=F,
        n_k=n_k,
        w=w,
        h=h,
        max_iter=max_iter,
        fitness=fitness,
        std=std,
        sample_id=sample_id
    )

    algorithm.solve()
