import os
import sys
import torch
from PIL import Image
sys.path.append('..')
from utils import DataLoader
from fitness import MultiScore
from algorithm import NSGAII





annotation_path = "../v1_anno.jsonl"
dataset_dir = "../../extracted/train"
loader = DataLoader(path=annotation_path,
                    img_dir=dataset_dir)   

pop_size = 10
mutation_rate = 0.1
F = 0.9
n_k = 1
w = 224
h = 224
max_iter = 10
std = 0.05


question, answer, paths, gt_paths = loader.take_data(183)
img_files = [Image.open(path).convert('RGB').resize((w, h)) for path in gt_paths]
fitness = MultiScore(reader_name="llava", 
                     retriever_name="clip", 
                     question=question, 
                     original_img=img_files[3], 
                     answer="The bird in the image has orange eyes.")

algorithm = NSGAII(
    population_size=pop_size,
    mutation_rate=mutation_rate,
    F=F,
    n_k=n_k,
    w=w,
    h=h,
    max_iter=max_iter,
    fitness=fitness,
    std=std
)

algorithm.solve()
