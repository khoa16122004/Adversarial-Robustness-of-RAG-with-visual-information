import os
import sys
import torch
from PIL import Image
import argparse

sys.path.append('..')
from util import DataLoader
from fitness import MultiScore
from algorithm import NSGAII
from tqdm import tqdm
import json


def main(args):
    loader = DataLoader(retri_dir=args.retri_dir)


    # fitness
    fitness = MultiScore(reader_name=args.reader_name, 
                         retriever_name=args.retriever_name
                         )
    
    # result_dir
    result_dir = f"attack_result"
    os.makedirs(result_dir, exist_ok=True)

    with open(args.sample_id_path) as f:
        lines = [int(line.strip()) for line in f.readlines()]
    
    for i in range(len(loader)):    
        # take data
        question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(i)
        json_path = os.path.join(args.retri_dir, str(i), "answer.json")
        with open(json_path, "r") as f:
            data = json.load(f)
            golder_answer =  data['topk_results'][f'top_{args.n_k}']['model_answer']
        original_image = retri_imgs[args.n_k]    
        
        # init fitness data
        fitness.init_data(question, original_image, golder_answer)
        
        # algorithm
        algorithm = NSGAII(
            population_size=args.pop_size,
            mutation_rate=args.mutation_rate,
            F=args.F,
            w=args.w,
            h=args.h,
            max_iter=args.max_iter,
            fitness=fitness,
            std=args.std,
            sample_id=str(i),
            log_dir=result_dir,
            n_k=args.n_k
        )

        algorithm.solve()
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader_dir", type=str, required=True)
    parser.add_argument("--retri_dir", type=str, required=True)
    parser.add_argument("--reader_name", type=str, default="llava")
    parser.add_argument("--retriever_name", type=str, default="clip")
    parser.add_argument("--w", type=int, default=312, help="Width to resize images")
    parser.add_argument("--h", type=int, default=312, help="Height to resize images")
    parser.add_argument("--pop_size", type=int, default=20, help="Population size for NSGA-II")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate for NSGA-II")
    parser.add_argument("--F", type=float, default=0.5, help="Differential weight for mutation")
    parser.add_argument("--n_k", type=int, default=1, help="Number of attack")
    parser.add_argument("--max_iter", type=int, default=100, help="Maximum iterations")
    parser.add_argument("--std", type=float, default=0.1, help="Standard deviation for initialization")
    args = parser.parse_args()
    main(args)
