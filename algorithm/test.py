import os
import sys
import torch
from PIL import Image
import argparse

sys.path.append('..')
from utils import DataLoader
from fitness import MultiScore
from algorithm import NSGAII



def main(args):
    loader = DataLoader(path=args.annotation_path,
                        img_dir=args.dataset_dir)


    # fitness
    fitness = MultiScore(reader_name="llava", 
                         retriever_name="clip"
                         )

    with open(args.sample_id_path) as f:
        lines = [int(line.strip()) for line in f.readlines()]
    
    for i in lines:    
        question, answer, paths, gt_paths = loader.take_data(i)
        corpus = [Image.open(path).convert('RGB').resize((args.w, args.h)) for path in paths]
        
        # top1 documents
        sim_scores = fitness.retriever(question, corpus)
        top1_img = corpus[sim_scores.argmax()]
        
        # answer form top1 documents
        top1_answer = fitness.reader.image_to_text(question, [top1_img])
        
        # init fitness data
        fitness.init_data(question, top1_img, top1_answer)
        
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
            sample_id=i
        )

        algorithm.solve()
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_path", type=str, required=True, help="Path to annotation file (json or txt)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory of the dataset images")
    parser.add_argument("--sample_id_path", type=str, required=True, help="Path to text file containing sample IDs to run")
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
