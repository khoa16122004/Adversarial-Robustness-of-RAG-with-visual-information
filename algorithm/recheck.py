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
    
    result_dir = f"attack_result_debug"

    
    for i in range(len(loader)):    
        # take data
        question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(i)
        print("query: ", query)
        print(fitness.retriever(query, retri_imgs))
        break
        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reader_dir", type=str, required=True)
    parser.add_argument("--retri_dir", type=str, required=True)
    parser.add_argument("--reader_name", type=str, default="llava")
    parser.add_argument("--retriever_name", type=str, default="clip")
    parser.add_argument("--n_k", type=int, default=1, help="Number of attack")
    parser.add_argument("--std", type=float, default=0.1, help="Number of attack")
    args = parser.parse_args()
    main(args)
