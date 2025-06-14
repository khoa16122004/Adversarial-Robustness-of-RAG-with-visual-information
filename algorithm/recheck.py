import os
import sys
import torch
from PIL import Image
import argparse
import pickle as pkl
sys.path.append('..')
from utils import DataLoader
from fitness import MultiScore
from algorithm import NSGAII
from tqdm import tqdm
import numpy as np

def main(args):
    loader = DataLoader(path=args.annotation_path,
                        img_dir=args.dataset_dir)
    with open("run.txt", "r") as f:
        lines = [int(line.strip()) for line in f.readlines()]

    # fitness
    fitness = MultiScore(reader_name="llava", 
                         retriever_name="clip"
                         )
    
    # output dir
    os.makedirs("results", exist_ok=True)
    for id in lines:
        

        question, answer, paths, gt_paths = loader.take_data(args.id)
        corpus = [Image.open(path).convert('RGB').resize((args.w, args.h)) for path in paths]
        
        # top1 documents
        sim_scores = fitness.retriever(question, corpus)
        top1_img = corpus[sim_scores.argmax()]
        
        # answer form top1 documents
        top1_answer = fitness.reader.image_to_text(question, [top1_img])
        
        # init fitness data
        fitness.init_data(question, top1_img, top1_answer)
            
        # data_runed
        dir = f"logs/clip_llava_0.1/{args.id}"
        images_dir = os.path.join(dir, "images")
        scores_path = os.path.join(dir, "scores.pkl")
        individual_path = os.path.join(dir, "individuals.pkl")
        # read
        with open(scores_path, 'rb') as f:
            history = pkl.load(open(scores_path, "rb"))[-1]
        
        with open(individual_path, 'rb') as f:
            individual = pkl.load(open(individual_path, "rb"))
        
        
        P_retri_score, P_reader_score, P_adv_imgs = fitness(individual)
        valid_mask = (P_retri_score < 1) & (P_reader_score < 1)
        print("Mask: ", valid_mask)
        print(history)
        imgs = []
        for i in range(len(valid_mask)):
            imgs.append(valid_mask)        
        # imgs = [P_adv_imgs[min_idx], fitness.original_img]
        imgs.append(fitness.original_img)
        outputs = []
        for i, img in enumerate(imgs):
            output = fitness.reader.image_to_text(question, [img])
            outputs.append(output)
        
        result_path = os.path.join("results", f"result_{id}.txt")
        with open(result_path, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n")
            f.write("Outputs:\n")
            for i, output in enumerate(outputs):
                f.write(f"{i+1}. {output}\n")    

    

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_path", type=str, required=True, help="Path to annotation file (json or txt)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory of the dataset images")
    parser.add_argument("--w", type=int, default=312, help="Width to resize images")
    parser.add_argument("--h", type=int, default=312, help="Height to resize images")
    args = parser.parse_args()
    main(args)
