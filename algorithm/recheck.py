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


    # fitness
    fitness = MultiScore(reader_name="llava", 
                         retriever_name="clip"
                         )


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
    print("score: ", history)
    # print("rerun score: ", P_retri_score, P_reader_score)
    
    # greedy selection
    # filtered = history[history[:, 0] < 1]
    # print(filtered)
    # if len(filtered) > 0:
    #     min_idx = np.argmin(filtered[:, 1])
    #     result = filtered[min_idx]
    #     print("Score greedy: ", result)
    # else:
    #     print("Không có dòng nào thỏa mãn điều kiện.")
    
    
    # imgs = [P_adv_imgs[min_idx], fitness.original_img]
    P_adv_imgs.append(fitness.original_img)
    outputs = []
    for i, img in enumerate(P_adv_imgs):
        output = fitness.reader.image_to_text(question, [img])
        outputs.append(output)
    
    print("Output: ", outputs)    

    

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_path", type=str, required=True, help="Path to annotation file (json or txt)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory of the dataset images")
    parser.add_argument("--w", type=int, default=312, help="Width to resize images")
    parser.add_argument("--h", type=int, default=312, help="Height to resize images")
    parser.add_argument("--id", type=int, required=True)
    args = parser.parse_args()
    main(args)
