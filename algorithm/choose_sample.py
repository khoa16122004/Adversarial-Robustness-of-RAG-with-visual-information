import os
import sys
import torch
from PIL import Image
import argparse

sys.path.append('..')
from utils import DataLoader
from fitness import MultiScore
from algorithm import NSGAII
from llm_service import LlamaService
from tqdm import tqdm

def main(args):
    # outptu path
    output_dir = "jerry_pick"
    os.makedirs(output_dir, exist_ok=True)
    
    # loader
    loader = DataLoader(path=args.annotation_path,
                        img_dir=args.dataset_dir)
    
    # llm_check
    llm = LlamaService(model_name="Llama-7b")
    system_prompt = (
        "You are an assistant that helps me compare two answers for a QA task. "
        "One answer is the ground truth, and the other is the model's prediction.\n"
        "You will return 'True' if the model answer matches the ground truth, otherwise 'False'."
    )
    
    prompt = (
        "Question: {question}\n"
        "Ground Truth Answer: {gt_answer}\n"
        "Model Answer: {model_answer}\n"
        )
    
    # fitness
    fitness = MultiScore(reader_name="llava", 
                         retriever_name="clip"
                         )


    
    for i in tqdm(range(len(loader))):    
        question, answer, paths, gt_paths = loader.take_data(i)
        
        corpus = []
        for path in paths:
            try:
                image = Image.open(path).convert('RGB').resize((args.w, args.h))
                corpus.append(image)
            except:
                continue
        
        
        # top1 documents
        sim_scores = fitness.retriever(question, corpus)
        top1_img = corpus[sim_scores.argmax()]
        
        # answer form top1 documents
        top1_answer = fitness.reader.image_to_text(question, [top1_img])
        print("Answer: ", top1_answer)
        print("GT: ", answer)
        # check answer
        recheck = top1_answer
        while recheck not in ["True", "False"]:
            recheck = llm.text_to_text(
                system_prompt=system_prompt,
                prompt=prompt.format(question=question, gt_answer=answer, model_answer=top1_answer)
            )[0]
        
        if recheck == "True":
            sample_dir = os.path.join(output_dir, i)
            os.makedirs(sample_dir, exist_ok=True)
            
            with open(os.path.join(sample_dir, "question.txt"), "w") as f:
                f.write(question)
            
            with open(os.path.join(sample_dir, "answer.txt"), "w") as f:
                f.write(answer)
            
            with open(os.path.join(sample_dir, "top1_answer.txt"), "w") as f:
                f.write(top1_answer)
                
            top1_img.save(os.path.join(sample_dir, "top1.png"))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_path", type=str, required=True, help="Path to annotation file (json or txt)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Directory of the dataset images")
    parser.add_argument("--w", type=int, default=312, help="Width to resize images")
    parser.add_argument("--h", type=int, default=312, help="Height to resize images")

    args = parser.parse_args()
    main(args)
