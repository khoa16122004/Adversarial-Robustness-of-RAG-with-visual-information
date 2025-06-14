import os
import sys
import torch
from PIL import Image
import argparse
import json
sys.path.append('..')
from utils import DataLoader
from retriever import Retriever
from tqdm import tqdm
from llm_service import LlamaService

def main(args):
    output_dir = "retri_result"
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(path=args.annotation_path, img_dir=args.dataset_dir)
    retriever = Retriever(model_name=args.model_name)
    llm = LlamaService(model_name="Llama-7b")
    
    system_prompt = (
        "You are a helpful assistant. Your task is to extract the specific physical parts or features mentioned in the question "
        "that are useful for image retrieval. These should be short terms or words like 'tail', 'claws', 'antennae', etc. "
        "Ignore any species names, stop words, and general descriptions. "
        "Return only the relevant physical features, not anything else."
    )
    prompt_template = "Question: {question}"
    
    for sample_id in tqdm(range(len(loader))):    
        question, answer, paths, gt_paths = loader.take_data(sample_id)
        path_basenames = [os.path.basename(path) for path in paths]
        gt_basenames = [os.path.basename(path) for path in gt_paths]
        
        # extract keywords
        keyword_query = llm.text_to_text(
            system_prompt=system_prompt,
            prompt=prompt_template.format(question=question),
        )[0].strip()
        
        # sims retri
        corpus = []
        basename_corpus = []
        for i, path in enumerate(paths):
            try:
                image = Image.open(path).convert('RGB').resize((args.w, args.h))
                basename_corpus.append(path_basenames[i])
                corpus.append(image)
            except:
                continue
        

        sims = retriever(question, corpus).flatten()
        topk_values, topk_indices = torch.topk(sims, 5)

        topk_basenames = [basename_corpus[i] for i in topk_indices]
        topk_imgs = [corpus[i] for i in topk_indices]
             
        # path
        metadata = {
            "question": question,
            "answer": answer,
            "keyword": keyword_query, 
            "gt_basenames": gt_basenames[:5],
            "topk_basenames": topk_basenames,
        }
        
        
        # save
        sample_dir = os.path.join(output_dir, str(sample_id))
        os.makedirs(sample_dir, exist_ok=True)
        for img, basename in zip(topk_imgs, topk_basenames):
            img.save(os.path.join(sample_dir, basename))
        
        with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--w", type=int, default=312)
    parser.add_argument("--h", type=int, default=312)
    args = parser.parse_args()
    main(args)
