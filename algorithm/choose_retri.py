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
from llm_service import LlamaService, GPTService

def main(args):
    # output_dir = f"retri_result_{args.model_name}"
    output_dir = f"retri_result_debug"
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(path=args.annotation_path, img_dir=args.dataset_dir)
    retriever = Retriever(model_name=args.model_name)
    llm = GPTService(model_name="gpt-4o")
    
    system_prompt = (
        "You are a helpful assistant. Your task is to extract the specific physical parts or features mentioned in the question "
        "that are useful for image retrieval. These should be short terms or words like 'tail', 'claws', 'antennae', etc. "
        "Ignore any species names, stop words, general descriptions or the words like colors"
        "Return only the most usefull relevant physical feature, not anything else."
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
        ).strip()
        print("keyword_query", keyword_query)
        
        # sims retri
        corpus = []
        basename_corpus = []
        for i, path in enumerate(paths):
            try:
                image = Image.open(path).resize((args.w, args.h))
                basename_corpus.append(path_basenames[i])
                corpus.append(image)
            except:
                continue
        

        sims = retriever(question, corpus).flatten()
        topk_values, topk_indices = torch.topk(sims, 5)
        print(topk_values)
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
        print("topk_basenames", topk_basenames)
        raise

# tensor([0.3416, 0.3347, 0.3318, 0.3301, 0.3289], device='cuda:0', dtype=torch.float16)
#topk_basenames ['582aa998-1f3c-4ea9-915c-b483d4f5afab.jpg', 'ac5fa8b9-05dd-4175-9dc6-dfbc23b39e78.jpg', '61703a7a-a43a-4dad-bcdd-f1f483e6a109.jpg', '12e3e429-c168-4f51-991e-b7e6ee10974d.jpg', '447b62d2-04bd-422a-bc4c-91fa60bb00ca.jpg']
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--w", type=int, default=312)
    parser.add_argument("--h", type=int, default=312)
    args = parser.parse_args()
    main(args)
