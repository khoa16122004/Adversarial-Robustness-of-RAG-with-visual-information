import argparse
from vl_models import *
from rag.db import Database
import json
from tqdm import tqdm
import os
from utils import DataLoader

def main(args):
    if args.model_name == 'clip':
        from vl_models import CLIPModel
        vs_model = CLIPModel()
        dim = 768
    elif args.model_name == 'git':
        from vl_models import GITModel
        vs_model = GITModel()
    elif args.model_name == 'open_clip':
        from vl_models import OpenCLIPModel
        vs_model = OpenCLIPModel()
    elif args.model_name == 'blip':
        from vl_models import BLIPModel
        dim = 768
        vs_model = BLIPModel()
    elif args.model_name == 'flava':
        from vl_models import FLAVAModel
        vs_model = FLAVAModel()
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    loader = DataLoader(path=args.annotation_path,
                        img_dir=args.dataset_dir)

    db = Database(
        data_loader=loader,
        database_dir=args.database_dir
    )


    output_dir = f"{args.model_name}_retrieval_result"
    os.makedirs(output_dir, exist_ok=True)
    
    
    
    for i in tqdm(range(args.start_index, args.end_index)): 
        question, answer, paths, gt_path = loader.take_data(i)
        db.read_db(
            qs_id=i,
            vs_model=vs_model,
        )
        
        D, I = db.search_index([question], 50)
        img_paths = db.get_image_paths(list(I))[0]
        with open(f"{output_dir}/{i}.json", 'w') as f:
            json.dump({'question': question , 'image_paths': img_paths}, f, indent=4)
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visual Semantic Embedding Pipeline")

    parser.add_argument('--model_name', type=str, required=True,
                        choices=['clip', 'git', 'open_clip', 'blip', 'flava'],
                        help='Model name to use')
    parser.add_argument('--pretrained', type=str, required=False, default='',
                        help='Pretrained model path or ID (if applicable)')
    parser.add_argument('--annotation_path', type=str, default="v1_anno.jsonl",
                        help='Path to the annotation .jsonl file')
    parser.add_argument('--dataset_dir', type=str, required=False,
                        help='Directory containing image dataset',
                        default="../extracted/train")
    parser.add_argument('--database_dir', type=str, required=False,
                        help='Directory to save extracted database',
                        default="database")
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for feature extraction')
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=0)

    args = parser.parse_args()
    main(args)
