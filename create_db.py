import argparse
from vl_models import *
from rag.db import Database
import json
from tqdm import tqdm
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
        vs_model = BLIPModel()
        dim = 768
    elif args.model_name == 'flava':
        from vl_models import FLAVAModel
        vs_model = FLAVAModel()
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    
    # Data-sample and Database
    loader = DataLoader(path=args.annotation_path,
                        img_dir=args.dataset_dir)

    db = Database(
        data_loader=loader,
        database_dir=args.database_dir
    )
    
    
    for i in tqdm(range(args.start_index, args.end_index)): 
        db.extract_db(
            qs_id=i,
            vs_model=vs_model,
            batch_size=args.batch_size
        )
        db.create_db(
            qs_id=i,
            dim=dim,
            vs_model=vs_model
        )
    
    
    

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
