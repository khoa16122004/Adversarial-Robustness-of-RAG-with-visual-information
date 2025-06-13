from config import Config
import argparse
from individual import OriginalTop, Individual
from torchvision import transforms  
from fitness import RetrievalObjective
from vl_models.clip_ import CLIPModel
import os
from attack import POPOP

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    config = Config(args.config)
    if not os.path.exists(config.get('output_path')):
        os.makedirs(config.get('output_path'))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    clip_model = CLIPModel()
    retrieval_obj = RetrievalObjective(clip_model, config.get('query'))
    org_top = OriginalTop(config=config, transform=transform, retrieval_obj=retrieval_obj)
    nk = config.get('nk')
    topk = config.get('top_k')

    atk_imgs = [org_top.get_image(i) for i in range(topk - nk, topk)]
    retrieval_scores = [org_top.get_fitness(i) for i in range(topk - nk, topk)]
    
    clean_idv = Individual(images=atk_imgs, 
                          retrieval_scores=retrieval_scores)
    
    popop = POPOP(clean_idv=clean_idv, 
                  retrieval_objective=retrieval_obj,
                  config=config)
    pop, fitness, best_fitness, best_individual = popop.attack()
    print(f"\nClean individual: {clean_idv.retrieval_scores}")
    print(f"Best fitness: {best_fitness}")
    print(f"Best individual: {best_individual.images}")
    print(f"Best individual fitness: {best_individual.retrieval_scores}")
    # print(f"Best individual images: {best_individual.images}")
    print(f"Best individual retrieval scores: {best_individual.retrieval_scores}")
    print(f"Best individual images: {best_individual.images}")
if __name__ == "__main__":
    main()
