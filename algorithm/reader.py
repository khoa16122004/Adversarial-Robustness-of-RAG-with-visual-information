import os
import sys
import torch
from PIL import Image
import pickle as pkl
sys.path.append('..')
from utils import DataLoader

class Reader(torch.nn.Module):
    def __init__(self, model_name="llava"):
        super().__init__()
        
        if model_name == "llava":
            from lvlm_models.llava_ import LLava
            
            # simple template
            self.template = "You will be given a question and somes images to help you answer the question. Please answer the question in the short ways."
            self.model = LLava(
                pretrained="llava-next-interleave-qwen-7b",
                model_name="llava_qwen",
            )
            
    @torch.no_grad()
    def image_to_text(self, qs, img_files):
  
        if not isinstance(img_files, list):
            img_files = [img_files]
        
        # input
        intruction = "You will be given a question and a image to help you answer the question."
        prompt = f"{intruction}\n question {qs}\n images: " +  "<image>" * len(img_files)
        
        outputs = self.model(prompt, img_files)[0]
        
        return outputs
    
    @torch.no_grad()
    def forward(self, qs, img_files, answer):
        # img_files = [[img_files]]
        # input
        intruction = "You will be given a question and a image to help you answer the question. Please answer the question in the short ways."
        prompt = f"{intruction}\n question {qs}\n images <image>"
        all_outputs = []
        for topk_imgs in img_files:
            text_output = self.model(prompt, topk_imgs, answer)
            print(text_output)
            raise
        
        all_outputs = torch.tensor(all_outputs)    
        return all_outputs
            
            
            
if __name__ == "__main__":
    annotation_path = "../v1_anno.jsonl"
    dataset_dir = "../../extracted/train"
    loader = DataLoader(path=annotation_path,
                        img_dir=dataset_dir)   
    
    question, answer, paths, gt_paths = loader.take_data(183)
    print(answer)

    img_files = [Image.open(path).convert('RGB').resize((428, 428)) for path in gt_paths]
    img_files[3].save("test_image.jpg")  # Save the image for testing purposes.
    reader = Reader(model_name="llava")
    # data
    ind_path = r"./logs/clip_lava_0.1/individuals.pkl"
    score_path = r"./logs/clip_lava_0.1/scores.pkl"
    
    history = pkl.load(open(score_path, "rb"))[-1]
    print("history: ", history)
    ind = torch.stack(pkl.load(open(ind_path, "rb")), dim=0)

    outputs = reader.image_to_text("what is color of the eyes of bird?", [img_files[3]])
    # print(outputs)  # Should print the answer to the question based on the image provided.
    # score = reader("what is color of the eyes of bird?", 
    #                [img_files[2]], 
    #                "The bird in the image has orange eyes.")
    # print(score)  # Should print the score or log probability of the answer.        

    # score = reader("what is color of the eyes of bird?", 
    #                [img_files[3]], 
    #                "The bird in the image has yellow eyes.")

    # print(score)  # Should print the score or log probability of the answer.        

    # score = reader("what is color of the eyes of bird?", 
    #                [img_files[3]], 
    #                "dont know")
    # print(score)  # Should print the score or log probability of the answer.        
        