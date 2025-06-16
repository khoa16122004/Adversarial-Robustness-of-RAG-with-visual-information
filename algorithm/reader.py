import os
import sys
import torch
from PIL import Image
import pickle as pkl
from bert_score import score as bert_score
sys.path.append('..')
from utils import DataLoader

class Reader(torch.nn.Module):
    def __init__(self, model_name="llava"):
        super().__init__()
        self.bert_model = "bert-base-uncased"  # hoặc "roberta-large" nếu muốn cao hơn
        if model_name == "llava":
            from lvlm_models.llava_ import LLava
            
            # simple template
            self.template = "You will be given a question and somes images to help you answer the question. Please answer the question in the short ways."
            self.model = LLava(
                pretrained="llava-next-interleave-qwen-7b",
                model_name="llava_qwen",
            )
    def compute_similarity(self, pred, gt):
        # BERTScore expects lists
        P, R, F1 = bert_score([pred], [gt], lang="en", model_type=self.bert_model, verbose=False)
        return F1[0].item()  # return scalar float      
    
          
    @torch.no_grad()
    def forward(self, qs, img_files, answer):
        instruction = "You will be given a question and an image to help you answer the question. Please answer the question in a short way."
        prompt = f"{instruction}\n question: {qs}\n images: <image>"
        
        all_outputs = []
        all_texts = []
        for topk_imgs in img_files:
            # topk_imgs: list of PIL.Image
            text_output = self.model(prompt, topk_imgs)[0]  # string output
            score = self.compute_similarity(text_output, answer)
            all_outputs.append(score)
            all_texts.append(text_output)

        return torch.tensor(all_outputs).cuda(), all_texts
    
    @torch.no_grad()
    def forward(self, qs, img_files, answer):
        # img_files = [[img_files]]
        # input
        intruction = "You will be given a question and a image to help you answer the question. Please answer the question in the short ways."
        prompt = f"{intruction}\n question {qs}\n images <image>"
        all_outputs = []
        for topk_imgs in img_files:
            text_output = self.model(prompt, topk_imgs)
            # calculate sim text_output and answer
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
        