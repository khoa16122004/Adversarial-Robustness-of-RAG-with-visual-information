import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from retriever import Retriever
from reader import Reader
import numpy as np
import pickle as pkl

class MultiScore:
    def __init__(self, reader_name, retriever_name, question, original_img, answer):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(retriever_name)
        self.original_img = original_img
        self.original_img_tensor = transforms.ToTensor()(original_img).cuda()
        self.retri_clean_reuslt = self.retriever(question, [original_img])
        self.reader_clean_result = self.reader(question, [original_img], answer)
        self.answer = answer
        self.question = question
        self.retriever_name = retriever_name
        self.reader_name = reader_name
        
    
    def __call__(self, pertubations):  # pertubations: tensor
        adv_img_tensors = pertubations + self.original_img_tensor
        adv_img_tensors = adv_img_tensors.clamp(0, 1)
        adv_imgs = [to_pil_image(img_tensor) for img_tensor in adv_img_tensors]

        retrieval_result = self.retriever(self.question, adv_imgs)
        reader_result = self.reader(self.question, adv_imgs, self.answer)

        retri_scores = (self.retri_clean_reuslt / retrieval_result).cpu().numpy()
        reader_scores = (reader_result / self.reader_clean_result).cpu().numpy()

        return retri_scores, reader_scores
    
    
if __name__ == "__main__":
    path = "logs/clip_llava_0.01_183_individuals.pkl"
    ind = pkl.load(open(path, "rb"))
    print(ind.shape)
    pass
    