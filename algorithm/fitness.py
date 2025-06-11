import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from retriever import Retriever
from reader import Reader
import numpy as np

class MultiScore:
    def __init__(self, reader_name, retriever_name, question, original_img, answer):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(retriever_name)
        self.original_img = original_img
        self.original_img_tensor = transforms.ToTensor()(original_img)
        self.retri_clean_reuslt = self.retriever(question, [original_img])
        self.reader_clean_result = self.reader(question, [original_img], answer)
        self.answer = answer
        self.question = question
        
    
    def __call__(self, pertubations):  # pertubations: tensor
        adv_img_tensors = pertubations + self.original_img_tensor
        adv_img_tensors = adv_img_tensors.clamp(0, 1)
        adv_imgs = [to_pil_image(img_tensor) for img_tensor in adv_img_tensors]

        retrieval_result = self.retriever(self.question, adv_imgs)
        reader_result = self.reader(self.question, adv_imgs, self.answer)

        retri_scores = np.array(self.retri_clean_reuslt) / np.array(retrieval_result)
        reader_scores = np.array(reader_result) / np.array(self.reader_clean_result)

        return retri_scores, reader_scores