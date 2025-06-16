import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from retriever import Retriever
from reader import Reader
import numpy as np
import pickle as pkl
from util import arkiv_proccess
from PIL import Image
import sys
sys.path.append('..')
from utils import DataLoader
from copy import deepcopy


class MultiScore:
    def __init__(self, reader_name, retriever_name):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(retriever_name)
        self.retriever_name = retriever_name
        self.reader_name = reader_name
    
    def init_data(self, query, question, top_adv_imgs, top_orginal_imgs, answer, n_k): # top_adv_imgs not inlucde current
        
        # top_adv_imgs: I'_0 , I'_1, ..., I'_{nk-2}
        # top_orginal_imgs: I_0, I_1, ..., I_{nk-1}
        
        self.original_img = deepcopy(top_orginal_imgs[-1]) # topk original img
        self.top1_img = deepcopy(top_orginal_imgs[0])
        self.top_adv_imgs = top_adv_imgs
        self.n_k = n_k
        self.original_img_tensor = transforms.ToTensor()(self.original_img).cuda()
        self.retri_clean_reuslt = self.retriever(query, [self.top1_img]) # s(q, I_0)
        self.reader_clean_result = self.reader(question, [top_orginal_imgs], answer) # p(a | I_nk, q)
        self.answer = answer
        self.question = question
        self.query = query
        
    def all_equal(self, perturbations: torch.Tensor) -> bool:
        return (perturbations == perturbations[0]).all().item()
    def __call__(self, pertubations):  # pertubations: tensor
        # kiểm tra pertubations có giống nmhau ko
        print("The same: ", self.all_equal(pertubations))
        raise
        adv_img_tensors = pertubations + self.original_img_tensor
        adv_img_tensors = adv_img_tensors.clamp(0, 1)
        adv_imgs = [to_pil_image(img_tensor) for img_tensor in adv_img_tensors]

        retrieval_result = self.retriever(self.query, adv_imgs)
        
        # adv_top_nk
        adv_topk_imgs = [self.top_adv_imgs + [adv_img] for adv_img in adv_imgs]
        print("adv_topk_imgs: ", len(adv_topk_imgs))
        print("adv_topk_imgs[0]: ", len(adv_topk_imgs[0]))
        reader_result = self.reader(self.question, adv_topk_imgs, self.answer)

        retri_scores = (self.retri_clean_reuslt / retrieval_result).cpu().numpy()
        reader_scores = (reader_result / self.reader_clean_result).cpu().numpy()

        return retri_scores, reader_scores,  adv_imgs  
    
    

