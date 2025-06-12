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

class MultiScore:
    def __init__(self, reader_name, retriever_name, question, original_img, answer):
        self.reader = Reader(reader_name)
        self.retriever = Retriever(retriever_name)
        self.original_img = original_img
        self.original_img_tensor = transforms.ToTensor()(original_img).cuda()
        # self.retri_clean_reuslt = self.retriever(question, [original_img])
        self.reader_clean_result = self.reader(question, [original_img], answer)
        self.answer = answer
        self.question = question
        self.retriever_name = retriever_name
        self.reader_name = reader_name
        
    
    def __call__(self, pertubations):  # pertubations: tensor
        adv_img_tensors = pertubations + self.original_img_tensor
        adv_img_tensors = adv_img_tensors.clamp(0, 1)
        adv_imgs = [to_pil_image(img_tensor) for img_tensor in adv_img_tensors]

        # retrieval_result = self.retriever(self.question, adv_imgs)
        reader_result = self.reader(self.question, adv_imgs, self.answer)

        # retri_scores = (self.retri_clean_reuslt / retrieval_result).cpu().numpy()
        retri_scores = None
        reader_scores = (reader_result / self.reader_clean_result).cpu().numpy()

        return retri_scores, reader_scores, adv_imgs
    
    
if __name__ == "__main__":
    # repair
    
    annotation_path = "../v1_anno.jsonl"
    dataset_dir = "../../extracted/train"
    loader = DataLoader(path=annotation_path,
                        img_dir=dataset_dir)  
    w = 224
    h = 224
    
    sample_id = 183
    question, answer, paths, gt_paths = loader.take_data(sample_id)
    img_files = [Image.open(path).convert('RGB').resize((w, h)) for path in gt_paths]
    fitnesse = MultiScore(reader_name="llava", 
                        retriever_name=None, 
                        question=question, 
                        original_img=img_files[3], 
                        answer="The bird in the image has orange eyes.")    
    
    
    
    ind_path = "logs/clip_lava_0.1/individuals.pkl"
    score_path = "logs/clip_lava_0.1/scores.pkl"
    
    history = pkl.load(open(score_path, "rb"))[-1]
    print("history: ", history)
    ind = torch.stack(pkl.load(open(ind_path, "rb")), dim=0)
    fit_output = fitnesse(ind)
    print("fitnesse: ", fit_output)
    
    
    adv_img_tensors = ind + fitnesse.original_img_tensor
    adv_img_tensors = adv_img_tensors.clamp(0, 1)
    adv_imgs = [to_pil_image(img_tensor) for img_tensor in adv_img_tensors]
    adv_imgs.append(fitnesse.original_img)
    outputs = []
    fitnesse.original_img.save("original.png")
    for i, img in enumerate(adv_imgs):
        output = fitnesse.reader.image_to_text(question, [img])
        outputs.append(output)
    print("outputs: ", outputs)
