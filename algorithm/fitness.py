import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

class RetrieverFitness:
    def __init__(self, vl_models, imgs, query):
        self.vl_models = vl_models
        self.imgs = imgs  # list of PIL Images
        self.query = query # text
        self.img_tensors = torch.stack([transforms.ToTensor()(img) for img in imgs]).cuda()  # (n, 3, 224, 224)
        with torch.no_grad():
            self.img_embeddings = self.vl_models.extract_visual_features(imgs).cuda()  # (n, dim)
            self.query_embedding = self.vl_models.extract_textual_features([query]).cuda()  # (1, dim)
            self.sim_clean = (self.img_embeddings @ self.query_embedding.T).squeeze(1)  # (n,)
            self.clean_score = self.sim_clean.mean().item()  # scalar

    def get_img_tensors(self):
        return self.img_tensors

    def __call__(self, perturbations):  
        popsize, n, c, h, w = perturbations.shape
        adv_tensors = self.img_tensors + perturbations  # (popsize, n, 3, 224, 224)
        print(adv_tensors.shape)
        adv_tensors = adv_tensors.clamp(0, 1)  # ensure valid image range

        adv_imgs = [
            [to_pil_image(adv_tensors[i, j].cpu()) for j in range(n)]
            for i in range(popsize)
        ]  

        fitness_scores = []
        with torch.no_grad():
            for i in range(popsize):
                adv_embeds = self.vl_models.extract_visual_features(adv_imgs[i]).cuda()  # (n, dim)
                sim = adv_embeds @ self.query_embedding.T  # (n, 1)
                score = sim.mean().item()
                fitness = self.clean_score / score
                fitness_scores.append(fitness)

        return torch.tensor(fitness_scores)  # (popsize,)
