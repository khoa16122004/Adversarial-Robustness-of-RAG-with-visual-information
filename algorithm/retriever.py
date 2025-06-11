import torch
from PIL import Image
import sys
sys.path.append('..')
from utils import DataLoader

class Retriever(torch.nn.Module):
    def __init__(self, model_name='clip'):
        super().__init__()
        if model_name == "clip":
            from vl_models import CLIPModel
            self.model = CLIPModel()
            
    @torch.no_grad()
    def forward(self, qs, img_files):
        adv_embeds = self.model.extract_visual_features(img_files)
        query_embedding = self.model.extract_textual_features([qs])
        sim = adv_embeds @ query_embedding.T
        
        return sim
    
if __name__ == "__main__":
    annotation_path = "../v1_anno.jsonl"
    dataset_dir = "../../extracted/train"
    loader = DataLoader(path=annotation_path,
                        img_dir=dataset_dir)   
    
    question, answer, paths, gt_paths = loader.take_data(183)
    print(answer)

    img_files = [Image.open(path).convert('RGB').resize((224, 224)) for path in gt_paths]
    img_files[3].save("test_image.jpg")  # Save the image for testing purposes.
    reader = Retriever(model_name="clip")
    
    score = reader("orange eyes", [img_files[3]])
    print(score)