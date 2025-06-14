from util import DataLoader

loader = DataLoader(
    retri_dir="retri_result"
)

hit_rate = 0
for i in range(len(loader)):
    question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(i)
    
    for retri_name in retri_basenames:
        if retri_name in gt_basenames:
            hit_rate += 1
            break
        
print(hit_rate)