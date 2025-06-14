from util import DataLoader

loader = DataLoader(
    retri_dir="retri_result"
)

for i in range(len(loader)):
    question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(i)
    print(question, answer, query, gt_basenames, retri_basenames, retri_imgs)