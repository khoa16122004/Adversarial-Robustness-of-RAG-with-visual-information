from util import DataLoader

loader = DataLoader(
    retri_dir="retri_result"
)

for i in range(len(loader)):
    question, answer, paths, gt_paths = loader.take_data(i)
    print(question, answer, paths, gt_paths)