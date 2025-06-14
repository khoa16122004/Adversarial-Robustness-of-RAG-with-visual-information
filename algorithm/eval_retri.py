from util import DataLoader

loader = DataLoader(
    retri_dir="retri_result"
)

hit_at_1 = 0
hit_at_2 = 0
hit_at_5 = 0

mrr_at_1 = 0
mrr_at_2 = 0
mrr_at_5 = 0

total = len(loader)

for i in range(total):
    question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(i)

    found = False
    for rank, retri_name in enumerate(retri_basenames[:5]):
        if retri_name in gt_basenames:
            if rank < 1:
                hit_at_1 += 1
                mrr_at_1 += 1.0 / (rank + 1)
            if rank < 2:
                hit_at_2 += 1
                mrr_at_2 += 1.0 / (rank + 1)
            if rank < 5:
                hit_at_5 += 1
                mrr_at_5 += 1.0 / (rank + 1)
            break  # stop at first hit

# Chia trung bình
print("Hit@1:", hit_at_1 / total)
print("Hit@2:", hit_at_2 / total)
print("Hit@5:", hit_at_5 / total)

print("MRR@1:", mrr_at_1 / total)
print("MRR@2:", mrr_at_2 / total)
print("MRR@5:", mrr_at_5 / total)
