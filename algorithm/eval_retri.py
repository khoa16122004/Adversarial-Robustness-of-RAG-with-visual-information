from util import DataLoader

loader = DataLoader(
    retri_dir="retri_result"
)

total = len(loader)

hit_at_1 = 0
hit_at_2 = 0
hit_at_5 = 0

mrr_at_1 = 0
mrr_at_2 = 0
mrr_at_5 = 0

hit_count_at_1 = 0
hit_count_at_2 = 0
hit_count_at_5 = 0

for i in range(total):
    question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(i)

    # MRR: tìm ảnh đúng đầu tiên
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
            break  # chỉ lấy ảnh đúng đầu tiên cho MRR

    # Hit count (có thể có nhiều ảnh đúng trong top-k)
    hit_count_at_1 += sum([1 for r in retri_basenames[:1] if r in gt_basenames])
    hit_count_at_2 += sum([1 for r in retri_basenames[:2] if r in gt_basenames])
    hit_count_at_5 += sum([1 for r in retri_basenames[:5] if r in gt_basenames])

# In kết quả
print("=== Hit Rate ===")
print("Hit@1:", hit_at_1 / total)
print("Hit@2:", hit_at_2 / total)
print("Hit@5:", hit_at_5 / total)

print("\n=== MRR ===")
print("MRR@1:", mrr_at_1 / total)
print("MRR@2:", mrr_at_2 / total)
print("MRR@5:", mrr_at_5 / total)

print("\n=== Hit Count ===")
print("HitCount@1:", hit_count_at_1)
print("HitCount@2:", hit_count_at_2)
print("HitCount@5:", hit_count_at_5)