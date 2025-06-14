import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import bert_score

def dominate(a, b):
    if a[0] < b[0] and a[1] < b[1]:
        return 1
    elif b[0] < a[0] and b[1] < a[1]:
        return -1
    else:
        return 0

def arkiv_proccess(history):
    arkiv = history[0][:]
    final_history = [arkiv[:]]  

    for i in range(1, len(history)):
        current_gen = history[i]
        valid_new_idx = []
        remove_arkv_idx = []

        for j in range(len(current_gen)):
            is_valid = True
            for k in range(len(arkiv)):
                check = dominate(arkiv[k], current_gen[j])
                if check == 1:
                    is_valid = False
                    break
                elif check == -1:
                    remove_arkv_idx.append(k)
            if is_valid:
                valid_new_idx.append(j)

        remove_arkv_idx = list(set(remove_arkv_idx))
        arkiv = [ind for idx, ind in enumerate(arkiv) if idx not in remove_arkv_idx]

        for j in valid_new_idx:
            arkiv.append(current_gen[j])

        final_history.append(arkiv[:])  
    return final_history

def visualize_process(final_history, objective_labels=["L_RSR", "L_GPR"], interval=500):
    num_generations = len(final_history)

    # Convert to numpy arrays if not already
    final_history = [np.array(gen) for gen in final_history]

    # Determine bounds across all generations
    min_obj1 = min(np.min(gen[:, 0]) for gen in final_history)
    max_obj1 = max(np.max(gen[:, 0]) for gen in final_history)
    min_obj2 = min(np.min(gen[:, 1]) for gen in final_history)
    max_obj2 = max(np.max(gen[:, 1]) for gen in final_history)

    padding_obj1 = (max_obj1 - min_obj1) * 0.1
    padding_obj2 = (max_obj2 - min_obj2) * 0.1
    xlim = (min_obj1 - padding_obj1, max_obj1 + padding_obj1)
    ylim = (min_obj2 - padding_obj2, max_obj2 + padding_obj2)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter([], [], c='red')
    ax.set_xlabel(objective_labels[0])
    ax.set_ylabel(objective_labels[1])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True)

    def update(frame):
        data = final_history[frame]
        scatter.set_offsets(data)
        ax.set_title(f"Pareto Front - Generation {frame + 1}")
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=num_generations, interval=interval, blit=True)
    plt.close(fig)
    return ani


class DataLoader:    
    def __init__(self, retri_dir):
        self.retri_dir = retri_dir
        

    def take_data(self, sample_id):
        sample_dir = os.path.join(self.retri_dir, str(sample_id))
        metadata_path = os.path.join(sample_dir, "metadata.json")
        with open(metadata_path, "r") as f:
            data = json.load(f)
            
        question = data["question"]
        answer = data["answer"]
        query = data["keyword"]
        gt_basenames = data["gt_basenames"]
        retri_basenames = data["topk_basenames"]
        
        retri_imgs = [Image.open(os.path.join(sample_dir, basename)) for basename in retri_basenames]
        
        return question, answer, query, gt_basenames, retri_basenames, retri_imgs
                                   
    def __len__(self):
        return len(os.listdir(self.retri_dir))
    
    
def compute_nlg_metrics(pred, refs):
    refs = [r.lower().strip() for r in refs]
    pred = pred.lower().strip()
    
    print(refs)
    print(pred)

    smoothing = SmoothingFunction().method1
    bleu = sentence_bleu([r.split() for r in refs], pred.split(), smoothing_function=smoothing)

    # METEOR expects single ref at a time, both tokenized
    meteor = max(meteor_score([ref.split()], pred.split()) for ref in refs)

    # ROUGE expects raw strings
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_1 = max(rouge.score(pred, ref)['rouge1'].fmeasure for ref in refs)
    rouge_l = max(rouge.score(pred, ref)['rougeL'].fmeasure for ref in refs)

    # BERTScore can accept list of refs and preds
    bert_P, bert_R, bert_F1 = bert_score.score([pred] * len(refs), refs, lang="en", verbose=False)
    bert_f1_max = max(bert_F1).item()

    return {
        "BLEU": bleu,
        "METEOR": meteor,
        "ROUGE-1": rouge_1,
        "ROUGE-L": rouge_l,
        "BERTScore": bert_f1_max
    }