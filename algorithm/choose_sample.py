import os
import sys
import torch
from PIL import Image
import argparse

sys.path.append('..')
from utils import DataLoader
from fitness import MultiScore
from algorithm import NSGAII
from llm_service import LlamaService
from tqdm import tqdm

def main(args):
    output_dir = "jerry_pick"
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(path=args.annotation_path, img_dir=args.dataset_dir)

    llm = LlamaService(model_name="Llama-7b")
    system_prompt = (
        "You are an assistant that helps compare two answers for a QA task. "
        "One answer is the ground truth, and the other is the model's prediction.\n"
        "Return 'True' if the model answer matches one of the ground truth answers, otherwise return 'False'."
    )

    keyword_prompt = (
        "Extract 2 to 3 main visual keywords from the question below, such as 'claws', 'tail', 'horns', 'fur', etc.\n"
        "Do not include any animal names or species.\n"
        "Return only the keywords, separated by commas.\n\n"
        "Question: {question}"
    )

    fitness = MultiScore(reader_name="llava", retriever_name="clip")

    for i in tqdm(range(len(loader))):    
        question, answer, paths, gt_paths = loader.take_data(i)

        corpus = []
        for path in paths:
            try:
                image = Image.open(path).convert('RGB').resize((args.w, args.h))
                corpus.append(image)
            except:
                continue
        
        if not corpus:
            continue

        keyword_text = llm.text_to_text(
            system_prompt="You are a helpful assistant for extracting key concepts.",
            prompt=keyword_prompt.format(question=question)
        )[0].strip()
        print("keyword:", keyword_text)
        
        sim_scores = fitness.retriever(keyword_text, corpus)
        top1_img = corpus[sim_scores.argmax()]

        top1_answer = fitness.reader.image_to_text(question, [top1_img])
        result = top1_answer
        while result not in ["True", "False"]:
            result = llm.text_to_text(
                system_prompt=system_prompt,
                prompt=(
                    f"Question: {question}\n"
                    f"Ground Truth Answers: {answer}\n"
                    f"Model Answer: {top1_answer}"
                )
            )[0].strip()

        if result == "True":
            sample_dir = os.path.join(output_dir, str(i))
            os.makedirs(sample_dir, exist_ok=True)

            with open(os.path.join(sample_dir, "question.txt"), "w") as f:
                f.write(question)

            with open(os.path.join(sample_dir, "answer.txt"), "w") as f:
                for item in answer:
                    f.write(item + "\n")

            with open(os.path.join(sample_dir, "top1_answer.txt"), "w") as f:
                f.write(top1_answer)

            with open(os.path.join(sample_dir, "keywords.txt"), "w") as f:
                f.write(keyword_text)

            top1_img.save(os.path.join(sample_dir, "top1.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--w", type=int, default=312)
    parser.add_argument("--h", type=int, default=312)
    args = parser.parse_args()
    main(args)
