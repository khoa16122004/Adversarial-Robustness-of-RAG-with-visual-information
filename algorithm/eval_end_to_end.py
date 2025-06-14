import argparse
import os
import json
from tqdm import tqdm
from util import DataLoader, compute_nlg_metrics
from llm_service import LlamaService
from fitness import MultiScore
from pathlib import Path




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retri_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="llava")
    parser.add_argument("--retriever_name", type=str, default="clip")
    parser.add_argument("--llm_model", type=str, default="Llama-7b")
    parser.add_argument("--topks", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--save_dir", type=str, default="reader_result")
    args = parser.parse_args()

    loader = DataLoader(retri_dir=args.retri_dir)
    llm = LlamaService(model_name=args.llm_model)

    system_prompt = (
        "You are an assistant that helps compare two answers for a QA task. "
        "One answer is the ground truth, and the other is the model's prediction.\n"
        "JUST ONLY Return 'True' if the model answer matches one of the ground truth answers, otherwise return 'False'."
    )
    
    fitness = MultiScore(
        reader_name=args.model_name,
        retriever_name=args.retriever_name
    )

    for i in tqdm(range(len(loader))):
        question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(i)

        answers_by_topk = {}

        for k in args.topks:
            imgs_k = retri_imgs[:k]
            pred_answer = fitness.reader.image_to_text(question, imgs_k)
            result = ""
            while result not in ["True", "False"]:
                result = llm.text_to_text(
                    system_prompt=system_prompt,
                    prompt=(
                        f"Question: {question}\n"
                        f"Ground Truth Answers: {answer}\n"
                        f"Model Answer: {pred_answer}"
                    )
                )[0].strip()

            nlg_scores = compute_nlg_metrics(pred_answer, answer)

            answers_by_topk[f"top_{k}"] = {
                "model_answer": pred_answer,
                "gt_answers": answer,
                "BLEU": nlg_scores["BLEU"],
                "METEOR": nlg_scores["METEOR"],
                "ROUGE-1": nlg_scores["ROUGE-1"],
                "ROUGE-L": nlg_scores["ROUGE-L"],
                "end_to_end": result,
                # "BERTScore": nlg_scores["BERTScore"]
            }

        save_path = Path(args.save_dir) / args.llm_model / args.retriever_name / str(i)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "answers.json", "w") as f:
            json.dump({
                "question": question,
                "ground_truth": answer,
                "topk_results": answers_by_topk
            }, f, indent=2)

if __name__ == "__main__":
    main()