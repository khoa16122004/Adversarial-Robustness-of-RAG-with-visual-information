import argparse
from util import DataLoader
from tqdm import tqdm
from llm_service import LlamaService
from fitness import MultiScore

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--retri_dir", type=str, required=True, help="Path to retrieval results")
    parser.add_argument("--model_name", type=str, default="blip", help="Name of the reader model")
    parser.add_argument("--retriever_name", type=str, default="clip", help="Name of the retriever model")
    parser.add_argument("--topks", type=int, nargs="+", default=[1, 2, 5], help="Top-k images to evaluate")

    args = parser.parse_args()

    loader = DataLoader(retri_dir=args.retri_dir)
    llm = LlamaService(model_name="Llama-7b")

    # Khởi tạo bộ đếm cho từng topk
    results = {k: {"e2e_acc": 0, "extract_match": 0, "clipscore": 0.0} for k in args.topks}

    system_prompt = (
        "You are an assistant that helps compare two answers for a QA task. "
        "One answer is the ground truth, and the other is the model's prediction.\n"
        "Return 'True' if the model answer matches one of the ground truth answers, otherwise return 'False'."
    )

    for i in tqdm(range(len(loader))):
        question, answer, query, gt_basenames, retri_basenames, retri_imgs = loader.take_data(i)

        fitness = MultiScore(
            reader=args.model_name,
            retriever=args.retriever_name
        )

        for k in args.topks:
            imgs_k = retri_imgs[:k]

            # Reader generates answer from top-k images
            model_answer = fitness.reader.image_to_text(question, imgs_k)

            # End-to-end evaluation using LLM
            result = ""
            while result not in ["True", "False"]:
                result = llm.text_to_text(
                    system_prompt=system_prompt,
                    prompt=(
                        f"Question: {question}\n"
                        f"Ground Truth Answers: {answer}\n"
                        f"Model Answer: {model_answer}"
                    )
                )[0].strip()

            if result == "True":
                results[k]["e2e_acc"] += 1

            # Extract match score
            if fitness.extract_match(model_answer, answer):
                results[k]["extract_match"] += 1

            # CLIPSCORE (similarity between query & top-k images)
            results[k]["clipscore"] += fitness.clipscore(query, imgs_k)

    total = len(loader)

    print("\n=== Evaluation Results ===")
    for k in args.topks:
        print(f"\nTop-{k}:")
        print(f"  End-to-End Accuracy: {results[k]['e2e_acc'] / total:.4f}")
        print(f"  Extract Match:        {results[k]['extract_match'] / total:.4f}")
        print(f"  Avg CLIPSCORE:        {results[k]['clipscore'] / total:.4f}")

if __name__ == "__main__":
    main()
