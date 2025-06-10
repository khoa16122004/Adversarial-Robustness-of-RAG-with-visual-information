import os
import json
import pickle

def main():
    output_path = "output"
    for file in os.listdir(output_path):
        if file.endswith(".pkl"):
            try:
                with open(os.path.join(output_path, file), "rb") as f:
                    data = pickle.load(f)
                    print(f"File: {file}")
                    print(f"Data: {data}")
                    print("-" * 50)
            except Exception as e:
                print(f"Lỗi khi đọc file {file}: {e}")
    for d in data:
        print(f"{d['gen']}: {d['pop'][:5]} ")
if __name__ == "__main__":
    main()