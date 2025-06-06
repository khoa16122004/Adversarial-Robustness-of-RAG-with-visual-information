import streamlit as st
from PIL import Image
import os
from utils import DataLoader

loader = DataLoader(
    path='v1_anno.jsonl',
    img_dir='../extracted/train'
)

st.title("Visual-RAG Viewer")

sample_id = st.number_input("Sample ID", min_value=0, max_value=len(loader.data)-1, step=1)

question, answer, paths, gt_paths = loader.take_data(sample_id)

if question:
    st.markdown(f"### 🧠 Question\n**{question}**")
    st.markdown(f"### ✅ Answer\n**{answer}**")
    
    st.markdown("### 🖼️ Images (50 per page)")
    page = st.number_input("Page", min_value=0, max_value=(len(paths)-1)//50, step=1)
    start = page * 50
    end = start + 50
    cols = st.columns(5)

    for i, path in enumerate(paths[start:end]):
        with cols[i % 5]:
            if os.path.exists(path):
                img = Image.open(path)
                is_gt = path in gt_paths
                st.image(img, caption="GT" if is_gt else "", use_column_width=True)
            else:
                st.text("Missing")
else:
    st.error("Invalid sample ID.")