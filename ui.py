import streamlit as st
from PIL import Image
import os
from utils import DataLoader
from rag.db import Database

# Load annotated data
loader = DataLoader(
    path='v1_anno.jsonl',
    img_dir='../extracted/train'
)

# Load database
db = Database(
    data_loader=loader,
    database_dir="database"
)

st.title("Visual-RAG Viewer")

# Choose a sample
sample_id = st.number_input("Sample ID", min_value=0, max_value=len(loader.data)-1, step=1)

# Load the selected sample
question, answer, _, gt_paths = loader.take_data(sample_id)

# Show question & answer
st.markdown(f"### 🧠 Question:\n**{question}**")
st.markdown(f"### ✅ Answer:\n**{answer}**")

# Search similar images
D, I = db.search_index([question], k=50)
retrieved_paths = db.get_image_paths(list(I))[0]

# Show retrieved images
st.markdown("### 🔍 Top-50 Retrieved Images")
cols = st.columns(5)
for i, path in enumerate(retrieved_paths):
    with cols[i % 5]:
        if os.path.exists(path):
            img = Image.open(path)
            caption = "GT" if path in gt_paths else ""
            st.image(img, caption=caption, use_column_width=True)
        else:
            st.text("Missing")

# Show GT images (separately)
if gt_paths:
    st.markdown("### 🎯 Ground Truth Images")
    gt_cols = st.columns(min(len(gt_paths), 5))
    for i, path in enumerate(gt_paths):
        with gt_cols[i % 5]:
            if os.path.exists(path):
                img = Image.open(path)
                st.image(img, caption="GT", use_column_width=True)
            else:
                st.text("Missing")
else:
    st.info("No GT image for this sample.")
