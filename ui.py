import streamlit as st
from PIL import Image
import os
from utils import DataLoader
from rag.db import Database

# Step 1: Select model
st.title("Visual-RAG Viewer")

model_name = st.selectbox("Choose Model", ["clip", "git", "open_clip", "blip", "flava"])

# Load visual-language model
if model_name == 'clip':
    from vl_models import CLIPModel
    vs_model = CLIPModel()
elif model_name == 'git':
    from vl_models import GITModel
    vs_model = GITModel()
elif model_name == 'open_clip':
    from vl_models import OpenCLIPModel
    vs_model = OpenCLIPModel()
elif model_name == 'blip':
    from vl_models import BLIPModel
    vs_model = BLIPModel()
elif model_name == 'flava':
    from vl_models import FLAVAModel
    vs_model = FLAVAModel()
else:
    st.error(f"Unknown model name: {model_name}")
    st.stop()

# Step 2: Load data
loader = DataLoader(
    path='v1_anno.jsonl',
    img_dir='../extracted/train'
)

# Step 3: Load database
db = Database(
    data_loader=loader,
    database_dir="database"
)

db.read_db(qs_id=None, vs_model=vs_model)  # Load index with selected model

# Step 4: Select sample
sample_id = st.number_input("Sample ID", min_value=0, max_value=len(loader.data)-1, step=1)

# Step 5: Load sample
question, answer, _, gt_paths = loader.take_data(sample_id)

# Step 6: Show Q&A
st.markdown(f"### 🧠 Question:\n**{question}**")
st.markdown(f"### ✅ Answer:\n**{answer}**")

# Step 7: Search similar images using selected model
D, I = db.search_index([question], k=50)
retrieved_paths = db.get_image_paths(list(I))[0]

# Step 8: Show retrieved images
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

# Step 9: Show ground truth images separately
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
