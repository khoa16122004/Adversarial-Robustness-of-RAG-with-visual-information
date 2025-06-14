import streamlit as st
from PIL import Image
import os
from utils import DataLoader

# Load data
loader = DataLoader(
    path='v1_anno.jsonl',
    img_dir='../extracted/train'
)

# Sidebar for sample and model selection
st.sidebar.title("⚙️ Configuration")
model_name = st.sidebar.selectbox("Model", ["clip", "git", "open_clip", "blip", "flava"])
sample_id = st.sidebar.number_input("Sample ID", min_value=0, max_value=len(loader.data) - 1, step=1)

# Load annotated sample
question, answer, paths, gt_paths = loader.take_data(sample_id)

# Display question and answer
st.markdown(f"### 🧠 Original Question:\n> {question}")
st.markdown(f"### ✅ Annotated Answer:\n> {answer}")

# Show corpus images
st.markdown("### 📚 Corpus Images")
cols = st.columns(5)
for i, path in enumerate(paths):
    with cols[i % 5]:
        if os.path.exists(path):
            img = Image.open(path).resize((256, 256))
            caption = "GT" if path in gt_paths else ""
            st.image(img, caption=caption, use_container_width=True)
        else:
            st.text("Missing")

# Show ground-truth images
if gt_paths:
    st.markdown("### 🎯 Ground Truth Images")
    gt_cols = st.columns(min(len(gt_paths), 5))
    for i, path in enumerate(gt_paths):
        with gt_cols[i % 5]:
            if os.path.exists(path):
                img = Image.open(path).resize((256, 256))
                st.image(img, caption="GT", use_container_width=True)
            else:
                st.text("Missing")
else:
    st.info("No GT image for this sample.")
