import streamlit as st
from PIL import Image
import os
from utils import DataLoader
from rag.db import Database

# Load data
loader = DataLoader(
    path='v1_anno.jsonl',
    img_dir='../extracted/train'
)

# Sidebar: chọn model và sample ID
st.sidebar.title("⚙️ Configuration")
model_name = st.sidebar.selectbox("Model", ["clip", "git", "open_clip", "blip", "flava"])
sample_id = st.sidebar.number_input("Sample ID", min_value=0, max_value=len(loader.data) - 1, step=1)

# Load model
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

# Tạo database instance
db = Database(
    data_loader=loader,
    database_dir="database",
)

# Load annotated sample
question, answer, _, gt_paths = loader.take_data(sample_id)

# Read database embedding cho sample_id
db.read_db(qs_id=sample_id, vs_model=vs_model)

# Hiện câu hỏi, câu trả lời từ file gốc
st.markdown(f"### 🧠 Original Question:\n> {question}")
st.markdown(f"### ✅ Annotated Answer:\n> {answer}")

# Nhập query mới
query = st.text_input("🔍 Enter your custom query", value=question)

# Search ảnh từ query
if query:
    D, I = db.search_index([query], k=50)
    retrieved_paths = db.get_image_paths(list(I))[0]

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

    # Hiện GT images riêng
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
else:
    st.info("Please enter a query to search.")
