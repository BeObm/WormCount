import streamlit as st
from ultralytics import YOLO
from utils import draw_detections
import tempfile
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
import time
import platform
import subprocess
import os

# --- CONFIGURATION ---
LOGO_URL = "logo.png"
APP_TITLE = "🐛 Welcome to the Worm Counter Interface"
DEFAULT_OUTPUT_DIR = "Worm_counter_output"

# --- PAGE SETUP ---
st.set_page_config(page_title=APP_TITLE, page_icon="🐛", layout="wide")

# --- HELPER FUNCTIONS ---
def open_folder(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as e:
        st.sidebar.error(f"Could not open folder: {e}")

def show_popup(message, duration=1.0):
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div style='position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background-color: #fffae6;
        padding: 25px 35px; border: 3px solid #f0ad4e; border-radius: 15px; z-index: 9999; font-size: 1.3rem;
        font-weight: 600; box-shadow: 0 0 20px rgba(240, 173, 78, 0.6); text-align: center; color: #8a6d3b;'>
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(duration)
    placeholder.empty()

def setup_sidebar(default_output_dir: str) -> Path:
    st.sidebar.image(LOGO_URL, use_container_width=True)
    st.sidebar.title("Settings")

    output_dir = st.sidebar.text_input("Output folder path:", default_output_dir)
    output_dir = Path(output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if st.sidebar.button("📂 Open Output Folder"):
        open_folder(str(output_dir.resolve()))

    if st.sidebar.button("🗑️ Clear Output Folder and Log"):
        for file in output_dir.glob("*"):
            try:
                file.unlink()
            except Exception as e:
                st.sidebar.warning(f"Could not delete {file.name}: {e}")
        pd.DataFrame(columns=["Image", "Worm Count"]).to_excel(output_dir / "worm_log.xlsx", index=False)
        st.session_state.page = 0
        st.session_state.image_data = []
        st.sidebar.success("✅ Output folder and log file cleared.")

    return output_dir

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

def detect_and_log_images(model, uploaded_files, output_dir: Path, log_file: Path):
    image_rows = []
    df_log = pd.read_excel(log_file)
    existing_names = df_log["Image"].str.extract(r"detected_(.*)_\d+\.jpg")[0].tolist()

    with st.spinner("Detecting worms in new images..."):
        for uploaded_file in reversed(uploaded_files):
            orig_name = Path(uploaded_file.name).stem
            if orig_name in existing_names:
                continue

            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, image_bgr)

            results = model(temp_path)
            result = results[0]
            worm_count = len(result.boxes)

            img_with_detections = draw_detections(result)
            img_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)

            save_filename = f"detected_{orig_name}_{worm_count}.jpg"
            save_path = output_dir / save_filename
            cv2.imwrite(str(save_path), img_with_detections)

            df_log.loc[len(df_log)] = [save_filename, worm_count]
            df_log.to_excel(log_file, index=False)

            show_popup(f"✅ {worm_count} worms detected in {uploaded_file.name}")
            image_rows.append((img_rgb, f"{uploaded_file.name} - Worms: {worm_count}"))

    return image_rows

def show_pagination():
    per_page = 2
    total_pages = (len(st.session_state.image_data) + per_page - 1) // per_page

    col1, _, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Prev") and st.session_state.page > 0:
            st.session_state.page -= 1
    with col3:
        if st.button("Next ➡️") and st.session_state.page < total_pages - 1:
            st.session_state.page += 1

    start = st.session_state.page * per_page
    end = start + per_page
    current_page_images = st.session_state.image_data[start:end]

    cols = st.columns(2)
    for i in range(per_page):
        if i < len(current_page_images):
            img, caption = current_page_images[i]
            pil_img = Image.fromarray(img)
            base_width = 450
            wpercent = base_width / float(pil_img.size[0])
            hsize = int(pil_img.size[1] * wpercent)
            pil_img_resized = pil_img.resize((base_width, hsize), Image.LANCZOS)
            with cols[i]:
                st.image(pil_img_resized, caption=caption, use_container_width=True)
        else:
            with cols[i]:
                st.empty()

# --- MAIN SCRIPT ---
st.title(APP_TITLE)
st.markdown("---")

output_dir = setup_sidebar(DEFAULT_OUTPUT_DIR)
log_file = output_dir / "worm_log.xlsx"
if not log_file.exists():
    pd.DataFrame(columns=["Image", "Worm Count"]).to_excel(log_file, index=False)

if "image_data" not in st.session_state:
    st.session_state.image_data = []
if "page" not in st.session_state:
    st.session_state.page = 0

model = load_model()

uploaded_files = st.file_uploader(
    "📄 Upload image(s) for worm detection and count",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if st.sidebar.button("🔢 Detect Worms in New Images"):
    if uploaded_files:
        new_data = detect_and_log_images(model, uploaded_files, output_dir, log_file)
        st.session_state.image_data.extend(new_data)
        st.session_state.page = 0
    else:
        st.warning("Please upload images first.")

if st.session_state.image_data:
    show_pagination()

# --- FOOTER ---
st.markdown("---")
st.markdown("Made with ❤️ using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io/).")
st.markdown("""<script>window.scrollTo({top: 0, behavior: 'smooth'});</script>""", unsafe_allow_html=True)
