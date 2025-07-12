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

# --- CONFIGURATION ---
LOGO_URL = "logo.png"  # example logo URL
APP_TITLE = "🐛 Welcome to the Worm Count Interface"
DEFAULT_OUTPUT_DIR = "Worm_counter_output"

# --- PAGE SETUP ---
st.set_page_config(page_title=APP_TITLE, page_icon="🐛", layout="wide")

# --- SIDEBAR ---
st.sidebar.image(LOGO_URL, use_container_width=True)
st.sidebar.title("Settings")

output_dir_str = st.sidebar.text_input("Output folder path:", DEFAULT_OUTPUT_DIR)
output_dir = Path(output_dir_str)
output_dir.mkdir(parents=True, exist_ok=True)

clear_button = st.sidebar.button("🗑️ Clear Output Folder and Log")

# --- LOG FILE SETUP ---
log_file = output_dir / "worm_log.xlsx"
if not log_file.exists():
    pd.DataFrame(columns=["Image", "Worm Count"]).to_excel(log_file, index=False)

# --- CLEAR OUTPUT HANDLER ---
if clear_button:
    for file in output_dir.glob("*"):
        try:
            file.unlink()
        except Exception as e:
            st.sidebar.warning(f"Could not delete {file.name}: {e}")
    pd.DataFrame(columns=["Image", "Worm Count"]).to_excel(log_file, index=False)
    st.session_state.page = 0
    st.session_state.image_data = []
    st.session_state.uploaded_files = []
    st.sidebar.success("✅ Output folder and log file cleared.")

# --- MODEL LOADING ---
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

model = load_model()

# --- MAIN APP HEADER ---
st.title(APP_TITLE)
st.markdown("---")

# --- SESSION STATE INIT ---
if "image_data" not in st.session_state:
    st.session_state.image_data = []

if "page" not in st.session_state:
    st.session_state.page = 0

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- POPUP FUNCTION ---
def show_popup(message, duration=1.0):
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div style='
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: #fffae6;
            padding: 25px 35px;
            border: 3px solid #f0ad4e;
            border-radius: 15px;
            z-index: 9999;
            font-size: 1.3rem;
            font-weight: 600;
            box-shadow: 0 0 20px rgba(240, 173, 78, 0.6);
            text-align: center;
            color: #8a6d3b;
        '>
            {message}
        </div>
        """,
        unsafe_allow_html=True
    )
    time.sleep(duration)
    placeholder.empty()

# --- IMAGE UPLOADER ---
uploaded_files = st.file_uploader(
    "📤 Upload image(s) for worm detection and count",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Store uploaded files persistently in session state
if uploaded_files:
    existing_names = {f.name for f in st.session_state.uploaded_files}
    for f in uploaded_files:
        if f.name not in existing_names:
            st.session_state.uploaded_files.append(f)

# --- SIDEBAR BUTTON TO RUN DETECTION ON NEW IMAGES ---
count_button = st.sidebar.button("🐛 Count Worms on New Images")

if count_button:
    if not st.session_state.uploaded_files:
        st.sidebar.warning("Please upload images first.")
    else:
        if "image_data" not in st.session_state:
            st.session_state.image_data = []

        # Track processed image names to avoid reprocessing
        processed_names = {caption.split(" - ")[0] for _, caption in st.session_state.image_data}

        new_image_rows = []
        with st.spinner("Detecting worms in new images..."):
            for uploaded_file in reversed(st.session_state.uploaded_files):
                if uploaded_file.name in processed_names:
                    continue  # Skip already processed images

                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Save temporarily
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    temp_path = tmp.name
                    cv2.imwrite(temp_path, image_bgr)

                # Run detection
                results = model(temp_path)
                result = results[0]
                worm_count = len(result.boxes)

                # Draw detections
                img_with_detections = draw_detections(result)
                img_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)

                # Save output image
                orig_name = Path(uploaded_file.name).stem
                save_filename = f"detected_{orig_name}_{worm_count}.jpg"
                save_path = output_dir / save_filename
                cv2.imwrite(str(save_path), img_with_detections)

                # Log to Excel
                df_log = pd.read_excel(log_file)
                df_log.loc[len(df_log)] = [save_filename, worm_count]
                df_log.to_excel(log_file, index=False)

                # Show popup briefly
                show_popup(f"✅ {worm_count} worms detected in {uploaded_file.name}")

                # Collect new images for display
                new_image_rows.append((img_rgb, f"{uploaded_file.name} - Worms: {worm_count}"))

        # Append new detections to existing data
        st.session_state.image_data.extend(new_image_rows)
        st.session_state.page = 0

# --- PAGINATION DISPLAY ---
if st.session_state.image_data:
    per_page = 2
    total_pages = (len(st.session_state.image_data) + per_page - 1) // per_page

    col1, col2, col3 = st.columns([1, 2, 1])
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

            # Resize image to fixed width for consistent layout (e.g., 450 px)
            pil_img = Image.fromarray(img)
            base_width = 450
            wpercent = (base_width / float(pil_img.size[0]))
            hsize = int(pil_img.size[1] * wpercent)
            pil_img_resized = pil_img.resize((base_width, hsize), Image.LANCZOS)

            with cols[i]:
                st.image(pil_img_resized, caption=caption, use_container_width=True)
        else:
            with cols[i]:
                st.empty()

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "Made with ❤️ using [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io/)."
)

# --- SCROLL TO TOP AFTER PROCESSING ---
st.markdown("""
    <script>
        window.scrollTo({top: 0, behavior: 'smooth'});
    </script>
""", unsafe_allow_html=True)
