# app.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Upload Photo + Predict", page_icon="📷")

st.title("Do you wear a Mask? SO, How do you feel today? Prediction")
st.write("Take a photo or upload one, then run it through the CNN model and optionally save it")

# ---------- Settings ----------
MODEL_PATH_MASK = r'C:\Users\USER\Desktop\Study\ML_True\Project_3_CNN_2\full_project\mask_new_model3_cnn.keras'
MODEL_PATH_EMOTIONS = r'C:\Users\USER\Desktop\Study\ML_True\Project_3_CNN_2\full_project\emotions_model3_cnn_5.keras'
IMG_SIZE = (64, 64)  # must match training target_size
#THRESHOLD = 0.5

# Folder where images will be saved (on the computer/server)
SAVE_DIR = Path("saved_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

st.session_state.setdefault('without_mask',None)
st.session_state.setdefault('mode',None)


@st.cache_resource
def get_model_mask(p):
    return load_model(p)

@st.cache_resource
def get_model_emotions(p):
    return load_model(p)

mask_model = get_model_mask(MODEL_PATH_MASK)
emotions_model = get_model_emotions(MODEL_PATH_EMOTIONS)


def save_uploaded_image(file_obj, prefix: str = "img") -> Path:
    original_name = getattr(file_obj, "name", "") or ""
    ext = Path(original_name).suffix.lower() if Path(original_name).suffix else ".jpg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique}{ext}"
    out_path = SAVE_DIR / filename

    data = file_obj.getvalue()
    out_path.write_bytes(data)
    return out_path

def preprocess_for_model(uploaded_file) -> np.ndarray:
    """
    Converts Streamlit uploaded image to model-ready numpy array:
    shape: (1, 64, 64, 3), dtype float32, scaled to [0,1]
    """
    # Streamlit UploadedFile -> bytes -> PIL Image
    img = Image.open(uploaded_file)

    # Ensure 3 channels (RGB). This also handles PNG with alpha (RGBA)
    img = img.convert("RGB")

    # Resize to model expected input
    img = img.resize(IMG_SIZE)

    # PIL -> np array
    arr = np.array(img, dtype=np.float32)

    # Normalize
    arr /= 255.0

    # Add batch dimension
    arr = np.expand_dims(arr, axis=0)  # (1, 64, 64, 3)
    return arr

def prediction_for_cnn(x1,model1,mode):
    if mode is None:
        return None, None


    emotions_classes = ['angry', 'happy', 'neutral', 'sad', 'surprise']
    mask_classes = ['incorrect mask','with mask','without mask']

    emotions_conf = 0.20
    mask_conf = 0.33

    classes = mask_classes if mode == 'mask' else emotions_classes

    result = model1.predict(x1)

    argmax = np.argmax(result[0])
    acc = result[0][argmax]
    cl = classes[argmax]

    if mode == 'mask':
        if acc <= mask_conf + 0.05:
            cl = 'UnKnown'
    else:
        if acc <= emotions_conf + 0.05:
            cl = 'UnKnown'


    return cl,acc


# ---------- UI ----------
st.subheader("1) Take a picture (phone camera)")
camera_photo = st.camera_input("Open camera and take a photo")

st.subheader("2) Or upload from gallery")
gallery_photo = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

chosen = camera_photo if camera_photo is not None else gallery_photo

if chosen is not None:
    st.image(chosen, caption="Preview", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧠 Predict Mask Or No (or false mask)"):
            st.session_state['mode'] = 'mask'
            try:
                x_mask = preprocess_for_model(chosen)
                mode = st.session_state['mode']
                cl_pred,cl_conf = prediction_for_cnn(x_mask,mask_model,mode)

                if cl_pred == 'without mask':
                    st.session_state['without_mask'] = True

                st.info(f"From Class {cl_pred}\nAccuracy = {cl_conf*100:.2f}% ")

                # Debug: show the shape to prove it's correct
                st.caption(f"Sent to model with shape: {x_mask.shape} (should be (1, 64, 64, 3))")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        if st.button("💾 Save image to computer"):
            try:
                saved_path = save_uploaded_image(chosen, prefix="phone")
                st.success(f"Saved ✅  {saved_path.resolve()}")
                st.info("The image was saved on the computer running Streamlit (server)")
            except Exception as e:
                st.error(f"Failed to save: {e}")

    if st.session_state['without_mask'] is None:
        st.info(f'For emotion diagnose first we have to see you dont wear a mask\nPush The predict Mask button')
        st.stop()

    if st.session_state['without_mask'] is True:
        if st.button("Tell me How I Feel"):
                st.session_state['mode'] = 'emotions'
                try:
                    x_feel = preprocess_for_model(chosen)
                    mode = st.session_state['mode']
                    cl_pred, cl_conf = prediction_for_cnn(x_feel, emotions_model, mode)

                    if cl_pred and cl_conf is not None:
                        st.session_state['mode'] = None
                        st.session_state['without_mask'] = None

                    st.info(f"From Class {cl_pred}\nConfidence = {cl_conf * 100:.2f}%")

                except Exception as e:
                    st.error(f"Failed to Predict: {e}")




st.caption(f"Images will be saved to: {SAVE_DIR.resolve()}")
st.caption(f"Model Mask Catcher file: {Path(MODEL_PATH_MASK).resolve()}")
st.caption(f"Model Feeling Recognization file: {Path(MODEL_PATH_EMOTIONS).resolve()}")