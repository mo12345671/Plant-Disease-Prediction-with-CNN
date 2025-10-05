import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown
import zipfile

# ---------------------------
# Configuration
# ---------------------------
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

# Google Drive file ID of the SavedModel zip
GOOGLE_DRIVE_ID = "YOUR_SAVEDMODEL_ZIP_FILE_ID"  # Replace this
MODEL_DIR = "trained_model"
MODEL_ZIP_PATH = os.path.join(MODEL_DIR, "plant_disease_model_saved.zip")
MODEL_FOLDER_PATH = os.path.join(MODEL_DIR, "plant_disease_model_saved")
CLASS_JSON_PATH = os.path.join(MODEL_DIR, "class_indices.json")

# ---------------------------
# Download & extract model
# ---------------------------
def download_and_extract_model():
    if os.path.exists(MODEL_FOLDER_PATH):
        st.success("✅ Model already exists — skipping download.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)
    st.info("📥 Downloading model from Google Drive...")

    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
    try:
        gdown.download(url, MODEL_ZIP_PATH, quiet=False, fuzzy=True)
    except Exception as e:
        st.error(f"❌ Model download failed: {e}")
        return

    if os.path.exists(MODEL_ZIP_PATH):
        with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        st.success("✅ Model downloaded and extracted successfully!")
    else:
        st.error("❌ Model zip not found after download!")

# ---------------------------
# Load model & class indices
# ---------------------------
@st.cache_resource
def load_model_and_classes():
    download_and_extract_model()

    st.info("🔄 Loading trained CNN model...")
    model = tf.keras.models.load_model(MODEL_FOLDER_PATH)

    if os.path.exists(CLASS_JSON_PATH):
        with open(CLASS_JSON_PATH, "r") as f:
            index_to_class = json.load(f)
    else:
        index_to_class = {}
        st.warning("⚠️ class_indices.json not found. Labels may not display correctly.")

    st.success("✅ Model loaded successfully!")
    return model, index_to_class

# ---------------------------
# Image preprocessing
# ---------------------------
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------------------
# Prediction
# ---------------------------
def predict_image_class(model, image, index_to_class):
    preprocessed_img = load_and_preprocess_image(image)
    preds = model.predict(preprocessed_img)
    predicted_index = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    predicted_label = index_to_class.get(str(predicted_index), f"Class {predicted_index}")
    return predicted_label, confidence

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌿 Plant Disease Prediction App")
st.markdown(
    "Upload a **leaf image** to identify plant diseases using a trained CNN model. "
    "The model will be downloaded automatically if not found."
)

model, index_to_class = load_model_and_classes()

uploaded_image = st.file_uploader("📸 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify"):
        with st.spinner("Analyzing image..."):
            label, confidence = predict_image_class(model, uploaded_image, index_to_class)
            st.success(f"🌱 Prediction: {label}")
            st.info(f"💡 Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption("Developed by Mohamed Saleh 🌿 | Powered by TensorFlow & Streamlit")
