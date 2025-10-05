import os
import json
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# -----------------------------
# 🌿 CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered"
)

GOOGLE_DRIVE_ID = "1iY6LYkfADGUjlUkOblqQQJzX0VRN8Gi5"  # Your Google Drive model ID
MODEL_DIR = "trained_model"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.h5")
CLASS_JSON_PATH = os.path.join(MODEL_DIR, "class_indices.json")


# -----------------------------
# 📦 DOWNLOAD MODEL (if missing)
# -----------------------------
def download_model_from_gdrive():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        st.info("📥 Downloading model (~344 MB) from Google Drive... Please wait.")

        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        progress_bar = st.progress(0)
        downloaded = 0

        with open(MODEL_PATH, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                downloaded += len(data)
                progress_bar.progress(min(downloaded / total_size, 1.0))

        st.success("✅ Model downloaded successfully!")


# -----------------------------
# 🧠 LOAD MODEL + CLASS LABELS
# -----------------------------
@st.cache_resource
def load_model_and_classes():
    download_model_from_gdrive()

    st.info("🔄 Loading trained CNN model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load class indices if available
    if os.path.exists(CLASS_JSON_PATH):
        with open(CLASS_JSON_PATH, "r") as f:
            index_to_class = json.load(f)
    else:
        index_to_class = {}

    st.success("✅ Model loaded successfully!")
    return model, index_to_class


# -----------------------------
# 🧩 IMAGE PREPROCESSING
# -----------------------------
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -----------------------------
# 🔍 PREDICTION
# -----------------------------
def predict_image_class(model, image, index_to_class):
    preprocessed_img = load_and_preprocess_image(image)
    preds = model.predict(preprocessed_img)
    predicted_index = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds) * 100

    if str(predicted_index) in index_to_class:
        predicted_label = index_to_class[str(predicted_index)]
    else:
        predicted_label = f"Class {predicted_index}"

    return predicted_label, confidence


# -----------------------------
# 🌿 STREAMLIT APP UI
# -----------------------------
st.title("🌿 Plant Disease Prediction App")
st.markdown("Upload a leaf image to identify **plant diseases** using a trained CNN model.")

model, index_to_class = load_model_and_classes()

uploaded_image = st.file_uploader("📸 Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    if st.button("🔍 Classify"):
        with st.spinner("Analyzing image..."):
            label, confidence = predict_image_class(model, uploaded_image, index_to_class)
            st.success(f"🌱 **Prediction:** {label}")
            st.info(f"💡 Confidence: {confidence:.2f}%")

st.markdown("---")
st.caption("Developed by Mohamed Saleh 🌿 | Powered by TensorFlow & Streamlit")
