import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# ------------------------
# Paths
# ------------------------
MODEL_FOLDER = "trained_model/plant_disease_model_saved"  # Use SavedModel folder if possible
CLASS_INDICES_PATH = "trained_model/class_indices.json"

# ------------------------
# Load model safely
# ------------------------
if os.path.exists(MODEL_FOLDER):
    model = tf.keras.models.load_model(MODEL_FOLDER)
    st.success("✅ Model loaded successfully.")
else:
    st.error(f"❌ Model not found at '{MODEL_FOLDER}'. Upload the SavedModel folder to 'trained_model/'.")
    st.stop()  # Stop execution if model is missing

# ------------------------
# Load class indices
# ------------------------
if os.path.exists(CLASS_INDICES_PATH):
    class_indices = json.load(open(CLASS_INDICES_PATH))
else:
    st.error(f"❌ Class indices file not found at '{CLASS_INDICES_PATH}'.")
    st.stop()

# ------------------------
# Image preprocessing
# ------------------------
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------
# Prediction
# ------------------------
def predict_image_class(model, image, class_indices):
    img_array = load_and_preprocess_image(image)
    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds, axis=1)[0])
    class_name = class_indices.get(str(class_idx), "Unknown")
    return class_name

# ------------------------
# Streamlit App
# ------------------------
st.title("🌿 Plant Disease Prediction App")
st.write("Upload a leaf image to identify plant diseases using the trained CNN model.")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        img = Image.open(uploaded_image)
        st.image(img.resize((200, 200)), caption="Uploaded Image")
    
    with col2:
        if st.button("Classify"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: **{prediction}**")
