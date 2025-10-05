import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown  # to download from Google Drive

# Working directory
working_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = f"{working_dir}/trained_model"
model_path = f"{model_dir}/plant_disease_prediction_model.h5"

# Google Drive file link (change this)
MODEL_DRIVE_LINK = "https://drive.google.com/uc?id=YOUR_FILE_ID"

# Download model if not found
if not os.path.exists(model_path):
    os.makedirs(model_dir, exist_ok=True)
    with st.spinner("Downloading model, please wait..."):
        gdown.download(MODEL_DRIVE_LINK, model_path, quiet=False)

# Load the model
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image)
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image_class(model, image, class_indices):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    return class_indices[str(predicted_class_index)]

# Streamlit UI
st.title("🌿 Plant Disease Classifier")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((150, 150)))

    with col2:
        if st.button("Classify"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: {prediction}")
