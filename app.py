import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# -------------------------------
# 🧠 Load Model and Class Labels
# -------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Load class indices
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Reverse mapping (optional)
id_to_class = {int(v): k for k, v in class_indices.items()}


# -------------------------------
# 🧩 Image Preprocessing Function
# -------------------------------
def load_and_preprocess_image(image, target_size=(224, 224)):
    img = Image.open(image).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -------------------------------
# 🔮 Prediction Function
# -------------------------------
def predict_image_class(model, image, id_to_class):
    img_array = load_and_preprocess_image(image)
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions) * 100
    predicted_label = id_to_class[predicted_idx]
    return predicted_label, confidence


# -------------------------------
# 🌿 Streamlit UI
# -------------------------------
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="wide")

st.title("🌱 Plant Disease Classification")
st.write("Upload an image of a plant leaf, and the model will predict the possible disease.")

uploaded_image = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_image)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        with st.spinner("Analyzing the image... 🔍"):
            predicted_label, confidence = predict_image_class(model, uploaded_image, id_to_class)

        st.success(f"✅ **Prediction:** {predicted_label}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")

        # Optional: Show model confidence as a progress bar
        st.progress(float(confidence) / 100.0)

        # Add a "Try another" button
        st.button("🔁 Try Another Image")
else:
    st.info("⬆️ Please upload a plant image to start classification.")
