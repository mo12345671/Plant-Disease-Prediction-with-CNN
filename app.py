import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# ------------------------------------------------------
# Helper Function to Load Model and Classes
# ------------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(working_dir, "trained_model", "plant_disease_model.h5")
    classes_path = os.path.join(working_dir, "trained_model", "class_indices.json")

    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Please upload 'plant_disease_model.h5' in the 'trained_model' folder.")
        st.stop()
    if not os.path.exists(classes_path):
        st.error("❌ Class indices file not found. Please upload 'class_indices.json' in the 'trained_model' folder.")
        st.stop()

    model = tf.keras.models.load_model(model_path)
    with open(classes_path, "r") as f:
        index_to_class = json.load(f)
    return model, index_to_class


# ------------------------------------------------------
# Image Preprocessing
# ------------------------------------------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    if img_array.ndim == 2:  # grayscale -> RGB
        img_array = np.stack([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array


# ------------------------------------------------------
# Streamlit App
# ------------------------------------------------------
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="wide")

st.title("🌿 Plant Disease Prediction App")
st.write("Upload a leaf image to identify plant diseases using a trained CNN model.")

model, index_to_class = load_model_and_classes()

uploaded_image = st.file_uploader("📸 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Classify"):
            try:
                preprocessed_img = preprocess_image(image)
                prediction = model.predict(preprocessed_img)
                predicted_index = np.argmax(prediction, axis=1)[0]
                predicted_class = index_to_class[str(predicted_index)]
                confidence = np.max(prediction) * 100

                st.success(f"✅ Prediction: **{predicted_class}**")
                st.info(f"Confidence: **{confidence:.2f}%**")

            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")
