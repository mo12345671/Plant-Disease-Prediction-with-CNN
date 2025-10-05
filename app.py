import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st

# -------------------------------
# 🌿 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🌱 Plant Disease Classifier",
    page_icon="🍃",
    layout="centered"
)

# -------------------------------
# 📂 Load Model and Classes
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_classes():
    """Load the trained CNN model and class indices."""
    working_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
    classes_path = os.path.join(working_dir, "class_indices.json")

    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load class names
    with open(classes_path, "r") as f:
        class_indices = json.load(f)

    # Reverse mapping from index → class name
    index_to_class = {v: k for k, v in class_indices.items()}
    return model, index_to_class


model, index_to_class = load_model_and_classes()

# -------------------------------
# 🖼️ Image Preprocessing
# -------------------------------
def load_and_preprocess_image(image_file, target_size=(244, 244)):
    """Preprocess uploaded image for model prediction."""
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -------------------------------
# 🔍 Prediction Function
# -------------------------------
def predict_image_class(image_file):
    """Run inference and return prediction + confidence."""
    img_array = load_and_preprocess_image(image_file)
    predictions = model.predict(img_array)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    predicted_class_name = index_to_class[predicted_class_index]
    confidence = float(np.max(predictions)) * 100
    return predicted_class_name, confidence


# -------------------------------
# 🎨 Streamlit App UI
# -------------------------------
st.title("🌿 Plant Disease Classifier")
st.markdown(
    """
    Upload a **leaf image** 🍃 and this app will use a Convolutional Neural Network (CNN)  
    to detect the **type of plant disease** or whether the plant is **healthy**.
    """
)

uploaded_image = st.file_uploader("📸 Upload a plant image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", width=250)

    with col2:
        if st.button("🔍 Classify"):
            with st.spinner("Analyzing the image... Please wait ⏳"):
                predicted_class, confidence = predict_image_class(uploaded_image)

            st.success(f"✅ **Prediction:** {predicted_class}")
            st.info(f"🎯 **Confidence:** {confidence:.2f}%")

# -------------------------------
# 📘 Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by Mohamed Saleh | Powered by TensorFlow & Streamlit 🌱")
