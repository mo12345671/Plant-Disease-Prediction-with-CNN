import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# -------------------------------
# 🧠 App Configuration
# -------------------------------
st.set_page_config(
    page_title="🌿 Plant Disease Classifier",
    page_icon="🌱",
    layout="centered"
)

# -------------------------------
# 📂 Paths
# -------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir,  "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(data, "class_indices.json")

# -------------------------------
# ⚙️ Load Model and Classes
# -------------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_classes():
    model = tf.keras.models.load_model(model_path)
    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)
    return model, class_indices

model, class_indices = load_model_and_classes()

# -------------------------------
# 🖼️ Image Preprocessing
# -------------------------------
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# 🔍 Prediction
# -------------------------------
def predict_image_class(image_file):
    img_array = load_and_preprocess_image(image_file)
    predictions = model.predict(img_array)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    predicted_class_name = class_indices[str(predicted_class_index)]
    confidence = float(np.max(predictions)) * 100
    return predicted_class_name, confidence

# -------------------------------
# 🎨 UI
# -------------------------------
st.title("🌿 Plant Disease Classifier")
st.markdown(
    """
    Upload a leaf image 🌱 and let the model detect whether it's healthy or diseased.  
    The model has been trained using a Convolutional Neural Network (CNN) for plant disease detection.
    """
)

uploaded_image = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("🔍 Classify"):
        with st.spinner("Analyzing the image... ⏳"):
            predicted_class, confidence = predict_image_class(uploaded_image)
        st.success(f"✅ **Prediction:** {predicted_class}")
        st.info(f"🎯 **Confidence:** {confidence:.2f}%")

# -------------------------------
# 📘 Footer
# -------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using TensorFlow & Streamlit | © 2025 Plant Disease Detection Project")
