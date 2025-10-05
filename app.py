import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
import requests

MODEL_URL = "https://huggingface.co/your_username/your_model_repo/resolve/main/plant_disease_model.h5"

def download_model_if_missing():
    os.makedirs("trained_model", exist_ok=True)
    model_path = "trained_model/plant_disease_model.h5"
    if not os.path.exists(model_path):
        st.info("📦 Downloading model, please wait...")
        response = requests.get(MODEL_URL, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("✅ Model downloaded successfully!")
    return model_path

@st.cache_resource
def load_model_and_classes():
    model_path = download_model_if_missing()
    with open("trained_model/class_indices.json") as f:
        index_to_class = json.load(f)
    model = tf.keras.models.load_model(model_path)
    return model, index_to_class
