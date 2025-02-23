import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import os
import streamlit as st
import tensorflow as tf
import urllib.request

@st.cache_resource
def load_model():
    file_id = "1g9o_InQ2add--WVq1PmMT7whqjD6dumR"  # Replace with actual file ID from Google Drive
    model_path = "densenet_model.h5"  # Change to .h5
    model_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Download only if the model is not already present
    if not os.path.exists(model_path):
        st.write("Downloading model...")
        urllib.request.urlretrieve(model_url, model_path)

    # Check if the file exists before loading
    if not os.path.exists(model_path):
        st.error("Model file was not found after download.")
        return None

    # Load the model from .h5
    model = tf.keras.models.load_model(model_path, compile=False)
    return model
    
model = load_model()


# Streamlit app UI
st.title("🖼️ Image Classification with CNN")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image to match model input size
    img = image.resize((224, 224))  # Change size according to your model
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Display results
    st.write(f"**Predicted Class:** {predicted_class[0]}")  # Map class index to label if needed
