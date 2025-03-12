import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

import os
import streamlit as st
import tensorflow as tf
import urllib.request

# Class labels (ensure this corresponds to the order of your model's classes. in my case, it's insects)
class_labels = ['Butterfly', 'Dragonfly', 'Grasshopper', 'Ladybird', 'Mosquito']

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

# Streamlit explanation of model
st.write("This is a CNN model that classifies insects. The relevant groups are: 'Butterfly', 'Dragonfly', 'Grasshopper', 'Ladybird', and 'Mosquito'.")

# File uploader for images
uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image to match model input size
    img = image.resize((224, 224))  
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0) 

    # Make predictions
    predictions = model.predict(img_array)

    
    
    # Get the predicted class index and its corresponding probability
    predicted_class_idx = np.argmax(predictions, axis=1)
    predicted_class_prob = predictions[0][predicted_class_idx]  # Probability of predicted class

    # Map the numerical prediction index to the actual class label
    predicted_class_label = class_labels[predicted_class_idx[0]]

    # Calculate the percentage likelihood
    predicted_class_percentage = predicted_class_prob[0] * 100  # Convert to percentage

    # Display results
    st.write(f"**Predicted Class:** {predicted_class_label}")
    st.write(f"**Confidence:** {predicted_class_percentage:.2f}%")
