import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained CNN model (update the path as needed)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("your_model.h5")  # Replace with your model file
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
