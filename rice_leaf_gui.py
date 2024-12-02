import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load your pretrained model
MODEL_PATH ="rice-disease-classifier.h5"
  # Replace with your actual .h5 file path
model = load_model(MODEL_PATH)

# Define class labels (update these with your actual class names)
class_labels = {0: "Leaf Blight", 1: "'Brown Spot",2:"'Leaf Smut"}  # Adjust based on your dataset

# Function to preprocess the image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)  # Resize the image
    image_array = img_to_array(image) / 255.0  # Convert to array and normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Streamlit app layout
st.title("Rice Disease Classifier")
st.write("Upload an image of a rice leaf to classify it as Healthy or Diseased.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = Image.open(uploaded_file)
    processed_image = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Display result
    st.write(f"Predicted Class: **{class_labels[predicted_class]}**")
