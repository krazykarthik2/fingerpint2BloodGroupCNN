import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Load the model
model = load_model("high_acc_model.h5")

# Streamlit App
st.title("Fingerprint Blood Group Classification")
st.write("Upload a fingerprint image to predict the blood group.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

# Classes (adjust based on your dataset)
classes = sorted(["A+","A-","AB+","AB-","B+","B-","O+","O-"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    
    # Preprocess the image
    img = load_img(uploaded_file, target_size=(64, 64))  # Resize to model input size
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.write(f"Predicted Blood Group: **{predicted_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
