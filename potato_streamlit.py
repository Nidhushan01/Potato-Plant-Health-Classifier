import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model(r"C:\Users\ASUS\Desktop\projects\potato\training\modele15.h5")

# Define class names
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Streamlit application
st.title("Potato Plant Health Classifier")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    # Predict the class
    prediction = model.predict(image)
    predicted_label = class_names[np.argmax(prediction)]

    # Display the prediction
    st.write(f"Predicted Label: {predicted_label}")
    
    # Plot the image with the prediction
    fig, ax = plt.subplots()
    ax.imshow(image[0])
    ax.set_title(f"Predicted Label: {predicted_label}")
    st.pyplot(fig)
