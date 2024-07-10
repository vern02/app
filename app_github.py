import streamlit as st
import joblib
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

# Define the image size for the model
IMG_SIZE = (224, 224)

# Define the paths to the models
feature_extractor_path = "/Users/vernsin/mobilenetv2.h5"  # Replace with your actual path
svm_model_path = "/Users/vernsin/best_model.joblib"  # Replace with your actual path

# Load the feature extractor model
feature_extractor = tf.keras.models.load_model(feature_extractor_path)

# Load the SVM model
svm_model = joblib.load(svm_model_path)

# Define a function to preprocess an image
def preprocess_image(image):
    # Convert image to RGB if it has more than 3 channels
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to IMG_SIZE
    img = image.resize(IMG_SIZE)
    
    # Convert image to numpy array and normalize
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array


# Define a function to predict image class
def predict_image(image_array):
    # Extract features using the pretrained model
    features = feature_extractor.predict(image_array)
    features = features.reshape((1, -1))

    # Predict class using SVM model
    prediction = svm_model.predict(features)
    class_labels = ['Glass Bottle', 'Plastic Bottle', 'Tin Can']
    
    return class_labels[prediction[0]]

# Streamlit app with dropdown menu for image input
def main():
    st.title("Object Classification")

    # Dropdown to select image input method
    input_method = st.selectbox("Choose Image Input Method", ("Please Select", "Upload Image", "Predict from URL"))

    if input_method == "Upload Image":
        # File uploader widget
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Convert uploaded file to PIL Image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Preprocess the image
            img_array = preprocess_image(image)

            # Make predictions on the uploaded image
            if st.button("Predict"):
                predicted_class = predict_image(img_array)
                st.success(f"Predicted Class: {predicted_class}")

    elif input_method == "Predict from URL":
        # Input URL for image
        url = st.text_input("Enter Image URL")

        if url:
            try:
                # Fetch image from URL
                response = requests.get(url)
                if response.status_code == 200:
                    # Read image from response content
                    image = Image.open(BytesIO(response.content))
                    st.image(image, caption="Image from URL", use_column_width=True)

                    # Preprocess the image
                    img_array = preprocess_image(image)

                    # Predict button to trigger prediction
                    if st.button("Predict"):
                        predicted_class = predict_image(img_array)
                        st.success(f"Predicted Class: {predicted_class}")
                else:
                    st.error(f"Error: Unable to fetch image from URL. Status code: {response.status_code}")

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
