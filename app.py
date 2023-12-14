import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
# Load the trained model
model = load_model("model.h5")

# Class mappings
class_mappings = {
    0: 'It is a Building',
    1: 'It is a Forest',
    2: 'It is a Glacier',
    3: 'It is a Mountain',
    4: 'It is a Sea',
    5: 'It is a Street'
}

st.title("Image Classification with CNN(Convolutional Neural Network)")
st.header("Classify the images of six different classes using CNN model")
st.text("Upload a picture of building, forest, glacier, mountain, sea or street and the model will predict the class of the image")

st.text("Name: Vishesh Phutela")
st.text("Roll No: 202066")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    st.image(img, caption="Uploaded Image.", use_column_width=True)
    st.write("Predicted Class:")
    st.write(class_mappings[predicted_class_index])
