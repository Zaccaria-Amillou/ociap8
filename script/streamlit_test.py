import streamlit as st
from PIL import Image
import requests
import io

st.title('Image Segmentation')

img_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button('Predict'):
        files = {'image': img_file.getvalue()}
        response = requests.post('http://localhost:5000/segment/', files=files)

        if response.status_code == 200:
            img = Image.open(io.BytesIO(response.content))
            st.image(img, caption='Segmented Image.', use_column_width=True)
        else:
            st.write("Failed to segment the image.")