import logging

import streamlit as st
from PIL import Image
from torchvision import transforms
from model_utils import get_similarities
normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

st.title("Similarity")

file_up1 = st.file_uploader("Upload image 1", type="jpg")
if file_up1 is not None:
    image1 = Image.open(file_up1)
    st.image(image1, caption='Uploaded Image 1', use_column_width=False)


file_up2 = st.file_uploader("Upload image 2", type="jpg")
if file_up2 is not None:
    image2 = Image.open(file_up2)
    st.image(image2, caption='Uploaded Image 2', use_column_width=False)

if file_up1 is not None and file_up2 is not None:
    weights = r"C:\Users\shiri\Documents\School\Master\Galit\epoch_60_weights.pt"
    # weights = "/home/Shirialmog/mysite/models/epoch_60_weights.pt"
    result = get_similarities(image1, image2)
    st.write(f"Similarity: {result}")


