import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gradcam import make_gradcam_heatmap, overlay_heatmap

model = load_model("medical_model.h5")

st.title("Pneumonia Detection from X-ray Images")

file = st.file_uploader("Upload X-ray Image")

if file:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image")

    img_resized = cv2.resize(img, (128,128)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    pred = model.predict(img_input)[0][0]

    if pred > 0.5:
        st.error("🩺 Pneumonia Detected")
    else:
        st.success("✅ Normal")

    heatmap = make_gradcam_heatmap(img_input, model)
    result = overlay_heatmap((img_resized*255).astype("uint8"), heatmap)

    st.image(result, caption="Grad-CAM Visualization")