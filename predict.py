import cv2
import numpy as np
from tensorflow.keras.models import load_model
from gradcam import make_gradcam_heatmap, overlay_heatmap
import matplotlib.pyplot as plt

model = load_model("medical_model.h5")

img = cv2.imread("test.jpg")
img_resized = cv2.resize(img, (128,128)) / 255.0
img_input = np.expand_dims(img_resized, axis=0)

pred = model.predict(img_input)[0][0]

print("Prediction:", "Pneumonia" if pred > 0.5 else "Normal")

heatmap = make_gradcam_heatmap(img_input, model)
result = overlay_heatmap((img_resized*255).astype("uint8"), heatmap)

plt.imshow(result)
plt.axis("off")
plt.show()