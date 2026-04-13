import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('dataset/train', target_size=(128,128), batch_size=32, class_mode='binary')
test_data = test_datagen.flow_from_directory('dataset/test', target_size=(128,128), batch_size=32, class_mode='binary')

model = create_model()

history = model.fit(train_data, validation_data=test_data, epochs=10)

model.save("medical_model.h5")

# Accuracy Plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.legend(["Train","Validation"])
plt.show()

# Predictions
Y_true = test_data.classes
Y_pred = model.predict(test_data)
Y_pred = (Y_pred > 0.5).astype(int).flatten()

# Confusion Matrix
cm = confusion_matrix(Y_true, Y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Report
print(classification_report(Y_true, Y_pred))