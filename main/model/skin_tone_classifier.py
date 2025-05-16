from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
import cv2
import numpy as np
import os 

class SkinToneClassifier:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "best_efficientnetv2s.h5")
        self.model = load_model(model_path)
        self.labels = ['dark', 'medium', 'light']

    def predict(self, face_img):
        if face_img is None:
            raise ValueError("Input image is None")

        # Resize image for EfficientNet
        img = cv2.resize(face_img, (300, 300))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_preprocessed = preprocess_input(img_rgb.astype(np.float32))
        img_input = np.expand_dims(img_preprocessed, axis=0)

        preds = self.model.predict(img_input, verbose=0)
        predicted_label = self.labels[np.argmax(preds)]
        confidence = float(np.max(preds))
        return predicted_label, confidence
