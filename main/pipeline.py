import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import joblib

from utils.skin_cropper import SkinCropper
from utils.face_detection import FaceDetector
from utils.color_palette import ColorPaletteRecommender

# === Load assets ===
model = load_model("model/best_efficientnetv2s.h5")
label_encoder = joblib.load("model/label_encoder.pkl")
face_detector = FaceDetector()
skin_cropper = SkinCropper(region="cheek", apply_mask=True)
palette_recommender = ColorPaletteRecommender()

# === Main processing function ===
def predict_skin_tone_pipeline(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image.")

    # Detect and crop face
    face_crop, _ = face_detector.detect_from_image(image)

    # Crop cheek region with HSV masking
    skin_patch = skin_cropper.crop(face_crop)
    if skin_patch is None or skin_patch.size == 0:
        raise ValueError("Failed to extract skin region.")

    # Preprocess for EfficientNetV2
    skin_patch = cv2.resize(skin_patch, (300, 300))
    skin_patch = cv2.cvtColor(skin_patch, cv2.COLOR_BGR2RGB)
    skin_patch = img_to_array(skin_patch).astype(np.float32)
    skin_patch = preprocess_input(skin_patch)
    skin_patch = np.expand_dims(skin_patch, axis=0)

    # Predict
    preds = model.predict(skin_patch, verbose=0)
    class_idx = np.argmax(preds)
    tone_label = label_encoder.inverse_transform([class_idx])[0]
    confidence = float(np.max(preds))
    palette = palette_recommender.recommend(tone_label)

    return {"skin_tone": tone_label, "confidence": confidence, "palette": palette}