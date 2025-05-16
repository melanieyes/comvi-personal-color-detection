import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from utils.skin_cropper import SkinCropper

# Load model
model = load_model("C:\Users\MELANIE\PycharmProjects\computer_vision\PersonalColorReccommendation\model\best_efficientnetv2s.h5")

# Load and preprocess original image
img_path = "images/test1.jpg"
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Init skin croppers
cropper_cheek = SkinCropper(region="cheek", apply_mask=True)
cropper_forehead = SkinCropper(region="forehead", apply_mask=True)

# Crop regions
cheek_crop = cropper_cheek.crop(img_bgr)
forehead_crop = cropper_forehead.crop(img_bgr)

def predict_region(crop_img, region_name):
    if crop_img is None:
        print(f"No valid {region_name} region found.")
        return None, None
    crop_resized = cv2.resize(crop_img, (300, 300))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    input_tensor = preprocess_input(crop_rgb.astype(np.float32))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    pred = model.predict(input_tensor, verbose=0)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    label_map = {0: "dark", 1: "medium", 2: "light"}
    return label_map[class_idx], confidence, crop_rgb

# Predict cheek
cheek_label, cheek_conf, cheek_img = predict_region(cheek_crop, "cheek")
# Predict forehead
forehead_label, forehead_conf, forehead_img = predict_region(forehead_crop, "forehead")

# Show results
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

if cheek_img is not None:
    axs[0].imshow(cheek_img)
    axs[0].set_title(f"Cheek: {cheek_label} ({cheek_conf:.2%})")
    axs[0].axis("off")
else:
    axs[0].set_visible(False)

if forehead_img is not None:
    axs[1].imshow(forehead_img)
    axs[1].set_title(f"Forehead: {forehead_label} ({forehead_conf:.2%})")
    axs[1].axis("off")
else:
    axs[1].set_visible(False)

plt.tight_layout()
plt.show()
