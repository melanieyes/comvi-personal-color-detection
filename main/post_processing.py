# from skimage.feature import hog
# import cv2
# import numpy as np
# import time
# import os
# from glob import glob
# # from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# import joblib
# from utils.skin_cropper import SkinCropper
# from pipeline import predict_skin_tone_pipeline


# def run_postprocessing(test_dir="results"):
#     # Load models
#     knn_model = joblib.load("model/knn_hog.pkl")
#     xgb_model = joblib.load("model/xgb_hog.pkl")
#     label_encoder = joblib.load("model/label_encoder.pkl")
#     skin_cropper = SkinCropper(region="cheek", apply_mask=True)

#     # Set parameters
#     IMG_SIZE = (64, 64)  # Match training size
#     y_true, knn_preds, xgb_preds, effnet_preds = [], [], [], []
#     times = {"knn": [], "xgb": [], "effnet": []}

#     print("\n--- Running Batch Inference on Saved Faces ---\n")

#     for path in glob(f"{test_dir}/*.jpg"):
#         basename = os.path.basename(path)
#         true_label = basename.split("_")[0].lower()

#         try:
#             img = cv2.imread(path)
#             skin_patch = skin_cropper.crop(img)

#             if skin_patch is None or skin_patch.size == 0:
#                 raise ValueError("Invalid crop")

#             resized = cv2.resize(skin_patch, IMG_SIZE)
#             hog_feat = hog(
#                 cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY),
#                 pixels_per_cell=(8, 8),
#                 cells_per_block=(2, 2),
#                 block_norm="L2-Hys"
#             ).reshape(1, -1)


#             knn_label, xgb_label, effnet_label = None, None, None

#             valid_prediction = False

#             # KNN
#             try:
#                 knn_idx = knn_model.predict(hog_feat)[0]
#                 knn_label = label_encoder.inverse_transform([knn_idx])[0]
#                 knn_preds.append(knn_label)
#                 valid_prediction = True
#             except Exception as e:
#                 print(f"{basename:25s}  KNN error: {e}")

#             # XGBoost
#             try:
#                 xgb_idx = xgb_model.predict(hog_feat)[0]
#                 xgb_label = label_encoder.inverse_transform([xgb_idx])[0]
#                 xgb_preds.append(xgb_label)
#                 valid_prediction = True
#             except Exception as e:
#                 print(f"{basename:25s}  XGB error: {e}")

#             # EfficientNet
#             try:
#                 result = predict_skin_tone_pipeline(path)
#                 effnet_label = result["skin_tone"]
#                 effnet_preds.append(effnet_label)
#                 valid_prediction = True
#             except Exception as e:
#                 print(f"{basename:25s}  EffNet error: {e}")

#             # Append y_true only if any succeeded
#             if valid_prediction:
#                 y_true.append("unknown")  # or actual label if known

#         except Exception as e:
#             print(f"{basename:25s}  General error: {e}")


#     def assess_model_bias(model_name, predictions):
#         if not predictions:
#             print(f"\n=== {model_name} Bias Summary ===")
#             print("No predictions available.")
#             return

#         print(f"\n=== {model_name} Bias Summary ===")
#         total = len(predictions)
#         counts = {label: predictions.count(label) for label in ['dark', 'medium', 'light']}

#         for tone, count in counts.items():
#             pct = (count / total) * 100
#             print(f"{tone.capitalize():>7}: {count:>3} images ({pct:>5.1f}%)")

#         # Simple heuristic: if one class > 50%, consider it biased
#         most_common = max(counts, key=counts.get)
#         if counts[most_common] / total > 0.5:
#             print(f"⚠️  Potential bias: Model leans towards '{most_common}' class.")
#         else:
#             print("✅  Fair distribution across tone classes.")

#     assess_model_bias("HOG + KNN", knn_preds)
#     assess_model_bias("HOG + XGBoost", xgb_preds)
#     assess_model_bias("EfficientNet", effnet_preds)

#     # print("\nAverage inference time (seconds):")
#     # for k in times:
#     #     if times[k]:
#     #         print(f"{k}: {np.mean(times[k]):.4f}")

#     return knn_preds, xgb_preds, effnet_preds


import cv2
import numpy as np
import os
from glob import glob
import joblib
from collections import Counter
from skimage.feature import hog
from utils.skin_cropper import SkinCropper
from pipeline import predict_skin_tone_pipeline
import json


def correct_white_balance(img):
    """Gray-world white balance correction."""
    result = img.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] *= avg_gray / avg_b
    result[:, :, 1] *= avg_gray / avg_g
    result[:, :, 2] *= avg_gray / avg_r
    return np.clip(result, 0, 255).astype(np.uint8)


def run_postprocessing(test_dir="results"):
    knn_model = joblib.load("model/knn_hog.pkl")
    xgb_model = joblib.load("model/xgb_hog.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
    skin_cropper = SkinCropper(region="cheek", apply_mask=True)

    IMG_SIZE = (64, 64)
    y_true, knn_preds, xgb_preds, effnet_preds = [], [], [], []

    vis_dir = os.path.join(test_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    for path in glob(f"{test_dir}/*.jpg"):
        basename = os.path.basename(path)

        try:
            img = cv2.imread(path)
            if img is None:
                continue

            img = correct_white_balance(img)
            skin_patch = skin_cropper.crop(img)
            if skin_patch is None or skin_patch.size == 0:
                continue

            resized = cv2.resize(skin_patch, IMG_SIZE)
            hog_feat = hog(
                cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY),
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            ).reshape(1, -1)

            # Initialize
            knn_label = xgb_label = effnet_label = "N/A"

            try:
                knn_idx = knn_model.predict(hog_feat)[0]
                knn_label = label_encoder.inverse_transform([knn_idx])[0]
                knn_preds.append(knn_label)
            except Exception as e:
                print(f"{basename} - KNN Error: {e}")

            try:
                xgb_idx = xgb_model.predict(hog_feat)[0]
                xgb_label = label_encoder.inverse_transform([xgb_idx])[0]
                xgb_preds.append(xgb_label)
            except Exception as e:
                print(f"{basename} - XGB Error: {e}")

            try:
                result = predict_skin_tone_pipeline(path)
                effnet_label = result["skin_tone"]
                effnet_preds.append(effnet_label)
            except Exception as e:
                print(f"{basename} - EffNet Error: {e}")

            # Draw and save annotated image
            annotated = img.copy()
            cv2.putText(
                annotated,
                f"KNN: {knn_label}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"XGB: {xgb_label}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"EffNet: {effnet_label}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

            cv2.imwrite(os.path.join(vis_dir, f"annotated_{basename}"), annotated)

        except Exception as e:
            print(f"{basename} - General error: {e}")

    def assess_model_bias(name, preds):
        print(f"\n📊 {name} Prediction Distribution")
        counts = Counter(preds)
        total = sum(counts.values())
        for label in ["dark", "medium", "light"]:
            pct = (counts[label] / total * 100) if total > 0 else 0
            print(f"  {label.title():<6}: {counts[label]:>2} images ({pct:.1f}%)")
        if total > 0 and max(counts.values()) / total > 0.5:
            print("⚠️  Possible model bias detected.")
        else:
            print("✅  Fair distribution.")

    assess_model_bias("KNN", knn_preds)
    assess_model_bias("XGBoost", xgb_preds)
    assess_model_bias("EfficientNet", effnet_preds)

    summary = {
        "KNN": dict(Counter(knn_preds)),
        "XGBoost": dict(Counter(xgb_preds)),
        "EfficientNet": dict(Counter(effnet_preds)),
    }

    with open(os.path.join(test_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return knn_preds, xgb_preds, effnet_preds
