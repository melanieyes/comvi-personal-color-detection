# import cv2
# import numpy as np
# from utils.face_detection import FaceDetector
# from utils.skin_cropper import SkinCropper
# from model.skin_tone_classifier import SkinToneClassifier
# from utils.color_palette import ColorPaletteRecommender
# from camera_view import CameraView
# # from HandTrackingModule import handDetector
# import os
# import sys
# import glob
# from post_processing import run_postprocessing
# import json
# from collections import Counter

# def create_palette_image(colors, width=300, height=50):
#     """Create a horizontal palette image from a list of HEX colors."""
#     bar = np.zeros((height, width, 3), dtype=np.uint8)
#     step = width // len(colors)
#     for i, hex_color in enumerate(colors):
#         rgb = tuple(int(hex_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
#         bar[:, i * step:(i + 1) * step] = rgb
#     return bar


# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python model_evaluation.py <folder_name>")
#         return
#     user_folder = sys.argv[1]
#     output_dir = os.path.join("results", user_folder)
#     os.makedirs(output_dir, exist_ok=True)

#     # Initialize components
#     camera_view = CameraView()
#     face_detector = FaceDetector()
#     skin_cropper = SkinCropper()
#     classifier = SkinToneClassifier()
#     palette_recommender = ColorPaletteRecommender()

#     detector = handDetector(detectionCon=0.8)
#     shutter_center = (290, 385)
#     shutter_radius = 25
#     captured = False

#     cap = cv2.VideoCapture(0)
#     saved_count = 0
#     # output_dir = "results"
#     # os.makedirs(output_dir, exist_ok=True)
#     # Accept folder name as CLI argument


#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print(" Failed to capture frame.")
#             break

#         # Detect hands and landmarks
#         frame = detector.findHands(frame)
#         lmList, _ = detector.findPosition(frame)

#         try:
#             # frame = cv2.imread(r"C:\Users\hanhn\Downloads\bluke.webp")

#             # Detect and crop face
#             face_crop, face_box = face_detector.detect_from_image(frame)

#             # Crop skin region from face
#             skin = skin_cropper.crop(face_crop)

#             # Classify skin tone
#             tone, conf = classifier.predict(skin)

#             # Get recommended palette
#             palette = palette_recommender.recommend(tone)
#             palette_img = create_palette_image(palette)

#             palette_img_resized = cv2.resize(palette_img, (frame.shape[1], 60))
#             combined = np.vstack((frame, palette_img_resized))

#             # Annotate result
#             x, y, w, h = face_box
#             cv2.rectangle(combined, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             cv2.putText(combined, f"{tone} ({conf*100:.1f}%)", (x, y - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         except Exception as e:
#             # camera_view.show_frame(frame)
#             combined = frame.copy()
#             face_box = None
#             print(" Error:", e)

#         # key = camera_view.get_key()

#         # if key == ord('s'):
#         #     camera_view.save_frame(frame, label="capture")

#         # elif key == ord('q'):
#         #     break
#         # Draw trendy shutter button
#         cv2.circle(combined, shutter_center, shutter_radius + 10, (255, 192, 203), 10)
#         cv2.circle(combined, shutter_center, shutter_radius, (255, 255, 255), -1)
#         cv2.circle(combined, shutter_center, shutter_radius - 10, (240, 128, 128), -1)
#         cv2.putText(combined, "Snap me!", (shutter_center[0] - 55, shutter_center[1] + shutter_radius + 35),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 3)
#         cv2.putText(combined, "Snap me!", (shutter_center[0] - 55, shutter_center[1] + shutter_radius + 35),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

#         # === Gesture-based capture logic ===
#         if len(lmList) != 0:
#             fingers = detector.fingersUp()
#             if fingers[1] == 1 and fingers[2] == 1:
#                 length, _, lineInfo = detector.findDistance(8, 12, frame)
#                 if length < 40:
#                     cx, cy = lineInfo[4], lineInfo[5]
#                     cv2.circle(combined, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
#                     dist_to_button = ((cx - shutter_center[0]) ** 2 + (cy - shutter_center[1]) ** 2) ** 0.5
#                     if dist_to_button < shutter_radius and not captured:
#                         if face_box:
#                             x, y, w, h = face_box
#                             extend_up = int(0.5 * h)
#                             extend_down = int(0.3 * h)
#                             y1 = max(y - extend_up, 0)
#                             y2 = min(y + h + extend_down, frame.shape[0])
#                             x1 = max(x - 20, 0)
#                             x2 = min(x + w + 20, frame.shape[1])
#                             # face_crop = frame[y1:y2, x1:x2]
#                             # camera_view.save_frame(face_crop, label="capture")
#                             # captured = True
#                             face_crop = frame[y1:y2, x1:x2]
#                             # save_path = os.path.join(output_dir, f"face_{saved_count+1:02d}.jpg")
#                             save_path = os.path.join(output_dir, f"{user_folder}_face_{saved_count+1:02d}.jpg")

#                             cv2.imwrite(save_path, face_crop)
#                             print(f"[INFO] Saved {save_path}")
#                             saved_count += 1
#                             captured = True
#                             cv2.putText(combined, f"Image {saved_count} saved!", (180, frame.shape[0] - 350),
#                                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#             else:
#                 captured = False

#         camera_view.show_frame(combined)
#         if camera_view.get_key() == ord('q'):
#             break

#     cap.release()
#     camera_view.release()

#     def summarize_predictions():
#         knn_preds, xgb_preds, effnet_preds = run_postprocessing(output_dir)

#         summary = {
#             "KNN": dict(Counter(knn_preds)),
#             "XGBoost": dict(Counter(xgb_preds)),
#             "EfficientNet": dict(Counter(effnet_preds))
#         }

#         summary_path = os.path.join(output_dir, "summary.json")
#         with open(summary_path, "w") as f:
#             json.dump(summary, f)

#         print(f"[INFO] Saved summary to {summary_path}")

#     if saved_count > 0:
#         print("\n--- Running Batch Inference on Saved Faces ---\n")
#         summarize_predictions()
#     # if saved_count > 0:
#     #     print("\n--- Running Batch Inference on Saved Faces ---\n")
#     #     run_postprocessing(output_dir)


# if __name__ == "__main__":
#     main()

import cv2
import numpy as np
import os
import sys
import json
from glob import glob
from collections import Counter

from utils.face_detection import FaceDetector
from utils.skin_cropper import SkinCropper
from model.skin_tone_classifier import SkinToneClassifier
from utils.color_palette import ColorPaletteRecommender
from camera_view import CameraView
from post_processing import run_postprocessing


def create_palette_image(colors, width=300, height=50):
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    step = width // len(colors)
    for i, hex_color in enumerate(colors):
        rgb = tuple(int(hex_color.lstrip("#")[j : j + 2], 16) for j in (0, 2, 4))
        bar[:, i * step : (i + 1) * step] = rgb
    return bar


def main():
    if len(sys.argv) < 2:
        print("Usage: python model_evaluation.py <folder_name>")
        return

    user_folder = sys.argv[1]
    output_dir = os.path.join("results", user_folder)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    skin_cropper = SkinCropper()
    classifier = SkinToneClassifier()
    palette_recommender = ColorPaletteRecommender()
    camera_view = CameraView()

    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        try:
            face_crop, face_box = face_detector.detect_from_image(frame)
            skin = skin_cropper.crop(face_crop)
            tone, conf = classifier.predict(skin)
            palette = palette_recommender.recommend(tone)
            palette_img = create_palette_image(palette)
            palette_img_resized = cv2.resize(palette_img, (frame.shape[1], 60))
            combined = np.vstack((frame, palette_img_resized))

            x, y, w, h = face_box
            cv2.rectangle(combined, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                combined,
                f"{tone} ({conf*100:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        except Exception as e:
            print("⚠️ Error:", e)
            combined = frame.copy()
            face_box = None

        cv2.putText(
            combined,
            "Press 'c' to capture, 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        key = camera_view.get_key()
        if key == ord("c") and face_box:
            x, y, w, h = face_box
            y1 = max(y - int(0.5 * h), 0)
            y2 = min(y + h + int(0.3 * h), frame.shape[0])
            x1 = max(x - 20, 0)
            x2 = min(x + w + 20, frame.shape[1])
            face_crop = frame[y1:y2, x1:x2]

            save_path = os.path.join(
                output_dir, f"{user_folder}_face_{saved_count+1:02d}.jpg"
            )
            cv2.imwrite(save_path, face_crop)

            if os.path.exists(save_path):
                saved_count += 1
                print(f"[INFO] Saved {save_path}")
                cv2.putText(
                    combined,
                    f"✅ Saved image {saved_count}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                print(f"❌ Failed to save image at {save_path}")

        elif key == ord("q"):
            break

        camera_view.show_frame(combined)

    cap.release()
    camera_view.release()

    if saved_count > 0:
        print("\n📊 Running Batch Inference on Saved Faces...")
        knn_preds, xgb_preds, effnet_preds = run_postprocessing(output_dir)

        summary = {
            "KNN": dict(Counter(knn_preds)),
            "XGBoost": dict(Counter(xgb_preds)),
            "EfficientNet": dict(Counter(effnet_preds)),
        }

        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[INFO] Summary saved to {summary_path}")
    else:
        print("⚠️ No images captured. Evaluation skipped.")


if __name__ == "__main__":
    main()
