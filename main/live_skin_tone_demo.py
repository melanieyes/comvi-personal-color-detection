import cv2
import numpy as np
import os
from utils.face_detection import FaceDetector
from utils.skin_cropper import SkinCropper
from model.skin_tone_classifier import SkinToneClassifier
from utils.color_palette import ColorPaletteRecommender
from camera_view import CameraView


def create_palette_image(colors, width=300, height=50):
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    step = width // len(colors)
    for i, hex_color in enumerate(colors):
        rgb = tuple(int(hex_color.lstrip("#")[j : j + 2], 16) for j in (0, 2, 4))
        bar[:, i * step : (i + 1) * step] = rgb
    return bar


def main():
    cap = cv2.VideoCapture(0)
    face_detector = FaceDetector()
    skin_cropper = SkinCropper()
    classifier = SkinToneClassifier()
    palette_recommender = ColorPaletteRecommender()
    camera_view = CameraView()

    os.makedirs("results", exist_ok=True)

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

        # UI hint
        cv2.putText(
            combined,
            "Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
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
            camera_view.save_frame(face_crop, label="capture")
            cv2.putText(
                combined,
                "✅ Image captured!",
                (180, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        elif key == ord("q"):
            break

        camera_view.show_frame(combined)

    cap.release()
    camera_view.release()


if __name__ == "__main__":
    main()
