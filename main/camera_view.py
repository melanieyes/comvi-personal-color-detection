
import cv2
import numpy as np

class CameraView:
    def __init__(self):
        self.window_name = "Skin Tone Classification"
        self.palette_window = "Recommended Colors"
        cv2.namedWindow(self.window_name)
        # cv2.namedWindow(self.palette_window)

    def show_frame(self, frame, faces=None):
        if faces is not None:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow(self.window_name, frame)

    def show_prediction(self, frame, face_rect, label, confidence):
        (x, y, w, h) = face_rect
        annotated = frame.copy()
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(annotated, f"{label} ({confidence*100:.1f}%)",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow(self.window_name, annotated)

    def show_palette(self, palette_img):
        cv2.imshow(self.palette_window, palette_img)

    def get_key(self):
        return cv2.waitKey(1) & 0xFF

    def release(self):
        cv2.destroyAllWindows()

    def save_frame(self, frame, label="frame"):
        filename = f"{label}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[INFO] Frame saved as {filename}")
