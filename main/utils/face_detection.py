import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self, cascade_path="haarcascade_frontalface_default.xml"):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.result_dir = "results"
        os.makedirs(self.result_dir, exist_ok=True)

    def detect_skin_in_color(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower, upper)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        return skin_mask

    def is_face(self, image, face_coords, threshold=0.3):
        x, y, w, h = face_coords
        face_img = image[y:y + h, x:x + w]
        skin_mask = self.detect_skin_in_color(face_img)
        skin_pixels = cv2.countNonZero(skin_mask)
        total_pixels = face_img.shape[0] * face_img.shape[1]
        skin_ratio = skin_pixels / total_pixels
        return skin_ratio >= threshold

    def detect_from_image(self, img, return_all=False):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        valid_faces = []
        for (x, y, w, h) in faces:
            if self.is_face(img, (x, y, w, h)):
                face_crop = img[y:y + h, x:x + w]
                if return_all:
                    valid_faces.append((face_crop, (x, y, w, h)))
                else:
                    return face_crop, (x, y, w, h)

        if return_all:
            return valid_faces
        raise ValueError("No valid face with skin detected.")

    def capture_and_save(self):
        cap = cv2.VideoCapture(0)
        save_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                face_crop, (x, y, w, h) = self.detect_from_image(frame)
                filename = f"{self.result_dir}/face_{save_count}.jpg"
                cv2.imwrite(filename, face_crop)
                print(f"Saved: {filename}")
                save_count += 1
            except ValueError:
                pass
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
