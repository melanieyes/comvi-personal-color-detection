import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import DirectoryIterator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

class SkinCropper:
    def __init__(self, region="cheek", apply_mask=True):
        self.region = region
        self.apply_mask = apply_mask

    def _mask_skin(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 48, 80], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        skin = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        return skin

    def crop(self, face_image):
        h, w, _ = face_image.shape

        if self.region == 'cheek':
            x1 = int(w * 0.25)
            x2 = int(w * 0.75)
            y1 = int(h * 0.4)
            y2 = int(h * 0.7)
        elif self.region == 'forehead':
            x1 = int(w * 0.3)
            x2 = int(w * 0.7)
            y1 = int(h * 0.1)
            y2 = int(h * 0.3)
        else:
            x1, y1, x2, y2 = 0, 0, w, h

        crop = face_image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None

        if self.apply_mask:
            crop = self._mask_skin(crop)

        return crop


class SkinCropDirectoryIterator(DirectoryIterator):
    def __init__(self, directory, image_data_generator, region="cheek", **kwargs):
        super().__init__(directory, image_data_generator, **kwargs)
        self.cropper = SkinCropper(region=region, apply_mask=True)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x, batch_y = super()._get_batches_of_transformed_samples(index_array)
        for i in range(batch_x.shape[0]):
            img = batch_x[i]
            bgr_img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            crop = self.cropper.crop(bgr_img)
            if crop is not None and crop.size > 0:
                crop = cv2.resize(crop, self.image_shape[:2][::-1])
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                batch_x[i] = preprocess_input(crop.astype(np.float32))
            else:
                batch_x[i] = np.zeros_like(batch_x[i])
        return batch_x, batch_y
