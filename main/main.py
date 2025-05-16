import fastapi
import cv2
from PIL import Image
from collections import Counter
import numpy as np
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
# import base64
# import skin_model as m
# import requests
# from ast import literal_eval       
# from model.skin_tone_classifier import SkinToneClassifier      
from skin_model import SkinToneClassifier
from utils.face_detection import FaceDetector
from utils.skin_cropper import SkinCropper
from utils.color_palette import ColorPaletteRecommender

face_detector = FaceDetector()
skin_cropper = SkinCropper()
classifier = SkinToneClassifier()
palette = ColorPaletteRecommender()

app = FastAPI()

origins = [
    "http://localhost:3000" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# @app.post("/image")
# async def image(data: dict):
#     if os.path.exists("temp.jpg"):
#         os.remove("temp.jpg")
#     if os.path.exists("saved.jpg"):
#         os.remove("saved.jpg")
#     try:
#         file_path = data["image"]
#         # Đọc file và encode base64
#         # file_path = file_path.replace("\\", "/")
#         with open(file_path, "rb") as image_file:
#             encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

#         image_data = f"data:image/jpeg;base64,{encoded_string}"

#         decoded_image = base64.b64decode(image_data.split(",")[1])

#         with open("saved.jpg","wb") as fi:
#             fi.write(decoded_image)
      
#         f.save_skin_mask("saved.jpg")
   
#         ans = m.get_season("temp.jpg")
#         # os.remove("temp.jpg")
#         # os.remove("saved.jpg")
   
#         ans = {0: 3, 3: 4}.get(ans, ans)

#         # Map số thành chữ
#         season_mapping = {
#             1: "Spring",
#             2: "Summer",
#             3: "Autumn",
#             4: "Winter"
#         }

#         tone_mapping = {
#             "Spring": "Warm tone: Vibrant, Alive, Energetic, Bright, Cute, Pop, Fresh",
#             "Summer": "Cool tone: Peaceful, Serene, Comforting, Pastel, Elegant, Soft, Romantic",
#             "Autumn": "Warm tone: Tawny, Coppery, Earthy, Deep, Natual, Chic, Mature",
#             "Winter": "Cool tone: Cool, Icy, Elegant, Regal, Brilliant, Sharp, Contrast, Fashion"
#         }
#         season_name = season_mapping.get(ans, "Unknown")

#         result = season_name
        
#         # Return response that map the season to the tone
#         return JSONResponse(content={
#             "season": season_name,
#             "tone": tone_mapping.get(season_name, "Unknown")
#         })
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
@app.post("/image")
async def image(data: dict):
    try:
        file_path = data["image"]

        # Load and decode image
        with open(file_path, "rb") as image_file:
            img_array = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Face detection & crop
        face_crop, _ = face_detector.detect_from_image(image)

        # Skin crop
        skin_patch = skin_cropper.crop(face_crop)

        # Predict skin tone
        tone_label, conf = classifier.predict(skin_patch)

        # Get palette & map to season
        color_palette = palette.recommend(tone_label)
        season_mapping = {
            'dark': "Dark",
            'medium': "Medium",
            'light': "Light"
        }
    
        tone_description = {
            "Dark": "Warm tone: Rich jewel tones, high contrast, bold prints",
            "Medium": "Neutral tone: Balanced earthy shades, dusty hues, warm denim",
            "Light": "Cool tone: Pastels, airy fabrics, soft color combinations"
        }

        season_name = season_mapping.get(tone_label.lower(), "Unknown")
        palette_colors = palette.recommend(tone_label)

        return {
            "season": season_name,
            "tone": tone_description.get(season_name, "Unknown"),
            "palette": palette_colors
        }


        # season_name = season_mapping.get(tone_label, "Unknown")

        # return {
        #     "season": season_name,
        #     "tone": style_tone_text.get(season_name, "Unknown")
        # }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/lip")
# async def image(data: dict):
#     try:
#         file_path = data["image"]

#         # Load and decode image
#         with open(file_path, "rb") as image_file:
#             img_array = np.frombuffer(image_file.read(), np.uint8)
#             image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

#         # Face detection & crop
#         face_crop, _ = face_detector.detect_from_image(image)

#         # Skin crop
#         skin_patch = skin_cropper.crop(face_crop)

#         # Predict skin tone
#         tone_label, conf = classifier.predict(skin_patch)

#         # Get palette & map to season
#         color_palette = palette.recommend(tone_label)
#         season_mapping = {
#             'dark': "Dark",
#             'medium': "Medium",
#             'light': "Light"
#         }
    
#         tone_description = {
#             "Dark": "Warm tone: Rich jewel tones, high contrast, bold prints",
#             "Medium": "Neutral tone: Balanced earthy shades, dusty hues, warm denim",
#             "Light": "Cool tone: Pastels, airy fabrics, soft color combinations"
#         }

#         season_name = season_mapping.get(tone_label.lower(), "Unknown")
#         palette_colors = palette.recommend(tone_label)

#         return {
#             "season": season_name,
#             "tone": tone_description.get(season_name, "Unknown"),
#             "palette": palette_colors
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))