# Melatone: Smart Skin Tone Classification and Personal Color Recommendation

Melatone is a computer vision system designed to classify facial skin tone into **Light**, **Medium**, or **Dark** and recommend a suitable color palette for fashion and personal styling. It includes a deep learning model (EfficientNet), traditional ML models (KNN, XGBoost with HOG), white balance correction, and a user-friendly interface built using Streamlit and FastAPI.

---

## 🎯 Features

- 🖼️ Real-time webcam image capture
- 📦 White balance correction for lighting normalization
- 📊 Skin tone classification using:
  - KNN + HOG features
  - XGBoost + HOG features
  - EfficientNet CNN
- 🎨 Color palette recommendation
- 🔁 Batch model evaluation with annotated output
- 💻 Dual frontend: Streamlit interface + FastAPI backend

---

## 🧠 Human vs. AI Contribution

The system architecture, preprocessing pipeline, and base model training were designed and implemented by the authors. AI tools (e.g., ChatGPT) were consulted to enhance performance during fine-tuning. For instance:

- Suggested switching from VGG16 to EfficientNet.
- Recommended use of techniques like early stopping, dropout, and Adam optimizer.

All feature engineering, webcam integration, and the overall evaluation interface were developed manually.

---

## 📁 Project Structure

Melatone/
├── app.py # Streamlit interface
├── main/ # FastAPI backend folder
│ └── main.py # FastAPI app
├── model_evaluation.py # Webcam image capture & evaluation
├── post_processing.py # Multi-model inference & visualization
├── model/
│ ├── best_efficientnetv2s.h5 # Tracked via Git LFS
│ ├── knn_hog.pkl
│ ├── xgb_hog.pkl
│ └── label_encoder.pkl
├── utils/
│ ├── face_detection.py
│ ├── skin_cropper.py
│ ├── color_palette.py
│ └── ...
├── results/
│ └── <folder>/ # Captured images & evaluation
└── README.md

yaml
Copy
Edit

---

## 💻 Setup Instructions

### 1. Install Git LFS (if cloning the repo)
```bash
git lfs install
git clone https://github.com/melanieyes/personal-color-recomm.git
2. Create and Activate Virtual Environment
bash
Copy
Edit
python -m venv colorenv
colorenv\Scripts\activate     # On Windows
source colorenv/bin/activate  # On Mac/Linux
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the Application
You will need two terminal windows for backend (FastAPI) and frontend (Streamlit):

Terminal 1 – Run FastAPI Server
bash
Copy
Edit
cd main
colorenv\Scripts\activate
python -m uvicorn main:app --reload
Terminal 2 – Run Streamlit Frontend
bash
Copy
Edit
cd main
colorenv\Scripts\activate
streamlit run ../app.py
🧪 Model Evaluation via Webcam
bash
Copy
Edit
python model_evaluation.py yourname_folder
Press c to capture image

Press q to quit

Outputs:

results/yourname_folder/summary.json

Annotated images in results/yourname_folder/vis/

🔧 Common Issues
❌ Git Push Fails due to Large File (>100MB)
EfficientNet .h5 model is >100MB. Tracked with Git LFS.

To use Git LFS:

bash
Copy
Edit
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add model/best_efficientnetv2s.h5
git commit -m "Track large model with Git LFS"
git push origin master
📜 License
This project is for academic and research purposes only.

👩‍💻 Author
Melanie
Email: [your_email@example.com]
GitHub: @melanieyes

yaml
Copy
Edit

---

Would you like me to export this as a `README.md` file now? Or copy it into a Word doc for final submission?






You said:
provide the whole awnser in readme format


ChatGPT said:
Absolutely! Here's the complete content in proper README.md format for direct use:

markdown
Copy
Edit
# 🎨 Melatone: Smart Skin Tone Classification and Personal Color Recommendation

Melatone is a computer vision system that classifies facial skin tone into **Light**, **Medium**, or **Dark**, and recommends a suitable personal color palette based on the prediction. The system integrates traditional machine learning and deep learning techniques with real-time webcam input, white balance correction, and a modern user interface built using **Streamlit** and **FastAPI**.

---

## 🌟 Features

- 🖼️ Real-time webcam image capture and classification
- ✨ White balance correction for consistent tone prediction
- 🤖 Multi-model skin tone classification:
  - KNN with HOG features
  - XGBoost with HOG features
  - EfficientNet-based CNN
- 🎨 Rule-based color palette recommendation
- 📊 Batch evaluation with model comparison and annotation
- 💻 Dual-interface system: FastAPI (backend) + Streamlit (frontend)

---

## 🧠 Human vs. AI Contribution

The core logic of Melatone—including face detection, skin cropping, white balance correction, dataset curation, and model integration—was developed manually by our team. AI tools such as ChatGPT were consulted during **fine-tuning and optimization**, for instance:

- Recommending the transition from **VGG16** (low performance) to **EfficientNet**.
- Suggesting training improvements such as **early stopping**, the **Adam optimizer**, and **dropout** regularization.

Thus, **AI served a supporting role**, while the **product’s foundation and implementation were done by us**.

---

## 🗂️ Project Structure

Melatone/
├── app.py # Streamlit interface
├── main/ # FastAPI backend folder
│ └── main.py # FastAPI app definition
├── model_evaluation.py # Webcam image capture + batch evaluation
├── post_processing.py # Model prediction, annotation, and summary
├── model/
│ ├── best_efficientnetv2s.h5 # Tracked via Git LFS
│ ├── knn_hog.pkl
│ ├── xgb_hog.pkl
│ └── label_encoder.pkl
├── utils/
│ ├── face_detection.py
│ ├── skin_cropper.py
│ ├── color_palette.py
│ └── ...
├── results/
│ └── <user_folder>/ # Captured images and model predictions
│ ├── summary.json
│ └── vis/
└── README.md

yaml
Copy
Edit

---

## ⚙️ Setup Instructions

### 1. Clone the Repository and Install Git LFS

```bash
git lfs install
git clone https://github.com/melanieyes/personal-color-recomm.git
2. Create and Activate Virtual Environment
bash
Copy
Edit
python -m venv colorenv
colorenv\Scripts\activate       # On Windows
# or
source colorenv/bin/activate    # On Mac/Linux
3. Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt
🚀 Run the Application
Open two terminals and activate your virtual environment in both:

bash
Copy
Edit
colorenv\Scripts\activate
Then in both terminals:

bash
Copy
Edit
cd main
Terminal 1 – Run FastAPI backend
bash
Copy
Edit
python -m uvicorn main:app --reload
Terminal 2 – Run Streamlit frontend
bash
Copy
Edit
streamlit run ../app.py
📷 Model Evaluation with Webcam
Capture and evaluate skin tone predictions using:

bash
Copy
Edit
python model_evaluation.py yourname_folder
Press c to capture an image

Press q to quit

Results will be saved to:

results/yourname_folder/summary.json

results/yourname_folder/vis/ (annotated images)

🔍 Batch Evaluation Output
For each captured image:

All three models predict tone class.

Annotated images are saved with overlayed predictions.

Bias summary is printed to terminal.

A summary JSON is saved showing class distributions per model.


🧪 Example Output (Visually Annotated)
Image	KNN	XGBoost	EfficientNet
annotated_face01.jpg	Medium	Dark	Medium

📜 License
This project is for academic, research, and educational purposes only. Commercial use is not permitted without permission.

👩‍💻 Author
Melanie - Hanh Nguyen

