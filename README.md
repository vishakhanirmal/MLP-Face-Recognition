🧠 AI Face Recognition Web Application

A complete Face Recognition System built using PCA (Eigenfaces), LDA (Fisherfaces), and MLP Neural Network, deployed as a professional Streamlit Web Application with Unknown Face Detection.

🚀 Project Overview
This project implements a machine learning–based face recognition pipeline that:
* Detects and processes facial images
* Reduces dimensionality using PCA
* Extracts discriminative features using LDA
* Classifies faces using a Multi-Layer Perceptron (MLP)
* Rejects unknown faces using confidence-based thresholding
* Provides a clean, interactive web interface
The system is designed to simulate a real-world AI-based face recognition solution.

🏗️ System Architecture
Input Image / Camera Snapshot
        ↓
Grayscale Conversion
        ↓
Resize (300x300)
        ↓
PCA (Dimensionality Reduction)
        ↓
LDA (Feature Extraction)
        ↓
MLP Classifier
        ↓
Confidence Threshold Logic
        ↓
Known Person / UNKNOWN

✨ Key Features
✅ Face Recognition using PCA + LDA + MLP
✅ Unknown Face Detection (Open-set logic)
✅ Confidence Threshold Control (User Adjustable)
✅ Snapshot Camera Detection
✅ Clean & Professional Streamlit UI
✅ Model Saving & Loading (.pkl)
✅ Classification Report & Confusion Matrix
✅ Real-Time Webcam Version (Notebook Version)

🛠️ Technologies Used
-Python
-NumPy
-OpenCV
-Scikit-learn
-Streamlit
-Joblib
-Matplotlib / Seaborn

📊 Model Details
* Image Size: 300 × 300 (Grayscale)
* Dimensionality Reduction: PCA (150 components)
* Feature Extraction: LDA
* Classifier: MLPClassifier
* Unknown Detection:
   -Confidence threshold
   -Probability gap analysis

🔍 Unknown Face Detection Strategy
The system improves real-world reliability by:
* Checking maximum prediction confidence
* Comparing difference between top two probabilities
* Rejecting overconfident predictions
* Marking uncertain results as UNKNOWN
This simulates open-set face recognition logic used in practical systems.

📦 Installation
* Clone the repository:
    git clone https://github.com/vishakhanirmal/MLP-Face-Recognition.git
    cd MLP-Face-Recognition
* Install dependencies: pip install -r requirements.txt
▶️ Run The Web App :- python -m streamlit run app.py
The browser will open automatically.

📷 How To Use
1. Upload an image OR use camera snapshot
2. Adjust confidence threshold (optional)
3. View prediction result
4. System displays:
    Name (if recognized)
    UNKNOWN (if not confident)
    Confidence score
   
📁 Project Structure
MLP-Face-Recognition/
│
├── app.py
├── MLP_Face_Recognition.ipynb
├── mlp_face_model.pkl
├── pca_model.pkl
├── lda_model.pkl
├── class_names.pkl
├── requirements.txt
└── README.md

📈 Future Improvements
-Real-time continuous streaming with bounding boxes
-CNN-based face embedding (FaceNet / Deep Learning)
-Deployment to cloud (Streamlit Cloud / Render)
-Face attendance system integration
-Improved lighting & pose robustness

👩‍💻 Author
Vishakha Nirmal
MSc Computer Application
AI & Machine Learning Enthusiast

📌 Project Status
✔ Completed
✔ Deployment-ready
✔ Portfolio-ready
✔ Industry-style evaluation included
