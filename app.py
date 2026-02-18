import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="AI Face Recognition",
    page_icon="🧠",
    layout="centered"
)

# ---------------------------
# Load Models
# ---------------------------
clf = joblib.load("mlp_face_model.pkl")
pca = joblib.load("pca_model.pkl")
lda = joblib.load("lda_model.pkl")
class_names = joblib.load("class_names.pkl")

h = w = 300

# ---------------------------
# Sidebar Settings
# ---------------------------
st.sidebar.title("⚙ Settings")
threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.80,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.write("Developed by Vishakha 🚀")

# ---------------------------
# Main Title
# ---------------------------

st.title("🧠 Face Recognition System")
st.markdown("Upload an image to recognize the person using PCA + LDA + MLP")

st.markdown("---")
st.subheader("📷 Live Camera Detection")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:

    image = Image.open(camera_image)
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (h, w))
    face_vector = resized.flatten().reshape(1, -1)

    # PCA → LDA
    face_pca = pca.transform(face_vector)
    face_lda = lda.transform(face_pca)

    prob = clf.predict_proba(face_lda)[0]
    sorted_prob = np.sort(prob)

    max_prob = sorted_prob[-1]
    second_max = sorted_prob[-2]
    class_id = np.argmax(prob)

    confidence_gap = max_prob - second_max

    if max_prob < threshold or confidence_gap < 0.20:
        label = "UNKNOWN"
        st.error("⚠ Person Not Recognized")
    else:
        label = class_names[class_id]
        st.success("✅ Person Recognized")

    st.write(f"**Name:** {label}")
    st.write(f"**Confidence:** {max_prob:.2f}")
    st.progress(float(max_prob))



uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------
# Prediction Logic (Upload Image)
# ---------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (h, w))

    # Normalize input (IMPORTANT)
    face_vector = resized.flatten().astype("float32") / 255.0
    face_vector = face_vector.reshape(1, -1)

    # PCA → LDA
    face_pca = pca.transform(face_vector)
    face_lda = lda.transform(face_pca)

    # Predict probabilities
    prob = clf.predict_proba(face_lda)[0]

    max_prob = np.max(prob)
    second_max = np.partition(prob, -2)[-2]
    confidence_gap = max_prob - second_max

    class_id = np.argmax(prob)

    # Stronger Unknown Logic
    if max_prob < threshold or confidence_gap < 0.20 or max_prob > 0.98:
        label = "UNKNOWN"
        st.error("⚠ Person Not Recognized")
    else:
        label = class_names[class_id]
        st.success("✅ Person Recognized")

    # Display Results
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Prediction Result")
        st.write(f"**Name:** {label}")
        st.write(f"**Confidence:** {max_prob:.2f}")
        st.progress(float(max_prob))

    st.markdown("---")
    st.info("Model: PCA → LDA → MLP Neural Network")
