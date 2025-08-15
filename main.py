import streamlit as st
import cv2 as cv
import numpy as np
import keras
import time
import pandas as pd

# --- Label Names ---
label_name = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 
    'Cherry Powdery mildew', 'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 
    'Corn Common rust', 'Corn Northern Leaf Blight', 'Corn healthy', 
    'Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 
    'Peach Bacterial spot', 'Peach healthy', 'Pepper bell Bacterial spot', 
    'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 'Potato healthy', 
    'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 
    'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 
    'Tomato Septoria leaf spot', 'Tomato Spider mites', 'Tomato Target Spot', 
    'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'Tomato healthy'
]

# --- Scientific Disease Descriptions with Bold Tags for Streamlit ---
disease_descriptions = pd.DataFrame({
    'disease': label_name,
    'description': [
        'Caused by the fungus Venturia inaequalis, apple scab results in olive-green to black velvety lesions on leaves, fruit, and stems.',
        'A fungal infection by Botryosphaeria obtusa causing circular black lesions with a target-like appearance on fruit and leaves.',
        'Caused by Gymnosporangium juniperi-virginianae, producing bright orange galls on junipers and yellow-orange leaf spots on apples.',
        'No observable disease symptoms; leaves and fruit are free from pathogens.',
        'Caused by Podosphaera spp., resulting in white, powdery fungal growth on the leaf surface.',
        'No signs of fungal or bacterial infection; plant tissue is healthy and normal.',
        'Caused by Cercospora zeae-maydis, producing gray to tan lesions on corn leaves.',
        'Caused by Puccinia sorghi, leading to cinnamon-brown pustules on both sides of the leaf.',
        'Triggered by Exserohilum turcicum, forming long, cigar-shaped lesions on leaves.',
        'Corn plants show normal morphology with no pathogenic symptoms.',
        'Caused by Guignardia bidwellii, grape black rot forms sunken, black lesions on berries and leaves.',
        'A trunk disease caused by Phaeoacremonium and Phaeomoniella, leading to tiger-stripe symptoms and internal decay.',
        'Caused by Pseudocercospora vitis, leading to brown spots and necrosis on grape leaves.',
        'Grape plant foliage and fruit are unaffected by disease.',
        'Caused by Xanthomonas arboricola pv. pruni, producing angular water-soaked lesions.',
        'Peach foliage exhibits normal color, texture, and development.',
        'Caused by Xanthomonas campestris pv. vesicatoria, forming greasy lesions on pepper leaves.',
        'Healthy pepper plant with no visible disease symptoms.',
        'Caused by Alternaria solani, forming dark concentric-ring lesions on lower leaves.',
        'A late-season disease caused by Phytophthora infestans, producing large, dark lesions on foliage.',
        'Potato foliage is free from disease with normal growth patterns.',
        'Leaf scorch may result from fungal pathogens or environmental stress, showing leaf margin browning.',
        'Strawberry plants exhibit healthy leaves and typical growth patterns.',
        'Caused by Xanthomonas campestris pv. vesicatoria, forming water-soaked spots that expand and coalesce.',
        'Caused by Alternaria solani, forming dark, concentric spots and blighted leaves.',
        'Caused by Phytophthora infestans, producing oily, water-soaked lesions on leaves and stems.',
        'Caused by Passalora fulva, resulting in yellow leaf spots and olive-colored mold on undersides.',
        'Caused by Septoria lycopersici, producing small, circular lesions with gray centers.',
        'Infestation by Tetranychus urticae results in stippling and fine webbing on the leaf surface.',
        'Caused by Corynespora cassiicola, creating circular lesions with concentric ring patterns.',
        'Caused by Tomato yellow leaf curl virus, transmitted by whiteflies and leading to leaf curling and stunted growth.',
        'Caused by Tobamovirus species, resulting in mottled leaf patterns and distortion.',
        'No signs of pathogen presence; tomato foliage appears healthy and vigorous.'
    ]
})

# --- Page Config ---
st.set_page_config(
    page_title="🌿 Plant Disease Detector",
    layout="wide",
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Professional Styling and Circular Loader ---
st.markdown("""
    <style>
    /* General Styling */
    .stApp {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #1a3c34;
        font-weight: 700;
        animation: fadeIn 1s ease-in;
    }
    .stButton>button {
        background-color: #2e8b57;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #1f6140;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stFileUploader {
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stFileUploader:hover {
        border-color: #2e8b57;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stRadio > label {
        font-size: 16px;
        font-weight: 500;
        color: #1a3c34;
        padding: 12px;
        border-radius: 8px;
        transition: all 0.3s ease;
        margin: 4px 0;
    }
    .stRadio > label:hover {
        background-color: #f1f5f9;
        transform: translateX(5px);
    }
    .card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: all 0.3s ease;
        animation: slideUp 0.5s ease-out;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    /* Custom Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes slideUp {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    /* Success Message Styling */
    .stSuccess {
        animation: slideUp 0.5s ease-out;
    }
    /* Warning Message Styling */
    .stWarning {
        animation: pulse 1s ease-in-out;
    }
    /* Info Message Styling */
    .stInfo {
        animation: fadeIn 0.5s ease-in;
    }
    /* Custom Markdown Styling */
    .custom-markdown {
        padding: 15px;
        border-left: 4px solid #2e8b57;
        background-color: #f8fafc;
        margin: 10px 0;
        animation: slideUp 0.5s ease-out;
    }
    /* Image Container Styling */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .image-container:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    return keras.models.load_model('Training/model/project_model.h5')

model = load_model()

# --- Prediction Function ---
def predict(image_bytes):
    try:
        img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
        normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
        predictions = model.predict(normalized_image)
        confidence = predictions[0][np.argmax(predictions)] * 100
        predicted_class = np.argmax(predictions)
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# --- Main App ---
def main():
    # Initialize session state for page navigation
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Home"
    if 'last_page' not in st.session_state:
        st.session_state.last_page = st.session_state.page

    # --- Header ---
    st.markdown("""
        <div style='text-align:center; padding: 20px 0;'>
            <h1>🌿 Plant Disease Detector AI</h1>
            <p style='font-size:18px; color:#4b5563; max-width:600px; margin:0 auto;'>
                Diagnose plant diseases of various crops using AI-powered image recognition technology.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # --- Sidebar Navigation ---
    st.sidebar.markdown("<h2 style='color:#1a3c34;'>📌 Navigation</h2>", unsafe_allow_html=True)
    page = st.sidebar.radio("", ["🏠 Home", "🩺 Diagnose"], label_visibility="collapsed")

    # --- Show Loader on Page Change ---
    if page != st.session_state.last_page:
        st.markdown("<div class='loader-container'><div class='loader'></div></div>", unsafe_allow_html=True)
        st.session_state.page = page
        st.session_state.last_page = page
        time.sleep(0.5)
        st.rerun()

    # --- Home Page ---
    if st.session_state.page == "🏠 Home":
        st.markdown("<div class='card'><h2>Welcome to Plant Disease Detector 🌱</h2>", unsafe_allow_html=True)
        st.markdown("""
            <p style='color:#4b5563;'>
            The plant disease detection model is built using deep learning techniques, leveraging transfer learning to utilize pre-trained knowledge. It is trained on a dataset containing images of 33 different types of plant diseases across crops like Apple, Cherry, Corn, Grape, Peach, Pepper, Potato, Strawberry, and Tomato.
            </p>
            <p style='color:#4b5563;'>
            This application allows you to:
            <ul>
                <li>📸 Upload a leaf image for instant analysis</li>
                <li>🧠 Use AI to identify potential diseases</li>
                <li>📝 Receive accurate disease predictions with confidence scores</li>
            </ul>
            </p>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Diagnose Page ---
    elif st.session_state.page == "🩺 Diagnose":
        st.markdown("<div class='card'><h2>🩺 Diagnose Your Leaf</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#4b5563;'>Upload a clear photo of a leaf from <b>Apple</b>, <b>Cherry</b>, <b>Corn</b>, <b>Grape</b>, <b>Peach</b>, <b>Pepper</b>, <b>Potato</b>, <b>Strawberry</b>, or <b>Tomato</b> for an AI-powered diagnosis.</p>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"], help="Upload a clear image of the leaf for best results.")

        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            col1, col2 = st.columns([1, 2], gap="medium")

            with col1:
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                st.image(image_bytes, caption="🖼 Uploaded Image", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                pred, confidence = predict(image_bytes)
                if pred is not None:
                    if confidence >= 80:
                        disease_name = label_name[pred]
                        disease_desc = disease_descriptions.loc[disease_descriptions['disease'] == disease_name, 'description'].iloc[0]

                        # Display the detected disease and confidence
                        st.success(f"🌾 **Detected Disease:** {disease_name} ({confidence:.2f}% confidence)")

                        # Display the description
                        st.markdown(f"<div class='custom-markdown'><strong>Description:</strong> {disease_desc}</div>", unsafe_allow_html=True)

                        # Create a ChatGPT search link
                        chatgpt_url = f"https://chat.openai.com/?q={disease_name.replace(' ', '+')}"
                        st.markdown(f"""
                            <div class='custom-markdown'>
                                <a href="{chatgpt_url}" target="_blank" style="text-decoration: none;">
                                    <button style="background-color: #10a37f; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                                        🔍 Learn more about {disease_name} on ChatGPT
                                    </button>
                                </a>
                            </div>
                        """, unsafe_allow_html=True)

                        st.balloons()

                    else:
                        st.warning("⚠️ Confidence too low. Please try another image.")
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()