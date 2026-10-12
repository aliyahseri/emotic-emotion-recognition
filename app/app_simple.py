"""
EMOTIC Demo App — Simplified (No YOLOv5 Auto-Download)
================================================================
Interactive web application for context-based emotion recognition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
import cv2
from PIL import Image
import torch

# Direct model loading without YOLOv5 auto-download
from src.model.emotic_model import EMOTICModel
from src.dataset.emotic_dataset import EMOTIC_CATEGORIES

st.set_page_config(
    page_title="EMOTIC — Emotion in Context",
    page_icon="🎭",
    layout="wide",
)

st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.title-block { text-align: center; padding: 20px 0; }
.emotion-chip {
    display: inline-block; padding: 4px 12px; margin: 3px;
    border-radius: 20px; background: #1e3a5f; color: #7ec8e3;
    font-size: 0.85rem; font-weight: 600;
}
.vad-bar { height: 8px; border-radius: 4px; background: #e0e0e0; margin: 4px 0; }
.vad-fill { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #1e3a5f, #7ec8e3); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-block"><h1>🎭 EMOTIC: Emotion in Context</h1>'
            '<p>Context-Based Human Emotion Recognition</p></div>',
            unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Settings")
ckpt_path = st.sidebar.text_input(
    "Checkpoint path", value="checkpoints/best_model.pth"
)
cat_threshold = st.sidebar.slider(
    "Category confidence threshold", 0.1, 0.9, 0.35, 0.05
)
show_top_k = st.sidebar.slider("Top-K categories to display", 1, 10, 5)

@st.cache_resource
def load_model(ckpt_path):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EMOTICModel(pretrained=False, dropout=0.0)
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        
        return model, device, None
    except Exception as e:
        return None, None, str(e)

model, device, load_err = load_model(ckpt_path)

if load_err:
    st.error(f"⚠️ Could not load model: {load_err}")
    st.stop()

# Image upload
uploaded = st.file_uploader(
    "Upload an image containing people",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

def preprocess_image(image):
    """Simple preprocessing without person detection"""
    img_array = np.array(image)
    
    # Resize to standard sizes
    body_img = cv2.resize(img_array, (128, 128))
    context_img = cv2.resize(img_array, (224, 224))
    
    # Convert to tensors
    body_tensor = torch.from_numpy(body_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    context_tensor = torch.from_numpy(context_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    return body_tensor.to(device), context_tensor.to(device), img_array

if uploaded is not None:
    pil_img = Image.open(uploaded)
    
    col_orig, col_result = st.columns(2)
    
    with col_orig:
        st.subheader("📷 Original Image")
        st.image(pil_img, use_column_width=True)
    
    # Preprocess and predict
    body_tensor, context_tensor, img_array = preprocess_image(pil_img)
    
    with torch.no_grad():
        cat_logits, vad_pred = model(body_tensor, context_tensor)
        cat_probs = torch.sigmoid(cat_logits).cpu().numpy()[0]
        vad_values = vad_pred.cpu().numpy()[0]
    
    with col_result:
        st.subheader("🎭 Emotion Predictions")
        
        # Top emotions
        top_indices = np.argsort(cat_probs)[-show_top_k:][::-1]
        for idx in top_indices:
            prob = cat_probs[idx]
            if prob > cat_threshold:
                emotion = EMOTIC_CATEGORIES[idx]
                st.markdown(
                    f'<span class="emotion-chip">{emotion}: {prob:.3f}</span>',
                    unsafe_allow_html=True
                )
        
        # VAD values
        st.subheader("📊 VAD Values")
        col_v, col_a, col_d = st.columns(3)
        
        with col_v:
            st.metric("Valence", f"{vad_values[0]:.3f}")
        with col_a:
            st.metric("Arousal", f"{vad_values[1]:.3f}")
        with col_d:
            st.metric("Dominance", f"{vad_values[2]:.3f}")
        
        # VAD bars
        st.subheader("📈 VAD Visualization")
        for name, value in [("Valence", vad_values[0]), ("Arousal", vad_values[1]), ("Dominance", vad_values[2])]:
            st.write(f"{name}:")
            st.progress(max(0, min(1, value)))

st.markdown("---")
st.caption("Note: This simplified demo processes the entire image as both body and context for testing purposes.")
