"""
EMOTIC Demo App — CPU Version (No CUDA Issues)
===============================================
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
.metric-card {
    background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-block"><h1>🎭 EMOTIC: Emotion in Context</h1>'
            '<p>RTX 5070-Trained Model (0.3989 mAP) - CPU Mode</p></div>',
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
def load_model_cpu(ckpt_path):
    try:
        # Force CPU mode to avoid CUDA issues
        device = torch.device("cpu")
        model = EMOTICModel(pretrained=False, dropout=0.0)
        
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        
        return model, device, None
    except Exception as e:
        return None, None, str(e)

model, device, load_err = load_model_cpu(ckpt_path)

if load_err:
    st.error(f"⚠️ Could not load model: {load_err}")
    st.stop()

st.sidebar.success(f"✅ Model loaded on CPU")
st.sidebar.info(f"📊 Best mAP: {torch.load(ckpt_path, map_location='cpu')['best_map']:.4f}")
st.sidebar.warning("⚡ Running in CPU mode (slower but compatible)")

def preprocess_image(image):
    """Simple preprocessing without person detection"""
    img_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Resize to standard sizes
    body_img = cv2.resize(img_array, (128, 128))
    context_img = cv2.resize(img_array, (224, 224))
    
    # Convert to tensors
    body_tensor = torch.from_numpy(body_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    context_tensor = torch.from_numpy(context_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    return body_tensor.to(device), context_tensor.to(device), img_array

# Image upload
uploaded = st.file_uploader(
    "📤 Upload an image containing people",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload any image with people to analyze emotions in context"
)

if uploaded is not None:
    pil_img = Image.open(uploaded)
    
    col_orig, col_result = st.columns(2)
    
    with col_orig:
        st.subheader("📷 Original Image")
        st.image(pil_img, use_column_width=True, caption=f"Uploaded: {uploaded.name}")
    
    # Preprocess and predict
    body_tensor, context_tensor, img_array = preprocess_image(pil_img)
    
    with torch.no_grad():
        cat_logits, vad_pred = model(body_tensor, context_tensor)
        cat_probs = torch.sigmoid(cat_logits).cpu().numpy()[0]
        vad_values = vad_pred.cpu().numpy()[0]
    
    with col_result:
        st.subheader("🎭 Emotion Predictions")
        
        # Top emotions
        st.write("**Emotion Categories:**")
        top_indices = np.argsort(cat_probs)[-show_top_k:][::-1]
        
        for i, idx in enumerate(top_indices):
            prob = cat_probs[idx]
            if prob > cat_threshold:
                emotion = EMOTIC_CATEGORIES[idx]
                st.markdown(
                    f'<span class="emotion-chip">{i+1}. {emotion}: {prob:.3f}</span>',
                    unsafe_allow_html=True
                )
        
        # VAD values
        st.write("**VAD (Continuous Emotions):**")
        col_v, col_a, col_d = st.columns(3)
        
        with col_v:
            st.metric("Valence", f"{vad_values[0]:.3f}")
        with col_a:
            st.metric("Arousal", f"{vad_values[1]:.3f}")
        with col_d:
            st.metric("Dominance", f"{vad_values[2]:.3f}")
        
        # VAD visualization
        st.write("**VAD Visualization:**")
        for name, value in [("Valence", vad_values[0]), ("Arousal", vad_values[1]), ("Dominance", vad_values[2])]:
            # Normalize to 0-1 range for progress bar
            normalized = max(0, min(1, (value + 1) / 2))
            st.write(f"{name}: {value:.3f}")
            st.progress(float(normalized))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 Trained on RTX 5070 with BF16 Mixed Precision</p>
    <p>📊 Model Performance: 0.3989 mAP on validation set</p>
    <p>⚡ Differential Learning Rates + Gradient Accumulation</p>
    <p>💻 Running in CPU mode for compatibility</p>
</div>
""", unsafe_allow_html=True)
