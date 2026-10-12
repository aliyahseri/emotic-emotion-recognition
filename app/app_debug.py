"""
EMOTIC Demo App — Debug Version (Low Threshold)
===============================================
Shows all emotion predictions for debugging
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
import cv2
from PIL import Image
import torch

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
.low-confidence { background: #666; color: #ccc; }
.high-confidence { background: #1e3a5f; color: #7ec8e3; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-block"><h1>🎭 EMOTIC: Emotion in Context</h1>'
            '<p>Debug Mode - Show All Predictions</p></div>',
            unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Settings")
ckpt_path = st.sidebar.text_input(
    "Checkpoint path", value="checkpoints/best_model.pth"
)
cat_threshold = st.sidebar.slider(
    "Category confidence threshold", 0.01, 0.9, 0.1, 0.01,  # Much lower threshold
    help="Show all predictions above this threshold"
)
show_all = st.sidebar.checkbox("Show ALL predictions (ignore threshold)", value=True)
show_top_k = st.sidebar.slider("Top-K categories to display", 1, 26, 10)  # Show more

@st.cache_resource
def load_model_cpu(ckpt_path):
    try:
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
st.sidebar.warning("🔍 Debug Mode - Showing all predictions")

def preprocess_image(image):
    img_array = np.array(image)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    body_img = cv2.resize(img_array, (128, 128))
    context_img = cv2.resize(img_array, (224, 224))
    
    body_tensor = torch.from_numpy(body_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    context_tensor = torch.from_numpy(context_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    return body_tensor.to(device), context_tensor.to(device), img_array

# Image upload
uploaded = st.file_uploader(
    "📤 Upload an image containing people",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Upload any image with people to analyze emotions"
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
        st.subheader("🎭 All Emotion Predictions")
        
        # Show ALL emotions with their confidence scores
        st.write("**All 26 Emotion Categories (sorted by confidence):**")
        
        # Sort all emotions by probability
        all_indices = np.argsort(cat_probs)[::-1]  # Descending order
        
        for i, idx in enumerate(all_indices):
            prob = cat_probs[idx]
            emotion = EMOTIC_CATEGORIES[idx]
            
            # Color code by confidence
            if prob >= cat_threshold:
                css_class = "emotion-chip high-confidence"
                threshold_text = f"✅ Above {cat_threshold:.2f}"
            else:
                css_class = "emotion-chip low-confidence"
                threshold_text = f"⚠️ Below {cat_threshold:.2f}"
            
            if show_all or prob >= cat_threshold:
                st.markdown(
                    f'<span class="{css_class}">{i+1}. {emotion}: {prob:.4f} {threshold_text}</span>',
                    unsafe_allow_html=True
                )
        
        # Show threshold info
        st.write(f"**Current threshold:** {cat_threshold:.2f}")
        st.write(f"**Emotions above threshold:** {sum(cat_probs >= cat_threshold)}/26")
        st.write(f"**Highest confidence:** {cat_probs.max():.4f}")
        
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
            normalized = max(0, min(1, (value + 1) / 2))
            st.write(f"{name}: {value:.3f}")
            st.progress(float(normalized))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🔍 Debug Mode - Shows ALL predictions regardless of confidence</p>
    <p>📊 Adjust threshold to filter low-confidence predictions</p>
    <p>💻 CPU mode for compatibility</p>
</div>
""", unsafe_allow_html=True)
