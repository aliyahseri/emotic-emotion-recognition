#!/usr/bin/env python3
"""
Enhanced EMOTIC Demo App - Streamlit
=====================================

Interactive web application for context-based emotion recognition with:
- Side-by-side model comparison
- VAD radar visualization
- Confidence bands and uncertainty indicators
- Example images for quick testing
- Inference time display

Run with:
    streamlit run app/demo_app.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import cv2
from PIL import Image

from src.inference.inference_pipeline import EMOTICInference, EMOTICEnsembleInference
from src.dataset.emotic_dataset import EMOTIC_CATEGORIES

# Page config
st.set_page_config(
    page_title="EMOTIC — Emotion in Context",
    page_icon="🎭",
    layout="wide",
)

# Styling
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.title-block { text-align: center; padding: 20px 0; }
.emotion-chip {
    display: inline-block; padding: 4px 12px; margin: 3px;
    border-radius: 20px; background: #1e3a5f; color: #7ec8e3;
    font-size: 0.85rem; font-weight: 600;
}
.confidence-bar { height: 8px; border-radius: 4px; background: #e0e0e0; margin: 4px 0; }
.confidence-fill { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #1e3a5f, #7ec8e3); }
.improvement-green { color: #28a745; font-weight: bold; }
.improvement-red { color: #dc3545; font-weight: bold; }
.uncertainty-low { color: #28a745; }
.uncertainty-medium { color: #ffc107; }
.uncertainty-high { color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 EMOTIC Emotion Recognition")
st.caption("Multi-label emotion classification with VAD regression | AffectNet-pretrained backbone")

# Sidebar
st.sidebar.header("⚙️ Settings")
use_ensemble = st.sidebar.checkbox("Use ensemble (2 checkpoints)", value=False)
use_comparison = st.sidebar.checkbox("Enable model comparison", value=True)

ckpt_path = st.sidebar.text_input(
    "Checkpoint path (A)", value="checkpoints_real/best_model.pth"
)
ckpt_path_b = st.sidebar.text_input(
    "Checkpoint path (B)", value="checkpoints/best_model.pth",
    disabled=not use_ensemble,
)

cat_threshold = st.sidebar.slider("Category confidence threshold", 0.1, 0.9, 0.35, 0.05)
show_top_k = st.sidebar.slider("Top-K categories to display", 1, 10, 5)
use_gradcam = st.sidebar.checkbox("Show Grad-CAM explanations", value=False)

# Model info section
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("- Backbone: ResNet-50")
st.sidebar.markdown("- Test mAP: 0.7634")
st.sidebar.markdown("- Training data: AffectNet")
st.sidebar.markdown("- Eval data: EMOTIC")

# Example images
EXAMPLE_IMAGES = {
    "Happy Person": "assets/examples/happy_person.jpg",
    "Sad Person": "assets/examples/sad_person.jpg", 
    "Angry Person": "assets/examples/angry_person.jpg",
    "Surprised Person": "assets/examples/surprised_person.jpg",
    "Neutral Person": "assets/examples/neutral_person.jpg",
    "Group Photo": "assets/examples/group_photo.jpg"
}

# Load models
@st.cache_resource
def load_pipeline(ckpt, use_ensemble_flag, ckpt_b, gradcam_flag):
    try:
        if use_ensemble_flag:
            pipeline = EMOTICEnsembleInference(
                checkpoint_path_a=ckpt,
                checkpoint_path_b=ckpt_b,
                use_gradcam=gradcam_flag,
            )
        else:
            pipeline = EMOTICInference(
                checkpoint_path=ckpt,
                use_gradcam=gradcam_flag,
            )
        return pipeline, None
    except Exception as e:
        return None, str(e)

# Load calibration parameters if available
@st.cache_resource
def load_calibration_params(checkpoint_path):
    calib_path = Path("results/calibration_params.json")
    if calib_path.exists():
        try:
            with open(calib_path, 'r') as f:
                params = json.load(f)
            return params.get("temperature", 1.0)
        except:
            return 1.0
    return 1.0

# Model loading
pipeline_a, load_err_a = load_pipeline(ckpt_path, use_ensemble, ckpt_path_b, use_gradcam)
temperature = load_calibration_params(ckpt_path)

if load_err_a:
    st.error(f"⚠️ Could not load model: {load_err_a}")
    st.info("Please train the model first or provide a valid checkpoint path.")

# Example image selector
st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Quick Examples")
selected_example = st.sidebar.selectbox("Choose an example image:", ["None"] + list(EXAMPLE_IMAGES.keys()))
load_example = st.sidebar.button("Load Example")

# Image upload
st.markdown("---")
col_upload, col_example = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader(
        "Upload an image containing people",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

with col_example:
    if load_example and selected_example != "None":
        example_path = EXAMPLE_IMAGES[selected_example]
        if Path(example_path).exists():
            uploaded = open(example_path, 'rb')
        else:
            st.warning(f"Example image not found: {example_path}")

# Process image
if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    # Display original image
    st.subheader("📷 Original Image")
    st.image(pil_img, use_container_width=True)
    
    # Add horizontal divider
    st.markdown("---")

    if pipeline_a is None:
        st.warning("Model not loaded — showing demo output only.")
        st.stop()

    # Apply temperature scaling if available
    if temperature != 1.0:
        pipeline_a.temperature = temperature

    # Set threshold
    pipeline_a.cat_threshold = float(cat_threshold)

    # Run inference with timing
    start_time = time.perf_counter()
    with st.spinner("🔍 Detecting people and predicting emotions…"):
        results = pipeline_a.predict_array(image_np)
    inference_time = (time.perf_counter() - start_time) * 1000

    # Display inference time
    st.info(f"⚡ Inference time: {inference_time:.1f}ms")
    
    # Add results summary box
    if results:
        res = results[0]  # Use first person
        top_emotion = res["categories"][0] if res["categories"] else "None"
        top_conf = res["cat_scores"][0] if res["cat_scores"] else 0.0
        uncertainty = compute_uncertainty(res["cat_probs"])
        uncertainty_label = get_uncertainty_class(uncertainty).replace("uncertainty-", "").capitalize()
        
        st.success(f"Top emotion: **{top_emotion}** ({top_conf:.1%} confidence) | Uncertainty: {uncertainty_label}")

    if use_comparison and not use_ensemble:
        # Load second model for comparison
        ckpt_old = "checkpoints/best_model.pth"
        pipeline_b, load_err_b = load_pipeline(ckpt_old, False, "", use_gradcam)
        
        if not load_err_b:
            pipeline_b.cat_threshold = float(cat_threshold)
            if temperature != 1.0:
                pipeline_b.temperature = temperature
            
            results_b = pipeline_b.predict_array(image_np)
        else:
            results_b = None
        
        # Side-by-side comparison
        st.markdown("---")
        st.subheader("🆚 Model Comparison")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### **New Model**")
            annotated_a = pipeline_a.visualize(image_np, results, show_top_k=show_top_k)
            st.image(cv2.cvtColor(annotated_a, cv2.COLOR_BGR2RGB), use_container_width=True)
            display_results(results, "New")
        
        with col_b:
            if results_b:
                st.markdown("### **Old Model**")
                annotated_b = pipeline_b.visualize(image_np, results_b, show_top_k=show_top_k)
                st.image(cv2.cvtColor(annotated_b, cv2.COLOR_BGR2RGB), use_container_width=True)
                display_results(results_b, "Old")
                
                # Show comparison differences
                st.markdown("### **Differences**")
                show_comparison_differences(results, results_b)
            else:
                st.error("Old model failed to load")
    else:
        # Single model or ensemble display
        annotated = pipeline_a.visualize(image_np, results, show_top_k=show_top_k)
        st.subheader("🎭 Emotion Predictions")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
        display_results(results, "Model")

def display_results(results, model_name):
    """Display detailed results for a model."""
    st.markdown(f"### 📊 {model_name} Results — {len(results)} person(s) detected")
    
    for idx, res in enumerate(results):
        with st.expander(f"Person {idx + 1} — BBox {res['box']}", expanded=True):
            c1, c2, c3 = st.columns([2, 2, 1])
            
            with c1:
                st.markdown("**Emotion Categories**")
                chips = " ".join(
                    f'<span class="emotion-chip">{cat} ({score:.2f})</span>'
                    for cat, score in zip(
                        res["categories"][:show_top_k],
                        res["cat_scores"][:show_top_k],
                    )
                )
                st.markdown(chips or "_No category above threshold_", unsafe_allow_html=True)
                
                # Confidence bands
                st.markdown("**Confidence Bands (Top 5)**")
                top_5_probs = sorted(res["cat_probs"], reverse=True)[:5]
                top_5_cats = [EMOTIC_CATEGORIES[np.where(res["cat_probs"] == p)[0][0]] 
                             for p in top_5_probs]
                
                for cat, prob in zip(top_5_cats, top_5_probs):
                    pct = int(prob * 100)
                    st.markdown(
                        f"{cat}: {prob:.3f}  \n"
                        f'<div class="confidence-bar"><div class="confidence-fill" style="width:{pct}%"></div></div>',
                        unsafe_allow_html=True,
                    )
                
                # Uncertainty indicator
                uncertainty = compute_uncertainty(res["cat_probs"])
                uncertainty_class = get_uncertainty_class(uncertainty)
                st.markdown(f"**Uncertainty**: <span class='uncertainty-{uncertainty_class}'>{uncertainty:.3f}</span>", 
                          unsafe_allow_html=True)
            
            with c2:
                st.markdown("**VAD Visualization**")
                create_vad_radar(res["vad"])
                
                # VAD values
                vad = res["vad"]
                st.markdown("**VAD Values**")
                for dim, val in vad.items():
                    st.markdown(f"**{dim.capitalize()}**: {val:.3f}")
            
            with c3:
                # Person crop
                st.markdown("**Person Crop**")
                x1, y1, x2, y2 = res["box"]
                h_img, w_img = image_np.shape[:2]
                crop = image_np[
                    max(0, y1):min(h_img, y2),
                    max(0, x1):min(w_img, x2),
                ]
                if crop.size > 0:
                    st.image(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                        caption="Person crop",
                        use_container_width=True,
                    )

def show_comparison_differences(results_a, results_b):
    """Show differences between two model predictions."""
    if len(results_a) != len(results_b):
        st.warning("Different number of people detected - cannot compare")
        return
    
    for i, (res_a, res_b) in enumerate(zip(results_a, results_b)):
        st.markdown(f"**Person {i+1} Differences**")
        
        # Compare top categories
        cats_a = set(res_a["categories"][:3])
        cats_b = set(res_b["categories"][:3])
        
        only_in_a = cats_a - cats_b
        only_in_b = cats_b - cats_a
        common = cats_a & cats_b
        
        if only_in_a:
            st.markdown(f"<span class='improvement-green'>Only in New: {', '.join(only_in_a)}</span>", 
                      unsafe_allow_html=True)
        if only_in_b:
            st.markdown(f"<span class='improvement-red'>Only in Old: {', '.join(only_in_b)}</span>", 
                      unsafe_allow_html=True)
        if common:
            st.markdown(f"Common: {', '.join(common)}")
        
        # Compare VAD
        vad_a = res_a["vad"]
        vad_b = res_b["vad"]
        
        for dim in ["valence", "arousal", "dominance"]:
            diff = vad_a[dim] - vad_b[dim]
            if abs(diff) > 0.1:
                color_class = "improvement-green" if diff > 0 else "improvement-red"
                st.markdown(f"{dim.capitalize()}: <span class='{color_class}'>{diff:+.3f}</span>", 
                          unsafe_allow_html=True)

def create_vad_radar(vad_dict):
    """Create radar chart for VAD visualization."""
    import matplotlib.pyplot as plt
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(projection='polar'))
    
    # Data
    categories = ['Valence', 'Arousal', 'Dominance']
    values = [vad_dict['valence'], vad_dict['arousal'], vad_dict['dominance']]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    # Values
    values += values[:1]  # Complete the loop
    
    # Plot with specified colors
    colors = {'Valence': 'green', 'Arousal': 'orange', 'Dominance': 'purple'}
    ax.plot(angles, values, 'o-', linewidth=2, color='#1e3a5f')
    ax.fill(angles, values, alpha=0.25, color='#7ec8e3')
    
    # Add labels with proper axis names
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Valence', 'Arousal', 'Dominance'])
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    # Remove y-axis labels
    ax.set_yticklabels([])
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def compute_uncertainty(probabilities):
    """Compute prediction uncertainty using entropy."""
    eps = 1e-8
    probs = np.clip(probabilities, eps, 1 - eps)
    entropy = -np.sum(probs * np.log(probs))
    max_entropy = np.log(len(probabilities))
    return entropy / max_entropy

def get_uncertainty_class(uncertainty):
    """Get CSS class for uncertainty level."""
    if uncertainty < 0.3:
        return "low"
    elif uncertainty < 0.7:
        return "medium"
    else:
        return "high"

# Global variables for display functions
show_top_k = 5
image_np = None  # Will be set in main flow
