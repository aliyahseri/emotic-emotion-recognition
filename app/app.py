"""
EMOTIC Demo App — Streamlit
============================
Interactive web application for context-based emotion recognition.

Run with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
import cv2
from PIL import Image

from src.inference.inference_pipeline import EMOTICInference, EMOTICEnsembleInference
from src.dataset.emotic_dataset import EMOTIC_CATEGORIES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EMOTIC — Emotion in Context",
    page_icon="🎭",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
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

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="title-block"><h1>🎭 EMOTIC: Emotion in Context</h1>'
            '<p>Context-Based Human Emotion Recognition</p></div>',
            unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
use_ensemble = st.sidebar.checkbox("Use ensemble (2 checkpoints)", value=False)
ckpt_path = st.sidebar.text_input(
    "Checkpoint path", value="checkpoints_real/best_model.pth"
)
ckpt_path_b = st.sidebar.text_input(
    "Checkpoint path (B)", value="checkpoints/best_model.pth",
    disabled=not use_ensemble,
)
cat_threshold = st.sidebar.slider(
    "Category confidence threshold", 0.1, 0.9, 0.35, 0.05
)
show_top_k = st.sidebar.slider("Top-K categories to display", 1, 10, 5)
use_gradcam = st.sidebar.checkbox("Show Grad-CAM explanations", value=False)

# ── Load model ─────────────────────────────────────────────────────────────────
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

pipeline, load_err = load_pipeline(ckpt_path, use_ensemble, ckpt_path_b, use_gradcam)

if load_err:
    st.error(f"⚠️ Could not load model: {load_err}")
    st.info("Please train the model first or provide a valid checkpoint path.")

# ── Image upload ──────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image containing people",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is not None:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image_np = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    pil_img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    col_orig, col_result = st.columns(2)

    with col_orig:
        st.subheader("📷 Original Image")
        st.image(pil_img, use_column_width=True)

    if pipeline is None:
        st.warning("Model not loaded — showing demo output only.")
        st.stop()

    if pipeline is not None:
        pipeline.cat_threshold = float(cat_threshold)

    with st.spinner("🔍 Detecting people and predicting emotions…"):
        results = pipeline.predict_array(image_np)
        annotated = pipeline.visualize(image_np, results, show_top_k=show_top_k)

    with col_result:
        st.subheader("🎭 Emotion Predictions")
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_column_width=True)

    st.markdown("---")
    st.subheader(f"📊 Detailed Results — {len(results)} person(s) detected")

    for idx, res in enumerate(results):
        with st.expander(f"Person {idx + 1}  —  BBox {res['box']}", expanded=True):
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

                st.markdown("**Full Probability Distribution**")
                prob_data = {
                    cat: float(res["cat_probs"][i])
                    for i, cat in enumerate(EMOTIC_CATEGORIES)
                }
                sorted_probs = dict(
                    sorted(prob_data.items(), key=lambda x: x[1], reverse=True)[:10]
                )
                import pandas as pd
                st.bar_chart(pd.Series(sorted_probs))

            with c2:
                st.markdown("**Valence / Arousal / Dominance**")
                vad = res["vad"]
                for dim, val in vad.items():
                    pct = int(val * 100)
                    st.markdown(
                        f"**{dim.capitalize()}**: {val:.3f}  \n"
                        f'<div class="vad-bar"><div class="vad-fill" style="width:{pct}%"></div></div>',
                        unsafe_allow_html=True,
                    )

            with c3:
                # Person crop
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
                        use_column_width=True,
                    )

            # Grad-CAM
            if use_gradcam and res["body_cam"] is not None:
                st.markdown("**Grad-CAM Explanations**")
                cam_c1, cam_c2 = st.columns(2)

                with cam_c1:
                    st.markdown("Body stream")
                    body_crop_np = image_np[
                        max(0, y1):min(h_img, y2),
                        max(0, x1):min(w_img, x2),
                    ]
                    body_cam_resized = cv2.resize(
                        res["body_cam"],
                        (body_crop_np.shape[1], body_crop_np.shape[0])
                    )
                    overlaid = pipeline.gradcam.overlay_heatmap(
                        body_cam_resized, body_crop_np
                    )
                    st.image(
                        cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB),
                        use_column_width=True,
                    )

                with cam_c2:
                    st.markdown("Context stream")
                    ctx_overlay = pipeline.gradcam.overlay_heatmap(
                        res["context_cam"], image_np
                    )
                    st.image(
                        cv2.cvtColor(ctx_overlay, cv2.COLOR_BGR2RGB),
                        use_column_width=True,
                    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Based on: Kosti et al., *Context Based Emotion Recognition using EMOTIC*, "
    "IEEE TPAMI 2020. · Dataset: [EMOTIC](http://sunai.uoc.edu/emotic/)"
)
