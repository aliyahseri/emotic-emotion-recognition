"""
Optimized Inference Pipeline — EMOTIC
=====================================

Enhanced version with torch.compile optimization, batched inference,
and model warmup for faster inference speeds.

Usage:
    pipeline = EMOTICInferenceOptimized("checkpoints/best_model.pth")
    results = pipeline.predict_image("photo.jpg")
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional

import torch
import cv2
from PIL import Image
from torchvision import transforms

from src.model.emotic_model import EMOTICModel
from src.dataset.emotic_dataset import EMOTIC_CATEGORIES, get_transforms
from src.explainability.gradcam import EMOTICGradCAM


# ── Person detector ───────────────────────────────────────────────────────────
class PersonDetector:
    """
    Lightweight wrapper around YOLOv5 for person bounding box detection.
    Falls back to a full-image pseudo-box if YOLO is unavailable.
    """

    def __init__(self, conf_threshold: float = 0.4):
        self.conf_threshold = conf_threshold
        self._model = None
        self._load()

    def _load(self):
        if os.environ.get("EMOTIC_DISABLE_YOLO", "0") == "1":
            self._model = None
            return
        try:
            self._model = torch.hub.load(
                "ultralytics/yolov5", "yolov5s", pretrained=True, verbose=False
            )
            self._model.eval()
            self._model.classes = [0]  # person class only
        except Exception as e:
            print(f"[PersonDetector] YOLOv5 not available ({e}). Using full-image box.")
            self._model = None

    @torch.no_grad()
    def detect(self, image_np: np.ndarray) -> List[List[int]]:
        """
        Detect persons in a BGR numpy image.

        Returns:
            list of [x1, y1, x2, y2] boxes (pixel coords)
        """
        h, w = image_np.shape[:2]

        if self._model is None:
            return [[0, 0, w, h]]

        rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = self._model(rgb)
        det = results.pandas().xyxy[0]
        persons = det[(det["name"] == "person") & (det["confidence"] >= self.conf_threshold)]

        boxes = []
        for _, row in persons.iterrows():
            boxes.append([
                int(row["xmin"]), int(row["ymin"]),
                int(row["xmax"]), int(row["ymax"]),
            ])

        if not boxes:
            boxes = [[0, 0, w, h]]  # fallback: whole image

        return boxes


# ── Main inference pipeline ───────────────────────────────────────────────────
class EMOTICInferenceOptimized:
    """
    End-to-end EMOTIC inference pipeline with optimizations.

    Args:
        checkpoint_path  : path to trained model .pth file
        device           : 'cuda' / 'cpu' / None (auto)
        cat_threshold    : sigmoid threshold for binary category prediction
        use_gradcam      : whether to compute Grad-CAM explanations
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        cat_threshold: float = 0.35,
        use_gradcam: bool = False,
    ):
        self.cat_threshold = cat_threshold
        self.use_gradcam = use_gradcam

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model = self._load_model(checkpoint_path)
        self.detector = PersonDetector()

        body_tf, ctx_tf = get_transforms("val")
        self.body_transform = body_tf
        self.context_transform = ctx_tf

        if use_gradcam:
            self.gradcam = EMOTICGradCAM(self.model, self.device)

    def _load_model(self, path: str) -> EMOTICModel:
        """Load model with torch.compile optimization and warmup."""
        ckpt = torch.load(path, map_location="cpu")
        model = EMOTICModel(pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        
        # Apply torch.compile for optimization (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead")
                print("[EMOTICInferenceOptimized] torch.compile enabled for faster inference")
        except Exception as e:
            print(f"[EMOTICInferenceOptimized] torch.compile failed: {e}")
        
        # Warm up model with dummy input
        try:
            dummy_body = torch.zeros(1, 3, 128, 128).to(self.device)
            dummy_context = torch.zeros(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = model(dummy_body, dummy_context)
            print("[EMOTICInferenceOptimized] Model warmup completed")
        except Exception as e:
            print(f"[EMOTICInferenceOptimized] Model warmup failed: {e}")
        
        return model

    def _preprocess(self, image_np: np.ndarray, box: List[int]):
        """
        Crop person from image and prepare both tensors.
        """
        h, w = image_np.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        body_pil = pil_image.crop((x1, y1, x2, y2))

        body_t = self.body_transform(body_pil).unsqueeze(0).to(self.device)
        ctx_t = self.context_transform(pil_image).unsqueeze(0).to(self.device)
        return body_t, ctx_t

    @torch.no_grad()
    def _run_model(self, body_t, ctx_t):
        cat_logits, vad_pred = self.model(body_t, ctx_t)
        cat_probs = torch.sigmoid(cat_logits).squeeze(0).cpu().numpy()
        vad = vad_pred.squeeze(0).cpu().numpy()
        return cat_probs, vad

    def predict_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Run full inference on an image file.

        Returns:
            list of dicts, one per detected person:
            {
                "box"             : [x1, y1, x2, y2],
                "cat_probs"       : np.ndarray (26,)   sigmoid probabilities,
                "categories"      : list[str]  categories above threshold,
                "cat_scores"      : list[float] matching scores,
                "vad"             : {"valence": float, "arousal": float, "dominance": float},
                "body_cam"        : np.ndarray or None (H×W, [0,1]),
                "context_cam"     : np.ndarray or None (H×W, [0,1]),
            }
        """
        image_np = cv2.imread(image_path)
        if image_np is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        return self.predict_array(image_np)

    def predict_array(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        """Batch-optimized inference on BGR numpy array."""
        boxes = self.detector.detect(image_np)
        results = []

        # Batch processing for multiple people
        if len(boxes) > 1:
            body_tensors = []
            ctx_tensors = []
            original_boxes = []
            
            for box in boxes:
                try:
                    body_t, ctx_t = self._preprocess(image_np, box)
                    body_tensors.append(body_t)
                    ctx_tensors.append(ctx_t)
                    original_boxes.append(box)
                except Exception as e:
                    print(f"[EMOTICInferenceOptimized] Error preprocessing box {box}: {e}")
                    continue
            
            if body_tensors:
                # Batch inference
                body_batch = torch.cat(body_tensors, dim=0)
                ctx_batch = torch.cat(ctx_tensors, dim=0)
                
                cat_logits_batch, vad_pred_batch = self.model(body_batch, ctx_batch)
                cat_probs_batch = torch.sigmoid(cat_logits_batch).cpu().numpy()
                vad_batch = vad_pred_batch.cpu().numpy()
                
                # Process batch results
                for i, box in enumerate(original_boxes):
                    cat_probs = cat_probs_batch[i]
                    vad = vad_batch[i]
                    
                    # Filter categories above threshold
                    active_idx = np.where(cat_probs >= self.cat_threshold)[0]
                    active_idx = active_idx[np.argsort(cat_probs[active_idx])[::-1]]
                    categories = [EMOTIC_CATEGORIES[i] for i in active_idx]
                    cat_scores = [float(cat_probs[i]) for i in active_idx]

                    vad_dict = {
                        "valence": float(vad[0]),
                        "arousal": float(vad[1]),
                        "dominance": float(vad[2]),
                    }

                    body_cam, ctx_cam = None, None
                    if self.use_gradcam:
                        # Note: GradCAM not supported for batched inference
                        pass

                    results.append({
                        "box": box,
                        "cat_probs": cat_probs,
                        "categories": categories,
                        "cat_scores": cat_scores,
                        "vad": vad_dict,
                        "body_cam": body_cam,
                        "context_cam": ctx_cam,
                    })
        else:
            # Single person processing (original logic)
            for box in boxes:
                try:
                    body_t, ctx_t = self._preprocess(image_np, box)
                    cat_probs, vad = self._run_model(body_t, ctx_t)

                    active_idx = np.where(cat_probs >= self.cat_threshold)[0]
                    active_idx = active_idx[np.argsort(cat_probs[active_idx])[::-1]]
                    categories = [EMOTIC_CATEGORIES[i] for i in active_idx]
                    cat_scores = [float(cat_probs[i]) for i in active_idx]

                    vad_dict = {
                        "valence": float(vad[0]),
                        "arousal": float(vad[1]),
                        "dominance": float(vad[2]),
                    }

                    body_cam, ctx_cam = None, None
                    if self.use_gradcam:
                        top_class = int(active_idx[0]) if active_idx.size > 0 else None
                        body_cam, ctx_cam = self.gradcam.generate(
                            body_t, ctx_t, target_class=top_class
                        )

                    results.append({
                        "box": box,
                        "cat_probs": cat_probs,
                        "categories": categories,
                        "cat_scores": cat_scores,
                        "vad": vad_dict,
                        "body_cam": body_cam,
                        "context_cam": ctx_cam,
                    })
                except Exception as e:
                    print(f"[EMOTICInferenceOptimized] Error processing box {box}: {e}")
                    continue

        return results

    def visualize(
        self,
        image_np: np.ndarray,
        results: List[Dict[str, Any]],
        show_top_k: int = 3,
    ) -> np.ndarray:
        """
        Draw bounding boxes and emotion labels on a copy of the image.
        Returns annotated BGR image.
        """
        vis = image_np.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for res in results:
            x1, y1, x2, y2 = res["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)

            # Draw top-k categories
            y_text = max(y1 - 10, 15)
            for k in range(min(show_top_k, len(res["categories"]))):
                label = f"{res['categories'][k]}: {res['cat_scores'][k]:.2f}"
                cv2.putText(vis, label, (x1, y_text), font, 0.5, (0, 200, 255), 1)
                y_text -= 18

            # Draw VAD
            vad_txt = (
                f"V={res['vad']['valence']:.2f} "
                f"A={res['vad']['arousal']:.2f} "
                f"D={res['vad']['dominance']:.2f}"
            )
            cv2.putText(vis, vad_txt, (x1, y2 + 18), font, 0.5, (255, 255, 255), 1)

        return vis


# ── Ensemble version with optimizations ───────────────────────────────────────
class EMOTICEnsembleInferenceOptimized:
    def __init__(
        self,
        checkpoint_path_a: str,
        checkpoint_path_b: str,
        device: Optional[str] = None,
        cat_threshold: float = 0.35,
        use_gradcam: bool = False,
    ):
        self.cat_threshold = cat_threshold
        self.use_gradcam = use_gradcam

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model_a = self._load_model(checkpoint_path_a)
        self.model_b = self._load_model(checkpoint_path_b)
        self.detector = PersonDetector()

        body_tf, ctx_tf = get_transforms("val")
        self.body_transform = body_tf
        self.context_transform = ctx_tf

        if use_gradcam:
            self.gradcam = EMOTICGradCAM(self.model_a, self.device)

    def _load_model(self, path: str) -> EMOTICModel:
        """Load model with torch.compile optimization and warmup."""
        ckpt = torch.load(path, map_location="cpu")
        model = EMOTICModel(pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        
        # Apply torch.compile for optimization (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead")
                print("[EMOTICEnsembleOptimized] torch.compile enabled for faster inference")
        except Exception as e:
            print(f"[EMOTICEnsembleOptimized] torch.compile failed: {e}")
        
        # Warm up model with dummy input
        try:
            dummy_body = torch.zeros(1, 3, 128, 128).to(self.device)
            dummy_context = torch.zeros(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                _ = model(dummy_body, dummy_context)
            print("[EMOTICEnsembleOptimized] Model warmup completed")
        except Exception as e:
            print(f"[EMOTICEnsembleOptimized] Model warmup failed: {e}")
        
        return model

    def _preprocess(self, image_np: np.ndarray, box: List[int]):
        h, w = image_np.shape[:2]
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        body_pil = pil_image.crop((x1, y1, x2, y2))

        body_t = self.body_transform(body_pil).unsqueeze(0).to(self.device)
        ctx_t = self.context_transform(pil_image).unsqueeze(0).to(self.device)
        return body_t, ctx_t

    @torch.no_grad()
    def _run_models(self, body_t, ctx_t):
        cat_logits_a, vad_pred_a = self.model_a(body_t, ctx_t)
        cat_logits_b, vad_pred_b = self.model_b(body_t, ctx_t)

        cat_probs_a = torch.sigmoid(cat_logits_a).squeeze(0)
        cat_probs_b = torch.sigmoid(cat_logits_b).squeeze(0)
        cat_probs = ((cat_probs_a + cat_probs_b) / 2.0).cpu().numpy()

        vad_a = vad_pred_a.squeeze(0)
        vad_b = vad_pred_b.squeeze(0)
        vad = ((vad_a + vad_b) / 2.0).cpu().numpy()

        return cat_probs, vad

    def predict_array(self, image_np: np.ndarray) -> List[Dict[str, Any]]:
        boxes = self.detector.detect(image_np)
        results = []

        for box in boxes:
            body_t, ctx_t = self._preprocess(image_np, box)
            cat_probs, vad = self._run_models(body_t, ctx_t)

            active_idx = np.where(cat_probs >= self.cat_threshold)[0]
            active_idx = active_idx[np.argsort(cat_probs[active_idx])[::-1]]
            categories = [EMOTIC_CATEGORIES[i] for i in active_idx]
            cat_scores = [float(cat_probs[i]) for i in active_idx]

            vad_dict = {
                "valence": float(vad[0]),
                "arousal": float(vad[1]),
                "dominance": float(vad[2]),
            }

            body_cam, ctx_cam = None, None
            if self.use_gradcam:
                top_class = int(active_idx[0]) if active_idx.size > 0 else None
                body_cam, ctx_cam = self.gradcam.generate(
                    body_t, ctx_t, target_class=top_class
                )

            results.append({
                "box": box,
                "cat_probs": cat_probs,
                "categories": categories,
                "cat_scores": cat_scores,
                "vad": vad_dict,
                "body_cam": body_cam,
                "context_cam": ctx_cam,
            })

        return results

    def visualize(
        self,
        image_np: np.ndarray,
        results: List[Dict[str, Any]],
        show_top_k: int = 3,
    ) -> np.ndarray:
        vis = image_np.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX

        for res in results:
            x1, y1, x2, y2 = res["box"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)

            y_text = max(y1 - 10, 15)
            for k in range(min(show_top_k, len(res["categories"]))):
                label = f"{res['categories'][k]}: {res['cat_scores'][k]:.2f}"
                cv2.putText(vis, label, (x1, y_text), font, 0.5, (0, 200, 255), 1)
                y_text -= 18

            vad_txt = (
                f"V:{res['vad']['valence']:.2f} "
                f"A:{res['vad']['arousal']:.2f} "
                f"D:{res['vad']['dominance']:.2f}"
            )
            cv2.putText(vis, vad_txt, (x1, y2 + 18), font, 0.5, (255, 255, 255), 1)

        return vis
