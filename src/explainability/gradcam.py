"""
Grad-CAM Explainability for EMOTIC
====================================
Implements Gradient-weighted Class Activation Mapping (Grad-CAM) for both
streams of the EMOTIC dual-stream model.

References:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", ICCV 2017.

Usage:
    gradcam = EMOTICGradCAM(model)
    body_heatmap, ctx_heatmap = gradcam.generate(
        body_tensor, context_tensor, target_class=12  # e.g. "Engagement"
    )
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Optional, Tuple


class GradCAMHook:
    """
    Register forward/backward hooks on a target layer to capture
    activations and gradients needed for Grad-CAM.
    """

    def __init__(self, layer: torch.nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_hook = layer.register_forward_hook(self._save_activations)
        self._bwd_hook = layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


class EMOTICGradCAM:
    """
    Grad-CAM for the EMOTIC dual-stream model.

    Hooks into the last convolutional layer of each ResNet backbone
    (layer4 → the last BasicBlock/Bottleneck).

    Args:
        model : EMOTICModel instance
        device: torch.device
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device

        # Target layers: last conv block of each ResNet
        body_target = list(model.body_stream.features.children())[-1]
        ctx_target = list(model.context_stream.features.children())[-1]

        self.body_hook = GradCAMHook(body_target)
        self.ctx_hook = GradCAMHook(ctx_target)

    def _compute_cam(
        self,
        hook: GradCAMHook,
        input_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Compute single Grad-CAM heatmap from stored activations/gradients.
        Returns H×W numpy array in [0, 1].
        """
        grads = hook.gradients        # (1, C, H', W')
        acts = hook.activations       # (1, C, H', W')

        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original input resolution
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        return cam

    @torch.enable_grad()
    def generate(
        self,
        body: torch.Tensor,
        context: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM heatmaps for body and context streams.

        Args:
            body         : (1, 3, H_b, W_b) body crop tensor
            context      : (1, 3, H_c, W_c) scene image tensor
            target_class : index [0, 25] of the emotion category to explain.
                           If None, uses the highest-scoring predicted class.

        Returns:
            body_cam    : H_b × W_b numpy heatmap in [0, 1]
            context_cam : H_c × W_c numpy heatmap in [0, 1]
        """
        self.model.eval()

        body = body.to(self.device).requires_grad_(True)
        context = context.to(self.device).requires_grad_(True)

        cat_logits, vad_pred = self.model(body, context)
        cat_probs = torch.sigmoid(cat_logits)  # (1, 26)

        if target_class is None:
            target_class = int(cat_probs.argmax(dim=1).item())

        # Backpropagate w.r.t. selected class score
        self.model.zero_grad()
        score = cat_probs[0, target_class]
        score.backward()

        body_h, body_w = body.shape[2], body.shape[3]
        ctx_h, ctx_w = context.shape[2], context.shape[3]

        body_cam = self._compute_cam(self.body_hook, (body_h, body_w))
        ctx_cam = self._compute_cam(self.ctx_hook, (ctx_h, ctx_w))

        return body_cam, ctx_cam

    def overlay_heatmap(
        self,
        heatmap: np.ndarray,
        image_np: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay a Grad-CAM heatmap on a BGR image.

        Args:
            heatmap  : H × W float in [0, 1]
            image_np : H × W × 3 BGR uint8 image (from cv2.imread)
            alpha    : blend weight for the heatmap
            colormap : OpenCV colormap

        Returns:
            blended BGR image (uint8)
        """
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, colormap)
        colored = cv2.resize(colored, (image_np.shape[1], image_np.shape[0]))

        blended = cv2.addWeighted(image_np, 1 - alpha, colored, alpha, 0)
        return blended

    def remove_hooks(self):
        self.body_hook.remove()
        self.ctx_hook.remove()
