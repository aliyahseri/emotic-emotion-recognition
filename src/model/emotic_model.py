"""
EMOTIC Dual-Stream CNN Model
============================
Faithfully replicates the architecture from:
    "Context Based Emotion Recognition using EMOTIC"
    Kosti et al., IEEE TPAMI 2020

Architecture:
    Body Stream   → ResNet-50  (pretrained on ImageNet) → 2048-d features
    Context Stream→ ResNet-101 (pretrained on ImageNet) → 2048-d features
    Fusion        → Concat [body_feat ‖ context_feat]  → 4096-d
    Cat Head      → FC(4096→256) → ReLU → Dropout → FC(256→26) → Sigmoid
    VAD Head      → FC(4096→256) → ReLU → Dropout → FC(256→3)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights, ResNet101_Weights


# ── Feature extractor wrapper ───────────────────────────────────────────────
class ResNetFeatureExtractor(nn.Module):
    """
    Wraps a ResNet backbone, removing its classification head.
    Returns a flattened 2048-d feature vector.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet50(weights=weights)
        elif backbone == "resnet101":
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.resnet101(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove average-pool and fc layers; keep everything up to layer4
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 2048

    def forward(self, x):
        x = self.features(x)    # (B, 2048, H', W')
        x = self.pool(x)        # (B, 2048, 1, 1)
        x = x.flatten(1)        # (B, 2048)
        return x


# ── Categorical head ─────────────────────────────────────────────────────────
class CategoricalHead(nn.Module):
    """
    Multi-label classification head.
    Input  : fused 4096-d feature vector
    Output : 26 logits (no sigmoid; applied in BCEWithLogitsLoss)
    """

    def __init__(self, in_dim: int = 4096, num_classes: int = 26, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # (B, 26)  — raw logits


# ── Continuous (VAD) head ────────────────────────────────────────────────────
class ContinuousHead(nn.Module):
    """
    VAD regression head.
    Input  : fused 4096-d feature vector
    Output : 3 continuous values (Valence, Arousal, Dominance)
    """

    def __init__(self, in_dim: int = 4096, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 3),
        )

    def forward(self, x):
        return self.net(x)  # (B, 3)


# ── Full EMOTIC model ────────────────────────────────────────────────────────
class EMOTICModel(nn.Module):
    """
    Full dual-stream EMOTIC model.

    Args:
        pretrained  : use ImageNet weights for backbones
        dropout     : dropout rate in prediction heads
        num_classes : number of emotion categories (default 26)
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.5,
        num_classes: int = 26,
    ):
        super().__init__()

        # ── Streams ──────────────────────────────────────────────────────
        self.body_stream = ResNetFeatureExtractor("resnet50", pretrained)
        self.context_stream = ResNetFeatureExtractor("resnet101", pretrained)

        fused_dim = self.body_stream.out_dim + self.context_stream.out_dim  # 4096

        # ── Prediction heads ──────────────────────────────────────────────
        self.cat_head = CategoricalHead(fused_dim, num_classes, dropout)
        self.vad_head = ContinuousHead(fused_dim, dropout)

    def forward(self, body, context):
        """
        Args:
            body    : tensor (B, 3, 128, 128) — person crop
            context : tensor (B, 3, 224, 224) — full scene image

        Returns:
            cat_logits : (B, 26)  raw logits for categorical emotions
            vad_pred   : (B, 3)   continuous VAD predictions
        """
        body_feat = self.body_stream(body)        # (B, 2048)
        context_feat = self.context_stream(context)  # (B, 2048)

        fused = torch.cat([body_feat, context_feat], dim=1)  # (B, 4096)

        cat_logits = self.cat_head(fused)   # (B, 26)
        vad_pred = self.vad_head(fused)     # (B, 3)

        return cat_logits, vad_pred

    # ── Convenience: predicted probabilities ─────────────────────────────
    @torch.no_grad()
    def predict(self, body, context, cat_threshold: float = 0.5):
        """
        Full inference: returns sigmoid probs, binary predictions, VAD.
        """
        self.eval()
        cat_logits, vad_pred = self.forward(body, context)
        cat_probs = torch.sigmoid(cat_logits)
        cat_binary = (cat_probs >= cat_threshold).float()
        return cat_probs, cat_binary, vad_pred


# ── Loss functions ───────────────────────────────────────────────────────────
class EMOTICLoss(nn.Module):
    """
    Joint loss as described in the paper:
        L_total = L_cat  +  λ * L_vad

    Args:
        lambda_vad       : weight for the regression loss
        cat_pos_weight   : optional tensor (26,) for class imbalance
    """

    def __init__(self, lambda_vad: float = 0.5, cat_pos_weight=None):
        super().__init__()
        self.lambda_vad = lambda_vad
        self.cat_loss_fn = nn.BCEWithLogitsLoss(pos_weight=cat_pos_weight)
        self.vad_loss_fn = nn.MSELoss()

    def forward(self, cat_logits, vad_pred, cat_targets, vad_targets):
        """
        Args:
            cat_logits  : (B, 26) raw logits
            vad_pred    : (B, 3)  predicted VAD
            cat_targets : (B, 26) ground-truth binary labels
            vad_targets : (B, 3)  ground-truth VAD in [0, 1]

        Returns:
            total_loss, cat_loss, vad_loss
        """
        cat_loss = self.cat_loss_fn(cat_logits, cat_targets)
        vad_loss = self.vad_loss_fn(vad_pred, vad_targets)
        total = cat_loss + self.lambda_vad * vad_loss
        return total, cat_loss, vad_loss


# ── Quick model summary ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EMOTICModel(pretrained=False)
    body = torch.randn(2, 3, 128, 128)
    ctx = torch.randn(2, 3, 224, 224)
    cat_logits, vad = model(body, ctx)
    print(f"cat_logits shape : {cat_logits.shape}")  # (2, 26)
    print(f"vad shape        : {vad.shape}")          # (2,  3)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters : {total_params:,}")
