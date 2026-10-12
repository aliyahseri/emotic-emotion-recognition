#!/usr/bin/env python3
"""
Confidence Calibration for EMOTIC Models
==========================================

Implements temperature scaling calibration to improve confidence estimates
and provides uncertainty quantification for emotion predictions.

Usage:
    python calibration.py --checkpoint checkpoints_real/best_model.pth --val-csv data/real_dataset/val.csv

Outputs:
    - calibration_params.json (learned temperature)
    - calibrated_model.pth (checkpoint with calibration parameters)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.dataset.emotic_dataset import EMOTICDataset, get_transforms
from src.model.emotic_model import EMOTICModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for confidence calibration.
    
    Learns a single temperature parameter T to scale logits before sigmoid:
        p = sigmoid(logits / T)
    
    When T > 1, probabilities become less confident (more calibrated).
    When T < 1, probabilities become more confident.
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature


class ModelCalibrator:
    """Calibrates EMOTIC model using temperature scaling."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize calibrator with a model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            device: Device to run calibration on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.temperature_scaling = None
        self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint."""
        try:
            logger.info(f"Loading model from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            
            self.model = EMOTICModel(pretrained=False)
            self.model.load_state_dict(checkpoint["model"])
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize temperature scaling
            self.temperature_scaling = TemperatureScaling().to(self.device)
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def fit_temperature(self, val_dataset: EMOTICDataset, 
                      learning_rate: float = 0.01, max_epochs: int = 100) -> float:
        """
        Fit temperature scaling on validation set.
        
        Args:
            val_dataset: Validation dataset
            learning_rate: Learning rate for temperature optimization
            max_epochs: Maximum number of optimization epochs
            
        Returns:
            Learned temperature value
        """
        logger.info("Fitting temperature scaling...")
        
        # Collect all logits and targets from validation set
        all_logits = []
        all_targets = []
        
        for i in tqdm(range(len(val_dataset)), desc="Collecting validation predictions"):
            try:
                sample = val_dataset[i]
                
                # Get tensors
                body = sample["body"].unsqueeze(0).to(self.device)
                context = sample["context"].unsqueeze(0).to(self.device)
                targets = sample["categories"].unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, _ = self.model(body, context)
                
                all_logits.append(logits.cpu())
                all_targets.append(targets.cpu())
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        if not all_logits:
            raise ValueError("No valid samples collected from validation set")
        
        # Concatenate all data
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        logger.info(f"Collected {len(all_logits)} samples for calibration")
        
        # Optimize temperature
        optimizer = optim.LBFGS(
            [self.temperature_scaling.temperature], 
            lr=learning_rate,
            max_iter=max_epochs
        )
        
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        def eval_loss():
            """Evaluate loss with current temperature."""
            self.temperature_scaling.train()
            optimizer.zero_grad()
            
            scaled_logits = self.temperature_scaling(all_logits.to(self.device))
            loss = criterion(scaled_logits, all_targets.to(self.device))
            
            loss.backward()
            return loss
        
        # Optimize
        optimizer.step(eval_loss)
        
        # Get final temperature
        final_temperature = self.temperature_scaling.temperature.item()
        logger.info(f"Learned temperature: {final_temperature:.4f}")
        
        return final_temperature
    
    def save_calibration_params(self, temperature: float, output_path: str):
        """Save calibration parameters to JSON file."""
        params = {
            "temperature": temperature,
            "checkpoint_path": self.checkpoint_path,
            "calibration_method": "temperature_scaling"
        }
        
        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Calibration parameters saved to {output_path}")
    
    def save_calibrated_model(self, temperature: float, output_path: str):
        """Save model with calibration parameters."""
        # Load original checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        
        # Add calibration parameters
        checkpoint["calibration"] = {
            "temperature": temperature,
            "method": "temperature_scaling"
        }
        
        # Save updated checkpoint
        torch.save(checkpoint, output_path)
        logger.info(f"Calibrated model saved to {output_path}")
    
    @staticmethod
    def compute_uncertainty(probabilities: np.ndarray) -> float:
        """
        Compute prediction uncertainty using entropy.
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Entropy-based uncertainty score
        """
        # Avoid log(0)
        eps = 1e-8
        probabilities = np.clip(probabilities, eps, 1 - eps)
        
        # Compute entropy
        entropy = -np.sum(probabilities * np.log(probabilities))
        
        # Normalize by maximum possible entropy (log(n_classes))
        max_entropy = np.log(len(probabilities))
        normalized_entropy = entropy / max_entropy
        
        return normalized_entropy
    
    @staticmethod
    def create_reliability_diagram(targets: np.ndarray, probs: np.ndarray, 
                                 confidences: np.ndarray, n_bins: int = 10,
                                 output_path: str = None) -> Dict[str, Any]:
        """
        Create reliability diagram for confidence calibration.
        
        Args:
            targets: Binary targets (n_samples, n_classes)
            probs: Predicted probabilities (n_samples, n_classes)
            confidences: Maximum confidence per sample (n_samples,)
            n_bins: Number of bins for reliability diagram
            output_path: Path to save diagram
            
        Returns:
            Dictionary with reliability data
        """
        bins = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Compute accuracy for samples in this bin
                accuracy = 0
                for i in range(len(targets)):
                    if in_bin[i]:
                        # Check if prediction is correct (any positive prediction matches target)
                        pred_binary = (probs[i] >= 0.35).astype(int)
                        target_binary = (targets[i] > 0).astype(int)
                        
                        # Simple accuracy: check if any predicted category matches any target category
                        if np.any(pred_binary & target_binary):
                            accuracy += 1
                
                accuracy /= in_bin.sum()
                bin_accuracies.append(accuracy)
                bin_confidences.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                bin_counts.append(0)
        
        # Create plot if output path provided
        if output_path:
            plt.figure(figsize=(10, 8))
            plt.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, markersize=8)
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
            plt.xlabel('Confidence')
            plt.ylabel('Accuracy')
            plt.title('Reliability Diagram')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        return {
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "bin_boundaries": list(bin_lowers)
        }


def create_val_dataset(val_csv: str, image_root: str) -> EMOTICDataset:
    """Create validation dataset for calibration."""
    _, val_transforms = get_transforms("val")
    return EMOTICDataset(
        annotation_csv=val_csv,
        image_root=image_root,
        body_transform=None,
        context_transform=val_transforms,
        normalize_vad=True
    )


def main():
    """Main calibration function."""
    parser = argparse.ArgumentParser(description="Calibrate EMOTIC model confidence")
    parser.add_argument("--checkpoint", default="checkpoints_real/best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--val-csv", default="data/real_dataset/val.csv", 
                       help="Path to validation CSV file")
    parser.add_argument("--image-root", default="data/real_dataset/images",
                       help="Path to validation images directory")
    parser.add_argument("--output-dir", default="results", 
                       help="Directory to save calibration results")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--lr", type=float, default=0.01, 
                       help="Learning rate for temperature optimization")
    parser.add_argument("--max-epochs", type=int, default=100,
                       help="Maximum epochs for temperature optimization")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Validate paths
    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not Path(args.val_csv).exists():
        logger.error(f"Validation CSV not found: {args.val_csv}")
        sys.exit(1)
    if not Path(args.image_root).exists():
        logger.error(f"Image root not found: {args.image_root}")
        sys.exit(1)
    
    # Create validation dataset
    logger.info("Creating validation dataset...")
    val_dataset = create_val_dataset(args.val_csv, args.image_root)
    
    # Initialize calibrator
    calibrator = ModelCalibrator(args.checkpoint, args.device)
    
    # Fit temperature scaling
    temperature = calibrator.fit_temperature(
        val_dataset, 
        learning_rate=args.lr, 
        max_epochs=args.max_epochs
    )
    
    # Save calibration parameters
    calibrator.save_calibration_params(
        temperature, 
        output_dir / "calibration_params.json"
    )
    
    # Save calibrated model
    calibrator.save_calibrated_model(
        temperature,
        output_dir / "calibrated_model.pth"
    )
    
    logger.info(f"Calibration complete! Temperature: {temperature:.4f}")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
