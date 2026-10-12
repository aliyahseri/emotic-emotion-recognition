#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for EMOTIC Models
=================================================

Evaluates two EMOTIC model checkpoints on a held-out test set and generates
detailed comparison reports with visualizations.

Usage:
    python evaluate.py --old checkpoints/best_model.pth --new checkpoints_real/best_model.pth

Outputs:
    - results/mAP_comparison.png
    - results/vad_mae_comparison.png  
    - results/confusion_matrix.png
    - results/reliability_diagram.png
    - evaluation_report.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from src.dataset.emotic_dataset import EMOTIC_CATEGORIES, EMOTICDataset, get_transforms
from src.model.emotic_model import EMOTICModel
from src.inference.inference_pipeline import EMOTICInference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


class ModelEvaluator:
    """Evaluates EMOTIC models with comprehensive metrics."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize evaluator with a model checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load model from checkpoint."""
        try:
            logger.info(f"Loading model from {self.checkpoint_path}")
            self.pipeline = EMOTICInference(
                checkpoint_path=self.checkpoint_path,
                device=str(self.device),
                cat_threshold=0.0  # Don't threshold for evaluation
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_dataset(self, dataset: EMOTICDataset) -> Dict[str, Any]:
        """
        Evaluate model on entire dataset.
        
        Args:
            dataset: EMOTICDataset instance
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating on {len(dataset)} samples")
        
        all_predictions = []
        all_targets_cat = []
        all_targets_vad = []
        all_confidences = []
        
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            try:
                sample = dataset[i]
                
                # Get image and boxes (assuming full image as one person for evaluation)
                image_path = sample["image_path"]
                
                # Load image using dataset's resolved path
                image_np = cv2.imread(sample["image_path"])
                if image_np is None:
                    logger.warning(f"Could not load image: {sample['image_path']}")
                    continue
                
                # Run inference
                results = self.pipeline.predict_array(image_np)
                
                if not results:
                    continue
                
                # Use first detected person
                result = results[0]
                
                # Store predictions
                all_predictions.append(result)
                
                # Store targets
                all_targets_cat.append(sample["categories"].numpy())
                all_targets_vad.append(sample["vad"].numpy())
                
                # Store confidence (max probability for calibration)
                probs = result["cat_probs"]
                all_confidences.append(np.max(probs))
                
                # Debug: print first successful sample
                if len(all_predictions) == 1:
                    print(f"DEBUG: First successful sample processed")
                    print(f"  Categories: {result.get('categories', [])}")
                    print(f"  VAD: {result.get('vad', {})}")
                    print(f"  Targets categories shape: {sample['categories'].numpy().shape}")
                    print(f"  Targets VAD shape: {sample['vad'].numpy().shape}")
                
            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue
        
        return self._compute_metrics(all_predictions, all_targets_cat, all_targets_vad, all_confidences)
    
    def _compute_metrics(self, predictions: List[Dict], targets_cat: List[np.ndarray], 
                        targets_vad: List[np.ndarray], confidences: List[float]) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics."""
        
        # Convert to numpy arrays
        pred_probs = np.array([p["cat_probs"] for p in predictions])
        pred_binary = (pred_probs >= 0.35).astype(int)  # Use default threshold
        
        targets_cat = np.array(targets_cat)
        targets_vad = np.array(targets_vad)
        confidences = np.array(confidences)
        
        # Ensure arrays are not empty
        if len(predictions) == 0 or len(targets_cat) == 0:
            logger.error("No valid predictions or targets collected")
            return {
                "overall_map": 0.0,
                "category_maps": {cat: 0.0 for cat in EMOTIC_CATEGORIES},
                "vad_mae": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                "vad_rmse": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                "confusion_matrix": [[0] * 10 for _ in range(10)],
                "top_10_categories": EMOTIC_CATEGORIES[:10],
                "reliability_data": {
                    "bin_accuracies": [0.0] * 10,
                    "bin_confidences": [0.0] * 10,
                    "bin_counts": [0] * 10,
                    "bin_boundaries": [i/10 for i in range(10)]
                }
            }
        
        # Ensure targets_cat is 2D
        if len(targets_cat.shape) == 1:
            targets_cat = np.array([targets_cat])
        
        # Ensure pred_probs is 2D
        if len(pred_probs.shape) == 1:
            pred_probs = np.array([pred_probs])
        
        # Compute mAP per category
        category_maps = []
        for i in range(len(EMOTIC_CATEGORIES)):
            # Average precision for this category
            ap = self._compute_average_precision(
                targets_cat[:, i], pred_probs[:, i]
            )
            category_maps.append(ap)
        
        # Overall mAP (honest - only categories with samples)
        categories_with_samples = [i for i in range(len(EMOTIC_CATEGORIES)) 
                               if np.sum(targets_cat[:, i]) > 0]
        if categories_with_samples:
            overall_map = np.mean([category_maps[i] for i in categories_with_samples])
        else:
            overall_map = 0.0
        
        # VAD metrics
        pred_vad = np.array([[p["vad"]["valence"], p["vad"]["arousal"], p["vad"]["dominance"]] 
                            for p in predictions])
        
        vad_mae = mean_absolute_error(targets_vad, pred_vad, multioutput='raw_values')
        vad_rmse = np.sqrt(mean_squared_error(targets_vad, pred_vad, multioutput='raw_values'))
        
        # Confusion matrix for top-10 most common emotions
        top_10_indices = np.argsort(np.sum(targets_cat, axis=0))[-10:]
        cm = confusion_matrix(
            targets_cat[:, top_10_indices].argmax(axis=1),
            pred_binary[:, top_10_indices].argmax(axis=1)
        )
        
        # Reliability diagram data
        reliability_data = self._compute_reliability_diagram(targets_cat, pred_probs, confidences)
        
        return {
            "overall_map": overall_map,
            "category_maps": dict(zip(EMOTIC_CATEGORIES, category_maps)),
            "vad_mae": dict(zip(["valence", "arousal", "dominance"], vad_mae)),
            "vad_rmse": dict(zip(["valence", "arousal", "dominance"], vad_rmse)),
            "confusion_matrix": cm.tolist(),
            "top_10_categories": [EMOTIC_CATEGORIES[i] for i in top_10_indices],
            "reliability_data": reliability_data
        }
    
    def _compute_average_precision(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Compute average precision for binary classification using sklearn."""
        try:
            return average_precision_score(y_true, y_scores)
        except ValueError:
            # Handle edge case where no positive samples
            return 0.0
    
    def _compute_reliability_diagram(self, targets: np.ndarray, probs: np.ndarray, 
                                   confidences: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
        """Compute reliability diagram data for confidence calibration."""
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
        
        return {
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "bin_boundaries": list(bin_lowers)
        }


def create_test_dataset(test_csv: str, image_root: str) -> EMOTICDataset:
    """Create test dataset for evaluation."""
    _, val_transforms = get_transforms("val")
    return EMOTICDataset(
        annotation_csv=test_csv,
        image_root=image_root,
        body_transform=None,  # Not used for evaluation
        context_transform=val_transforms,
        normalize_vad=True
    )


def save_comparison_chart(old_metrics: Dict, new_metrics: Dict, output_path: Path):
    """Create side-by-side mAP comparison chart."""
    categories = list(old_metrics["category_maps"].keys())
    old_maps = list(old_metrics["category_maps"].values())
    new_maps = list(new_metrics["category_maps"].values())
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - width/2, old_maps, width, label='Old Model', alpha=0.8)
    ax.bar(x + width/2, new_maps, width, label='New Model', alpha=0.8)
    
    ax.set_xlabel('Emotion Categories')
    ax.set_ylabel('Average Precision (AP)')
    ax.set_title('Per-Category mAP Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_vad_comparison(old_metrics: Dict, new_metrics: Dict, output_path: Path):
    """Create VAD MAE comparison chart."""
    dimensions = ["valence", "arousal", "dominance"]
    old_mae = [old_metrics["vad_mae"][dim] for dim in dimensions]
    new_mae = [new_metrics["vad_mae"][dim] for dim in dimensions]
    
    x = np.arange(len(dimensions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, old_mae, width, label='Old Model', alpha=0.8)
    ax.bar(x + width/2, new_mae, width, label='New Model', alpha=0.8)
    
    ax.set_xlabel('VAD Dimensions')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('VAD MAE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Valence', 'Arousal', 'Dominance'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_confusion_matrix(metrics: Dict, output_path: Path):
    """Create confusion matrix heatmap."""
    cm = np.array(metrics["confusion_matrix"])
    categories = metrics["top_10_categories"]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix (Top 10 Emotions)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_reliability_diagram(old_metrics: Dict, new_metrics: Dict, output_path: Path):
    """Create reliability diagram for confidence calibration."""
    old_data = old_metrics["reliability_data"]
    new_data = new_metrics["reliability_data"]
    
    plt.figure(figsize=(10, 8))
    
    # Plot old model
    plt.plot(old_data["bin_confidences"], old_data["bin_accuracies"], 
             'o-', label='Old Model', linewidth=2, markersize=8)
    
    # Plot new model
    plt.plot(new_data["bin_confidences"], new_data["bin_accuracies"], 
             'o-', label='New Model', linewidth=2, markersize=8)
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def print_comparison_table(old_metrics: Dict, new_metrics: Dict):
    """Print formatted comparison table to stdout."""
    print("\n" + "="*80)
    print("MODEL EVALUATION COMPARISON")
    print("="*80)
    
    # Overall metrics
    print(f"\nOverall mAP:")
    print(f"  Old Model: {old_metrics['overall_map']:.4f}")
    print(f"  New Model: {new_metrics['overall_map']:.4f}")
    print(f"  Improvement: {new_metrics['overall_map'] - old_metrics['overall_map']:+.4f}")
    
    # VAD metrics
    print(f"\nVAD MAE:")
    for dim in ["valence", "arousal", "dominance"]:
        old_val = old_metrics["vad_mae"][dim]
        new_val = new_metrics["vad_mae"][dim]
        print(f"  {dim.capitalize()}:")
        print(f"    Old: {old_val:.4f} | New: {new_val:.4f} | Diff: {new_val - old_val:+.4f}")
    
    # Top 5 categories by improvement
    print(f"\nTop 5 Categories by mAP Improvement:")
    improvements = {}
    for cat in EMOTIC_CATEGORIES:
        old_val = old_metrics["category_maps"][cat]
        new_val = new_metrics["category_maps"][cat]
        improvements[cat] = new_val - old_val
    
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
    for i, (cat, improvement) in enumerate(sorted_improvements[:5]):
        print(f"  {i+1}. {cat}: {improvement:+.4f}")
    
    print("="*80)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate EMOTIC models")
    parser.add_argument("--old", default="checkpoints/best_model.pth", help="Path to old model checkpoint")
    parser.add_argument("--new", default="checkpoints_real/best_model.pth", help="Path to new model checkpoint")
    parser.add_argument("--test-csv", default="data/real_dataset/test.csv", 
                       help="Path to test CSV file")
    parser.add_argument("--image-root", default="data/real_dataset/images",
                       help="Path to test images directory")
    parser.add_argument("--device", default="cpu", help="Device to use")
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.old).exists():
        logger.error(f"Old checkpoint not found: {args.old}")
        sys.exit(1)
    if not Path(args.new).exists():
        logger.error(f"New checkpoint not found: {args.new}")
        sys.exit(1)
    if not Path(args.test_csv).exists():
        logger.error(f"Test CSV not found: {args.test_csv}")
        sys.exit(1)
    if not Path(args.image_root).exists():
        logger.error(f"Image root not found: {args.image_root}")
        sys.exit(1)
    
    # Create test dataset
    logger.info("Creating test dataset...")
    test_dataset = create_test_dataset(args.test_csv, args.image_root + "/Train")
    
    # Evaluate old model
    logger.info("Evaluating old model...")
    old_evaluator = ModelEvaluator(args.old, args.device)
    old_metrics = old_evaluator.evaluate_dataset(test_dataset)
    
    # Evaluate new model
    logger.info("Evaluating new model...")
    new_evaluator = ModelEvaluator(args.new, args.device)
    new_metrics = new_evaluator.evaluate_dataset(test_dataset)
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    save_comparison_chart(old_metrics, new_metrics, RESULTS_DIR / "mAP_comparison.png")
    save_vad_comparison(old_metrics, new_metrics, RESULTS_DIR / "vad_mae_comparison.png")
    save_confusion_matrix(new_metrics, RESULTS_DIR / "confusion_matrix.png")  # Use new model
    save_reliability_diagram(old_metrics, new_metrics, RESULTS_DIR / "reliability_diagram.png")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if 'float' in str(type(obj)) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    # Save detailed results
    results = {
        "old_model": {
            "checkpoint": args.old,
            "metrics": convert_numpy(old_metrics)
        },
        "new_model": {
            "checkpoint": args.new,
            "metrics": convert_numpy(new_metrics)
        },
        "dataset_info": {
            "test_csv": args.test_csv,
            "image_root": args.image_root,
            "num_samples": len(test_dataset)
        }
    }
    
    with open(RESULTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print_comparison_table(old_metrics, new_metrics)
    
    logger.info(f"Evaluation complete! Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
