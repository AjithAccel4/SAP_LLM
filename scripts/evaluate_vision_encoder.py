#!/usr/bin/env python3
"""
Evaluation script for Vision Encoder.

This script provides comprehensive evaluation metrics for the trained vision encoder:
- Document type classification accuracy, precision, recall, F1
- PO subtype classification accuracy, precision, recall, F1
- Token classification (field extraction) precision, recall, F1
- Inference latency benchmarking
- Confusion matrices and classification reports

Usage:
    python scripts/evaluate_vision_encoder.py \
        --model_path ./models/vision_encoder/best \
        --data_dir ./data/vision_encoder/test \
        --output_dir ./evaluation_results
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Config, LayoutLMv3Processor
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sap_llm.models.vision_encoder import MultiTaskLayoutLMv3, DOCUMENT_TYPES, PO_SUBTYPES, benchmark_model
from sap_llm.training.vision_dataset import VisionEncoderDataset, SAP_FIELD_LABELS
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Vision Encoder")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing test data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmarking",
    )
    parser.add_argument(
        "--benchmark_iterations",
        type=int,
        default=100,
        help="Number of iterations for benchmarking",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )

    return parser.parse_args()


def load_model(model_path: str, device: str) -> MultiTaskLayoutLMv3:
    """Load trained model from checkpoint."""
    logger.info(f"Loading model from {model_path}")

    # Load config
    config_path = Path(model_path) / "training_state.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            training_state = json.load(f)
            args = training_state.get("args", {})
            num_doc_types = args.get("num_doc_types", 15)
            num_po_subtypes = args.get("num_po_subtypes", 35)
            num_token_labels = args.get("num_token_labels", 181)
            model_name = args.get("model_name", "microsoft/layoutlmv3-base")
    else:
        # Use defaults
        num_doc_types = 15
        num_po_subtypes = 35
        num_token_labels = 181
        model_name = "microsoft/layoutlmv3-base"

    # Create model
    config = LayoutLMv3Config.from_pretrained(model_name)
    model = MultiTaskLayoutLMv3(
        config=config,
        num_doc_types=num_doc_types,
        num_po_subtypes=num_po_subtypes,
        num_token_labels=num_token_labels,
    )

    # Load weights
    weights_path = Path(model_path) / "pytorch_model.bin"
    if not weights_path.exists():
        raise ValueError(f"Model weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded successfully")
    return model


def evaluate_classification(
    model: MultiTaskLayoutLMv3,
    dataloader: DataLoader,
    device: str,
) -> Tuple[Dict, Dict, Dict]:
    """
    Evaluate document type and PO subtype classification.

    Returns:
        Tuple of (doc_type_results, po_subtype_results, token_results)
    """
    logger.info("Running classification evaluation...")

    # Collect predictions
    doc_type_preds = []
    doc_type_labels = []
    po_subtype_preds = []
    po_subtype_labels = []
    token_preds = []
    token_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Document type predictions
            doc_type_pred = torch.argmax(outputs["doc_type_logits"], dim=-1)
            doc_type_preds.extend(doc_type_pred.cpu().numpy())
            doc_type_labels.extend(batch["doc_type_labels"].cpu().numpy())

            # PO subtype predictions
            po_subtype_pred = torch.argmax(outputs["po_subtype_logits"], dim=-1)
            po_subtype_preds.extend(po_subtype_pred.cpu().numpy())
            po_subtype_labels.extend(batch["po_subtype_labels"].cpu().numpy())

            # Token predictions (only for active tokens)
            token_logits = outputs["token_logits"]
            token_pred = torch.argmax(token_logits, dim=-1)

            # Flatten and filter by attention mask
            attention_mask = batch["attention_mask"]
            for i in range(token_pred.size(0)):
                active_mask = attention_mask[i] == 1
                active_preds = token_pred[i][active_mask].cpu().numpy()
                active_labels = batch["token_labels"][i][active_mask].cpu().numpy()
                token_preds.extend(active_preds)
                token_labels.extend(active_labels)

    # Convert to numpy
    doc_type_preds = np.array(doc_type_preds)
    doc_type_labels = np.array(doc_type_labels)
    po_subtype_preds = np.array(po_subtype_preds)
    po_subtype_labels = np.array(po_subtype_labels)
    token_preds = np.array(token_preds)
    token_labels = np.array(token_labels)

    # Calculate metrics for document type classification
    doc_type_results = {
        "accuracy": accuracy_score(doc_type_labels, doc_type_preds),
        "precision": precision_recall_fscore_support(
            doc_type_labels, doc_type_preds, average="weighted", zero_division=0
        )[0],
        "recall": precision_recall_fscore_support(
            doc_type_labels, doc_type_preds, average="weighted", zero_division=0
        )[1],
        "f1": precision_recall_fscore_support(
            doc_type_labels, doc_type_preds, average="weighted", zero_division=0
        )[2],
        "predictions": doc_type_preds,
        "labels": doc_type_labels,
    }

    # Calculate metrics for PO subtype classification
    po_subtype_results = {
        "accuracy": accuracy_score(po_subtype_labels, po_subtype_preds),
        "precision": precision_recall_fscore_support(
            po_subtype_labels, po_subtype_preds, average="weighted", zero_division=0
        )[0],
        "recall": precision_recall_fscore_support(
            po_subtype_labels, po_subtype_preds, average="weighted", zero_division=0
        )[1],
        "f1": precision_recall_fscore_support(
            po_subtype_labels, po_subtype_preds, average="weighted", zero_division=0
        )[2],
        "predictions": po_subtype_preds,
        "labels": po_subtype_labels,
    }

    # Calculate metrics for token classification
    token_results = {
        "accuracy": accuracy_score(token_labels, token_preds),
        "precision": precision_recall_fscore_support(
            token_labels, token_preds, average="weighted", zero_division=0
        )[0],
        "recall": precision_recall_fscore_support(
            token_labels, token_preds, average="weighted", zero_division=0
        )[1],
        "f1": precision_recall_fscore_support(
            token_labels, token_preds, average="weighted", zero_division=0
        )[2],
        "predictions": token_preds,
        "labels": token_labels,
    }

    return doc_type_results, po_subtype_results, token_results


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)

    # Normalize
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved to {output_path}")


def save_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: List[str],
    output_path: Path,
):
    """Save detailed classification report."""
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Classification report saved to {output_path}")


def benchmark_latency(
    model: MultiTaskLayoutLMv3,
    sample_input: Dict[str, torch.Tensor],
    device: str,
    num_iterations: int = 100,
) -> Dict[str, float]:
    """Benchmark model inference latency."""
    logger.info("Benchmarking inference latency...")

    # Warmup
    warmup_iterations = 10
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(**sample_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in tqdm(range(num_iterations), desc="Benchmarking"):
            start = time.time()
            _ = model(**sample_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    results = {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }

    return results


def main():
    """Main evaluation function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.model_path, args.device)

    # Load processor
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    # Load test dataset
    logger.info(f"Loading test data from {args.data_dir}")
    test_dataset = VisionEncoderDataset(
        data_dir=args.data_dir,
        processor=processor,
        max_length=args.max_length,
        mode="test",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    # Evaluate classification
    doc_type_results, po_subtype_results, token_results = evaluate_classification(
        model, test_loader, args.device
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("DOCUMENT TYPE CLASSIFICATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {doc_type_results['accuracy']:.4f}")
    logger.info(f"Precision: {doc_type_results['precision']:.4f}")
    logger.info(f"Recall:    {doc_type_results['recall']:.4f}")
    logger.info(f"F1 Score:  {doc_type_results['f1']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("PO SUBTYPE CLASSIFICATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {po_subtype_results['accuracy']:.4f}")
    logger.info(f"Precision: {po_subtype_results['precision']:.4f}")
    logger.info(f"Recall:    {po_subtype_results['recall']:.4f}")
    logger.info(f"F1 Score:  {po_subtype_results['f1']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("TOKEN CLASSIFICATION (FIELD EXTRACTION) RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy:  {token_results['accuracy']:.4f}")
    logger.info(f"Precision: {token_results['precision']:.4f}")
    logger.info(f"Recall:    {token_results['recall']:.4f}")
    logger.info(f"F1 Score:  {token_results['f1']:.4f}")

    # Save results to JSON
    results_summary = {
        "document_type_classification": {
            "accuracy": doc_type_results["accuracy"],
            "precision": doc_type_results["precision"],
            "recall": doc_type_results["recall"],
            "f1": doc_type_results["f1"],
        },
        "po_subtype_classification": {
            "accuracy": po_subtype_results["accuracy"],
            "precision": po_subtype_results["precision"],
            "recall": po_subtype_results["recall"],
            "f1": po_subtype_results["f1"],
        },
        "token_classification": {
            "accuracy": token_results["accuracy"],
            "precision": token_results["precision"],
            "recall": token_results["recall"],
            "f1": token_results["f1"],
        },
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(f"\nResults saved to {output_dir / 'evaluation_results.json'}")

    # Plot confusion matrices
    plot_confusion_matrix(
        doc_type_results["labels"],
        doc_type_results["predictions"],
        DOCUMENT_TYPES,
        output_dir / "doc_type_confusion_matrix.png",
        "Document Type Classification - Confusion Matrix",
    )

    plot_confusion_matrix(
        po_subtype_results["labels"],
        po_subtype_results["predictions"],
        PO_SUBTYPES[:min(len(PO_SUBTYPES), 20)],  # Limit for readability
        output_dir / "po_subtype_confusion_matrix.png",
        "PO Subtype Classification - Confusion Matrix (Top 20)",
    )

    # Save classification reports
    save_classification_report(
        doc_type_results["labels"],
        doc_type_results["predictions"],
        DOCUMENT_TYPES,
        output_dir / "doc_type_classification_report.txt",
    )

    save_classification_report(
        po_subtype_results["labels"],
        po_subtype_results["predictions"],
        PO_SUBTYPES,
        output_dir / "po_subtype_classification_report.txt",
    )

    # Benchmark latency
    if args.benchmark:
        logger.info("\n" + "=" * 80)
        logger.info("LATENCY BENCHMARKING")
        logger.info("=" * 80)

        # Get sample input
        sample_batch = next(iter(test_loader))
        sample_input = {
            k: v[0:1].to(args.device) if isinstance(v, torch.Tensor) else v
            for k, v in sample_batch.items()
        }

        latency_results = benchmark_latency(
            model, sample_input, args.device, args.benchmark_iterations
        )

        logger.info(f"Mean latency:   {latency_results['mean_ms']:.2f} ms")
        logger.info(f"Std latency:    {latency_results['std_ms']:.2f} ms")
        logger.info(f"Min latency:    {latency_results['min_ms']:.2f} ms")
        logger.info(f"Max latency:    {latency_results['max_ms']:.2f} ms")
        logger.info(f"P50 latency:    {latency_results['p50_ms']:.2f} ms")
        logger.info(f"P95 latency:    {latency_results['p95_ms']:.2f} ms")
        logger.info(f"P99 latency:    {latency_results['p99_ms']:.2f} ms")

        # Save latency results
        with open(output_dir / "latency_results.json", "w") as f:
            json.dump(latency_results, f, indent=2)

        logger.info(f"\nLatency results saved to {output_dir / 'latency_results.json'}")

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
