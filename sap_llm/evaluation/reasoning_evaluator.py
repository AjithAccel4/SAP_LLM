"""
Evaluation System for Reasoning Engine.

Measures:
1. Routing Accuracy (target: ≥97%)
2. SAP API Selection Accuracy (target: 100% - CRITICAL)
3. Payload Generation Accuracy (target: ≥99%)
4. Decision Explanation Quality (human validation on 100 samples)
5. Inference Latency (target: <500ms)

Generates comprehensive accuracy reports.
"""

import json
import time
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics."""
    # Overall metrics
    total_examples: int
    routing_accuracy: float
    api_selection_accuracy: float
    payload_accuracy: float
    avg_confidence: float
    avg_inference_latency_ms: float

    # Per-document-type metrics
    accuracy_by_doc_type: Dict[str, float]

    # Latency statistics
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Error analysis
    total_errors: int
    endpoint_errors: int
    payload_errors: int
    timeout_errors: int

    # Detailed breakdown
    confusion_matrix: Dict[str, Dict[str, int]]

    # Pass/Fail for success criteria
    passes_routing_threshold: bool  # ≥97%
    passes_api_selection_threshold: bool  # 100%
    passes_payload_threshold: bool  # ≥99%
    passes_latency_threshold: bool  # <500ms avg


class ReasoningEngineEvaluator:
    """
    Comprehensive evaluator for Reasoning Engine.

    Evaluates:
    - Routing accuracy
    - API selection accuracy
    - Payload generation correctness
    - Inference latency
    - Explanation quality (manual validation)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        precision: str = "int8",
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model
            device: Device to run on
            precision: Model precision
        """
        logger.info(f"Loading model from {model_path}")
        self.model = ReasoningEngine.load(
            model_path=model_path,
            device=device,
            precision=precision,
        )

        logger.info("Evaluator initialized")

    def evaluate(
        self,
        test_data_file: str,
        num_samples: Optional[int] = None,
        output_dir: str = "./evaluation_results",
    ) -> EvaluationMetrics:
        """
        Run comprehensive evaluation.

        Args:
            test_data_file: Path to test data
            num_samples: Number of samples to evaluate (None = all)
            output_dir: Output directory for results

        Returns:
            Evaluation metrics
        """
        logger.info("Starting evaluation...")

        # Load test data
        test_examples = []
        with open(test_data_file) as f:
            for i, line in enumerate(f):
                if num_samples and i >= num_samples:
                    break
                test_examples.append(json.loads(line))

        logger.info(f"Loaded {len(test_examples)} test examples")

        # Run evaluation
        results = []
        latencies = []

        for example in tqdm(test_examples, desc="Evaluating"):
            # Time inference
            start_time = time.time()

            # Generate prediction
            prediction = self._predict(example)

            inference_time_ms = (time.time() - start_time) * 1000
            latencies.append(inference_time_ms)

            # Evaluate prediction
            eval_result = self._evaluate_prediction(prediction, example)
            eval_result["latency_ms"] = inference_time_ms

            results.append(eval_result)

        # Compute metrics
        metrics = self._compute_metrics(results, latencies, test_examples)

        # Generate report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._generate_report(metrics, results, output_path)

        # Log summary
        self._log_summary(metrics)

        return metrics

    def _predict(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prediction for example."""
        # Create prompt
        adc_json = example["adc_json"]
        doc_type = example["doc_type"]
        api_schemas = example["api_schemas"]
        similar_cases = example.get("similar_cases", [])

        # Get routing decision
        decision = self.model.decide_routing(
            adc_json=adc_json,
            doc_type=doc_type,
            api_schemas=api_schemas,
            similar_cases=similar_cases,
        )

        return decision

    def _evaluate_prediction(
        self,
        prediction: Dict[str, Any],
        ground_truth: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate a single prediction.

        Args:
            prediction: Model prediction
            ground_truth: Ground truth example

        Returns:
            Evaluation result
        """
        result = {
            "doc_id": ground_truth.get("doc_id"),
            "doc_type": ground_truth["doc_type"],
            "predicted_endpoint": prediction.get("endpoint"),
            "target_endpoint": ground_truth["target_endpoint"],
            "confidence": prediction.get("confidence", 0.0),
        }

        # Check endpoint correctness
        endpoint_correct = prediction.get("endpoint") == ground_truth["target_endpoint"]
        result["endpoint_correct"] = endpoint_correct

        # Check payload correctness
        if "payload" in prediction:
            payload_correct = self._check_payload_correctness(
                prediction["payload"],
                ground_truth["target_payload"],
            )
        else:
            payload_correct = False

        result["payload_correct"] = payload_correct

        # Overall routing correctness (endpoint + payload)
        result["routing_correct"] = endpoint_correct and payload_correct

        return result

    def _check_payload_correctness(
        self,
        predicted: Dict[str, Any],
        target: Dict[str, Any],
    ) -> bool:
        """
        Check if payload is correct.

        Args:
            predicted: Predicted payload
            target: Target payload

        Returns:
            True if ≥90% of fields match
        """
        if not predicted or not target:
            return False

        # Extract data
        pred_data = predicted.get("d", predicted)
        target_data = target.get("d", target)

        if not isinstance(pred_data, dict) or not isinstance(target_data, dict):
            return False

        # Count matching fields
        total_fields = len(target_data)
        if total_fields == 0:
            return True

        matching_fields = 0
        for key, value in target_data.items():
            if key in pred_data and pred_data[key] == value:
                matching_fields += 1

        # Require ≥90% match
        return (matching_fields / total_fields) >= 0.9

    def _compute_metrics(
        self,
        results: List[Dict[str, Any]],
        latencies: List[float],
        test_examples: List[Dict[str, Any]],
    ) -> EvaluationMetrics:
        """Compute evaluation metrics."""
        total = len(results)

        # Overall accuracy
        routing_correct = sum(1 for r in results if r["routing_correct"])
        endpoint_correct = sum(1 for r in results if r["endpoint_correct"])
        payload_correct = sum(1 for r in results if r["payload_correct"])

        routing_accuracy = routing_correct / total
        api_selection_accuracy = endpoint_correct / total
        payload_accuracy = payload_correct / total

        # Confidence
        avg_confidence = np.mean([r["confidence"] for r in results])

        # Latency
        avg_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        # Per-document-type accuracy
        doc_types = set(r["doc_type"] for r in results)
        accuracy_by_doc_type = {}

        for doc_type in doc_types:
            type_results = [r for r in results if r["doc_type"] == doc_type]
            type_correct = sum(1 for r in type_results if r["routing_correct"])
            accuracy_by_doc_type[doc_type] = type_correct / len(type_results) if type_results else 0.0

        # Error counts
        total_errors = total - routing_correct
        endpoint_errors = total - endpoint_correct
        payload_errors = total - payload_correct

        # Confusion matrix (simplified)
        confusion_matrix = {}
        for result in results:
            pred = result["predicted_endpoint"]
            target = result["target_endpoint"]

            if target not in confusion_matrix:
                confusion_matrix[target] = {}

            if pred not in confusion_matrix[target]:
                confusion_matrix[target][pred] = 0

            confusion_matrix[target][pred] += 1

        # Success criteria checks
        passes_routing = routing_accuracy >= 0.97
        passes_api_selection = api_selection_accuracy >= 1.0  # 100%
        passes_payload = payload_accuracy >= 0.99
        passes_latency = avg_latency < 500  # <500ms

        return EvaluationMetrics(
            total_examples=total,
            routing_accuracy=routing_accuracy,
            api_selection_accuracy=api_selection_accuracy,
            payload_accuracy=payload_accuracy,
            avg_confidence=avg_confidence,
            avg_inference_latency_ms=avg_latency,
            accuracy_by_doc_type=accuracy_by_doc_type,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            total_errors=total_errors,
            endpoint_errors=endpoint_errors,
            payload_errors=payload_errors,
            timeout_errors=0,
            confusion_matrix=confusion_matrix,
            passes_routing_threshold=passes_routing,
            passes_api_selection_threshold=passes_api_selection,
            passes_payload_threshold=passes_payload,
            passes_latency_threshold=passes_latency,
        )

    def _generate_report(
        self,
        metrics: EvaluationMetrics,
        results: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Generate evaluation report."""
        logger.info(f"Generating report in {output_dir}")

        # Save metrics as JSON
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        # Save detailed results
        with open(output_dir / "detailed_results.jsonl", "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        # Generate markdown report
        self._generate_markdown_report(metrics, output_dir / "EVALUATION_REPORT.md")

        # Generate visualizations
        self._generate_visualizations(metrics, results, output_dir)

        logger.info(f"Report saved to {output_dir}")

    def _generate_markdown_report(self, metrics: EvaluationMetrics, output_file: Path) -> None:
        """Generate markdown report."""
        with open(output_file, "w") as f:
            f.write("# Reasoning Engine Evaluation Report\n\n")

            # Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Examples Evaluated:** {metrics.total_examples}\n")
            f.write(f"- **Overall Routing Accuracy:** {metrics.routing_accuracy * 100:.2f}%\n")
            f.write(f"- **API Selection Accuracy:** {metrics.api_selection_accuracy * 100:.2f}%\n")
            f.write(f"- **Payload Generation Accuracy:** {metrics.payload_accuracy * 100:.2f}%\n")
            f.write(f"- **Average Confidence:** {metrics.avg_confidence:.3f}\n")
            f.write(f"- **Average Inference Latency:** {metrics.avg_inference_latency_ms:.2f}ms\n\n")

            # Success Criteria
            f.write("## Success Criteria\n\n")
            f.write("| Metric | Target | Actual | Status |\n")
            f.write("|--------|--------|--------|--------|\n")
            f.write(f"| Routing Accuracy | ≥97% | {metrics.routing_accuracy * 100:.2f}% | {'✅ PASS' if metrics.passes_routing_threshold else '❌ FAIL'} |\n")
            f.write(f"| API Selection Accuracy | 100% | {metrics.api_selection_accuracy * 100:.2f}% | {'✅ PASS' if metrics.passes_api_selection_threshold else '❌ FAIL'} |\n")
            f.write(f"| Payload Accuracy | ≥99% | {metrics.payload_accuracy * 100:.2f}% | {'✅ PASS' if metrics.passes_payload_threshold else '❌ FAIL'} |\n")
            f.write(f"| Inference Latency | <500ms | {metrics.avg_inference_latency_ms:.2f}ms | {'✅ PASS' if metrics.passes_latency_threshold else '❌ FAIL'} |\n\n")

            # Latency statistics
            f.write("## Latency Statistics\n\n")
            f.write(f"- **P50 (Median):** {metrics.p50_latency_ms:.2f}ms\n")
            f.write(f"- **P95:** {metrics.p95_latency_ms:.2f}ms\n")
            f.write(f"- **P99:** {metrics.p99_latency_ms:.2f}ms\n\n")

            # Per-document-type accuracy
            f.write("## Accuracy by Document Type\n\n")
            f.write("| Document Type | Accuracy |\n")
            f.write("|---------------|----------|\n")
            for doc_type, accuracy in sorted(metrics.accuracy_by_doc_type.items()):
                f.write(f"| {doc_type} | {accuracy * 100:.2f}% |\n")
            f.write("\n")

            # Error analysis
            f.write("## Error Analysis\n\n")
            f.write(f"- **Total Errors:** {metrics.total_errors}\n")
            f.write(f"- **Endpoint Errors:** {metrics.endpoint_errors}\n")
            f.write(f"- **Payload Errors:** {metrics.payload_errors}\n\n")

            # Confusion matrix
            f.write("## Confusion Matrix (Top Errors)\n\n")
            f.write("| True Endpoint | Predicted Endpoint | Count |\n")
            f.write("|---------------|-------------------|-------|\n")

            # Find top misclassifications
            errors = []
            for true_endpoint, preds in metrics.confusion_matrix.items():
                for pred_endpoint, count in preds.items():
                    if pred_endpoint != true_endpoint:
                        errors.append((true_endpoint, pred_endpoint, count))

            errors.sort(key=lambda x: x[2], reverse=True)

            for true_ep, pred_ep, count in errors[:10]:  # Top 10 errors
                f.write(f"| {true_ep} | {pred_ep} | {count} |\n")

            f.write("\n")

            # Overall status
            f.write("## Overall Status\n\n")
            all_pass = all([
                metrics.passes_routing_threshold,
                metrics.passes_api_selection_threshold,
                metrics.passes_payload_threshold,
                metrics.passes_latency_threshold,
            ])

            if all_pass:
                f.write("**✅ ALL SUCCESS CRITERIA MET**\n\n")
                f.write("The Reasoning Engine is ready for production deployment.\n")
            else:
                f.write("**❌ SOME SUCCESS CRITERIA NOT MET**\n\n")
                f.write("Further training or optimization required before production deployment.\n")

    def _generate_visualizations(
        self,
        metrics: EvaluationMetrics,
        results: List[Dict[str, Any]],
        output_dir: Path,
    ) -> None:
        """Generate visualization plots."""
        # Plot 1: Accuracy metrics
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        metrics_data = {
            "Routing\nAccuracy": metrics.routing_accuracy * 100,
            "API Selection\nAccuracy": metrics.api_selection_accuracy * 100,
            "Payload\nAccuracy": metrics.payload_accuracy * 100,
        }

        thresholds = {
            "Routing\nAccuracy": 97,
            "API Selection\nAccuracy": 100,
            "Payload\nAccuracy": 99,
        }

        x = range(len(metrics_data))
        values = list(metrics_data.values())
        threshold_values = [thresholds[k] for k in metrics_data.keys()]

        bars = ax.bar(x, values, color=['green' if v >= t else 'red' for v, t in zip(values, threshold_values)])
        ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

        # Add threshold lines
        for i, (metric, threshold) in enumerate(thresholds.items()):
            ax.plot([i - 0.4, i + 0.4], [threshold, threshold], 'b--', linewidth=2, label=f'{metric} threshold' if i == 0 else '')

        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Reasoning Engine Accuracy Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_data.keys())
        ax.set_ylim([0, 105])

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_dir / "accuracy_metrics.png", dpi=300)
        plt.close()

        # Plot 2: Per-document-type accuracy
        if metrics.accuracy_by_doc_type:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            doc_types = list(metrics.accuracy_by_doc_type.keys())
            accuracies = [metrics.accuracy_by_doc_type[dt] * 100 for dt in doc_types]

            bars = ax.bar(range(len(doc_types)), accuracies, color='steelblue')
            ax.axhline(y=97, color='red', linestyle='--', linewidth=2, label='Target (97%)')

            ax.set_ylabel('Accuracy (%)')
            ax.set_title('Accuracy by Document Type')
            ax.set_xticks(range(len(doc_types)))
            ax.set_xticklabels(doc_types, rotation=45, ha='right')
            ax.set_ylim([0, 105])
            ax.legend()

            # Add value labels
            for bar, val in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

            plt.tight_layout()
            plt.savefig(output_dir / "accuracy_by_doc_type.png", dpi=300)
            plt.close()

        logger.info("Visualizations generated")

    def _log_summary(self, metrics: EvaluationMetrics) -> None:
        """Log evaluation summary."""
        logger.info("="*60)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Total Examples: {metrics.total_examples}")
        logger.info(f"Routing Accuracy: {metrics.routing_accuracy * 100:.2f}% (target: ≥97%)")
        logger.info(f"API Selection Accuracy: {metrics.api_selection_accuracy * 100:.2f}% (target: 100%)")
        logger.info(f"Payload Accuracy: {metrics.payload_accuracy * 100:.2f}% (target: ≥99%)")
        logger.info(f"Avg Latency: {metrics.avg_inference_latency_ms:.2f}ms (target: <500ms)")
        logger.info("="*60)

        if all([
            metrics.passes_routing_threshold,
            metrics.passes_api_selection_threshold,
            metrics.passes_payload_threshold,
            metrics.passes_latency_threshold,
        ]):
            logger.info("✅ ALL SUCCESS CRITERIA MET")
        else:
            logger.warning("❌ SOME SUCCESS CRITERIA NOT MET")

        logger.info("="*60)


def main():
    """Main evaluation script."""
    evaluator = ReasoningEngineEvaluator(
        model_path="./models/reasoning_engine_rlhf/final",
        device="cuda",
        precision="int8",
    )

    metrics = evaluator.evaluate(
        test_data_file="data/training/reasoning_engine/test_routing_examples.jsonl",
        num_samples=1000,  # Evaluate on 1000 samples
        output_dir="./evaluation_results/reasoning_engine",
    )

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: ./evaluation_results/reasoning_engine")
    print("="*60)


if __name__ == "__main__":
    main()
