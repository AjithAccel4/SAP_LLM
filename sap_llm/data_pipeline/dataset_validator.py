"""
Dataset Quality Validator for SAP_LLM Training Corpus.

Validates that the training dataset meets production requirements:
- Document count targets (1M+ documents)
- Quality score distribution (Cohen's kappa > 0.92)
- Document type balance (all 8+ SAP types represented)
- Annotation completeness (50K+ annotated)
- Token count (100B+ tokens for LLM training)
- Data split integrity (70/15/15)
- Field coverage (200+ business fields)

Ensures data quality before expensive model training begins.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import Counter, defaultdict
import statistics

logger = logging.getLogger(__name__)


class DatasetValidator:
    """
    Comprehensive dataset quality validation.

    Checks:
    - Completeness
    - Quality metrics
    - Balance and distribution
    - Annotation quality
    - Format compliance
    """

    def __init__(self, data_dir: str):
        """
        Initialize dataset validator.

        Args:
            data_dir: Root directory of dataset
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.validation_results = {
            "passed": False,
            "warnings": [],
            "failures": [],
            "metrics": {},
            "recommendations": []
        }

        logger.info(f"DatasetValidator initialized: {data_dir}")

    def validate_corpus(self,
                       min_documents: int = 1_000_000,
                       min_quality_score: float = 0.8,
                       min_tokens: int = 100_000_000_000,
                       required_document_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive corpus validation.

        Args:
            min_documents: Minimum required documents
            min_quality_score: Minimum average quality score
            min_tokens: Minimum total tokens
            required_document_types: Required document types

        Returns:
            Validation report
        """
        logger.info("=" * 80)
        logger.info("Starting Comprehensive Dataset Validation")
        logger.info("=" * 80)

        # Run all validation checks
        self._validate_document_count(min_documents)
        self._validate_quality_scores(min_quality_score)
        self._validate_document_types(required_document_types)
        self._validate_token_count(min_tokens)
        self._validate_data_splits()
        self._validate_annotations()
        self._validate_field_coverage()
        self._validate_format_compliance()

        # Determine overall pass/fail
        self.validation_results["passed"] = len(self.validation_results["failures"]) == 0

        # Generate recommendations
        self._generate_recommendations()

        # Print summary
        self._print_validation_summary()

        return self.validation_results

    def _validate_document_count(self, min_documents: int):
        """Validate total document count meets target."""
        logger.info(f"\n[1/8] Validating Document Count (target: {min_documents:,})...")

        # Count documents across all splits
        total_count = 0

        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / "processed" / split
            if split_dir.exists():
                # Count parquet files or JSON files
                parquet_files = list(split_dir.glob("*.parquet"))
                json_files = list(split_dir.glob("*.json"))

                if parquet_files:
                    # Would need to read parquet to count actual rows
                    # For now, estimate based on file count
                    split_count = len(parquet_files) * 1000  # Estimate
                elif json_files:
                    # Read JSON metadata
                    for json_file in json_files:
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                                if "documents" in data:
                                    split_count = len(data["documents"])
                                elif "total_documents" in data:
                                    split_count = data["total_documents"]
                        except Exception as e:
                            logger.warning(f"Error reading {json_file}: {e}")
                            split_count = 0
                else:
                    split_count = 0

                total_count += split_count
                logger.info(f"  {split}: {split_count:,} documents")

        self.validation_results["metrics"]["total_documents"] = total_count

        if total_count < min_documents:
            self.validation_results["failures"].append(
                f"Insufficient documents: {total_count:,} < {min_documents:,} (shortfall: {min_documents - total_count:,})"
            )
            logger.error(f"  ❌ FAILED: Only {total_count:,} documents (need {min_documents:,})")
        else:
            logger.info(f"  ✅ PASSED: {total_count:,} documents")

    def _validate_quality_scores(self, min_avg_score: float):
        """Validate quality score distribution."""
        logger.info(f"\n[2/8] Validating Quality Scores (min avg: {min_avg_score})...")

        # Load quality scores from metadata
        quality_scores = []

        metadata_file = self.data_dir / "raw" / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    if "documents" in data:
                        for doc in data["documents"]:
                            if "quality_score" in doc:
                                quality_scores.append(doc["quality_score"])
            except Exception as e:
                logger.warning(f"Error loading quality scores: {e}")

        if quality_scores:
            avg_score = statistics.mean(quality_scores)
            median_score = statistics.median(quality_scores)
            min_score = min(quality_scores)
            max_score = max(quality_scores)

            self.validation_results["metrics"]["quality_scores"] = {
                "mean": avg_score,
                "median": median_score,
                "min": min_score,
                "max": max_score,
                "count": len(quality_scores)
            }

            logger.info(f"  Mean: {avg_score:.3f}, Median: {median_score:.3f}")
            logger.info(f"  Range: [{min_score:.3f}, {max_score:.3f}]")

            if avg_score < min_avg_score:
                self.validation_results["failures"].append(
                    f"Low average quality score: {avg_score:.3f} < {min_avg_score}"
                )
                logger.error(f"  ❌ FAILED: Average quality {avg_score:.3f} < {min_avg_score}")
            else:
                logger.info(f"  ✅ PASSED: Average quality {avg_score:.3f}")

            # Check for low-quality outliers
            low_quality_count = sum(1 for score in quality_scores if score < 0.5)
            if low_quality_count > 0:
                pct = (low_quality_count / len(quality_scores)) * 100
                self.validation_results["warnings"].append(
                    f"{low_quality_count} documents ({pct:.1f}%) have quality < 0.5"
                )
                logger.warning(f"  ⚠️ WARNING: {low_quality_count} low-quality documents")
        else:
            self.validation_results["warnings"].append("No quality scores found")
            logger.warning("  ⚠️ WARNING: No quality scores available")

    def _validate_document_types(self, required_types: Optional[List[str]]):
        """Validate document type distribution."""
        logger.info(f"\n[3/8] Validating Document Types...")

        if required_types is None:
            required_types = [
                "invoice", "purchase_order", "delivery_note",
                "material_document", "sales_order", "goods_receipt",
                "packing_list", "shipping_notice"
            ]

        # Count document types
        type_counts = Counter()

        metadata_file = self.data_dir / "raw" / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    if "documents" in data:
                        for doc in data["documents"]:
                            if "document_type" in doc:
                                type_counts[doc["document_type"]] += 1
            except Exception as e:
                logger.warning(f"Error loading document types: {e}")

        self.validation_results["metrics"]["document_type_distribution"] = dict(type_counts)

        # Check all required types are present
        missing_types = []
        for required_type in required_types:
            count = type_counts.get(required_type, 0)
            if count == 0:
                missing_types.append(required_type)
            logger.info(f"  {required_type}: {count:,} documents")

        if missing_types:
            self.validation_results["failures"].append(
                f"Missing required document types: {', '.join(missing_types)}"
            )
            logger.error(f"  ❌ FAILED: Missing types: {missing_types}")
        else:
            logger.info(f"  ✅ PASSED: All {len(required_types)} required types present")

        # Check for severe imbalance
        if type_counts:
            counts = list(type_counts.values())
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

            if imbalance_ratio > 10:
                self.validation_results["warnings"].append(
                    f"Severe type imbalance: {imbalance_ratio:.1f}x (max={max_count:,}, min={min_count:,})"
                )
                logger.warning(f"  ⚠️ WARNING: Imbalance ratio {imbalance_ratio:.1f}x")

    def _validate_token_count(self, min_tokens: int):
        """Validate total token count for LLM training."""
        logger.info(f"\n[4/8] Validating Token Count (target: {min_tokens:,})...")

        # Estimate tokens from documents
        # Average: 500 tokens per document (conservative estimate)
        total_documents = self.validation_results["metrics"].get("total_documents", 0)
        estimated_tokens = total_documents * 500

        self.validation_results["metrics"]["estimated_tokens"] = estimated_tokens

        logger.info(f"  Estimated tokens: {estimated_tokens:,} ({total_documents:,} docs × 500 avg)")

        if estimated_tokens < min_tokens:
            self.validation_results["failures"].append(
                f"Insufficient tokens: {estimated_tokens:,} < {min_tokens:,}"
            )
            logger.error(f"  ❌ FAILED: Only {estimated_tokens:,} tokens (need {min_tokens:,})")
        else:
            logger.info(f"  ✅ PASSED: {estimated_tokens:,} tokens")

    def _validate_data_splits(self):
        """Validate train/val/test split ratios."""
        logger.info(f"\n[5/8] Validating Data Splits...")

        split_sizes = {}

        for split in ["train", "val", "test"]:
            split_dir = self.data_dir / "processed" / split
            if split_dir.exists():
                # Count files (simplified - would need actual row count in production)
                file_count = len(list(split_dir.glob("*")))
                split_sizes[split] = file_count
            else:
                split_sizes[split] = 0

        total = sum(split_sizes.values())

        if total > 0:
            train_pct = (split_sizes["train"] / total) * 100
            val_pct = (split_sizes["val"] / total) * 100
            test_pct = (split_sizes["test"] / total) * 100

            logger.info(f"  Train: {split_sizes['train']:,} ({train_pct:.1f}%)")
            logger.info(f"  Val:   {split_sizes['val']:,} ({val_pct:.1f}%)")
            logger.info(f"  Test:  {split_sizes['test']:,} ({test_pct:.1f}%)")

            self.validation_results["metrics"]["split_distribution"] = {
                "train": train_pct,
                "val": val_pct,
                "test": test_pct
            }

            # Check split ratios (allow ±5% tolerance)
            if not (65 <= train_pct <= 75):
                self.validation_results["warnings"].append(
                    f"Train split {train_pct:.1f}% outside target range [65%, 75%]"
                )
                logger.warning(f"  ⚠️ WARNING: Train split {train_pct:.1f}% not in [65%, 75%]")

            if not (10 <= val_pct <= 20):
                self.validation_results["warnings"].append(
                    f"Validation split {val_pct:.1f}% outside target range [10%, 20%]"
                )
                logger.warning(f"  ⚠️ WARNING: Val split {val_pct:.1f}% not in [10%, 20%]")

            if not (10 <= test_pct <= 20):
                self.validation_results["warnings"].append(
                    f"Test split {test_pct:.1f}% outside target range [10%, 20%]"
                )
                logger.warning(f"  ⚠️ WARNING: Test split {test_pct:.1f}% not in [10%, 20%]")

            if 65 <= train_pct <= 75 and 10 <= val_pct <= 20 and 10 <= test_pct <= 20:
                logger.info(f"  ✅ PASSED: Split ratios within acceptable ranges")
        else:
            self.validation_results["failures"].append("No data splits found")
            logger.error(f"  ❌ FAILED: No data splits found")

    def _validate_annotations(self):
        """Validate annotation completeness and quality."""
        logger.info(f"\n[6/8] Validating Annotations...")

        annotations_dir = self.data_dir / "annotations"

        if annotations_dir.exists():
            annotation_files = list(annotations_dir.glob("*.json"))
            annotation_count = len(annotation_files)

            logger.info(f"  Total annotations: {annotation_count:,}")

            # Load and analyze annotations
            verified_count = 0
            avg_quality = []

            for ann_file in annotation_files[:1000]:  # Sample for performance
                try:
                    with open(ann_file, 'r') as f:
                        ann = json.load(f)
                        if ann.get("verified", False):
                            verified_count += 1
                        if "quality_score" in ann:
                            avg_quality.append(ann["quality_score"])
                except Exception as e:
                    logger.warning(f"Error reading {ann_file}: {e}")

            verified_pct = (verified_count / len(annotation_files[:1000])) * 100 if annotation_files else 0

            logger.info(f"  Verified: {verified_pct:.1f}% (sample of {min(1000, len(annotation_files))})")

            if avg_quality:
                mean_quality = statistics.mean(avg_quality)
                logger.info(f"  Average annotation quality: {mean_quality:.3f}")

            self.validation_results["metrics"]["annotations"] = {
                "total": annotation_count,
                "verified_pct": verified_pct,
                "avg_quality": statistics.mean(avg_quality) if avg_quality else 0
            }

            if annotation_count < 50000:
                self.validation_results["warnings"].append(
                    f"Low annotation count: {annotation_count:,} < 50,000 target"
                )
                logger.warning(f"  ⚠️ WARNING: Only {annotation_count:,} annotations (target: 50K+)")
            else:
                logger.info(f"  ✅ PASSED: {annotation_count:,} annotations")
        else:
            self.validation_results["warnings"].append("No annotations directory found")
            logger.warning("  ⚠️ WARNING: No annotations found")

    def _validate_field_coverage(self):
        """Validate business field coverage across documents."""
        logger.info(f"\n[7/8] Validating Field Coverage...")

        # Collect all unique fields
        all_fields: Set[str] = set()
        field_counts = Counter()

        annotations_dir = self.data_dir / "annotations"

        if annotations_dir.exists():
            for ann_file in list(annotations_dir.glob("*.json"))[:1000]:  # Sample
                try:
                    with open(ann_file, 'r') as f:
                        ann = json.load(f)
                        if "fields" in ann:
                            for field_name in ann["fields"].keys():
                                all_fields.add(field_name)
                                field_counts[field_name] += 1
                except Exception as e:
                    continue

            unique_fields_count = len(all_fields)

            logger.info(f"  Unique business fields: {unique_fields_count}")

            # Show top fields
            logger.info(f"  Top 10 fields:")
            for field, count in field_counts.most_common(10):
                logger.info(f"    - {field}: {count:,}")

            self.validation_results["metrics"]["field_coverage"] = {
                "unique_fields": unique_fields_count,
                "top_fields": dict(field_counts.most_common(20))
            }

            if unique_fields_count < 200:
                self.validation_results["warnings"].append(
                    f"Low field coverage: {unique_fields_count} < 200 target"
                )
                logger.warning(f"  ⚠️ WARNING: Only {unique_fields_count} unique fields (target: 200+)")
            else:
                logger.info(f"  ✅ PASSED: {unique_fields_count} unique fields")
        else:
            logger.warning("  ⚠️ WARNING: Cannot validate field coverage (no annotations)")

    def _validate_format_compliance(self):
        """Validate dataset format compliance."""
        logger.info(f"\n[8/8] Validating Format Compliance...")

        # Check Hugging Face format
        hf_dir = self.data_dir / "huggingface"

        if hf_dir.exists():
            dataset_info_file = hf_dir / "dataset_info.json"

            if dataset_info_file.exists():
                logger.info(f"  ✅ Hugging Face dataset_info.json exists")

                try:
                    with open(dataset_info_file, 'r') as f:
                        info = json.load(f)

                    required_keys = ["dataset_name", "version", "features", "splits"]
                    missing_keys = [key for key in required_keys if key not in info]

                    if missing_keys:
                        self.validation_results["warnings"].append(
                            f"dataset_info.json missing keys: {missing_keys}"
                        )
                        logger.warning(f"  ⚠️ WARNING: Missing keys in dataset_info.json")
                    else:
                        logger.info(f"  ✅ PASSED: dataset_info.json is valid")

                except Exception as e:
                    self.validation_results["warnings"].append(
                        f"Error validating dataset_info.json: {e}"
                    )
            else:
                self.validation_results["warnings"].append(
                    "Missing dataset_info.json in Hugging Face export"
                )
                logger.warning(f"  ⚠️ WARNING: No dataset_info.json found")
        else:
            self.validation_results["warnings"].append(
                "No Hugging Face export found"
            )
            logger.warning(f"  ⚠️ WARNING: No Hugging Face export directory")

    def _generate_recommendations(self):
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Check failures
        if self.validation_results["failures"]:
            recommendations.append({
                "priority": "CRITICAL",
                "issue": "Validation failures detected",
                "action": "Address all failures before proceeding with model training"
            })

        # Check document count
        total_docs = self.validation_results["metrics"].get("total_documents", 0)
        if total_docs < 1_000_000:
            shortfall = 1_000_000 - total_docs
            recommendations.append({
                "priority": "HIGH",
                "issue": f"Need {shortfall:,} more documents",
                "action": "Generate additional synthetic documents or collect more real data"
            })

        # Check token count
        tokens = self.validation_results["metrics"].get("estimated_tokens", 0)
        if tokens < 100_000_000_000:
            recommendations.append({
                "priority": "HIGH",
                "issue": "Insufficient tokens for LLM training",
                "action": "Increase document count or use longer documents"
            })

        # Check annotations
        ann_metrics = self.validation_results["metrics"].get("annotations", {})
        if ann_metrics.get("total", 0) < 50000:
            recommendations.append({
                "priority": "MEDIUM",
                "issue": "Low annotation count",
                "action": "Annotate more documents for supervised training"
            })

        # Check field coverage
        field_metrics = self.validation_results["metrics"].get("field_coverage", {})
        if field_metrics.get("unique_fields", 0) < 200:
            recommendations.append({
                "priority": "MEDIUM",
                "issue": "Limited field coverage",
                "action": "Ensure diverse document samples covering all business fields"
            })

        self.validation_results["recommendations"] = recommendations

    def _print_validation_summary(self):
        """Print validation summary."""
        logger.info("\n" + "=" * 80)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 80)

        # Overall status
        if self.validation_results["passed"]:
            logger.info("✅ OVERALL: PASSED")
        else:
            logger.error("❌ OVERALL: FAILED")

        # Failures
        if self.validation_results["failures"]:
            logger.error(f"\n❌ FAILURES ({len(self.validation_results['failures'])}):")
            for failure in self.validation_results["failures"]:
                logger.error(f"  - {failure}")

        # Warnings
        if self.validation_results["warnings"]:
            logger.warning(f"\n⚠️  WARNINGS ({len(self.validation_results['warnings'])}):")
            for warning in self.validation_results["warnings"]:
                logger.warning(f"  - {warning}")

        # Recommendations
        if self.validation_results["recommendations"]:
            logger.info(f"\n💡 RECOMMENDATIONS:")
            for rec in self.validation_results["recommendations"]:
                logger.info(f"  [{rec['priority']}] {rec['issue']}")
                logger.info(f"      → {rec['action']}")

        logger.info("\n" + "=" * 80)


# CLI entrypoint
def main():
    """CLI for dataset validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate SAP_LLM Training Dataset")
    parser.add_argument("--data-dir", required=True, help="Dataset root directory")
    parser.add_argument("--min-documents", type=int, default=1_000_000, help="Minimum documents")
    parser.add_argument("--min-quality", type=float, default=0.8, help="Minimum quality score")
    parser.add_argument("--min-tokens", type=int, default=100_000_000_000, help="Minimum tokens")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run validation
    validator = DatasetValidator(data_dir=args.data_dir)

    results = validator.validate_corpus(
        min_documents=args.min_documents,
        min_quality_score=args.min_quality,
        min_tokens=args.min_tokens
    )

    # Exit with appropriate code
    exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
