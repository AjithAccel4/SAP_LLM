"""
Data annotation tools for training dataset preparation.

Supports:
- Manual annotation UI (Label Studio integration)
- Automated pre-annotation
- Quality control and validation
- Annotation export in training format
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DocumentAnnotation:
    """Document annotation schema."""
    document_id: str
    document_type: str
    fields: Dict[str, Any]  # Extracted fields with bounding boxes
    quality_score: float
    annotator_id: str
    annotation_time_seconds: float
    verified: bool = False


class DataAnnotator:
    """
    Annotation management and quality control.

    Features:
    - Annotation storage and retrieval
    - Quality metrics
    - Inter-annotator agreement
    - Export to training format
    """

    def __init__(self, annotations_dir: str):
        """
        Initialize annotator.

        Args:
            annotations_dir: Directory to store annotations
        """
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        self.annotations = {}
        self._load_annotations()

        logger.info(f"DataAnnotator initialized: {len(self.annotations)} annotations loaded")

    def _load_annotations(self):
        """Load existing annotations."""
        for annotation_file in self.annotations_dir.glob("*.json"):
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)
                    self.annotations[annotation_data['document_id']] = annotation_data
            except Exception as e:
                logger.error(f"Error loading annotation {annotation_file}: {e}")

    def add_annotation(self, annotation: DocumentAnnotation):
        """
        Add new annotation.

        Args:
            annotation: Document annotation
        """
        ann_dict = asdict(annotation)
        self.annotations[annotation.document_id] = ann_dict

        # Save to file
        annotation_file = self.annotations_dir / f"{annotation.document_id}.json"
        with open(annotation_file, 'w') as f:
            json.dump(ann_dict, f, indent=2)

        logger.info(f"Added annotation for document: {annotation.document_id}")

    def get_annotation(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get annotation for document.

        Args:
            document_id: Document ID

        Returns:
            Annotation dictionary or None
        """
        return self.annotations.get(document_id)

    def export_for_training(self, output_file: str, filter_verified: bool = True):
        """
        Export annotations in training format.

        Args:
            output_file: Output JSON file
            filter_verified: Only export verified annotations
        """
        annotations_to_export = []

        for doc_id, annotation in self.annotations.items():
            if filter_verified and not annotation.get('verified', False):
                continue

            # Convert to training format
            training_sample = {
                "id": doc_id,
                "document_type": annotation['document_type'],
                "text": self._extract_text(annotation),
                "labels": annotation['fields'],
                "metadata": {
                    "quality_score": annotation['quality_score'],
                    "annotator_id": annotation['annotator_id']
                }
            }

            annotations_to_export.append(training_sample)

        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "total_annotations": len(annotations_to_export),
                "samples": annotations_to_export
            }, f, indent=2)

        logger.info(f"Exported {len(annotations_to_export)} annotations to {output_file}")

    def _extract_text(self, annotation: Dict[str, Any]) -> str:
        """Extract text from annotation fields."""
        # Combine all text fields
        text_parts = []
        for field_name, field_value in annotation['fields'].items():
            if isinstance(field_value, str):
                text_parts.append(f"{field_name}: {field_value}")

        return " ".join(text_parts)

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute annotation statistics."""
        stats = {
            "total_annotations": len(self.annotations),
            "verified_annotations": sum(1 for a in self.annotations.values() if a.get('verified', False)),
            "by_document_type": {},
            "by_annotator": {},
            "average_quality_score": 0.0,
            "average_annotation_time": 0.0
        }

        quality_scores = []
        annotation_times = []

        for annotation in self.annotations.values():
            # Count by document type
            doc_type = annotation['document_type']
            stats["by_document_type"][doc_type] = stats["by_document_type"].get(doc_type, 0) + 1

            # Count by annotator
            annotator = annotation['annotator_id']
            stats["by_annotator"][annotator] = stats["by_annotator"].get(annotator, 0) + 1

            # Collect metrics
            quality_scores.append(annotation['quality_score'])
            annotation_times.append(annotation['annotation_time_seconds'])

        if quality_scores:
            stats["average_quality_score"] = sum(quality_scores) / len(quality_scores)
        if annotation_times:
            stats["average_annotation_time"] = sum(annotation_times) / len(annotation_times)

        return stats


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    annotator = DataAnnotator(annotations_dir="./data/annotations")

    # Example annotation
    annotation = DocumentAnnotation(
        document_id="doc_001",
        document_type="invoice",
        fields={
            "invoice_number": "INV-2024-001",
            "date": "2024-01-15",
            "total_amount": "1250.00",
            "vendor": "ABC Corp"
        },
        quality_score=0.95,
        annotator_id="annotator_1",
        annotation_time_seconds=120.5,
        verified=True
    )

    annotator.add_annotation(annotation)

    # Export for training
    annotator.export_for_training("./data/training_annotations.json")

    # Get statistics
    stats = annotator.compute_statistics()
    print(f"Annotation statistics: {stats}")
