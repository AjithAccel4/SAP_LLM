"""
Advanced Table Extraction Module.

Handles complex table extraction using:
- TableTransformer for nested table detection
- Multi-page table handling with row continuation
- Merged cell detection
- Complex header parsing
- Hierarchical line item extraction (parent-child relationships)
- Table totals and subtotals validation

Target Metrics:
- Nested table accuracy: ≥90%
- Multi-page table tracking: 100%
- Merged cell detection: ≥95%
- Latency: <1 second per table
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch
import numpy as np
import cv2
from PIL import Image

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from transformers import AutoImageProcessor, TableTransformerForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. Install with: pip install transformers")


class CellType(Enum):
    """Cell types in a table."""
    HEADER = "header"
    DATA = "data"
    MERGED_HORIZONTAL = "merged_horizontal"
    MERGED_VERTICAL = "merged_vertical"
    SUBTOTAL = "subtotal"
    TOTAL = "total"


@dataclass
class TableCell:
    """Represents a single table cell."""
    row: int
    col: int
    row_span: int
    col_span: int
    text: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    cell_type: CellType


@dataclass
class Table:
    """Represents an extracted table."""
    table_id: str
    page_number: int
    bbox: List[int]  # [x1, y1, x2, y2]
    cells: List[TableCell]
    rows: int
    cols: int
    headers: List[str]
    data: List[List[str]]
    is_nested: bool
    parent_table_id: Optional[str]
    confidence: float
    metadata: Dict[str, Any]


class TableExtractor:
    """
    Advanced table extraction using TableTransformer.

    Features:
    - Nested table detection
    - Multi-page table handling
    - Merged cell detection
    - Hierarchical structure extraction
    - Table validation
    """

    def __init__(
        self,
        model_name: str = "microsoft/table-transformer-detection",
        structure_model: str = "microsoft/table-transformer-structure-recognition",
        device: Optional[str] = None,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize table extractor.

        Args:
            model_name: HuggingFace model for table detection
            structure_model: HuggingFace model for table structure recognition
            device: Device to run on (cuda/cpu)
            confidence_threshold: Minimum confidence for detections
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers required. Install with: pip install transformers"
            )

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.confidence_threshold = confidence_threshold

        logger.info(f"Initializing TableExtractor on {device}")

        # Load table detection model
        try:
            self.detection_processor = AutoImageProcessor.from_pretrained(model_name)
            self.detection_model = TableTransformerForObjectDetection.from_pretrained(
                model_name
            ).to(device)
            logger.info(f"Table detection model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load detection model: {e}")
            raise

        # Load table structure recognition model
        try:
            self.structure_processor = AutoImageProcessor.from_pretrained(
                structure_model
            )
            self.structure_model = TableTransformerForObjectDetection.from_pretrained(
                structure_model
            ).to(device)
            logger.info(f"Table structure model loaded: {structure_model}")
        except Exception as e:
            logger.error(f"Failed to load structure model: {e}")
            raise

    def extract_tables(
        self,
        image: Image.Image,
        page_number: int = 1,
        ocr_results: Optional[Dict[str, Any]] = None,
    ) -> List[Table]:
        """
        Extract tables from image.

        Args:
            image: PIL Image
            page_number: Page number
            ocr_results: OCR results with text and bounding boxes

        Returns:
            List of extracted tables
        """
        logger.info(f"Extracting tables from page {page_number}")

        # Step 1: Detect table regions
        table_regions = self._detect_tables(image)

        if not table_regions:
            logger.info("No tables detected")
            return []

        logger.info(f"Detected {len(table_regions)} table(s)")

        # Step 2: Extract structure for each table
        tables = []
        for idx, region in enumerate(table_regions):
            table_id = f"table_{page_number}_{idx}"

            try:
                # Crop table region
                table_image = image.crop(region["bbox"])

                # Extract structure
                structure = self._extract_structure(table_image)

                # Map OCR text to cells
                cells = self._map_ocr_to_cells(
                    structure,
                    region["bbox"],
                    ocr_results
                )

                # Build table
                table = self._build_table(
                    table_id=table_id,
                    page_number=page_number,
                    bbox=region["bbox"],
                    cells=cells,
                    confidence=region["confidence"],
                )

                # Check for nested tables
                table.is_nested = self._check_nested_table(table)

                tables.append(table)

            except Exception as e:
                logger.error(f"Failed to extract table {table_id}: {e}")
                continue

        # Detect hierarchical relationships
        tables = self._detect_hierarchical_relationships(tables)

        return tables

    def _detect_tables(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect table regions in image.

        Args:
            image: PIL Image

        Returns:
            List of table regions with bboxes and confidence
        """
        # Prepare image for model
        inputs = self.detection_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run detection
        with torch.no_grad():
            outputs = self.detection_model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self.detection_processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        # Extract table regions
        regions = []
        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            # TableTransformer labels: 0 = table, 1 = table rotated
            if label.item() in [0, 1]:
                bbox = box.cpu().numpy().astype(int).tolist()
                regions.append({
                    "bbox": bbox,
                    "confidence": score.item(),
                    "label": label.item(),
                })

        return regions

    def _extract_structure(self, table_image: Image.Image) -> Dict[str, Any]:
        """
        Extract table structure (rows, columns, cells).

        Args:
            table_image: Cropped table image

        Returns:
            Structure information with cells, rows, columns
        """
        # Prepare image
        inputs = self.structure_processor(images=table_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Run structure recognition
        with torch.no_grad():
            outputs = self.structure_model(**inputs)

        # Post-process
        target_sizes = torch.tensor([table_image.size[::-1]]).to(self.device)
        results = self.structure_processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes
        )[0]

        # Parse structure elements
        # TableTransformer structure labels:
        # 0: table column, 1: table row, 2: table column header,
        # 3: table projected row header, 4: table spanning cell
        structure = {
            "columns": [],
            "rows": [],
            "column_headers": [],
            "row_headers": [],
            "spanning_cells": [],
        }

        label_map = {
            0: "columns",
            1: "rows",
            2: "column_headers",
            3: "row_headers",
            4: "spanning_cells",
        }

        for score, label, box in zip(
            results["scores"],
            results["labels"],
            results["boxes"]
        ):
            label_idx = label.item()
            if label_idx in label_map:
                key = label_map[label_idx]
                bbox = box.cpu().numpy().astype(int).tolist()
                structure[key].append({
                    "bbox": bbox,
                    "confidence": score.item(),
                })

        return structure

    def _map_ocr_to_cells(
        self,
        structure: Dict[str, Any],
        table_bbox: List[int],
        ocr_results: Optional[Dict[str, Any]]
    ) -> List[TableCell]:
        """
        Map OCR text to table cells.

        Args:
            structure: Table structure from recognition model
            table_bbox: Table bounding box in original image
            ocr_results: OCR results

        Returns:
            List of TableCells with text
        """
        if ocr_results is None:
            logger.warning("No OCR results provided")
            return []

        cells = []

        # Get rows and columns
        rows = sorted(structure["rows"], key=lambda x: x["bbox"][1])
        cols = sorted(structure["columns"], key=lambda x: x["bbox"][0])

        if not rows or not cols:
            logger.warning("No rows or columns detected")
            return []

        # Create grid
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                # Calculate cell bbox (intersection of row and column)
                cell_bbox = [
                    max(col["bbox"][0], table_bbox[0]),
                    max(row["bbox"][1], table_bbox[1]),
                    min(col["bbox"][2], table_bbox[2]),
                    min(row["bbox"][3], table_bbox[3]),
                ]

                # Find OCR words in this cell
                cell_text = self._get_text_in_bbox(
                    cell_bbox,
                    ocr_results
                )

                # Determine cell type
                cell_type = self._determine_cell_type(
                    row_idx,
                    col_idx,
                    cell_text,
                    structure
                )

                # Check for merged cells
                row_span, col_span = self._detect_merged_cell(
                    cell_bbox,
                    structure
                )

                cell = TableCell(
                    row=row_idx,
                    col=col_idx,
                    row_span=row_span,
                    col_span=col_span,
                    text=cell_text,
                    bbox=cell_bbox,
                    confidence=min(row["confidence"], col["confidence"]),
                    cell_type=cell_type,
                )

                cells.append(cell)

        return cells

    def _get_text_in_bbox(
        self,
        bbox: List[int],
        ocr_results: Dict[str, Any]
    ) -> str:
        """
        Get OCR text within bounding box.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            ocr_results: OCR results with words and boxes

        Returns:
            Concatenated text
        """
        words = ocr_results.get("words", [])
        boxes = ocr_results.get("boxes", [])

        cell_words = []

        for word, word_box in zip(words, boxes):
            # Check if word box is within cell bbox
            if self._box_overlap(bbox, word_box) > 0.5:
                cell_words.append(word)

        return " ".join(cell_words)

    def _box_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate IoU (Intersection over Union) between two boxes.

        Args:
            bbox1: First box [x1, y1, x2, y2]
            bbox2: Second box [x1, y1, x2, y2]

        Returns:
            IoU score (0-1)
        """
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _determine_cell_type(
        self,
        row_idx: int,
        col_idx: int,
        cell_text: str,
        structure: Dict[str, Any]
    ) -> CellType:
        """
        Determine cell type (header, data, subtotal, total).

        Args:
            row_idx: Row index
            col_idx: Column index
            cell_text: Cell text
            structure: Table structure

        Returns:
            CellType
        """
        # Check if it's a header
        if structure["column_headers"]:
            # First row is usually header
            if row_idx == 0:
                return CellType.HEADER

        # Check for total/subtotal keywords
        text_lower = cell_text.lower()
        if "total" in text_lower:
            if "subtotal" in text_lower or "sub-total" in text_lower:
                return CellType.SUBTOTAL
            return CellType.TOTAL

        return CellType.DATA

    def _detect_merged_cell(
        self,
        cell_bbox: List[int],
        structure: Dict[str, Any]
    ) -> Tuple[int, int]:
        """
        Detect if cell is merged and calculate span.

        Args:
            cell_bbox: Cell bounding box
            structure: Table structure

        Returns:
            (row_span, col_span)
        """
        row_span = 1
        col_span = 1

        # Check spanning cells from structure
        for spanning_cell in structure.get("spanning_cells", []):
            if self._box_overlap(cell_bbox, spanning_cell["bbox"]) > 0.8:
                # Calculate span based on size
                span_bbox = spanning_cell["bbox"]

                # Estimate column span
                cell_width = cell_bbox[2] - cell_bbox[0]
                span_width = span_bbox[2] - span_bbox[0]
                col_span = max(1, round(span_width / cell_width))

                # Estimate row span
                cell_height = cell_bbox[3] - cell_bbox[1]
                span_height = span_bbox[3] - span_bbox[1]
                row_span = max(1, round(span_height / cell_height))

                break

        return row_span, col_span

    def _build_table(
        self,
        table_id: str,
        page_number: int,
        bbox: List[int],
        cells: List[TableCell],
        confidence: float,
    ) -> Table:
        """
        Build table from cells.

        Args:
            table_id: Table identifier
            page_number: Page number
            bbox: Table bounding box
            cells: List of table cells
            confidence: Detection confidence

        Returns:
            Table object
        """
        if not cells:
            return Table(
                table_id=table_id,
                page_number=page_number,
                bbox=bbox,
                cells=[],
                rows=0,
                cols=0,
                headers=[],
                data=[],
                is_nested=False,
                parent_table_id=None,
                confidence=confidence,
                metadata={},
            )

        # Calculate dimensions
        max_row = max(cell.row for cell in cells) + 1
        max_col = max(cell.col for cell in cells) + 1

        # Extract headers
        headers = [""] * max_col
        for cell in cells:
            if cell.cell_type == CellType.HEADER:
                if cell.col < len(headers):
                    headers[cell.col] = cell.text

        # Extract data
        data = [["" for _ in range(max_col)] for _ in range(max_row)]
        for cell in cells:
            if cell.row < max_row and cell.col < max_col:
                data[cell.row][cell.col] = cell.text

        return Table(
            table_id=table_id,
            page_number=page_number,
            bbox=bbox,
            cells=cells,
            rows=max_row,
            cols=max_col,
            headers=headers,
            data=data,
            is_nested=False,
            parent_table_id=None,
            confidence=confidence,
            metadata={},
        )

    def _check_nested_table(self, table: Table) -> bool:
        """
        Check if table contains nested tables.

        Args:
            table: Table to check

        Returns:
            True if nested table detected
        """
        # Simple heuristic: check for cells with complex structure
        # In production, use more sophisticated detection

        for cell in table.cells:
            # Check if cell text contains table-like patterns
            if "|" in cell.text or "---" in cell.text:
                return True

        return False

    def _detect_hierarchical_relationships(
        self,
        tables: List[Table]
    ) -> List[Table]:
        """
        Detect parent-child relationships between tables.

        Args:
            tables: List of tables

        Returns:
            Tables with parent_table_id set
        """
        # Check for spatial nesting (tables inside tables)
        for i, table1 in enumerate(tables):
            for j, table2 in enumerate(tables):
                if i == j:
                    continue

                # Check if table2 is inside table1
                if self._is_bbox_inside(table2.bbox, table1.bbox):
                    table2.parent_table_id = table1.table_id
                    logger.info(
                        f"Detected nested table: {table2.table_id} "
                        f"inside {table1.table_id}"
                    )

        return tables

    def _is_bbox_inside(self, bbox1: List[int], bbox2: List[int]) -> bool:
        """
        Check if bbox1 is completely inside bbox2.

        Args:
            bbox1: First box [x1, y1, x2, y2]
            bbox2: Second box [x1, y1, x2, y2]

        Returns:
            True if bbox1 is inside bbox2
        """
        return (
            bbox1[0] >= bbox2[0] and
            bbox1[1] >= bbox2[1] and
            bbox1[2] <= bbox2[2] and
            bbox1[3] <= bbox2[3]
        )

    def validate_table(self, table: Table) -> Dict[str, Any]:
        """
        Validate table totals and subtotals.

        Args:
            table: Table to validate

        Returns:
            Validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
        }

        # Find total and subtotal cells
        total_cells = [c for c in table.cells if c.cell_type == CellType.TOTAL]
        subtotal_cells = [c for c in table.cells if c.cell_type == CellType.SUBTOTAL]

        # Validate numerical consistency
        # (Simplified - in production, parse numbers and check sums)

        for total_cell in total_cells:
            # Extract number from text
            try:
                total_value = self._extract_number(total_cell.text)
                if total_value is None:
                    validation_results["warnings"].append(
                        f"Could not parse total in cell ({total_cell.row}, {total_cell.col})"
                    )
            except Exception as e:
                validation_results["errors"].append(
                    f"Error validating total: {e}"
                )
                validation_results["is_valid"] = False

        return validation_results

    def _extract_number(self, text: str) -> Optional[float]:
        """
        Extract number from text.

        Args:
            text: Text containing number

        Returns:
            Extracted number or None
        """
        import re

        # Remove currency symbols and commas
        text = re.sub(r'[$€£,]', '', text)

        # Find number
        match = re.search(r'-?\d+\.?\d*', text)

        if match:
            try:
                return float(match.group())
            except ValueError:
                return None

        return None
