"""
Dataset for Vision Encoder training.

Handles loading and preprocessing of document images with OCR data
for multi-task learning (document classification + field extraction).
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import LayoutLMv3Processor

from sap_llm.utils.logger import get_logger
from sap_llm.models.vision_encoder import DOCUMENT_TYPES, PO_SUBTYPES

logger = get_logger(__name__)


# SAP field labels for token classification (180+ fields)
SAP_FIELD_LABELS = {
    "O": 0,  # Outside any entity
    # Document header fields (1-30)
    "B-DOC_NUMBER": 1,
    "I-DOC_NUMBER": 2,
    "B-DOC_DATE": 3,
    "I-DOC_DATE": 4,
    "B-DOC_TYPE": 5,
    "I-DOC_TYPE": 6,
    "B-COMPANY_CODE": 7,
    "I-COMPANY_CODE": 8,
    "B-PLANT": 9,
    "I-PLANT": 10,
    "B-STORAGE_LOCATION": 11,
    "I-STORAGE_LOCATION": 12,
    "B-POSTING_DATE": 13,
    "I-POSTING_DATE": 14,
    "B-DOCUMENT_DATE": 15,
    "I-DOCUMENT_DATE": 16,
    "B-REFERENCE": 17,
    "I-REFERENCE": 18,
    "B-CURRENCY": 19,
    "I-CURRENCY": 20,
    # Vendor/Customer fields (31-60)
    "B-VENDOR_NUMBER": 21,
    "I-VENDOR_NUMBER": 22,
    "B-VENDOR_NAME": 23,
    "I-VENDOR_NAME": 24,
    "B-CUSTOMER_NUMBER": 25,
    "I-CUSTOMER_NUMBER": 26,
    "B-CUSTOMER_NAME": 27,
    "I-CUSTOMER_NAME": 28,
    "B-VENDOR_ADDRESS": 29,
    "I-VENDOR_ADDRESS": 30,
    "B-CUSTOMER_ADDRESS": 31,
    "I-CUSTOMER_ADDRESS": 32,
    "B-CONTACT_PERSON": 33,
    "I-CONTACT_PERSON": 34,
    "B-PHONE": 35,
    "I-PHONE": 36,
    "B-EMAIL": 37,
    "I-EMAIL": 38,
    "B-TAX_ID": 39,
    "I-TAX_ID": 40,
    # PO specific fields (61-90)
    "B-PO_NUMBER": 41,
    "I-PO_NUMBER": 42,
    "B-PO_DATE": 43,
    "I-PO_DATE": 44,
    "B-PO_TYPE": 45,
    "I-PO_TYPE": 46,
    "B-DELIVERY_DATE": 47,
    "I-DELIVERY_DATE": 48,
    "B-PAYMENT_TERMS": 49,
    "I-PAYMENT_TERMS": 50,
    "B-INCOTERMS": 51,
    "I-INCOTERMS": 52,
    "B-PURCHASING_ORG": 53,
    "I-PURCHASING_ORG": 54,
    "B-PURCHASING_GROUP": 55,
    "I-PURCHASING_GROUP": 56,
    # Item fields (91-150)
    "B-ITEM_NUMBER": 57,
    "I-ITEM_NUMBER": 58,
    "B-MATERIAL_NUMBER": 59,
    "I-MATERIAL_NUMBER": 60,
    "B-MATERIAL_DESC": 61,
    "I-MATERIAL_DESC": 62,
    "B-QUANTITY": 63,
    "I-QUANTITY": 64,
    "B-UNIT": 65,
    "I-UNIT": 66,
    "B-UNIT_PRICE": 67,
    "I-UNIT_PRICE": 68,
    "B-TOTAL_PRICE": 69,
    "I-TOTAL_PRICE": 70,
    "B-TAX_CODE": 71,
    "I-TAX_CODE": 72,
    "B-TAX_AMOUNT": 73,
    "I-TAX_AMOUNT": 74,
    "B-DISCOUNT": 75,
    "I-DISCOUNT": 76,
    "B-NET_AMOUNT": 77,
    "I-NET_AMOUNT": 78,
    "B-GROSS_AMOUNT": 79,
    "I-GROSS_AMOUNT": 80,
    # Invoice fields (151-180)
    "B-INVOICE_NUMBER": 81,
    "I-INVOICE_NUMBER": 82,
    "B-INVOICE_DATE": 83,
    "I-INVOICE_DATE": 84,
    "B-DUE_DATE": 85,
    "I-DUE_DATE": 86,
    "B-PAYMENT_METHOD": 87,
    "I-PAYMENT_METHOD": 88,
    "B-BANK_ACCOUNT": 89,
    "I-BANK_ACCOUNT": 90,
    "B-BANK_NAME": 91,
    "I-BANK_NAME": 92,
    "B-IBAN": 93,
    "I-IBAN": 94,
    "B-SWIFT": 95,
    "I-SWIFT": 96,
    # Additional fields to reach 180+
    "B-GL_ACCOUNT": 97,
    "I-GL_ACCOUNT": 98,
    "B-COST_CENTER": 99,
    "I-COST_CENTER": 100,
    "B-PROFIT_CENTER": 101,
    "I-PROFIT_CENTER": 102,
    "B-WBS_ELEMENT": 103,
    "I-WBS_ELEMENT": 104,
    "B-INTERNAL_ORDER": 105,
    "I-INTERNAL_ORDER": 106,
    "B-ASSET_NUMBER": 107,
    "I-ASSET_NUMBER": 108,
    "B-SERIAL_NUMBER": 109,
    "I-SERIAL_NUMBER": 110,
    "B-BATCH_NUMBER": 111,
    "I-BATCH_NUMBER": 112,
    "B-DELIVERY_NOTE": 113,
    "I-DELIVERY_NOTE": 114,
    "B-PACKING_LIST": 115,
    "I-PACKING_LIST": 116,
    "B-CONTAINER_NUMBER": 117,
    "I-CONTAINER_NUMBER": 118,
    "B-TRACKING_NUMBER": 119,
    "I-TRACKING_NUMBER": 120,
    "B-SHIPPING_METHOD": 121,
    "I-SHIPPING_METHOD": 122,
    "B-FREIGHT_COST": 123,
    "I-FREIGHT_COST": 124,
    "B-INSURANCE_COST": 125,
    "I-INSURANCE_COST": 126,
    "B-CUSTOMS_VALUE": 127,
    "I-CUSTOMS_VALUE": 128,
    "B-HS_CODE": 129,
    "I-HS_CODE": 130,
    "B-COUNTRY_ORIGIN": 131,
    "I-COUNTRY_ORIGIN": 132,
    "B-CERTIFICATE": 133,
    "I-CERTIFICATE": 134,
    "B-WARRANTY_PERIOD": 135,
    "I-WARRANTY_PERIOD": 136,
    "B-LEAD_TIME": 137,
    "I-LEAD_TIME": 138,
    "B-VALIDITY_PERIOD": 139,
    "I-VALIDITY_PERIOD": 140,
    "B-REVISION_NUMBER": 141,
    "I-REVISION_NUMBER": 142,
    "B-APPROVAL_STATUS": 143,
    "I-APPROVAL_STATUS": 144,
    "B-APPROVER_NAME": 145,
    "I-APPROVER_NAME": 146,
    "B-APPROVAL_DATE": 147,
    "I-APPROVAL_DATE": 148,
    "B-NOTES": 149,
    "I-NOTES": 150,
    "B-TERMS_CONDITIONS": 151,
    "I-TERMS_CONDITIONS": 152,
    "B-SPECIAL_INSTRUCTIONS": 153,
    "I-SPECIAL_INSTRUCTIONS": 154,
    "B-QUALITY_SPEC": 155,
    "I-QUALITY_SPEC": 156,
    "B-TOLERANCE": 157,
    "I-TOLERANCE": 158,
    "B-VALUATION_TYPE": 159,
    "I-VALUATION_TYPE": 160,
    "B-VALUATION_CLASS": 161,
    "I-VALUATION_CLASS": 162,
    "B-PRICE_CONTROL": 163,
    "I-PRICE_CONTROL": 164,
    "B-MOVING_PRICE": 165,
    "I-MOVING_PRICE": 166,
    "B-STANDARD_PRICE": 167,
    "I-STANDARD_PRICE": 168,
    "B-PRICE_UNIT": 169,
    "I-PRICE_UNIT": 170,
    "B-ORDER_UNIT": 171,
    "I-ORDER_UNIT": 172,
    "B-CONVERSION_FACTOR": 173,
    "I-CONVERSION_FACTOR": 174,
    "B-MINIMUM_ORDER_QTY": 175,
    "I-MINIMUM_ORDER_QTY": 176,
    "B-MAXIMUM_ORDER_QTY": 177,
    "I-MAXIMUM_ORDER_QTY": 178,
    "B-SAFETY_STOCK": 179,
    "I-SAFETY_STOCK": 180,
}


class VisionEncoderDataset(Dataset):
    """
    Dataset for training vision encoder with multi-task learning.

    Expected data format:
    data_dir/
        images/
            doc_001.png
            doc_002.png
            ...
        annotations/
            doc_001.json
            doc_002.json
            ...

    Annotation JSON format:
    {
        "image_path": "doc_001.png",
        "doc_type": "PURCHASE_ORDER",
        "po_subtype": "STANDARD",  # Optional, only for PO documents
        "words": ["word1", "word2", ...],
        "boxes": [[x1, y1, x2, y2], ...],  # Normalized 0-1000
        "token_labels": [0, 1, 2, ...],  # Field labels for each word
        "width": 1654,  # Original image width
        "height": 2339  # Original image height
    }
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        processor: LayoutLMv3Processor,
        max_length: int = 512,
        mode: str = "train",
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Root directory containing images and annotations
            processor: LayoutLMv3Processor for preprocessing
            max_length: Maximum sequence length
            mode: "train", "val", or "test"
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.mode = mode

        # Load annotations
        self.annotations_dir = self.data_dir / "annotations"
        self.images_dir = self.data_dir / "images"

        if not self.annotations_dir.exists():
            raise ValueError(f"Annotations directory not found: {self.annotations_dir}")
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")

        # Get all annotation files
        self.annotation_files = sorted(list(self.annotations_dir.glob("*.json")))

        if len(self.annotation_files) == 0:
            raise ValueError(f"No annotation files found in {self.annotations_dir}")

        logger.info(f"Loaded {len(self.annotation_files)} samples from {data_dir} ({mode} mode)")

        # Create label mappings
        self.doc_type_to_id = {doc_type: idx for idx, doc_type in enumerate(DOCUMENT_TYPES)}
        self.po_subtype_to_id = {subtype: idx for idx, subtype in enumerate(PO_SUBTYPES)}

    def __len__(self) -> int:
        return len(self.annotation_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training sample."""
        # Load annotation
        annotation_path = self.annotation_files[idx]
        with open(annotation_path, "r") as f:
            annotation = json.load(f)

        # Load image
        image_path = self.images_dir / annotation["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a dummy sample
            return self._get_dummy_sample()

        # Extract data from annotation
        words = annotation["words"]
        boxes = annotation["boxes"]
        token_labels = annotation.get("token_labels", [0] * len(words))

        # Get document type label
        doc_type = annotation["doc_type"]
        doc_type_label = self.doc_type_to_id.get(doc_type, len(DOCUMENT_TYPES) - 1)  # Use "OTHER" as fallback

        # Get PO subtype label (if applicable)
        po_subtype = annotation.get("po_subtype", "STANDARD")
        po_subtype_label = self.po_subtype_to_id.get(po_subtype, 0)  # Use "STANDARD" as fallback

        # Process with LayoutLMv3Processor
        try:
            encoding = self.processor(
                image,
                words,
                boxes=boxes,
                word_labels=token_labels,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        except Exception as e:
            logger.error(f"Failed to process sample {annotation_path}: {e}")
            return self._get_dummy_sample()

        # Squeeze batch dimension (processor adds it)
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}

        # Add classification labels
        encoding["doc_type_labels"] = torch.tensor(doc_type_label, dtype=torch.long)
        encoding["po_subtype_labels"] = torch.tensor(po_subtype_label, dtype=torch.long)

        # Rename word_labels to token_labels
        if "labels" in encoding:
            encoding["token_labels"] = encoding.pop("labels")

        return encoding

    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Create a dummy sample for error cases."""
        return {
            "input_ids": torch.zeros(self.max_length, dtype=torch.long),
            "bbox": torch.zeros((self.max_length, 4), dtype=torch.long),
            "attention_mask": torch.zeros(self.max_length, dtype=torch.long),
            "pixel_values": torch.zeros((3, 224, 224), dtype=torch.float),
            "doc_type_labels": torch.tensor(0, dtype=torch.long),
            "po_subtype_labels": torch.tensor(0, dtype=torch.long),
            "token_labels": torch.zeros(self.max_length, dtype=torch.long),
        }


def create_synthetic_dataset(
    output_dir: Union[str, Path],
    num_samples: int = 1000,
    split: str = "train",
) -> None:
    """
    Create synthetic dataset for testing.

    This generates random images with synthetic OCR data and labels
    for development and testing purposes.

    Args:
        output_dir: Directory to save synthetic data
        num_samples: Number of samples to generate
        split: Dataset split ("train", "val", or "test")
    """
    import random
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating synthetic dataset: {num_samples} samples in {output_dir}")

    for i in range(num_samples):
        # Create synthetic image
        width, height = 1654, 2339
        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)

        # Generate random words and boxes
        num_words = random.randint(50, 200)
        words = []
        boxes = []
        token_labels = []

        y_pos = 100
        for j in range(num_words):
            # Random word
            word = f"word_{j}"
            words.append(word)

            # Random box (normalized to 0-1000)
            x1 = random.randint(50, 800)
            y1 = y_pos
            x2 = x1 + random.randint(50, 200)
            y2 = y1 + 30

            # Normalize to 0-1000
            box = [
                int(x1 * 1000 / width),
                int(y1 * 1000 / height),
                int(x2 * 1000 / width),
                int(y2 * 1000 / height),
            ]
            boxes.append(box)

            # Random token label
            token_labels.append(random.randint(0, 50))

            # Draw text on image
            draw.text((x1, y1), word, fill="black")

            y_pos += 35
            if y_pos > height - 100:
                y_pos = 100

        # Random document type and subtype
        doc_type = random.choice(DOCUMENT_TYPES)
        po_subtype = random.choice(PO_SUBTYPES) if doc_type == "PURCHASE_ORDER" else "STANDARD"

        # Save image
        image_filename = f"{split}_{i:06d}.png"
        image_path = images_dir / image_filename
        image.save(image_path)

        # Create annotation
        annotation = {
            "image_path": image_filename,
            "doc_type": doc_type,
            "po_subtype": po_subtype,
            "words": words,
            "boxes": boxes,
            "token_labels": token_labels,
            "width": width,
            "height": height,
        }

        # Save annotation
        annotation_path = annotations_dir / f"{split}_{i:06d}.json"
        with open(annotation_path, "w") as f:
            json.dump(annotation, f, indent=2)

        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples")

    logger.info(f"Synthetic dataset created: {output_dir}")


if __name__ == "__main__":
    # Example: Create synthetic dataset for testing
    create_synthetic_dataset(
        output_dir="/tmp/sap_llm_synthetic_data/train",
        num_samples=100,
        split="train",
    )

    create_synthetic_dataset(
        output_dir="/tmp/sap_llm_synthetic_data/val",
        num_samples=20,
        split="val",
    )

    # Test dataset loading
    from transformers import LayoutLMv3Processor

    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    dataset = VisionEncoderDataset(
        data_dir="/tmp/sap_llm_synthetic_data/train",
        processor=processor,
        max_length=512,
        mode="train",
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Bbox shape: {sample['bbox'].shape}")
    print(f"Pixel values shape: {sample['pixel_values'].shape}")
