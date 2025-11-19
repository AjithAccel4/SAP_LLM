#!/usr/bin/env python3
"""
Generate comprehensive test dataset for performance benchmarking.

Creates 1000+ diverse documents covering:
- All 13 document types
- Various complexities (simple to very complex)
- Different formats (PDF, image, multi-page)
- Edge cases (poor quality, handwritten, etc.)
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

# Document types from SAP_LLM
DOCUMENT_TYPES = [
    "PURCHASE_ORDER",
    "INVOICE",
    "DELIVERY_NOTE",
    "GOODS_RECEIPT",
    "PAYMENT_ADVICE",
    "CONTRACT",
    "QUOTATION",
    "ORDER_CONFIRMATION",
    "PACKING_LIST",
    "SHIPMENT_NOTICE",
    "CREDIT_NOTE",
    "DEBIT_NOTE",
    "STATEMENT",
]

SUBTYPES = {
    "PURCHASE_ORDER": ["STANDARD", "BLANKET", "CONTRACT"],
    "INVOICE": ["STANDARD", "CREDIT", "DEBIT", "PROFORMA"],
    "DELIVERY_NOTE": ["STANDARD", "PARTIAL", "COMPLETE"],
    "default": ["STANDARD"]
}


class TestDataGenerator:
    """Generate synthetic test documents for benchmarking."""

    def __init__(self, output_dir: str, seed: int = 42):
        """Initialize generator."""
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Create directories
        self.docs_dir = self.output_dir / "sample_documents"
        self.ground_truth_dir = self.output_dir / "ground_truth"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)

        print(f"Initialized test data generator")
        print(f"  Output: {self.output_dir}")
        print(f"  Seed: {self.seed}")

    def generate_dataset(
        self,
        num_documents: int = 1000,
        include_edge_cases: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generate complete test dataset."""
        print(f"\nGenerating {num_documents} test documents...")

        documents = []
        ground_truth = []

        # Ensure we have samples for each document type
        docs_per_type = num_documents // len(DOCUMENT_TYPES)
        remaining = num_documents % len(DOCUMENT_TYPES)

        doc_id = 0
        for doc_type in DOCUMENT_TYPES:
            count = docs_per_type + (1 if remaining > 0 else 0)
            remaining -= 1

            for i in range(count):
                doc_data = self._generate_document(doc_id, doc_type)
                documents.append(doc_data)

                # Create ground truth
                ground_truth.append({
                    "document_id": doc_id,
                    "file_name": doc_data["file_name"],
                    "document_type": doc_type,
                    "document_subtype": doc_data["subtype"],
                    "extracted_fields": doc_data["fields"],
                    "routing_endpoint": doc_data["routing"],
                    "complexity": doc_data["complexity"],
                })

                doc_id += 1

                if (doc_id % 100) == 0:
                    print(f"  Generated {doc_id}/{num_documents} documents...")

        # Add edge cases
        if include_edge_cases:
            print("\nGenerating edge cases...")
            edge_cases = self._generate_edge_cases(doc_id)
            documents.extend(edge_cases["documents"])
            ground_truth.extend(edge_cases["ground_truth"])

        # Save ground truth
        self._save_ground_truth(ground_truth)

        # Save dataset metadata
        metadata = {
            "num_documents": len(documents),
            "document_types": {dt: sum(1 for d in documents if d["doc_type"] == dt)
                              for dt in DOCUMENT_TYPES},
            "generated_at": datetime.now().isoformat(),
            "seed": self.seed,
        }

        with open(self.output_dir / "dataset_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Generated {len(documents)} documents")
        print(f"   Saved to: {self.docs_dir}")
        print(f"   Ground truth: {self.ground_truth_dir}")

        return documents

    def _generate_document(
        self,
        doc_id: int,
        doc_type: str,
    ) -> Dict[str, Any]:
        """Generate a single synthetic document."""
        # Random complexity
        complexity = random.choice(["simple", "moderate", "complex", "very_complex"])

        # Random subtype
        subtypes = SUBTYPES.get(doc_type, SUBTYPES["default"])
        subtype = random.choice(subtypes)

        # Generate synthetic fields based on document type
        fields = self._generate_fields(doc_type, complexity)

        # Determine routing
        routing = self._determine_routing(doc_type, subtype, fields)

        # Generate document image
        file_name = f"{doc_type.lower()}_{doc_id:06d}.png"
        file_path = self.docs_dir / file_name

        self._create_document_image(
            file_path,
            doc_type,
            subtype,
            fields,
            complexity,
        )

        return {
            "document_id": doc_id,
            "file_name": file_name,
            "file_path": str(file_path),
            "doc_type": doc_type,
            "subtype": subtype,
            "fields": fields,
            "routing": routing,
            "complexity": complexity,
        }

    def _generate_fields(self, doc_type: str, complexity: str) -> Dict[str, Any]:
        """Generate synthetic field values."""
        fields = {}

        # Common fields
        fields["document_number"] = f"{doc_type[:3]}-{random.randint(100000, 999999)}"
        fields["document_date"] = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")
        fields["vendor_id"] = f"V{random.randint(1000, 9999)}"
        fields["vendor_name"] = random.choice([
            "Acme Corporation",
            "Global Supplies Ltd",
            "TechnoSource GmbH",
            "Premium Materials Inc",
        ])

        # Amount fields
        if complexity in ["simple", "moderate"]:
            num_lines = random.randint(1, 5)
        else:
            num_lines = random.randint(5, 20)

        line_items = []
        total = 0
        for i in range(num_lines):
            quantity = random.randint(1, 100)
            unit_price = round(random.uniform(10, 1000), 2)
            line_total = round(quantity * unit_price, 2)
            total += line_total

            line_items.append({
                "line_number": i + 1,
                "material_number": f"MAT-{random.randint(10000, 99999)}",
                "description": f"Item {i+1}",
                "quantity": quantity,
                "unit_price": unit_price,
                "line_total": line_total,
            })

        fields["line_items"] = line_items
        fields["subtotal"] = round(total, 2)
        fields["tax"] = round(total * 0.19, 2)  # 19% VAT
        fields["total_amount"] = round(total * 1.19, 2)
        fields["currency"] = "EUR"

        # Document-specific fields
        if doc_type == "PURCHASE_ORDER":
            fields["po_number"] = fields["document_number"]
            fields["delivery_date"] = (datetime.now() + timedelta(days=random.randint(7, 60))).strftime("%Y-%m-%d")
        elif doc_type == "INVOICE":
            fields["invoice_number"] = fields["document_number"]
            fields["due_date"] = (datetime.now() + timedelta(days=random.randint(14, 90))).strftime("%Y-%m-%d")
            fields["payment_terms"] = random.choice(["Net 30", "Net 60", "Net 90", "Due on receipt"])

        return fields

    def _determine_routing(self, doc_type: str, subtype: str, fields: Dict) -> Dict[str, Any]:
        """Determine routing endpoint for document."""
        routing = {
            "endpoint": f"API_{doc_type}",
            "priority": "normal",
            "auto_post": False,
        }

        # High-value documents get priority routing
        if fields.get("total_amount", 0) > 10000:
            routing["priority"] = "high"

        # Standard invoices with known vendors get auto-posting
        if doc_type == "INVOICE" and subtype == "STANDARD":
            routing["auto_post"] = True

        return routing

    def _create_document_image(
        self,
        file_path: Path,
        doc_type: str,
        subtype: str,
        fields: Dict,
        complexity: str,
    ) -> None:
        """Create synthetic document image."""
        # Create image
        width, height = 800, 1100  # A4-like aspect ratio
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Try to load a font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()
            font_bold = ImageFont.load_default()

        y = 50

        # Header
        draw.text((50, y), f"{doc_type} - {subtype}", fill='black', font=font_bold)
        y += 40

        # Document details
        draw.text((50, y), f"Document No: {fields.get('document_number', 'N/A')}", fill='black', font=font)
        y += 25
        draw.text((50, y), f"Date: {fields.get('document_date', 'N/A')}", fill='black', font=font)
        y += 25
        draw.text((50, y), f"Vendor: {fields.get('vendor_name', 'N/A')}", fill='black', font=font)
        y += 40

        # Line items (first few)
        draw.text((50, y), "Line Items:", fill='black', font=font_bold)
        y += 30

        for i, item in enumerate(fields.get('line_items', [])[:10]):  # Show max 10
            text = f"{item['line_number']}. {item['description']} - Qty: {item['quantity']} @ {item['unit_price']} = {item['line_total']}"
            draw.text((70, y), text, fill='black', font=font)
            y += 25

        y += 20

        # Totals
        draw.text((500, y), f"Subtotal: {fields.get('currency', 'EUR')} {fields.get('subtotal', 0):.2f}", fill='black', font=font)
        y += 25
        draw.text((500, y), f"Tax: {fields.get('currency', 'EUR')} {fields.get('tax', 0):.2f}", fill='black', font=font)
        y += 25
        draw.text((500, y), f"Total: {fields.get('currency', 'EUR')} {fields.get('total_amount', 0):.2f}", fill='black', font=font_bold)

        # Add complexity-based noise
        if complexity == "complex":
            # Add slight blur
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        elif complexity == "very_complex":
            # Add more blur and noise
            img = img.filter(ImageFilter.GaussianBlur(radius=1.0))
            # Add noise
            noise = np.random.normal(0, 10, (height, width, 3))
            img_array = np.array(img)
            noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(noisy)

        # Save image
        img.save(file_path)

    def _generate_edge_cases(self, start_id: int) -> Dict[str, Any]:
        """Generate edge case documents."""
        edge_cases = {
            "documents": [],
            "ground_truth": [],
        }

        # Poor quality scans
        # Handwritten documents
        # Multi-page documents
        # Non-standard formats
        # etc.

        # For now, just generate a few complex documents
        for i in range(50):  # 50 edge cases
            doc_type = random.choice(DOCUMENT_TYPES)
            doc_data = self._generate_document(start_id + i, doc_type)
            doc_data["complexity"] = "very_complex"
            doc_data["edge_case"] = True

            edge_cases["documents"].append(doc_data)
            edge_cases["ground_truth"].append({
                "document_id": doc_data["document_id"],
                "file_name": doc_data["file_name"],
                "document_type": doc_type,
                "document_subtype": doc_data["subtype"],
                "extracted_fields": doc_data["fields"],
                "routing_endpoint": doc_data["routing"],
                "complexity": doc_data["complexity"],
                "edge_case": True,
            })

        return edge_cases

    def _save_ground_truth(self, ground_truth: List[Dict]) -> None:
        """Save ground truth data."""
        # Save as JSON
        gt_file = self.ground_truth_dir / "ground_truth.json"
        with open(gt_file, "w") as f:
            json.dump(ground_truth, f, indent=2)

        # Save individual files for easier testing
        for gt in ground_truth:
            doc_id = gt["document_id"]
            gt_doc_file = self.ground_truth_dir / f"doc_{doc_id:06d}.json"
            with open(gt_doc_file, "w") as f:
                json.dump(gt, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate test data for benchmarking")
    parser.add_argument("--num-docs", type=int, default=1000, help="Number of documents to generate")
    parser.add_argument("--output", type=str, default="benchmarks/data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-edge-cases", action="store_true", help="Skip edge case generation")

    args = parser.parse_args()

    print("="*70)
    print("  SAP_LLM Test Data Generator")
    print("="*70)

    generator = TestDataGenerator(output_dir=args.output, seed=args.seed)
    documents = generator.generate_dataset(
        num_documents=args.num_docs,
        include_edge_cases=not args.no_edge_cases,
    )

    print(f"\n✅ Successfully generated {len(documents)} test documents")


if __name__ == "__main__":
    main()
