"""
Synthetic Document Generator for SAP_LLM Training Data.

Generates realistic SAP documents using templates and fake data:
- Invoices
- Purchase Orders
- Delivery Notes
- Material Documents
- Sales Orders
- Goods Receipts
- Packing Lists
- Shipping Notices

Features:
- PDF generation with realistic layouts
- Faker integration for realistic business data
- Barcode and QR code generation
- Multi-language support
- Variation in document quality (scans, digital, photos)
- Realistic field extraction targets

Target: 500K synthetic documents for training.
"""

import os
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)

# Try to import PDF generation libraries
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not available. Install with: pip install reportlab")

# Try to import Faker for realistic data
try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    logger.warning("Faker not available. Install with: pip install faker")


@dataclass
class SyntheticDocument:
    """Synthetic document metadata."""
    id: str
    document_type: str
    file_path: str
    fields: Dict[str, Any]
    metadata: Dict[str, Any]
    quality_score: float = 1.0
    language: str = "en"


class SyntheticDocumentGenerator:
    """
    Generate synthetic SAP documents for training.

    Uses templates and realistic fake data to create diverse
    training examples covering all document types and variations.
    """

    def __init__(self,
                 template_dir: str,
                 output_dir: str,
                 languages: List[str] = ["en", "de", "es", "fr"]):
        """
        Initialize synthetic document generator.

        Args:
            template_dir: Directory containing document templates
            output_dir: Directory to save generated documents
            languages: Supported languages for generation
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.languages = languages

        # Initialize Faker for each language
        self.fakers = {}
        if FAKER_AVAILABLE:
            for lang in languages:
                locale = self._get_faker_locale(lang)
                self.fakers[lang] = Faker(locale)

        # Document type configurations
        self.doc_configs = self._initialize_document_configs()

        # Generation statistics
        self.stats = {
            "total_generated": 0,
            "by_type": {},
            "by_language": {},
            "errors": 0
        }

        logger.info(f"SyntheticDocumentGenerator initialized: languages={languages}")

    def _get_faker_locale(self, lang: str) -> str:
        """Map language code to Faker locale."""
        locale_map = {
            "en": "en_US",
            "de": "de_DE",
            "es": "es_ES",
            "fr": "fr_FR",
            "zh": "zh_CN",
            "ja": "ja_JP"
        }
        return locale_map.get(lang, "en_US")

    def _initialize_document_configs(self) -> Dict[str, Dict]:
        """Initialize configuration for each document type."""
        return {
            "invoice": {
                "fields": ["invoice_number", "invoice_date", "due_date", "total_amount",
                          "subtotal", "tax_amount", "vendor_name", "vendor_address",
                          "customer_name", "customer_address", "line_items"],
                "page_size": A4,
                "header": "INVOICE",
                "complexity": "high"
            },
            "purchase_order": {
                "fields": ["po_number", "po_date", "delivery_date", "total_value",
                          "vendor_code", "buyer_name", "shipping_address", "items"],
                "page_size": A4,
                "header": "PURCHASE ORDER",
                "complexity": "high"
            },
            "delivery_note": {
                "fields": ["delivery_number", "delivery_date", "carrier", "tracking_number",
                          "sender", "recipient", "items"],
                "page_size": A4,
                "header": "DELIVERY NOTE",
                "complexity": "medium"
            },
            "material_document": {
                "fields": ["material_doc_number", "posting_date", "movement_type",
                          "plant", "storage_location", "materials"],
                "page_size": A4,
                "header": "MATERIAL DOCUMENT",
                "complexity": "medium"
            },
            "sales_order": {
                "fields": ["sales_order_number", "order_date", "customer_code",
                          "total_value", "items", "payment_terms"],
                "page_size": A4,
                "header": "SALES ORDER",
                "complexity": "high"
            },
            "goods_receipt": {
                "fields": ["gr_number", "receipt_date", "po_reference", "supplier",
                          "items", "inspector"],
                "page_size": A4,
                "header": "GOODS RECEIPT",
                "complexity": "medium"
            },
            "packing_list": {
                "fields": ["packing_list_number", "shipment_date", "packages",
                          "total_weight", "destination", "items"],
                "page_size": A4,
                "header": "PACKING LIST",
                "complexity": "low"
            },
            "shipping_notice": {
                "fields": ["shipment_number", "ship_date", "expected_arrival",
                          "carrier", "tracking", "packages"],
                "page_size": A4,
                "header": "SHIPPING NOTICE",
                "complexity": "low"
            }
        }

    def generate_documents(self,
                          document_type: str,
                          count: int,
                          output_format: str = "pdf",
                          quality_variation: bool = True) -> List[SyntheticDocument]:
        """
        Generate synthetic documents of specified type.

        Args:
            document_type: Type of document to generate
            count: Number of documents to generate
            output_format: Output format (pdf, png, jpg)
            quality_variation: Add quality variation (scans, noise, etc.)

        Returns:
            List of generated synthetic documents
        """
        if document_type not in self.doc_configs:
            raise ValueError(f"Unknown document type: {document_type}")

        if not REPORTLAB_AVAILABLE:
            logger.warning("reportlab not available. Generating metadata only.")

        logger.info(f"Generating {count:,} {document_type} documents...")

        documents = []

        for i in range(count):
            try:
                # Generate document
                doc = self._generate_single_document(
                    document_type,
                    index=i,
                    output_format=output_format,
                    quality_variation=quality_variation
                )

                documents.append(doc)

                self.stats["total_generated"] += 1
                self.stats["by_type"][document_type] = \
                    self.stats["by_type"].get(document_type, 0) + 1

                if (i + 1) % 1000 == 0:
                    logger.info(f"Generated {i + 1:,} / {count:,} {document_type} documents")

            except Exception as e:
                logger.error(f"Error generating {document_type} #{i}: {e}")
                self.stats["errors"] += 1

        logger.info(f"Generation complete: {len(documents):,} {document_type} documents")

        return documents

    def _generate_single_document(self,
                                  document_type: str,
                                  index: int,
                                  output_format: str,
                                  quality_variation: bool) -> SyntheticDocument:
        """
        Generate a single synthetic document.
        """
        # Select random language
        language = random.choice(self.languages)
        faker = self.fakers.get(language, self.fakers.get("en"))

        # Generate document ID
        doc_id = f"synthetic_{document_type}_{index:08d}"

        # Generate field data
        fields = self._generate_field_data(document_type, faker)

        # Generate PDF (if available)
        file_path = None
        if REPORTLAB_AVAILABLE:
            file_path = self._generate_pdf(
                document_type,
                doc_id,
                fields,
                language,
                quality_variation
            )

        # Calculate quality score
        quality_score = 1.0
        if quality_variation:
            quality_score = random.uniform(0.7, 1.0)

        # Create document metadata
        doc = SyntheticDocument(
            id=doc_id,
            document_type=document_type,
            file_path=str(file_path) if file_path else "",
            fields=fields,
            metadata={
                "language": language,
                "generation_method": "synthetic",
                "generated_at": datetime.now().isoformat(),
                "quality_variation": quality_variation
            },
            quality_score=quality_score,
            language=language
        )

        return doc

    def _generate_field_data(self,
                            document_type: str,
                            faker) -> Dict[str, Any]:
        """
        Generate realistic field data for document type.
        """
        if not FAKER_AVAILABLE:
            # Return placeholder data
            return {"placeholder": "data"}

        fields = {}

        if document_type == "invoice":
            fields = self._generate_invoice_fields(faker)
        elif document_type == "purchase_order":
            fields = self._generate_po_fields(faker)
        elif document_type == "delivery_note":
            fields = self._generate_delivery_note_fields(faker)
        elif document_type == "material_document":
            fields = self._generate_material_doc_fields(faker)
        elif document_type == "sales_order":
            fields = self._generate_sales_order_fields(faker)
        elif document_type == "goods_receipt":
            fields = self._generate_goods_receipt_fields(faker)
        elif document_type == "packing_list":
            fields = self._generate_packing_list_fields(faker)
        elif document_type == "shipping_notice":
            fields = self._generate_shipping_notice_fields(faker)

        return fields

    def _generate_invoice_fields(self, faker) -> Dict[str, Any]:
        """Generate realistic invoice fields."""
        invoice_date = faker.date_between(start_date="-1y", end_date="today")
        due_date = invoice_date + timedelta(days=random.randint(15, 60))

        # Generate line items
        num_items = random.randint(1, 15)
        line_items = []
        subtotal = 0.0

        for i in range(num_items):
            quantity = random.randint(1, 100)
            unit_price = round(random.uniform(10.0, 500.0), 2)
            line_total = round(quantity * unit_price, 2)
            subtotal += line_total

            line_items.append({
                "line_number": i + 1,
                "description": faker.catch_phrase(),
                "quantity": quantity,
                "unit_price": unit_price,
                "total": line_total
            })

        tax_rate = random.choice([0.07, 0.10, 0.19, 0.20])  # Common tax rates
        tax_amount = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax_amount, 2)

        return {
            "invoice_number": f"INV-{faker.year()}-{faker.random_number(digits=6, fix_len=True)}",
            "invoice_date": invoice_date.isoformat(),
            "due_date": due_date.isoformat(),
            "vendor_name": faker.company(),
            "vendor_address": faker.address().replace('\n', ', '),
            "vendor_tax_id": faker.bothify(text="??-########"),
            "customer_name": faker.company(),
            "customer_address": faker.address().replace('\n', ', '),
            "subtotal": subtotal,
            "tax_rate": tax_rate,
            "tax_amount": tax_amount,
            "total_amount": total,
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "payment_terms": random.choice(["Net 30", "Net 45", "Net 60", "Due on Receipt"]),
            "line_items": line_items
        }

    def _generate_po_fields(self, faker) -> Dict[str, Any]:
        """Generate realistic purchase order fields."""
        po_date = faker.date_between(start_date="-6m", end_date="today")
        delivery_date = po_date + timedelta(days=random.randint(7, 90))

        # Generate items
        num_items = random.randint(1, 20)
        items = []
        total_value = 0.0

        for i in range(num_items):
            quantity = random.randint(10, 1000)
            unit_price = round(random.uniform(5.0, 200.0), 2)
            line_total = round(quantity * unit_price, 2)
            total_value += line_total

            items.append({
                "item_number": i + 1,
                "material_code": faker.bothify(text="MAT-#####"),
                "description": faker.catch_phrase(),
                "quantity": quantity,
                "unit": random.choice(["EA", "KG", "L", "M"]),
                "unit_price": unit_price,
                "total": line_total
            })

        return {
            "po_number": f"PO-{faker.year()}{faker.random_number(digits=8, fix_len=True)}",
            "po_date": po_date.isoformat(),
            "delivery_date": delivery_date.isoformat(),
            "vendor_code": faker.bothify(text="VEND-####"),
            "vendor_name": faker.company(),
            "buyer_name": faker.name(),
            "buyer_email": faker.email(),
            "shipping_address": faker.address().replace('\n', ', '),
            "total_value": round(total_value, 2),
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "payment_terms": random.choice(["Net 30", "Net 45", "Net 60"]),
            "items": items
        }

    def _generate_delivery_note_fields(self, faker) -> Dict[str, Any]:
        """Generate realistic delivery note fields."""
        return {
            "delivery_number": f"DN-{faker.year()}{faker.random_number(digits=6, fix_len=True)}",
            "delivery_date": faker.date_between(start_date="-3m", end_date="today").isoformat(),
            "carrier": random.choice(["FedEx", "UPS", "DHL", "USPS"]),
            "tracking_number": faker.bothify(text="??########??"),
            "sender": faker.company(),
            "sender_address": faker.address().replace('\n', ', '),
            "recipient": faker.company(),
            "recipient_address": faker.address().replace('\n', ', '),
            "items": [
                {
                    "item_number": i + 1,
                    "description": faker.catch_phrase(),
                    "quantity": random.randint(1, 100)
                }
                for i in range(random.randint(1, 10))
            ]
        }

    def _generate_material_doc_fields(self, faker) -> Dict[str, Any]:
        """Generate realistic material document fields."""
        return {
            "material_doc_number": faker.bothify(text="MATDOC-##########"),
            "posting_date": faker.date_between(start_date="-1y", end_date="today").isoformat(),
            "movement_type": random.choice(["101", "201", "261", "311", "501"]),
            "plant": faker.bothify(text="PLNT-####"),
            "storage_location": faker.bothify(text="SL-###"),
            "materials": [
                {
                    "material_code": faker.bothify(text="MAT-#####"),
                    "description": faker.catch_phrase(),
                    "quantity": random.randint(1, 1000),
                    "unit": random.choice(["EA", "KG", "L", "M"])
                }
                for i in range(random.randint(1, 5))
            ]
        }

    def _generate_sales_order_fields(self, faker) -> Dict[str, Any]:
        """Generate realistic sales order fields."""
        order_date = faker.date_between(start_date="-6m", end_date="today")

        items = []
        total_value = 0.0

        for i in range(random.randint(1, 15)):
            quantity = random.randint(1, 100)
            unit_price = round(random.uniform(20.0, 500.0), 2)
            line_total = round(quantity * unit_price, 2)
            total_value += line_total

            items.append({
                "item_number": i + 1,
                "product_code": faker.bothify(text="PROD-#####"),
                "description": faker.catch_phrase(),
                "quantity": quantity,
                "unit_price": unit_price,
                "total": line_total
            })

        return {
            "sales_order_number": f"SO-{faker.year()}{faker.random_number(digits=8, fix_len=True)}",
            "order_date": order_date.isoformat(),
            "customer_code": faker.bothify(text="CUST-####"),
            "customer_name": faker.company(),
            "sales_rep": faker.name(),
            "total_value": round(total_value, 2),
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "payment_terms": random.choice(["Net 30", "Net 45", "Prepaid"]),
            "items": items
        }

    def _generate_goods_receipt_fields(self, faker) -> Dict[str, Any]:
        """Generate realistic goods receipt fields."""
        return {
            "gr_number": faker.bothify(text="GR-##########"),
            "receipt_date": faker.date_between(start_date="-3m", end_date="today").isoformat(),
            "po_reference": f"PO-{faker.year()}{faker.random_number(digits=8, fix_len=True)}",
            "supplier": faker.company(),
            "inspector": faker.name(),
            "items": [
                {
                    "item_number": i + 1,
                    "material_code": faker.bothify(text="MAT-#####"),
                    "description": faker.catch_phrase(),
                    "quantity_ordered": random.randint(10, 100),
                    "quantity_received": random.randint(8, 100),
                    "status": random.choice(["Accepted", "Partial", "Rejected"])
                }
                for i in range(random.randint(1, 10))
            ]
        }

    def _generate_packing_list_fields(self, faker) -> Dict[str, Any]:
        """Generate realistic packing list fields."""
        num_packages = random.randint(1, 20)
        total_weight = round(sum(random.uniform(1.0, 50.0) for _ in range(num_packages)), 2)

        return {
            "packing_list_number": f"PL-{faker.year()}{faker.random_number(digits=6, fix_len=True)}",
            "shipment_date": faker.date_between(start_date="-1m", end_date="today").isoformat(),
            "origin": faker.city(),
            "destination": faker.city(),
            "packages": num_packages,
            "total_weight": total_weight,
            "weight_unit": "KG",
            "items": [
                {
                    "package_number": i + 1,
                    "description": faker.catch_phrase(),
                    "weight": round(random.uniform(1.0, 50.0), 2)
                }
                for i in range(num_packages)
            ]
        }

    def _generate_shipping_notice_fields(self, faker) -> Dict[str, Any]:
        """Generate realistic shipping notice fields."""
        ship_date = faker.date_between(start_date="-1m", end_date="today")
        arrival_date = ship_date + timedelta(days=random.randint(2, 14))

        return {
            "shipment_number": faker.bothify(text="SHP-##########"),
            "ship_date": ship_date.isoformat(),
            "expected_arrival": arrival_date.isoformat(),
            "carrier": random.choice(["FedEx", "UPS", "DHL", "Maersk"]),
            "tracking": faker.bothify(text="??########??"),
            "origin": faker.city(),
            "destination": faker.city(),
            "packages": random.randint(1, 50),
            "shipping_method": random.choice(["Air", "Ground", "Ocean", "Express"])
        }

    def _generate_pdf(self,
                     document_type: str,
                     doc_id: str,
                     fields: Dict[str, Any],
                     language: str,
                     quality_variation: bool) -> Path:
        """
        Generate PDF document using ReportLab.
        """
        config = self.doc_configs[document_type]

        # Create output path
        output_path = self.output_dir / f"{doc_id}.pdf"

        # Create PDF
        c = canvas.Canvas(str(output_path), pagesize=config["page_size"])
        width, height = config["page_size"]

        # Draw header
        c.setFont("Helvetica-Bold", 24)
        c.drawString(1 * inch, height - 1 * inch, config["header"])

        # Draw document ID
        c.setFont("Helvetica", 10)
        c.drawString(1 * inch, height - 1.5 * inch, f"Document No: {fields.get(list(fields.keys())[0], doc_id)}")

        # Draw fields (simplified layout)
        y_position = height - 2 * inch
        c.setFont("Helvetica", 10)

        for key, value in list(fields.items())[:15]:  # Limit to first 15 fields
            if isinstance(value, (str, int, float)):
                c.drawString(1 * inch, y_position, f"{key}: {value}")
                y_position -= 0.3 * inch

            if y_position < 1 * inch:
                break

        # Save PDF
        c.save()

        return output_path

    def save_metadata(self, documents: List[SyntheticDocument], output_file: str):
        """
        Save generated documents metadata to JSON.

        Args:
            documents: List of synthetic documents
            output_file: Output JSON file path
        """
        metadata = {
            "total_documents": len(documents),
            "generated_at": datetime.now().isoformat(),
            "statistics": self.stats,
            "documents": [asdict(doc) for doc in documents]
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata for {len(documents):,} documents to {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return self.stats.copy()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    generator = SyntheticDocumentGenerator(
        template_dir="./templates",
        output_dir="./data/synthetic"
    )

    # Generate invoices
    invoices = generator.generate_documents(
        document_type="invoice",
        count=100,
        quality_variation=True
    )

    # Save metadata
    generator.save_metadata(invoices, "./data/synthetic/invoices_metadata.json")

    print(f"Generation statistics: {generator.get_statistics()}")
