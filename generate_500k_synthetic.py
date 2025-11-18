#!/usr/bin/env python3
"""
Production-scale synthetic document generation: 500K documents.

This script generates a realistic training dataset without requiring
the full SAP_LLM package imports (avoiding PyTorch dependency for now).

Distribution:
- 150K Invoices (30%)
- 150K Purchase Orders (30%)
- 100K Goods Receipts (20%)
- 50K Sales Orders (10%)
- 25K Delivery Notes (5%)
- 25K Material Documents (5%)
"""

import json
import logging
from pathlib import Path
from faker import Faker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, letter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import random
from tqdm import tqdm
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SyntheticDocument:
    """Document metadata."""
    id: str
    document_type: str
    file_path: str
    fields: dict
    language: str = "en"
    quality_score: float = 1.0

class ProductionSyntheticGenerator:
    """Production-scale synthetic document generator."""

    def __init__(self, output_dir: Path, languages=["en", "de", "es", "fr"]):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.languages = languages

        # Initialize Faker for each language
        self.fakers = {}
        locale_map = {
            "en": "en_US",
            "de": "de_DE",
            "es": "es_ES",
            "fr": "fr_FR"
        }
        for lang in languages:
            self.fakers[lang] = Faker(locale_map.get(lang, "en_US"))

        self.stats = {
            "total_generated": 0,
            "by_type": {},
            "by_language": {},
            "errors": 0
        }

    def generate_invoice(self, doc_id: int, lang: str = "en") -> SyntheticDocument:
        """Generate a synthetic invoice."""
        faker = self.fakers[lang]

        invoice_number = f"INV-{doc_id:08d}"
        invoice_date = datetime.now() - timedelta(days=random.randint(0, 1095))
        due_date = invoice_date + timedelta(days=random.randint(15, 90))

        # Generate line items
        num_items = random.randint(1, 10)
        line_items = []
        subtotal = 0.0
        for i in range(num_items):
            qty = random.randint(1, 100)
            price = random.uniform(10, 1000)
            amount = qty * price
            subtotal += amount
            line_items.append({
                "line_number": i + 1,
                "description": faker.catch_phrase(),
                "quantity": qty,
                "unit_price": f"${price:.2f}",
                "amount": f"${amount:.2f}"
            })

        tax_amount = subtotal * random.uniform(0.05, 0.20)
        total_amount = subtotal + tax_amount

        fields = {
            "invoice_number": invoice_number,
            "invoice_date": invoice_date.strftime("%Y-%m-%d"),
            "due_date": due_date.strftime("%Y-%m-%d"),
            "vendor_name": faker.company(),
            "vendor_address": faker.address(),
            "vendor_tax_id": faker.ssn() if lang == "en" else faker.random_number(digits=9),
            "customer_name": faker.company(),
            "customer_address": faker.address(),
            "subtotal": f"${subtotal:.2f}",
            "tax_amount": f"${tax_amount:.2f}",
            "total_amount": f"${total_amount:.2f}",
            "currency": "USD" if lang == "en" else "EUR",
            "line_items": line_items
        }

        # Create PDF
        pdf_path = self.output_dir / f"invoice_{doc_id:08d}.pdf"
        self._create_invoice_pdf(pdf_path, fields, lang)

        return SyntheticDocument(
            id=f"doc_{doc_id:08d}",
            document_type="invoice",
            file_path=str(pdf_path),
            fields=fields,
            language=lang
        )

    def generate_purchase_order(self, doc_id: int, lang: str = "en") -> SyntheticDocument:
        """Generate a synthetic purchase order."""
        faker = self.fakers[lang]

        po_number = f"PO-{doc_id:08d}"
        po_date = datetime.now() - timedelta(days=random.randint(0, 730))
        delivery_date = po_date + timedelta(days=random.randint(7, 60))

        fields = {
            "po_number": po_number,
            "po_date": po_date.strftime("%Y-%m-%d"),
            "delivery_date": delivery_date.strftime("%Y-%m-%d"),
            "vendor_code": f"VEN{random.randint(1000, 9999)}",
            "buyer_name": faker.name(),
            "shipping_address": faker.address(),
            "total_value": f"${random.uniform(1000, 50000):.2f}",
            "payment_terms": random.choice(["Net 30", "Net 60", "Net 90", "COD"]),
            "delivery_method": random.choice(["Standard", "Express", "Overnight"])
        }

        pdf_path = self.output_dir / f"po_{doc_id:08d}.pdf"
        self._create_simple_pdf(pdf_path, "PURCHASE ORDER", fields)

        return SyntheticDocument(
            id=f"doc_{doc_id:08d}",
            document_type="purchase_order",
            file_path=str(pdf_path),
            fields=fields,
            language=lang
        )

    def generate_goods_receipt(self, doc_id: int, lang: str = "en") -> SyntheticDocument:
        """Generate a synthetic goods receipt."""
        faker = self.fakers[lang]

        gr_number = f"GR-{doc_id:08d}"
        receipt_date = datetime.now() - timedelta(days=random.randint(0, 365))

        fields = {
            "gr_number": gr_number,
            "receipt_date": receipt_date.strftime("%Y-%m-%d"),
            "po_reference": f"PO-{random.randint(10000000, 99999999)}",
            "vendor_name": faker.company(),
            "receiver_name": faker.name(),
            "warehouse_location": faker.city(),
            "total_quantity": random.randint(10, 1000),
            "condition": random.choice(["Good", "Damaged", "Partial"])
        }

        pdf_path = self.output_dir / f"gr_{doc_id:08d}.pdf"
        self._create_simple_pdf(pdf_path, "GOODS RECEIPT", fields)

        return SyntheticDocument(
            id=f"doc_{doc_id:08d}",
            document_type="goods_receipt",
            file_path=str(pdf_path),
            fields=fields,
            language=lang
        )

    def _create_invoice_pdf(self, pdf_path: Path, fields: dict, lang: str):
        """Create an invoice PDF."""
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4

        # Header
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, height - 50, "INVOICE" if lang == "en" else "RECHNUNG")

        # Invoice details
        c.setFont("Helvetica", 12)
        y = height - 100
        for key, value in fields.items():
            if key != "line_items":
                text = f"{key.replace('_', ' ').title()}: {value}"
                if len(str(text)) > 80:
                    text = text[:80] + "..."
                c.drawString(50, y, text)
                y -= 20
                if y < 100:
                    break

        c.save()

    def _create_simple_pdf(self, pdf_path: Path, title: str, fields: dict):
        """Create a simple PDF document."""
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4

        # Header
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, height - 50, title)

        # Fields
        c.setFont("Helvetica", 11)
        y = height - 100
        for key, value in fields.items():
            text = f"{key.replace('_', ' ').title()}: {value}"
            if len(str(text)) > 80:
                text = text[:80] + "..."
            c.drawString(50, y, text)
            y -= 18
            if y < 50:
                break

        c.save()

    def generate_batch(self, doc_type: str, count: int, batch_size: int = 1000):
        """Generate a batch of documents of a specific type."""
        logger.info(f"Generating {count:,} {doc_type} documents...")

        documents = []
        errors = 0

        for i in tqdm(range(count), desc=f"Generating {doc_type}"):
            try:
                doc_id = int(f"{hash(doc_type) % 100}{i:08d}")
                lang = random.choice(self.languages)

                if doc_type == "invoice":
                    doc = self.generate_invoice(doc_id, lang)
                elif doc_type == "purchase_order":
                    doc = self.generate_purchase_order(doc_id, lang)
                elif doc_type == "goods_receipt":
                    doc = self.generate_goods_receipt(doc_id, lang)
                else:
                    # Fallback to simple PO
                    doc = self.generate_purchase_order(doc_id, lang)

                documents.append(doc)

                # Save batch metadata every batch_size documents
                if (i + 1) % batch_size == 0:
                    self._save_batch_metadata(doc_type, documents[-batch_size:], i // batch_size)

            except Exception as e:
                logger.error(f"Error generating {doc_type} {i}: {e}")
                errors += 1

        self.stats["total_generated"] += len(documents)
        self.stats["by_type"][doc_type] = len(documents)
        self.stats["errors"] += errors

        return documents

    def _save_batch_metadata(self, doc_type: str, documents: list, batch_num: int):
        """Save metadata for a batch of documents."""
        batch_dir = self.output_dir / "batches"
        batch_dir.mkdir(exist_ok=True)

        batch_file = batch_dir / f"{doc_type}_batch_{batch_num:04d}.json"
        with open(batch_file, 'w') as f:
            json.dump({
                "batch_number": batch_num,
                "document_type": doc_type,
                "count": len(documents),
                "documents": [asdict(doc) for doc in documents]
            }, f, indent=2)

def main():
    logger.info("="*80)
    logger.info("PRODUCTION SYNTHETIC DOCUMENT GENERATION: 500K DOCUMENTS")
    logger.info("="*80)

    output_dir = Path("/home/user/SAP_LLM/data/raw/synthetic")
    generator = ProductionSyntheticGenerator(output_dir)

    start_time = datetime.now()

    # Generate documents according to distribution
    distribution = {
        "invoice": 150_000,
        "purchase_order": 150_000,
        "goods_receipt": 100_000,
        "sales_order": 50_000,
        "delivery_note": 25_000,
        "material_document": 25_000
    }

    # For the first version, we'll generate the first 3 types
    # The others (sales_order, delivery_note, material_document) will use PO template
    all_documents = []

    for doc_type, count in distribution.items():
        docs = generator.generate_batch(doc_type, count)
        all_documents.extend(docs)

    # Save final metadata
    metadata_file = output_dir / "metadata.json"
    metadata = {
        "total_generated": len(all_documents),
        "distribution": distribution,
        "generation_time": datetime.now().isoformat(),
        "duration_seconds": (datetime.now() - start_time).total_seconds(),
        "stats": generator.stats,
        "documents_sample": [asdict(doc) for doc in all_documents[:100]]  # Save first 100 as sample
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    duration = datetime.now() - start_time
    logger.info("\n" + "="*80)
    logger.info("GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Total Documents: {len(all_documents):,}")
    logger.info(f"Duration: {duration}")
    logger.info(f"Rate: {len(all_documents) / duration.total_seconds():.2f} docs/second")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Metadata: {metadata_file}")
    logger.info(f"Errors: {generator.stats['errors']}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
