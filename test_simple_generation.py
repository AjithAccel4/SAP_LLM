#!/usr/bin/env python3
"""
Simple test of synthetic document generation without full package imports.
"""

import json
import logging
from pathlib import Path
from faker import Faker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimpleDocument:
    """Simple document metadata."""
    id: str
    document_type: str
    file_path: str
    fields: dict

def generate_simple_invoice(output_dir: Path, doc_id: int) -> SimpleDocument:
    """Generate a simple test invoice."""
    faker = Faker()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate invoice data
    invoice_number = f"INV-{doc_id:06d}"
    invoice_date = datetime.now() - timedelta(days=random.randint(0, 365))

    fields = {
        "invoice_number": invoice_number,
        "invoice_date": invoice_date.strftime("%Y-%m-%d"),
        "vendor_name": faker.company(),
        "vendor_address": faker.address(),
        "customer_name": faker.company(),
        "total_amount": f"${random.uniform(100, 10000):.2f}",
        "tax_amount": f"${random.uniform(10, 1000):.2f}",
    }

    # Create PDF
    pdf_path = output_dir / f"invoice_{doc_id:06d}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)

    # Add content
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "INVOICE")

    c.setFont("Helvetica", 12)
    y = 750
    for key, value in fields.items():
        c.drawString(100, y, f"{key}: {value}")
        y -= 20

    c.save()

    # Create document metadata
    doc = SimpleDocument(
        id=f"doc_{doc_id:06d}",
        document_type="invoice",
        file_path=str(pdf_path),
        fields=fields
    )

    return doc

def main():
    logger.info("="*60)
    logger.info("SIMPLE SYNTHETIC DATA GENERATION TEST")
    logger.info("="*60)

    output_dir = Path("./data/test_simple")

    # Generate 10 test invoices
    logger.info("Generating 10 test invoices...")
    documents = []

    for i in range(1, 11):
        try:
            doc = generate_simple_invoice(output_dir, i)
            documents.append(doc)
            logger.info(f"  ✅ Generated invoice {i}: {doc.fields['invoice_number']}")
        except Exception as e:
            logger.error(f"  ❌ Error generating invoice {i}: {e}")

    # Save metadata
    metadata_file = output_dir / "metadata.json"
    metadata = {
        "total_generated": len(documents),
        "generation_time": datetime.now().isoformat(),
        "documents": [asdict(doc) for doc in documents]
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ Generated {len(documents)} documents successfully!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Metadata file: {metadata_file}")
    logger.info("="*60)

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
