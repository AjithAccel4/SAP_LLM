#!/usr/bin/env python3
"""
Test generation: 10K documents to validate production script.
"""

# Same imports as 500K script
import json
import logging
from pathlib import Path
from faker import Faker
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import random
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the generator from the 500K script
import sys
sys.path.insert(0, str(Path(__file__).parent))

# We'll use simplified inline version for testing
@dataclass
class Doc:
    id: str
    doc_type: str
    file_path: str
    fields: dict

def generate_doc(doc_id, doc_type, output_dir, faker):
    """Generate a single test document."""
    fields = {
        "number": f"{doc_type.upper()}-{doc_id:08d}",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "company": faker.company(),
        "amount": f"${random.uniform(100, 10000):.2f}"
    }

    pdf_path = output_dir / f"{doc_type}_{doc_id:08d}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, doc_type.upper())
    c.setFont("Helvetica", 12)
    y = 750
    for k, v in fields.items():
        c.drawString(100, y, f"{k}: {v}")
        y -= 20
    c.save()

    return Doc(
        id=f"doc_{doc_id:08d}",
        doc_type=doc_type,
        file_path=str(pdf_path),
        fields=fields
    )

def main():
    logger.info("="*60)
    logger.info("TEST GENERATION: 10,000 DOCUMENTS")
    logger.info("="*60)

    output_dir = Path("/home/user/SAP_LLM/data/test_10k")
    output_dir.mkdir(parents=True, exist_ok=True)

    faker = Faker()
    documents = []

    # Distribution for 10K
    distribution = {
        "invoice": 3000,
        "purchase_order": 3000,
        "goods_receipt": 2000,
        "sales_order": 1000,
        "delivery_note": 500,
        "material_document": 500
    }

    start_time = datetime.now()
    doc_counter = 0

    for doc_type, count in distribution.items():
        logger.info(f"Generating {count:,} {doc_type} documents...")
        for i in tqdm(range(count), desc=doc_type):
            try:
                doc = generate_doc(doc_counter, doc_type, output_dir, faker)
                documents.append(doc)
                doc_counter += 1
            except Exception as e:
                logger.error(f"Error: {e}")

    # Save metadata
    metadata = {
        "total": len(documents),
        "distribution": distribution,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": (datetime.now() - start_time).total_seconds()
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    duration = datetime.now() - start_time
    logger.info(f"\n✅ Generated {len(documents):,} documents in {duration}")
    logger.info(f"Rate: {len(documents) / duration.total_seconds():.1f} docs/sec")
    logger.info(f"Output: {output_dir}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
