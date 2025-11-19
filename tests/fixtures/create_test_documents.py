"""
Create test documents with ground truth for integration tests.

Generates realistic invoice/PO images with known field values for accuracy testing.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta
import random

from PIL import Image, ImageDraw, ImageFont


class TestDocumentGenerator:
    """Generate test documents with ground truth labels."""

    def __init__(self, output_dir: str = "tests/fixtures"):
        """
        Initialize the test document generator.

        Args:
            output_dir: Directory to save generated documents
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track generated documents
        self.documents: List[Dict[str, Any]] = []

    def generate_supplier_invoice(
        self,
        invoice_id: str,
        vendor_id: str = "VENDOR-123",
        total_amount: float = 1000.00,
    ) -> Dict[str, Any]:
        """
        Generate a supplier invoice image with ground truth.

        Args:
            invoice_id: Invoice number
            vendor_id: Vendor identifier
            total_amount: Total invoice amount

        Returns:
            Dictionary with image path and ground truth
        """
        # Create image
        img = Image.new('RGB', (850, 1100), color='white')
        draw = ImageDraw.Draw(img)

        # Try to load a font, fall back to default if not available
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            normal_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            normal_font = ImageFont.load_default()

        # Header
        draw.text((50, 30), "SUPPLIER INVOICE", fill='black', font=title_font)
        draw.line([(50, 65), (800, 65)], fill='black', width=2)

        # Invoice details
        y = 100
        line_height = 30

        invoice_date = (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
        due_date = (datetime.now() + timedelta(days=random.randint(15, 45))).strftime("%Y-%m-%d")

        fields = [
            ("Invoice Number:", invoice_id),
            ("Invoice Date:", invoice_date),
            ("Due Date:", due_date),
            ("Vendor ID:", vendor_id),
            ("Vendor Name:", "Acme Corporation"),
            ("Vendor Address:", "123 Business Street, Suite 100"),
            ("City, State ZIP:", "San Francisco, CA 94105"),
            ("Tax ID:", "XX-XXXXXXX"),
        ]

        for label, value in fields:
            draw.text((50, y), label, fill='black', font=normal_font)
            draw.text((250, y), str(value), fill='black', font=normal_font)
            y += line_height

        # Line items
        y += 30
        draw.text((50, y), "LINE ITEMS", fill='black', font=title_font)
        y += 40

        # Table header
        draw.rectangle([(50, y), (800, y + 25)], fill='lightgray')
        draw.text((60, y + 5), "Description", fill='black', font=normal_font)
        draw.text((400, y + 5), "Quantity", fill='black', font=normal_font)
        draw.text((500, y + 5), "Unit Price", fill='black', font=normal_font)
        draw.text((650, y + 5), "Amount", fill='black', font=normal_font)
        y += 30

        # Line items
        net_amount = total_amount / 1.1  # Assuming 10% tax
        line_items = [
            ("Professional Services", 10, net_amount / 10),
        ]

        for desc, qty, unit_price in line_items:
            amount = qty * unit_price
            draw.text((60, y), desc, fill='black', font=normal_font)
            draw.text((400, y), str(qty), fill='black', font=normal_font)
            draw.text((500, y), f"${unit_price:.2f}", fill='black', font=normal_font)
            draw.text((650, y), f"${amount:.2f}", fill='black', font=normal_font)
            y += 25

        # Totals
        y += 30
        tax_amount = total_amount - net_amount

        draw.text((550, y), "Subtotal:", fill='black', font=normal_font)
        draw.text((650, y), f"${net_amount:.2f}", fill='black', font=normal_font)
        y += 25

        draw.text((550, y), "Tax (10%):", fill='black', font=normal_font)
        draw.text((650, y), f"${tax_amount:.2f}", fill='black', font=normal_font)
        y += 25

        draw.rectangle([(550, y), (800, y + 30)], outline='black', width=2)
        draw.text((560, y + 5), "TOTAL:", fill='black', font=title_font)
        draw.text((650, y + 5), f"${total_amount:.2f}", fill='black', font=title_font)

        # Payment terms
        y += 60
        draw.text((50, y), "Payment Terms: Net 30", fill='black', font=normal_font)
        y += 25
        draw.text((50, y), "Please remit payment to: Bank of America, Account #XXXXXX", fill='black', font=normal_font)

        # Save image
        image_path = self.output_dir / f"invoice_{invoice_id.replace('/', '_')}.png"
        img.save(image_path)

        # Ground truth
        ground_truth = {
            "doc_type": "SUPPLIER_INVOICE",
            "subtype": "STANDARD",
            "fields": {
                "invoice_number": invoice_id,
                "invoice_date": invoice_date,
                "due_date": due_date,
                "vendor_id": vendor_id,
                "vendor_name": "Acme Corporation",
                "vendor_address": "123 Business Street, Suite 100",
                "city_state_zip": "San Francisco, CA 94105",
                "tax_id": "XX-XXXXXXX",
                "subtotal": round(net_amount, 2),
                "tax_amount": round(tax_amount, 2),
                "total_amount": total_amount,
                "currency": "USD",
                "payment_terms": "Net 30",
            },
        }

        document_info = {
            "image_path": str(image_path),
            "ground_truth": ground_truth,
            "doc_id": invoice_id,
        }

        self.documents.append(document_info)

        return document_info

    def generate_purchase_order(
        self,
        po_id: str,
        vendor_id: str = "VENDOR-456",
        total_amount: float = 2500.00,
    ) -> Dict[str, Any]:
        """
        Generate a purchase order image with ground truth.

        Args:
            po_id: Purchase order number
            vendor_id: Vendor identifier
            total_amount: Total PO amount

        Returns:
            Dictionary with image path and ground truth
        """
        # Create image
        img = Image.new('RGB', (850, 1100), color='white')
        draw = ImageDraw.Draw(img)

        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            normal_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except:
            title_font = ImageFont.load_default()
            normal_font = ImageFont.load_default()

        # Header
        draw.text((50, 30), "PURCHASE ORDER", fill='black', font=title_font)
        draw.line([(50, 65), (800, 65)], fill='black', width=2)

        # PO details
        y = 100
        line_height = 30

        po_date = datetime.now().strftime("%Y-%m-%d")
        delivery_date = (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d")

        fields = [
            ("PO Number:", po_id),
            ("PO Date:", po_date),
            ("Delivery Date:", delivery_date),
            ("Vendor ID:", vendor_id),
            ("Vendor Name:", "Global Supplies Inc."),
            ("Ship To:", "Our Company, 456 Main St"),
            ("Bill To:", "Our Company, Accounts Payable"),
            ("Buyer:", "John Doe"),
        ]

        for label, value in fields:
            draw.text((50, y), label, fill='black', font=normal_font)
            draw.text((250, y), str(value), fill='black', font=normal_font)
            y += line_height

        # Line items
        y += 30
        draw.text((50, y), "ITEMS ORDERED", fill='black', font=title_font)
        y += 40

        # Table header
        draw.rectangle([(50, y), (800, y + 25)], fill='lightgray')
        draw.text((60, y + 5), "Item", fill='black', font=normal_font)
        draw.text((400, y + 5), "Qty", fill='black', font=normal_font)
        draw.text((500, y + 5), "Unit Price", fill='black', font=normal_font)
        draw.text((650, y + 5), "Amount", fill='black', font=normal_font)
        y += 30

        # Line items
        line_items = [
            ("Office Supplies", 50, 20.00),
            ("Computer Equipment", 5, 300.00),
        ]

        for desc, qty, unit_price in line_items:
            amount = qty * unit_price
            draw.text((60, y), desc, fill='black', font=normal_font)
            draw.text((400, y), str(qty), fill='black', font=normal_font)
            draw.text((500, y), f"${unit_price:.2f}", fill='black', font=normal_font)
            draw.text((650, y), f"${amount:.2f}", fill='black', font=normal_font)
            y += 25

        # Total
        y += 30
        draw.rectangle([(550, y), (800, y + 30)], outline='black', width=2)
        draw.text((560, y + 5), "TOTAL:", fill='black', font=title_font)
        draw.text((650, y + 5), f"${total_amount:.2f}", fill='black', font=title_font)

        # Save image
        image_path = self.output_dir / f"po_{po_id.replace('/', '_')}.png"
        img.save(image_path)

        # Ground truth
        ground_truth = {
            "doc_type": "PURCHASE_ORDER",
            "subtype": "STANDARD",
            "fields": {
                "po_number": po_id,
                "po_date": po_date,
                "delivery_date": delivery_date,
                "vendor_id": vendor_id,
                "vendor_name": "Global Supplies Inc.",
                "ship_to": "Our Company, 456 Main St",
                "bill_to": "Our Company, Accounts Payable",
                "buyer": "John Doe",
                "total_amount": total_amount,
                "currency": "USD",
            },
        }

        document_info = {
            "image_path": str(image_path),
            "ground_truth": ground_truth,
            "doc_id": po_id,
        }

        self.documents.append(document_info)

        return document_info

    def generate_test_dataset(
        self,
        num_invoices: int = 50,
        num_pos: int = 50,
    ) -> str:
        """
        Generate complete test dataset.

        Args:
            num_invoices: Number of supplier invoices to generate
            num_pos: Number of purchase orders to generate

        Returns:
            Path to dataset manifest JSON
        """
        print(f"Generating test dataset: {num_invoices} invoices + {num_pos} POs")

        # Generate invoices
        for i in range(num_invoices):
            invoice_id = f"INV-2025-{i+1:04d}"
            vendor_id = f"VENDOR-{random.randint(100, 999)}"
            total_amount = round(random.uniform(100, 10000), 2)

            self.generate_supplier_invoice(invoice_id, vendor_id, total_amount)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{num_invoices} invoices")

        # Generate POs
        for i in range(num_pos):
            po_id = f"PO-2025-{i+1:04d}"
            vendor_id = f"VENDOR-{random.randint(100, 999)}"
            total_amount = round(random.uniform(500, 50000), 2)

            self.generate_purchase_order(po_id, vendor_id, total_amount)

            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{num_pos} POs")

        # Save manifest
        manifest_path = self.output_dir / "test_dataset_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump({
                "total_documents": len(self.documents),
                "num_invoices": num_invoices,
                "num_pos": num_pos,
                "documents": self.documents,
                "created_at": datetime.now().isoformat(),
            }, f, indent=2)

        print(f"\nDataset manifest saved to: {manifest_path}")
        print(f"Total documents: {len(self.documents)}")

        return str(manifest_path)


def main():
    """Generate test dataset."""
    import sys

    generator = TestDocumentGenerator()

    # Check command line arguments for dataset size
    if len(sys.argv) > 1 and sys.argv[1] == '--enterprise':
        # Generate enterprise-level test set (100 documents)
        print("Generating enterprise-level test dataset (100 documents)...")
        generator.generate_test_dataset(num_invoices=50, num_pos=50)
    else:
        # Generate small test set for quick tests
        print("Generating small test dataset (10 documents)...")
        print("Use --enterprise flag to generate 100 documents for production testing")
        generator.generate_test_dataset(num_invoices=5, num_pos=5)

    print("\n✅ Test dataset generated successfully!")


if __name__ == "__main__":
    main()
