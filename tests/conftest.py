"""
Pytest fixtures and configuration for SAP_LLM tests.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from PIL import Image
import numpy as np

from sap_llm.config import Config, load_config


@pytest.fixture(scope="session")
def test_config() -> Config:
    """Load test configuration."""
    # Use test config if available, otherwise use default
    config_path = Path(__file__).parent / "test_config.yaml"
    if not config_path.exists():
        config = load_config()
    else:
        config = load_config(str(config_path))

    return config


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image() -> Image.Image:
    """Create sample image for testing."""
    # Create a simple test image
    img_array = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_document_image() -> Image.Image:
    """Create sample document image with text-like patterns."""
    # Create a white image
    img_array = np.ones((1200, 800, 3), dtype=np.uint8) * 255

    # Add some black rectangles to simulate text
    for i in range(10):
        y = 100 + i * 100
        for j in range(5):
            x = 50 + j * 150
            img_array[y:y+20, x:x+100] = 0

    return Image.fromarray(img_array)


@pytest.fixture
def sample_ocr_text() -> str:
    """Sample OCR text."""
    return """
    PURCHASE ORDER

    PO Number: 4500123456
    Date: 2024-01-15
    Vendor: ACME Corp
    Vendor ID: 100001

    Items:
    1. Widget A - Qty: 100 - Price: $10.00 - Total: $1,000.00
    2. Widget B - Qty: 50 - Price: $20.00 - Total: $1,000.00

    Subtotal: $2,000.00
    Tax (10%): $200.00
    Total: $2,200.00
    """


@pytest.fixture
def sample_adc() -> Dict[str, Any]:
    """Sample ADC (Aggregated Document Content)."""
    return {
        "document_type": "purchase_order",
        "document_subtype": "standard",
        "po_number": "4500123456",
        "po_date": "2024-01-15",
        "vendor_id": "100001",
        "vendor_name": "ACME Corp",
        "company_code": "1000",
        "currency": "USD",
        "subtotal": 2000.00,
        "tax_amount": 200.00,
        "total_amount": 2200.00,
        "line_items": [
            {
                "line_number": 1,
                "material": "Widget A",
                "quantity": 100,
                "unit_price": 10.00,
                "total": 1000.00,
            },
            {
                "line_number": 2,
                "material": "Widget B",
                "quantity": 50,
                "unit_price": 20.00,
                "total": 1000.00,
            },
        ],
    }


@pytest.fixture
def sample_invoice_adc() -> Dict[str, Any]:
    """Sample invoice ADC."""
    return {
        "document_type": "invoice",
        "document_subtype": "standard",
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-01-20",
        "vendor_id": "100001",
        "vendor_name": "ACME Corp",
        "po_reference": "4500123456",
        "currency": "USD",
        "subtotal": 2200.00,
        "tax_amount": 220.00,
        "total_amount": 2420.00,
        "due_date": "2024-02-20",
    }


@pytest.fixture
def sample_cluster() -> Dict[str, Any]:
    """Sample exception cluster for SHWL testing."""
    return {
        "id": "cluster_001",
        "label": 0,
        "size": 25,
        "category": "VALIDATION_ERROR",
        "severity": "HIGH",
        "exceptions": [
            {
                "id": f"exc_{i}",
                "category": "VALIDATION_ERROR",
                "severity": "HIGH",
                "field": "total_amount",
                "message": "Total amount does not match subtotal + tax",
                "expected": "2200.00",
                "actual": "2000.00",
                "timestamp": "2024-01-15T10:00:00Z",
            }
            for i in range(25)
        ],
    }


@pytest.fixture
def mock_pmg(mocker):
    """Mock Process Memory Graph."""
    mock = mocker.Mock()
    mock.store_transaction.return_value = "tx_001"
    mock.query_similar.return_value = []
    mock.query_exceptions.return_value = []
    return mock


@pytest.fixture
def mock_reasoning_engine(mocker):
    """Mock Reasoning Engine."""
    mock = mocker.Mock()
    mock.generate.return_value = '{"decision": "approve", "confidence": 0.95}'
    return mock


@pytest.fixture
def mock_redis(mocker):
    """Mock Redis client."""
    mock = mocker.Mock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = True
    mock.exists.return_value = False
    return mock


# Skip markers for CI/CD
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "requires_models: mark test as requiring model files"
    )
    config.addinivalue_line(
        "markers", "requires_cosmos: mark test as requiring Cosmos DB"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on environment."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_models = pytest.mark.skip(reason="Model files not available")
    skip_cosmos = pytest.mark.skip(reason="Cosmos DB not available")

    # Check environment
    has_gpu = os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    has_models = Path("models").exists()
    has_cosmos = os.environ.get("COSMOS_ENDPOINT") is not None

    for item in items:
        if "requires_gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)

        if "requires_models" in item.keywords and not has_models:
            item.add_marker(skip_models)

        if "requires_cosmos" in item.keywords and not has_cosmos:
            item.add_marker(skip_cosmos)
