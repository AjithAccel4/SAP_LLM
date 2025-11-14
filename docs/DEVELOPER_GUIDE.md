# SAP_LLM Developer Guide

Comprehensive technical documentation for developers working with or extending SAP_LLM.

## Table of Contents

1. [Introduction](#introduction)
2. [Development Environment Setup](#development-environment-setup)
3. [Code Structure and Organization](#code-structure-and-organization)
4. [Core Components](#core-components)
5. [Adding New Document Types](#adding-new-document-types)
6. [Extending Pipeline Stages](#extending-pipeline-stages)
7. [Working with Models](#working-with-models)
8. [Testing Guide](#testing-guide)
9. [Debugging Tips](#debugging-tips)
10. [Performance Optimization](#performance-optimization)
11. [API Development](#api-development)
12. [Best Practices](#best-practices)

---

## Introduction

This guide is for developers who want to:
- Contribute to SAP_LLM development
- Extend the system with custom functionality
- Integrate SAP_LLM into their applications
- Understand the codebase architecture

### Prerequisites

- Strong Python knowledge (3.10+)
- Understanding of machine learning concepts
- Familiarity with PyTorch and Transformers
- Experience with async programming (asyncio)
- Basic knowledge of REST APIs and FastAPI

---

## Development Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/qorsync/sap-llm.git
cd sap-llm
```

### 2. Set Up Python Environment

We recommend using `pyenv` for Python version management:

```bash
# Install Python 3.10 or higher
pyenv install 3.10.12
pyenv local 3.10.12

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### 4. Configure Environment Variables

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

MONGODB_URI=mongodb://localhost:27017/sap_llm
COSMOS_ENDPOINT=your-cosmos-endpoint
COSMOS_KEY=your-cosmos-key

# Model Configuration
MODEL_PATH=/path/to/models
VISION_MODEL=microsoft/layoutlmv3-base
LANGUAGE_MODEL=meta-llama/Llama-2-7b-hf
REASONING_MODEL=mistralai/Mixtral-8x7B-v0.1

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_DEVICE=cuda
```

### 5. Set Up Local Services

Using Docker Compose:

```bash
# Start required services
docker-compose up -d redis mongodb

# Verify services are running
docker-compose ps
```

### 6. Download Models

```bash
# Download pre-trained models
python scripts/download_models.py

# Or manually specify models
python scripts/download_models.py --vision microsoft/layoutlmv3-base \
  --language meta-llama/Llama-2-7b-hf \
  --reasoning mistralai/Mixtral-8x7B-v0.1
```

### 7. Initialize Database

```bash
# Initialize databases with schemas
python scripts/init_databases.py

# Build knowledge base
python scripts/build_knowledge_base.py
```

### 8. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sap_llm --cov-report=html

# Open coverage report
open htmlcov/index.html  # Mac
# or
xdg-open htmlcov/index.html  # Linux
```

### 9. Start Development Server

```bash
# Start FastAPI server with auto-reload
uvicorn sap_llm.api.server:app --reload --host 0.0.0.0 --port 8000

# Or use the convenience script
python -m sap_llm.api.server
```

### 10. Verify Installation

```bash
# Check health endpoint
curl http://localhost:8000/health

# Run verification script
python scripts/health_check.py
```

---

## Code Structure and Organization

### Project Layout

```
sap_llm/
├── sap_llm/                      # Main package
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration management
│   │
│   ├── models/                   # AI Models
│   │   ├── __init__.py
│   │   ├── vision_encoder.py    # Vision model (LayoutLMv3)
│   │   ├── language_decoder.py  # Language model (LLaMA-2)
│   │   ├── reasoning_engine.py  # Reasoning model (Mixtral)
│   │   └── unified_model.py     # Unified model orchestration
│   │
│   ├── stages/                   # 8 Pipeline Stages
│   │   ├── __init__.py
│   │   ├── base_stage.py        # Base stage interface
│   │   ├── inbox.py             # Stage 1: Document intake
│   │   ├── preprocessing.py     # Stage 2: OCR & enhancement
│   │   ├── classification.py    # Stage 3: Document classification
│   │   ├── type_identifier.py   # Stage 4: Subtype identification
│   │   ├── extraction.py        # Stage 5: Field extraction
│   │   ├── quality_check.py     # Stage 6: Quality validation
│   │   ├── validation.py        # Stage 7: Business rules
│   │   └── routing.py           # Stage 8: SAP routing
│   │
│   ├── pmg/                      # Process Memory Graph
│   │   ├── __init__.py
│   │   ├── graph_manager.py     # Graph operations
│   │   ├── query_engine.py      # Graph queries
│   │   └── embeddings.py        # Vector embeddings
│   │
│   ├── apop/                     # APOP Orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Main orchestrator
│   │   ├── cloud_events.py      # CloudEvents protocol
│   │   ├── stage_agents.py      # Autonomous stage agents
│   │   └── workflow.py          # Workflow management
│   │
│   ├── shwl/                     # Self-Healing Workflow Loop
│   │   ├── __init__.py
│   │   ├── healing_loop.py      # Main healing loop
│   │   ├── clusterer.py         # Exception clustering
│   │   ├── config_loader.py     # Configuration management
│   │   └── deployment_manager.py # Deployment automation
│   │
│   ├── knowledge_base/           # SAP Knowledge Base
│   │   ├── __init__.py
│   │   ├── schema_registry.py   # Document schemas
│   │   ├── sap_mappings.py      # SAP field mappings
│   │   └── business_rules.py    # Validation rules
│   │
│   ├── api/                      # REST API
│   │   ├── __init__.py
│   │   ├── server.py            # FastAPI application
│   │   ├── routes/              # API routes
│   │   │   ├── documents.py     # Document endpoints
│   │   │   ├── auth.py          # Authentication
│   │   │   ├── metrics.py       # Metrics endpoints
│   │   │   └── websocket.py     # WebSocket handlers
│   │   ├── middleware/          # Middleware
│   │   │   ├── auth.py          # Auth middleware
│   │   │   ├── rate_limit.py    # Rate limiting
│   │   │   └── logging.py       # Request logging
│   │   └── models/              # Pydantic models
│   │       ├── requests.py      # Request models
│   │       └── responses.py     # Response models
│   │
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── image_utils.py       # Image processing
│   │   ├── text_utils.py        # Text processing
│   │   ├── validation.py        # Validation utilities
│   │   └── cache.py             # Caching utilities
│   │
│   ├── training/                 # Training scripts
│   │   ├── __init__.py
│   │   ├── train.py             # Main training script
│   │   ├── dataset.py           # Dataset loaders
│   │   └── metrics.py           # Training metrics
│   │
│   ├── monitoring/               # Monitoring
│   │   ├── __init__.py
│   │   ├── prometheus.py        # Prometheus metrics
│   │   ├── tracing.py           # OpenTelemetry tracing
│   │   └── logging.py           # Structured logging
│   │
│   ├── security/                 # Security components
│   │   ├── __init__.py
│   │   ├── encryption.py        # Encryption utilities
│   │   ├── pii_detector.py      # PII detection
│   │   └── auth.py              # Authentication
│   │
│   └── advanced/                 # Advanced features
│       ├── __init__.py
│       ├── multilingual.py      # Multi-language support
│       ├── explainable_ai.py    # Explainability
│       ├── federated_learning.py # Federated learning
│       └── online_learning.py   # Online learning
│
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── performance/             # Performance tests
│   └── conftest.py              # Pytest configuration
│
├── configs/                      # Configuration files
│   └── default_config.yaml      # Default configuration
│
├── scripts/                      # Utility scripts
│   ├── download_models.py
│   ├── init_databases.py
│   └── build_knowledge_base.py
│
├── docs/                         # Documentation
├── examples/                     # Example code
├── docker/                       # Docker files
├── k8s/                         # Kubernetes manifests
├── helm/                        # Helm charts
├── .github/                     # GitHub Actions
│
├── pyproject.toml               # Project metadata
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── .env.example                 # Example environment file
└── README.md                    # Project README
```

### Key Design Patterns

#### 1. Stage Pattern
All pipeline stages inherit from `BaseStage`:

```python
from sap_llm.stages.base_stage import BaseStage

class MyCustomStage(BaseStage):
    async def process(self, context: ProcessingContext) -> ProcessingContext:
        # Your processing logic
        return context
```

#### 2. Configuration Pattern
Configuration is centralized in `config.py`:

```python
from sap_llm.config import get_config

config = get_config()
model_path = config.models.vision_encoder_path
```

#### 3. Dependency Injection
Dependencies are injected via constructors:

```python
class ExtractionStage:
    def __init__(
        self,
        model: UnifiedExtractorModel,
        cache: CacheManager,
        pmg: ProcessMemoryGraph
    ):
        self.model = model
        self.cache = cache
        self.pmg = pmg
```

---

## Core Components

### Configuration Management

The configuration system uses Pydantic for validation and supports multiple sources:

```python
# sap_llm/config.py
from pydantic import BaseSettings
from typing import Optional

class ModelConfig(BaseSettings):
    vision_encoder_path: str = "microsoft/layoutlmv3-base"
    language_decoder_path: str = "meta-llama/Llama-2-7b-hf"
    device: str = "cuda"
    batch_size: int = 4

class Config(BaseSettings):
    environment: str = "development"
    models: ModelConfig = ModelConfig()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

# Usage
config = Config()
print(config.models.device)  # "cuda"
```

Environment variables can override config:
```bash
export MODELS__DEVICE=cpu
```

### Model Loading

Models are loaded lazily for efficiency:

```python
# sap_llm/models/unified_model.py
class UnifiedExtractorModel:
    def __init__(self, config: Config):
        self.config = config
        self._vision_encoder = None
        self._language_decoder = None
        self._reasoning_engine = None

    @property
    def vision_encoder(self):
        if self._vision_encoder is None:
            self._vision_encoder = self._load_vision_encoder()
        return self._vision_encoder

    def _load_vision_encoder(self):
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            self.config.models.vision_encoder_path,
            torch_dtype=torch.float16
        )
```

### Processing Context

All stages share a processing context:

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class ProcessingContext:
    """Shared context passed through pipeline stages."""

    # Input
    document_id: str
    file_path: str
    file_bytes: bytes

    # Metadata
    document_type: Optional[str] = None
    language: Optional[str] = None
    confidence: float = 0.0

    # Extracted data
    ocr_text: Optional[str] = None
    words: List[str] = field(default_factory=list)
    boxes: List[List[int]] = field(default_factory=list)
    extracted_fields: Dict[str, Any] = field(default_factory=dict)

    # Validation
    validation_results: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0

    # Routing
    routing_decision: Optional[Dict[str, Any]] = None

    # Timing
    stage_timings: Dict[str, float] = field(default_factory=dict)

    # Errors
    errors: List[str] = field(default_factory=list)
```

---

## Adding New Document Types

### 1. Define Document Schema

Create a schema in `sap_llm/knowledge_base/schemas/`:

```python
# sap_llm/knowledge_base/schemas/purchase_requisition.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class PurchaseRequisitionLineItem(BaseModel):
    line_number: int
    material: str
    description: str
    quantity: float
    unit: str
    price: Optional[float] = None
    delivery_date: Optional[date] = None

class PurchaseRequisitionSchema(BaseModel):
    """Schema for Purchase Requisition documents."""

    # Header
    pr_number: str = Field(..., description="Purchase Requisition Number")
    pr_date: date = Field(..., description="PR Creation Date")
    requestor: str = Field(..., description="Requestor Name")
    department: Optional[str] = None
    cost_center: Optional[str] = None

    # Items
    line_items: List[PurchaseRequisitionLineItem] = Field(default_factory=list)

    # Totals
    total_amount: Optional[float] = None
    currency: str = "USD"

    # Metadata
    priority: Optional[str] = None
    notes: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "pr_number": "PR-2025-00123",
                "pr_date": "2025-11-10",
                "requestor": "John Doe",
                "line_items": [
                    {
                        "line_number": 1,
                        "material": "MAT-001",
                        "description": "Office Supplies",
                        "quantity": 100,
                        "unit": "EA"
                    }
                ]
            }
        }
```

### 2. Register Schema

```python
# sap_llm/knowledge_base/schema_registry.py
from .schemas.purchase_requisition import PurchaseRequisitionSchema

DOCUMENT_SCHEMAS = {
    "invoice": InvoiceSchema,
    "purchase_order": PurchaseOrderSchema,
    "purchase_requisition": PurchaseRequisitionSchema,  # Add new schema
    # ... other schemas
}

def get_schema(document_type: str):
    """Get schema for document type."""
    schema = DOCUMENT_SCHEMAS.get(document_type.lower())
    if schema is None:
        raise ValueError(f"Unknown document type: {document_type}")
    return schema
```

### 3. Add SAP Mapping

```python
# sap_llm/knowledge_base/sap_mappings.py
PURCHASE_REQUISITION_MAPPING = {
    "sap_endpoint": "/sap/opu/odata/sap/API_PURCHASE_REQ",
    "method": "POST",
    "field_mapping": {
        "pr_number": "PurchaseRequisition",
        "pr_date": "PurchaseRequisitionDate",
        "requestor": "CreatedByUser",
        "department": "Department",
        "cost_center": "CostCenter",
        "line_items": {
            "target": "to_PurchaseRequisitionItem",
            "fields": {
                "line_number": "PurchaseRequisitionItem",
                "material": "Material",
                "quantity": "RequestedQuantity",
                "unit": "BaseUnit"
            }
        }
    }
}

SAP_MAPPINGS = {
    "invoice": INVOICE_MAPPING,
    "purchase_order": PURCHASE_ORDER_MAPPING,
    "purchase_requisition": PURCHASE_REQUISITION_MAPPING,
    # ... other mappings
}
```

### 4. Add Validation Rules

```python
# sap_llm/knowledge_base/business_rules.py
def validate_purchase_requisition(data: dict) -> List[str]:
    """Validate purchase requisition business rules."""
    errors = []

    # Required fields
    if not data.get("pr_number"):
        errors.append("PR number is required")

    # Date validation
    pr_date = data.get("pr_date")
    if pr_date and pr_date > date.today():
        errors.append("PR date cannot be in the future")

    # Line items validation
    line_items = data.get("line_items", [])
    if not line_items:
        errors.append("At least one line item is required")

    for item in line_items:
        if item.get("quantity", 0) <= 0:
            errors.append(f"Line {item['line_number']}: Quantity must be > 0")

    return errors
```

### 5. Add Training Examples

Create training data in `data/training/purchase_requisition/`:

```
data/training/purchase_requisition/
├── annotations.json
├── pr_001.pdf
├── pr_002.pdf
└── pr_003.pdf
```

```json
// annotations.json
[
  {
    "file": "pr_001.pdf",
    "document_type": "purchase_requisition",
    "annotations": {
      "pr_number": "PR-2025-00123",
      "pr_date": "2025-11-10",
      "requestor": "John Doe",
      "line_items": [...]
    }
  }
]
```

### 6. Test the New Document Type

```python
# tests/integration/test_purchase_requisition.py
import pytest
from sap_llm import initialize
from sap_llm.models import UnifiedExtractorModel

@pytest.fixture
def model():
    config = initialize("configs/test_config.yaml")
    return UnifiedExtractorModel(config)

def test_purchase_requisition_extraction(model):
    # Load test document
    with open("data/test/pr_sample.pdf", "rb") as f:
        result = model.process_document(
            file_bytes=f.read(),
            document_type="purchase_requisition"
        )

    # Verify extraction
    assert result["document_type"] == "purchase_requisition"
    assert "pr_number" in result["extracted_fields"]
    assert result["confidence"] > 0.85
    assert len(result["extracted_fields"]["line_items"]) > 0
```

---

## Extending Pipeline Stages

### Creating a Custom Stage

```python
# sap_llm/stages/custom_validation.py
from typing import Any, Dict
from sap_llm.stages.base_stage import BaseStage
from sap_llm.models import ProcessingContext

class CustomValidationStage(BaseStage):
    """Custom validation stage for specific business rules."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("custom_validation", config)
        self.strict_mode = config.get("strict_mode", False)

    async def process(self, context: ProcessingContext) -> ProcessingContext:
        """
        Apply custom validation rules.

        Args:
            context: Current processing context

        Returns:
            Updated processing context
        """
        self.logger.info(f"Starting custom validation for {context.document_id}")

        # Extract fields
        fields = context.extracted_fields

        # Validation logic
        errors = []

        # Example: Check total matches sum
        if "line_items" in fields and "total_amount" in fields:
            calculated_total = sum(
                item.get("total", 0) for item in fields["line_items"]
            )
            actual_total = fields["total_amount"]

            tolerance = 0.01 if not self.strict_mode else 0.001

            if abs(calculated_total - actual_total) > tolerance:
                errors.append(
                    f"Total amount mismatch: "
                    f"calculated={calculated_total}, actual={actual_total}"
                )

        # Update context
        context.validation_results["custom_validation"] = {
            "status": "passed" if not errors else "failed",
            "errors": errors
        }

        if errors:
            self.logger.warning(f"Validation errors: {errors}")
            context.errors.extend(errors)

        return context

    async def validate(self, context: ProcessingContext) -> bool:
        """Validate stage prerequisites."""
        # Check if extraction is complete
        if not context.extracted_fields:
            self.logger.error("No extracted fields available")
            return False
        return True
```

### Integrating the Custom Stage

```python
# sap_llm/stages/__init__.py
from .custom_validation import CustomValidationStage

# Update pipeline configuration
# configs/default_config.yaml
stages:
  - name: inbox
    class: InboxStage
    enabled: true

  # ... other stages ...

  - name: custom_validation
    class: CustomValidationStage
    enabled: true
    config:
      strict_mode: false

  - name: routing
    class: RoutingStage
    enabled: true
```

### Testing the Custom Stage

```python
# tests/unit/test_custom_validation.py
import pytest
from sap_llm.stages.custom_validation import CustomValidationStage
from sap_llm.models import ProcessingContext

@pytest.fixture
def stage():
    return CustomValidationStage({"strict_mode": False})

@pytest.fixture
def context():
    return ProcessingContext(
        document_id="test_001",
        file_path="/test/invoice.pdf",
        file_bytes=b"",
        extracted_fields={
            "total_amount": 1000.00,
            "line_items": [
                {"total": 500.00},
                {"total": 499.99}  # Intentional mismatch
            ]
        }
    )

@pytest.mark.asyncio
async def test_validation_detects_mismatch(stage, context):
    result = await stage.process(context)

    assert result.validation_results["custom_validation"]["status"] == "failed"
    assert len(result.validation_results["custom_validation"]["errors"]) > 0
```

---

## Working with Models

### Loading Custom Models

```python
# sap_llm/models/custom_classifier.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class CustomDocumentClassifier:
    """Custom document classifier."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def classify(self, text: str) -> dict:
        """Classify document text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        predicted_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][predicted_class].item()

        return {
            "class_id": predicted_class,
            "class_name": self.model.config.id2label[predicted_class],
            "confidence": confidence,
            "all_probabilities": probs[0].cpu().numpy().tolist()
        }
```

### Model Fine-Tuning

```python
# sap_llm/training/train.py
from transformers import Trainer, TrainingArguments
from sap_llm.training.dataset import DocumentDataset

def train_model(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 3
):
    """Fine-tune a model on custom data."""

    # Load dataset
    train_dataset = DocumentDataset(train_data_path)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        fp16=True,  # Mixed precision training
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(output_dir)

    return trainer
```

---

## Testing Guide

### Unit Tests

```python
# tests/unit/test_extraction_stage.py
import pytest
from unittest.mock import Mock, AsyncMock
from sap_llm.stages.extraction import ExtractionStage
from sap_llm.models import ProcessingContext

@pytest.fixture
def mock_model():
    model = Mock()
    model.extract = AsyncMock(return_value={
        "invoice_number": "INV-123",
        "total": 1000.00
    })
    return model

@pytest.fixture
def extraction_stage(mock_model):
    return ExtractionStage(
        model=mock_model,
        cache=Mock(),
        pmg=Mock()
    )

@pytest.mark.asyncio
async def test_extraction_stage(extraction_stage):
    context = ProcessingContext(
        document_id="test_001",
        file_path="/test/invoice.pdf",
        file_bytes=b"",
        document_type="invoice",
        ocr_text="Invoice #INV-123 Total: $1000.00"
    )

    result = await extraction_stage.process(context)

    assert "invoice_number" in result.extracted_fields
    assert result.extracted_fields["invoice_number"] == "INV-123"
    assert result.extracted_fields["total"] == 1000.00
```

### Integration Tests

```python
# tests/integration/test_full_pipeline.py
import pytest
from sap_llm import initialize
from sap_llm.models import UnifiedExtractorModel

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_invoice():
    """Test complete pipeline with real invoice."""

    # Initialize system
    config = initialize("configs/test_config.yaml")
    model = UnifiedExtractorModel(config)

    # Load test invoice
    with open("tests/data/sample_invoice.pdf", "rb") as f:
        file_bytes = f.read()

    # Process
    result = await model.process_document(
        file_bytes=file_bytes,
        document_type="invoice"
    )

    # Assertions
    assert result["status"] == "completed"
    assert result["document_type"] == "invoice"
    assert result["confidence"] > 0.85

    fields = result["extracted_fields"]
    assert "invoice_number" in fields
    assert "total_amount" in fields
    assert len(fields.get("line_items", [])) > 0
```

### Performance Tests

```python
# tests/performance/test_throughput.py
import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from sap_llm.models import UnifiedExtractorModel

@pytest.mark.performance
def test_throughput(model, sample_documents):
    """Test processing throughput."""

    start_time = time.time()
    num_documents = len(sample_documents)

    # Process in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(
            lambda doc: model.process_document_sync(doc),
            sample_documents
        ))

    end_time = time.time()
    duration = end_time - start_time

    # Calculate metrics
    throughput = num_documents / duration
    avg_latency = duration / num_documents

    print(f"Processed {num_documents} documents in {duration:.2f}s")
    print(f"Throughput: {throughput:.2f} docs/sec")
    print(f"Average latency: {avg_latency*1000:.2f}ms")

    # Assertions
    assert throughput > 50, "Throughput below target (50 docs/sec)"
    assert avg_latency < 0.5, "Average latency above target (500ms)"
```

### Test Coverage

Run tests with coverage:

```bash
# Generate coverage report
pytest --cov=sap_llm --cov-report=html --cov-report=term

# View coverage
open htmlcov/index.html

# Check coverage threshold
pytest --cov=sap_llm --cov-fail-under=80
```

---

## Debugging Tips

### Enable Debug Logging

```python
# Set in .env or code
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or configure specific loggers
logging.getLogger('sap_llm.stages').setLevel(logging.DEBUG)
logging.getLogger('sap_llm.models').setLevel(logging.INFO)
```

### Use IPython Debugger

```python
# Add breakpoint in code
import ipdb; ipdb.set_trace()

# Or use built-in breakpoint() (Python 3.7+)
breakpoint()
```

### Profile Performance

```python
import cProfile
import pstats

# Profile a function
profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = model.process_document(doc)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile script
python -m memory_profiler your_script.py
```

```python
# Add decorator to functions
from memory_profiler import profile

@profile
def process_large_batch(documents):
    # Your code
    pass
```

### Debug API Requests

```bash
# Enable FastAPI debug mode
uvicorn sap_llm.api.server:app --reload --log-level debug

# View request/response with httpx
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/v1/documents/extract",
        files={"file": open("invoice.pdf", "rb")}
    )
    print(response.status_code)
    print(response.json())
```

---

## Performance Optimization

### Batch Processing

```python
# Process documents in batches
async def process_batch(documents, batch_size=16):
    results = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        batch_results = await model.process_batch(batch)
        results.extend(batch_results)
    return results
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_document_schema(document_type: str):
    """Cache document schemas."""
    return load_schema(document_type)
```

### Model Quantization

```python
# Use INT8 quantization
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,  # INT8 quantization
    device_map="auto"
)
```

---

## API Development

### Adding New Endpoints

```python
# sap_llm/api/routes/custom.py
from fastapi import APIRouter, Depends, HTTPException
from sap_llm.api.models.requests import CustomRequest
from sap_llm.api.models.responses import CustomResponse
from sap_llm.api.middleware.auth import get_current_user

router = APIRouter(prefix="/v1/custom", tags=["custom"])

@router.post("/process", response_model=CustomResponse)
async def process_custom(
    request: CustomRequest,
    current_user: dict = Depends(get_current_user)
):
    """Custom processing endpoint."""
    try:
        # Your logic here
        result = await process_custom_logic(request)
        return CustomResponse(
            status="success",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Middleware

```python
# sap_llm/api/middleware/custom.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

class CustomMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Pre-processing
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Post-processing
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        return response
```

---

## Best Practices

### 1. Code Style
- Follow PEP 8
- Use type hints
- Write docstrings (Google style)
- Keep functions small and focused

### 2. Error Handling
```python
# Use specific exceptions
class ExtractionError(Exception):
    """Raised when field extraction fails."""
    pass

# Log errors properly
try:
    result = extract_field(document)
except ExtractionError as e:
    logger.error(f"Extraction failed: {e}", exc_info=True)
    raise
```

### 3. Async Programming
```python
# Use async/await properly
async def process_document(doc):
    # Await async operations
    result = await async_operation()

    # Use asyncio.gather for parallel operations
    results = await asyncio.gather(
        operation1(),
        operation2(),
        operation3()
    )
```

### 4. Resource Management
```python
# Use context managers
async with aiofiles.open('file.txt', 'r') as f:
    content = await f.read()

# Clean up resources
try:
    resource = acquire_resource()
    use_resource(resource)
finally:
    release_resource(resource)
```

### 5. Testing
- Write tests first (TDD)
- Test edge cases
- Use fixtures for common setups
- Mock external dependencies

---

## Next Steps

- Explore [API Documentation](API_DOCUMENTATION.md) for API details
- Read [Architecture](ARCHITECTURE.md) for system design
- Check [Contributing Guidelines](../CONTRIBUTING.md) before submitting PRs
- Join our developer community

---

**Questions?** Contact dev@qorsync.com or open an issue on GitHub.
