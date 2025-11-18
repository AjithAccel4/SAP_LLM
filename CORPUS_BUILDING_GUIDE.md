# SAP_LLM Training Corpus Building Guide

Complete guide for building the 1M+ document training corpus for SAP_LLM.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [Data Sources](#data-sources)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Quality Metrics](#quality-metrics)
- [Troubleshooting](#troubleshooting)

## Overview

The SAP_LLM training corpus builder creates a comprehensive dataset for training multimodal document understanding models. Following PLAN_02.md Phase 2 specifications, it collects, annotates, and prepares 1M+ documents across 13 SAP document types.

### Target Composition

- **500K Synthetic Documents** - Generated from templates with realistic SAP data
- **200K Public Dataset Documents** - RVL-CDIP, FUNSD, CORD, SROIE
- **300K SAP Documents** - From production systems (when available)
- **400+ SAP API Schemas** - For knowledge base enrichment
- **100B+ Tokens** - For LLM training
- **Cohen's Kappa > 0.92** - Annotation quality target

### Document Types Supported

1. Invoice (Supplier/Customer)
2. Purchase Order
3. Sales Order
4. Delivery Note
5. Goods Receipt
6. Material Document
7. Packing List
8. Shipping Notice
9. Advanced Shipping Notice (ASN)
10. Receipt
11. Expense Report
12. Form (Generic)
13. Letter/Email

## Quick Start

### Prerequisites

```bash
# Install Python 3.10+
python --version  # Should be 3.10 or higher

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Full corpus build (1M+ documents, ~7 days processing time)
python build_training_corpus.py --output-dir ./data/training_corpus

# Quick test (reduced counts, ~30 minutes)
python build_training_corpus.py \
    --output-dir ./data/test \
    --quick-test

# Development/testing with sample limit
python build_training_corpus.py \
    --output-dir ./data/sample \
    --sample-limit 1000
```

### Expected Output

```
data/training_corpus/
├── raw/                          # Raw collected documents
│   ├── public/                   # Public datasets
│   ├── synthetic/                # Generated documents
│   └── metadata.json
├── annotations/                  # Field-level annotations
│   └── *.json
├── sap_knowledge_base/           # SAP API schemas
│   ├── schemas/
│   │   └── *.json
│   ├── document_schemas.json
│   ├── field_index.json
│   └── field_embeddings.json
├── processed/                    # Processed & split datasets
│   ├── train/
│   ├── val/
│   └── test/
├── huggingface/                  # HF Datasets format
│   ├── dataset_info.json
│   ├── train.arrow
│   ├── validation.arrow
│   └── test.arrow
├── statistics/                   # Quality metrics
│   └── quality_report.json
├── dataset_splits.json           # Split manifest
├── CORPUS_BUILD_REPORT.md        # Build summary
└── corpus_build.log              # Detailed logs
```

## Pipeline Components

### 1. Document Collection

**Module:** `sap_llm.data_pipeline.collector`

Collects documents from multiple sources:
- Local filesystem
- S3/MinIO object storage
- SAP systems via APIs
- Public datasets

**Individual Tools:**

```bash
# Download public datasets only
python -m sap_llm.data_pipeline.public_datasets_downloader \
    --output-dir ./data/public \
    --dataset rvl-cdip  # or funsd, cord, sroie

# Download all public datasets
python -m sap_llm.data_pipeline.public_datasets_downloader \
    --output-dir ./data/public
```

### 2. SAP API Schema Extraction

**Module:** `sap_llm.data_pipeline.sap_api_scraper`

Scrapes SAP Business Accelerator Hub for API schemas and field definitions.

```bash
python -m sap_llm.data_pipeline.sap_api_scraper \
    --output-dir ./data/sap_schemas \
    --max-apis 400 \
    --api-key YOUR_SAP_API_KEY  # Optional
```

**Extracts:**
- OData entity definitions
- Field types and validations
- API relationships
- Business object metadata

### 3. Synthetic Document Generation

**Module:** `sap_llm.data_pipeline.synthetic_generator`

Generates realistic SAP documents using templates and Faker library.

```bash
# Generate invoices
python -m sap_llm.data_pipeline.synthetic_generator \
    --document-type invoice \
    --count 10000 \
    --output-dir ./data/synthetic

# Supported types:
# invoice, purchase_order, delivery_note, sales_order,
# goods_receipt, material_document, packing_list, shipping_notice
```

**Features:**
- Multi-language support (EN, DE, ES, FR, ZH, JA)
- Realistic field data (Faker)
- PDF/PNG output
- Quality variation (simulated scans)
- Line item generation

### 4. SAP Knowledge Base Building

**Module:** `sap_llm.data_pipeline.knowledge_base_builder`

Constructs comprehensive knowledge base from SAP API schemas.

```bash
python -m sap_llm.data_pipeline.knowledge_base_builder \
    --output-dir ./data/knowledge_base \
    --api-schemas-dir ./data/sap_schemas \
    --no-embeddings  # Disable for faster builds
```

**Outputs:**
- `document_schemas.json` - 13 document type schemas
- `field_index.json` - 200+ business fields
- `field_embeddings.json` - Semantic embeddings (optional)
- `knowledge_base_metadata.json` - Statistics

### 5. Data Annotation

**Module:** `sap_llm.data_pipeline.annotator`

Annotates documents with field-level labels.

**Annotation Methods:**
- Automated (OCR + heuristics)
- Manual (Label Studio integration)
- Active learning loop

**Quality Control:**
- Inter-annotator agreement (Cohen's kappa)
- Triple-check verification
- Confidence scoring

### 6. Data Augmentation

**Module:** `sap_llm.data_pipeline.data_augmentation`

Applies realistic augmentations for model robustness.

```bash
python -m sap_llm.data_pipeline.data_augmentation \
    --input-dir ./data/synthetic \
    --output-dir ./data/augmented \
    --augmentations-per-image 2
```

**Augmentation Types:**

| Type | Description | Probability |
|------|-------------|-------------|
| Rotation | ±2° random rotation | 70% |
| Brightness | ±30% brightness adjustment | 60% |
| Noise | Gaussian noise (σ=5) | 50% |
| JPEG Compression | Quality 75-95 | 40% |
| Photocopy Simulation | Increased contrast, speckles | 10% |
| Fax Simulation | Low resolution, scan lines | 5% |
| Watermarks | Semi-transparent stamps | 15% |

### 7. Dataset Validation

**Module:** `sap_llm.data_pipeline.dataset_validator`

Validates corpus meets production requirements.

```bash
python -m sap_llm.data_pipeline.dataset_validator \
    --data-dir ./data/training_corpus \
    --min-documents 1000000 \
    --min-quality 0.8
```

**Validation Checks:**
- ✅ Document count (≥1M)
- ✅ Quality scores (avg ≥0.8)
- ✅ Document type coverage (all 13 types)
- ✅ Token count (≥100B)
- ✅ Data split ratios (70/15/15)
- ✅ Annotation completeness
- ✅ Field coverage (≥200 fields)
- ✅ Format compliance (HF Datasets)

## Data Sources

### Public Datasets

| Dataset | Documents | Description | License |
|---------|-----------|-------------|---------|
| **RVL-CDIP** | 400K | 16 document categories | Academic |
| **FUNSD** | 199 | Form understanding | MIT |
| **CORD** | 11K | Receipt parsing | MIT |
| **SROIE** | 1K | Receipt OCR | Competition |

**Download:** Automatically via Hugging Face Datasets library

### SAP API Schemas

**Source:** SAP Business Accelerator Hub (api.sap.com)

**Coverage:**
- S/4HANA Cloud APIs (100+)
- SAP Ariba Procurement (50+)
- SAP Concur Expense (30+)
- SAP EWM Logistics (40+)
- SAP MDG Master Data (20+)

**Authentication:** Optional API key for increased rate limits

### Synthetic Documents

**Generation:** Template-based with Faker library

**Languages:** EN, DE, ES, FR, ZH, JA

**Quality Levels:**
- High (1.0) - Perfect digital documents
- Medium (0.85) - Minor artifacts
- Low (0.70) - Scanned/photocopied appearance

## Usage

### Full Production Build

```bash
#!/bin/bash
# Full production corpus build

python build_training_corpus.py \
    --output-dir /data/sap_llm/training_corpus \
    --verbose \
    2>&1 | tee corpus_build.log

# Estimated time: 5-7 days
# Estimated storage: 500GB+
```

### Development Build

```bash
# Quick development build for testing
python build_training_corpus.py \
    --output-dir ./data/dev_corpus \
    --sample-limit 1000 \
    --verbose

# Time: ~30 minutes
# Storage: ~5GB
```

### CI/CD Integration

```yaml
# .github/workflows/corpus_build.yml
name: Build Training Corpus

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 10080  # 7 days

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Build corpus
        run: |
          python build_training_corpus.py \
            --output-dir ./data/corpus \
            --sample-limit 5000

      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: training-corpus
          path: ./data/corpus
```

## Output Structure

### Dataset Splits

**Stratified by:**
- Document type (balanced)
- Source (proportional)
- Quality (20% low-quality)
- Complexity (30% multi-page)

**Ratios:**
- Train: 70% (700K documents)
- Validation: 15% (150K documents)
- Test: 15% (150K documents)

### Hugging Face Format

```python
from datasets import load_from_disk

# Load dataset
dataset = load_from_disk('./data/training_corpus/huggingface')

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Example item
item = train_data[0]
# {
#   'id': 'synthetic_invoice_00001234',
#   'document_type': 'invoice',
#   'image': <PIL.Image>,
#   'text': 'extracted text...',
#   'labels': {
#     'invoice_number': 'INV-2024-001',
#     'total_amount': 1250.00,
#     ...
#   },
#   'metadata': {...}
# }
```

## Quality Metrics

### Target Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| Total Documents | ≥1,000,000 | CRITICAL |
| Total Tokens | ≥100B | HIGH |
| Cohen's Kappa | ≥0.92 | CRITICAL |
| Avg Quality Score | ≥0.80 | HIGH |
| Document Types | 13 types | CRITICAL |
| Field Coverage | ≥200 fields | MEDIUM |
| SAP API Coverage | ≥400 schemas | MEDIUM |

### Annotation Quality

**Cohen's Kappa Calculation:**
```python
from sklearn.metrics import cohen_kappa_score

# Inter-annotator agreement
kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)

# Target: kappa > 0.92 (almost perfect agreement)
# 0.81-1.00: Almost perfect
# 0.61-0.80: Substantial
# 0.41-0.60: Moderate
```

**Quality Control Process:**
1. Initial annotation (BPO team)
2. 10% random sample review
3. Triple-check verification (senior annotators)
4. Cohen's kappa computation
5. Conflict resolution (third annotator)

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Error:** `MemoryError: Unable to allocate...`

**Solution:**
```bash
# Reduce sample limit
python build_training_corpus.py \
    --output-dir ./data/corpus \
    --sample-limit 500

# Or use Spark for distributed processing
# (requires cluster setup)
```

#### 2. Public Dataset Download Fails

**Error:** `ConnectionError: Failed to download dataset`

**Solution:**
```bash
# Manual download and extraction
wget https://huggingface.co/datasets/...
unzip dataset.zip -d ./data/public/

# Then point corpus builder to local data
```

#### 3. Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'reportlab'`

**Solution:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or install missing package
pip install reportlab
```

#### 4. SAP API Rate Limiting

**Error:** `HTTP 429: Too Many Requests`

**Solution:**
```bash
# Use API key for higher limits
export SAP_API_HUB_KEY="your-api-key"

# Or reduce concurrent requests
# (edit sap_api_scraper.py max_workers)
```

### Performance Optimization

```bash
# Use multiple CPU cores for synthetic generation
export OMP_NUM_THREADS=16

# Enable GPU for image processing (if available)
pip install opencv-python-headless

# Use SSD for temporary files
export TMPDIR=/path/to/ssd/tmp
```

### Logging

```bash
# Enable debug logging
python build_training_corpus.py \
    --output-dir ./data/corpus \
    --verbose \
    2>&1 | tee -a corpus_build_debug.log

# Check log file
tail -f ./data/corpus/corpus_build.log
```

## Next Steps

After corpus building completes:

1. **Review Quality Report**
   ```bash
   cat ./data/training_corpus/CORPUS_BUILD_REPORT.md
   ```

2. **Validate Dataset**
   ```bash
   python -m sap_llm.data_pipeline.dataset_validator \
       --data-dir ./data/training_corpus
   ```

3. **Train Model**
   ```bash
   python train_sap_llm.py \
       --data-dir ./data/training_corpus/huggingface \
       --output-dir ./models/sap_llm_v1
   ```

4. **Monitor & Iterate**
   - Track model performance
   - Collect production feedback
   - Continuously update corpus
   - Retrain with improved data

## Support

For issues or questions:
- Check this guide first
- Review PLAN_02.md Phase 2 specifications
- Check logs in `corpus_build.log`
- Open GitHub issue with:
  - Error message
  - Command used
  - System info (OS, Python version)
  - Log excerpt

## References

- [PLAN_02.md](./PLAN_02.md) - Full implementation plan
- [SAP API Hub](https://api.sap.com) - API documentation
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) - Dataset format
- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387) - Model architecture

---

**Last Updated:** 2024-01-15
**Version:** 1.0.0
**Author:** SAP_LLM Team
