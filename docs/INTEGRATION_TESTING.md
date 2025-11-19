# Real Model Integration Testing Guide

## Overview

This guide covers the comprehensive integration testing infrastructure for SAP_LLM that uses **real ML models** (no mocks) to validate end-to-end functionality.

**Key Features:**
- ✅ Real model inference with LayoutLMv3, LLaMA-2, and Mixtral
- ✅ Accuracy validation against ground truth dataset
- ✅ Performance benchmarking with latency targets
- ✅ Error handling and robustness testing
- ✅ GPU-enabled CI/CD integration
- ✅ Comprehensive test coverage across all pipeline stages

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Test Structure](#test-structure)
4. [Running Tests](#running-tests)
5. [Test Suites](#test-suites)
6. [Performance Targets](#performance-targets)
7. [Troubleshooting](#troubleshooting)
8. [CI/CD Integration](#cicd-integration)
9. [Advanced Topics](#advanced-topics)

---

## Prerequisites

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA GPU with 16GB VRAM (e.g., Tesla T4)
- CPU: 8 cores
- RAM: 32GB
- Disk: 50GB free space

**Recommended:**
- GPU: NVIDIA A100 (40GB VRAM)
- CPU: 16+ cores
- RAM: 64GB
- Disk: 100GB free space (for models + test data)

### Software Requirements

- **Operating System:** Linux (Ubuntu 20.04+ recommended)
- **CUDA:** 11.8 or 12.1
- **Python:** 3.10+
- **PyTorch:** 2.0+

### Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# HuggingFace libraries
pip install transformers accelerate bitsandbytes

# Testing dependencies
pip install pytest pytest-timeout pytest-asyncio pytest-cov

# Utilities
pip install pillow numpy pyyaml rich
```

### HuggingFace Token

Some models (e.g., LLaMA-2) require authentication:

1. Create account at https://huggingface.co
2. Request access to gated models
3. Generate access token
4. Set environment variable:

```bash
export HF_TOKEN=your_huggingface_token
```

---

## Quick Start

### 1. Download Models

```bash
# Download all required models
python scripts/download_models.py --all

# Or download specific models
python scripts/download_models.py \
  --models vision_encoder language_decoder reasoning_engine
```

**Expected output:**
```
Downloading models...
  ✓ Vision Encoder (LayoutLMv3-base): 1.2 GB
  ✓ Language Decoder (LLaMA-2-7B): 13.5 GB
  ✓ Reasoning Engine (Mixtral-8x7B): 87 GB
Total: ~102 GB

All models downloaded successfully!
Cache directory: /models/huggingface_cache
```

### 2. Generate Test Fixtures

```bash
# Generate test documents with ground truth
python tests/fixtures/create_test_documents.py

# Verify test dataset
ls tests/fixtures/*.png
# Should show 10 test images
```

### 3. Run Basic Integration Tests

```bash
# Run model loading tests (fast, ~2 minutes)
pytest tests/integration/test_real_models.py::TestRealModelLoading -v

# Run inference tests (medium, ~10 minutes)
pytest tests/integration/test_real_models.py::TestRealModelInference -v -s

# Run full test suite (slow, ~30 minutes)
pytest tests/integration/test_real_models.py -v -s
```

---

## Test Structure

### Directory Layout

```
SAP_LLM/
├── config/
│   └── models.yaml                    # Model configuration
├── tests/
│   ├── utils/
│   │   ├── __init__.py
│   │   └── model_loader.py            # Real model loader utility
│   ├── fixtures/
│   │   ├── create_test_documents.py   # Test data generator
│   │   ├── test_dataset_manifest.json # Ground truth labels
│   │   ├── invoice_*.png              # Test images
│   │   └── po_*.png
│   └── integration/
│       ├── test_real_models.py        # Main integration tests
│       ├── test_real_model_accuracy.py       # Accuracy validation
│       ├── test_real_model_performance.py    # Performance benchmarks
│       └── test_real_model_error_handling.py # Error handling tests
├── .github/
│   └── workflows/
│       └── integration-tests-gpu.yml  # CI/CD pipeline
└── docs/
    └── INTEGRATION_TESTING.md         # This document
```

### Test File Overview

| File | Purpose | Test Count | Duration |
|------|---------|------------|----------|
| `test_real_models.py` | Core integration tests | 15 | ~10 min |
| `test_real_model_accuracy.py` | Accuracy validation | 5 | ~20 min |
| `test_real_model_performance.py` | Performance benchmarks | 10 | ~15 min |
| `test_real_model_error_handling.py` | Error handling | 12 | ~5 min |

---

## Running Tests

### Basic Usage

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_real_models.py -v

# Run specific test class
pytest tests/integration/test_real_models.py::TestRealModelInference -v

# Run specific test
pytest tests/integration/test_real_models.py::TestRealModelInference::test_vision_encoder_inference -v
```

### Common Options

```bash
# Show output in real-time
pytest tests/integration/test_real_models.py -v -s

# Run with coverage
pytest tests/integration/ --cov=sap_llm --cov-report=html

# Run with timeout (prevent hanging)
pytest tests/integration/ --timeout=600

# Skip slow tests
pytest tests/integration/ -v -m "not slow"

# Run only performance tests
pytest tests/integration/ -v -m "performance"

# Run with detailed output
pytest tests/integration/ -v -s --tb=short
```

### Test Markers

Tests are marked with pytest markers for filtering:

| Marker | Description | Usage |
|--------|-------------|-------|
| `integration` | All integration tests | `-m integration` |
| `real_models` | Tests requiring real models | `-m real_models` |
| `slow` | Slow tests (>1 minute) | `-m "not slow"` to skip |
| `gpu` | Tests requiring GPU | `-m gpu` |
| `accuracy` | Accuracy validation tests | `-m accuracy` |
| `performance` | Performance benchmarks | `-m performance` |
| `error_handling` | Error handling tests | `-m error_handling` |

---

## Test Suites

### 1. Model Loading Tests

**Purpose:** Validate models load correctly

```bash
pytest tests/integration/test_real_models.py::TestRealModelLoading -v
```

**Tests:**
- ✅ Load LayoutLMv3 vision encoder
- ✅ Load LLaMA-2 language decoder
- ✅ Load Mixtral reasoning engine
- ✅ Model info and metadata
- ✅ GPU memory allocation

**Expected Duration:** 2-5 minutes

### 2. Inference Tests

**Purpose:** Validate real model inference

```bash
pytest tests/integration/test_real_models.py::TestRealModelInference -v -s
```

**Tests:**
- ✅ Vision encoder classification
- ✅ Language decoder extraction
- ✅ Reasoning engine routing
- ✅ Confidence scores
- ✅ Output validation

**Expected Duration:** 10-15 minutes

### 3. Accuracy Tests

**Purpose:** Measure accuracy against ground truth

```bash
pytest tests/integration/test_real_model_accuracy.py -v -s
```

**Tests:**
- ✅ Classification accuracy (target: ≥70%)
- ✅ Extraction F1 score per field (target: ≥50%)
- ✅ End-to-end accuracy (target: ≥60%)
- ✅ Per-class metrics
- ✅ Confusion matrix

**Expected Duration:** 15-25 minutes

**Note:** Accuracy targets are adjusted for synthetic test data. Real production data should achieve higher accuracy.

### 4. Performance Tests

**Purpose:** Benchmark latency and throughput

```bash
pytest tests/integration/test_real_model_performance.py -v -s
```

**Tests:**
- ✅ Vision encoder latency (P95 < 500ms)
- ✅ Language decoder latency (P95 < 5s)
- ✅ Reasoning engine latency (P95 < 10s)
- ✅ Throughput (docs/second)
- ✅ GPU memory usage
- ✅ End-to-end pipeline latency

**Expected Duration:** 10-20 minutes

### 5. Error Handling Tests

**Purpose:** Validate error handling and recovery

```bash
pytest tests/integration/test_real_model_error_handling.py -v -s
```

**Tests:**
- ✅ Corrupted/invalid inputs
- ✅ Empty/blank images
- ✅ Very large images
- ✅ Low quality scans
- ✅ Invalid prompts
- ✅ Resource cleanup
- ✅ Timeout handling
- ✅ Edge cases

**Expected Duration:** 5-10 minutes

### 6. End-to-End Tests

**Purpose:** Full pipeline validation

```bash
pytest tests/integration/test_real_models.py::TestRealPipelineE2E -v -s
```

**Tests:**
- ✅ Complete supplier invoice pipeline
- ✅ Complete purchase order pipeline
- ✅ Classification → Extraction → Routing
- ✅ Quality checking
- ✅ Self-correction

**Expected Duration:** 15-20 minutes

---

## Performance Targets

### Latency Targets (with Quantization)

| Component | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Vision Encoder (LayoutLMv3) | <200ms | <500ms | <1000ms |
| Language Decoder (LLaMA-2) | <1500ms | <5000ms | <8000ms |
| Reasoning Engine (Mixtral) | <3000ms | <10000ms | <15000ms |
| **Full Pipeline** | **<5000ms** | **<10000ms** | **<15000ms** |

### Throughput Targets

| Component | Throughput |
|-----------|------------|
| Vision Encoder | >2 docs/sec |
| Language Decoder | >0.5 docs/sec |
| Full Pipeline | >0.2 docs/sec |

### GPU Memory Targets

| Component | Peak Memory |
|-----------|-------------|
| Vision Encoder | <10 GB |
| Language Decoder (8-bit) | <15 GB |
| Reasoning Engine (4-bit) | <25 GB |
| **All Models Loaded** | **<40 GB** |

### Accuracy Targets

| Metric | Target | Current (Synthetic) |
|--------|--------|---------------------|
| Classification Accuracy | ≥99% | ≥70% |
| Extraction F1 Score | ≥97% | ≥50% |
| End-to-End Accuracy | ≥95% | ≥60% |

**Note:** Current accuracy is measured on synthetic test data. Real production accuracy should be higher with proper training.

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**
```bash
# Use more aggressive quantization
# Edit config/models.yaml:
language_decoder:
  quantization: 8bit  # or 4bit

reasoning_engine:
  quantization: 4bit

# Reduce batch size
language_decoder:
  batch_size: 1

# Enable CPU offloading
memory:
  cpu_offload: true
```

#### 2. Models Not Downloaded

**Error:**
```
pytest.skip: Models not downloaded: ['llama2', 'mixtral']
```

**Solution:**
```bash
# Download missing models
python scripts/download_models.py --all

# Or specific model
python scripts/download_models.py --models language_decoder
```

#### 3. HuggingFace Authentication Error

**Error:**
```
401 Client Error: Unauthorized for url: https://huggingface.co/...
```

**Solution:**
```bash
# Set HuggingFace token
export HF_TOKEN=your_token_here

# Or pass directly
python scripts/download_models.py --token your_token_here
```

#### 4. Test Fixtures Not Found

**Error:**
```
pytest.skip: Test dataset not found
```

**Solution:**
```bash
# Generate test fixtures
python tests/fixtures/create_test_documents.py

# Verify
ls tests/fixtures/*.png
```

#### 5. Slow Test Execution

**Symptom:** Tests take >1 hour

**Solutions:**
```bash
# Skip slow tests
pytest tests/integration/ -m "not slow"

# Run only fast tests
pytest tests/integration/test_real_models.py::TestRealModelLoading

# Use smaller test dataset
# Edit tests/fixtures/create_test_documents.py
generator.generate_test_dataset(num_invoices=5, num_pos=5)  # Reduce from 50
```

#### 6. GPU Not Detected

**Error:**
```
pytest.skip: GPU not available - required for real model tests
```

**Solutions:**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check CUDA version
nvcc --version
```

---

## CI/CD Integration

### GitHub Actions Workflow

The real model integration tests run automatically via GitHub Actions on GPU runners.

**Workflow File:** `.github/workflows/integration-tests-gpu.yml`

**Triggers:**
- ✅ Nightly at 2 AM UTC
- ✅ Manual trigger via workflow_dispatch
- ✅ Push to main branch
- ✅ Pull requests to main

**Jobs:**
1. **Preflight Checks** - Verify GPU runner availability
2. **Model Download** - Download and cache models (~2 hours, cached)
3. **Integration Tests** - Run all test suites in parallel
4. **E2E Tests** - Run full pipeline tests
5. **Test Report** - Generate and publish test results
6. **Cleanup** - Free GPU resources

### Manual Trigger

```bash
# Via GitHub UI
Actions → Integration Tests (Real Models - GPU) → Run workflow

# Via GitHub CLI
gh workflow run integration-tests-gpu.yml

# With specific test subset
gh workflow run integration-tests-gpu.yml -f test_subset=performance
```

### Viewing Results

1. Go to **Actions** tab in GitHub
2. Select **Integration Tests (Real Models - GPU)** workflow
3. Click on latest run
4. View job logs and artifacts
5. Download test results XML

---

## Advanced Topics

### Custom Model Configuration

Edit `config/models.yaml` to customize model settings:

```yaml
vision_encoder:
  model: microsoft/layoutlmv3-base
  precision: float16  # or float32, bfloat16
  quantization: null  # or 8bit, 4bit

language_decoder:
  model: meta-llama/Llama-2-7b-hf
  quantization: 8bit
  lora:
    enabled: true
    r: 16
```

### Custom Test Dataset

Create your own test dataset with ground truth:

```python
from tests.fixtures.create_test_documents import TestDocumentGenerator

generator = TestDocumentGenerator(output_dir="tests/fixtures/custom")

# Generate custom documents
generator.generate_supplier_invoice(
    invoice_id="CUSTOM-001",
    vendor_id="VENDOR-999",
    total_amount=5000.00
)

# Save manifest
generator.generate_test_dataset(num_invoices=100, num_pos=100)
```

### Profiling Model Performance

```python
from tests.utils.model_loader import RealModelLoader
import torch

loader = RealModelLoader(config_path="config/models.yaml")
model, processor = loader.load_vision_encoder()

# Enable profiling
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    # Run inference
    encoding = processor(image, return_tensors="pt")
    outputs = model(**encoding)

# Print profile
print(prof.key_averages().table(sort_by="cuda_time_total"))

# Export trace
prof.export_chrome_trace("trace.json")
```

### Distributed Testing

For large-scale testing across multiple GPUs:

```bash
# Run tests on specific GPU
CUDA_VISIBLE_DEVICES=0 pytest tests/integration/test_real_models.py

# Run tests on multiple GPUs in parallel
CUDA_VISIBLE_DEVICES=0 pytest tests/integration/test_real_models.py::TestVisionEncoder &
CUDA_VISIBLE_DEVICES=1 pytest tests/integration/test_real_models.py::TestLanguageDecoder &
wait
```

---

## Appendix

### Model Details

| Model | Parameters | Quantized Size | Inference Time | Accuracy |
|-------|-----------|----------------|----------------|----------|
| LayoutLMv3-base | 300M | 1.2 GB (FP16) | ~200ms | High |
| LLaMA-2-7B | 7B | 7 GB (8-bit) | ~2s | High |
| Mixtral-8x7B | 47B (6B active) | 24 GB (4-bit) | ~3s | Very High |

### Resource Requirements Summary

**Minimum Setup:**
- GPU: 16GB VRAM
- Disk: 50GB
- Duration: ~30 min full test suite

**Recommended Setup:**
- GPU: 40GB VRAM (A100)
- Disk: 100GB
- Duration: ~20 min full test suite

### Support

For issues and questions:
- **GitHub Issues:** https://github.com/AjithAccel4/SAP_LLM/issues
- **Documentation:** https://docs.sap-llm.com
- **CI/CD Logs:** GitHub Actions tab

---

**Last Updated:** 2025-01-15
**Version:** 1.0.0
**Maintained by:** SAP_LLM Team
