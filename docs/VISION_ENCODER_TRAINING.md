# Vision Encoder Training Guide

## Overview

This guide provides detailed instructions for training the Vision Encoder component of SAP_LLM. The Vision Encoder is based on LayoutLMv3 and performs multi-task learning for SAP document understanding.

## Architecture

**Base Model:** LayoutLMv3-base (300M parameters)

**Tasks:**
1. **Document Type Classification** - 15 classes (Invoice, PO, Sales Order, etc.)
2. **PO Subtype Classification** - 35 classes (Standard, Blanket, Contract, etc.)
3. **Token Classification** - 180+ SAP field labels for field extraction

**Model Components:**
- Base LayoutLMv3 encoder (visual + text + layout)
- Document type classification head
- PO subtype classification head
- Token classification head for field extraction

## Hardware Requirements

### Recommended Configuration

**Option 1: High-End (Fastest training)**
- GPUs: 4x NVIDIA A100 80GB
- RAM: 256GB+
- Storage: 500GB SSD
- Training time: ~36 hours

**Option 2: Mid-Range (Good performance)**
- GPUs: 8x NVIDIA A10 40GB
- RAM: 128GB+
- Storage: 500GB SSD
- Training time: ~60 hours

**Option 3: Budget (Slower but viable)**
- GPUs: 4x NVIDIA V100 32GB
- RAM: 128GB+
- Storage: 500GB SSD
- Training time: ~80 hours

## Installation

### Prerequisites

```bash
# Python 3.9+
python --version

# CUDA 11.8+ (for GPU training)
nvidia-smi
```

### Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Transformers and other dependencies
pip install transformers datasets pillow scikit-learn
pip install onnx onnxruntime-gpu
pip install wandb  # For experiment tracking

# Optional: Install DeepSpeed for advanced distributed training
pip install deepspeed
```

## Data Preparation

### Data Format

The training data should be organized as follows:

```
data/vision_encoder/
├── train/
│   ├── images/
│   │   ├── doc_001.png
│   │   ├── doc_002.png
│   │   └── ...
│   └── annotations/
│       ├── doc_001.json
│       ├── doc_002.json
│       └── ...
├── val/
│   ├── images/
│   │   └── ...
│   └── annotations/
│       └── ...
└── test/
    ├── images/
    │   └── ...
    └── annotations/
        └── ...
```

### Annotation Format

Each annotation JSON file should have the following structure:

```json
{
  "image_path": "doc_001.png",
  "doc_type": "PURCHASE_ORDER",
  "po_subtype": "STANDARD",
  "words": ["word1", "word2", "word3", ...],
  "boxes": [[x1, y1, x2, y2], ...],
  "token_labels": [0, 1, 2, ...],
  "width": 1654,
  "height": 2339
}
```

**Field Descriptions:**
- `image_path`: Filename of the document image
- `doc_type`: Document type (one of 15 types: INVOICE, PURCHASE_ORDER, etc.)
- `po_subtype`: PO subtype (optional, one of 35 subtypes)
- `words`: List of OCR words from the document
- `boxes`: Bounding boxes for each word, normalized to 0-1000 scale
- `token_labels`: Field label ID for each word (0-180)
- `width`: Original image width in pixels
- `height`: Original image height in pixels

### Document Types (15 classes)

1. INVOICE
2. PURCHASE_ORDER
3. SALES_ORDER
4. GOODS_RECEIPT
5. DELIVERY_NOTE
6. CREDIT_MEMO
7. DEBIT_MEMO
8. PAYMENT_ADVICE
9. REMITTANCE_ADVICE
10. STATEMENT
11. CONTRACT
12. QUOTATION
13. RFQ
14. ASN
15. OTHER

### Creating Synthetic Data (for testing)

```bash
# Generate synthetic dataset for testing
python -c "
from sap_llm.training.vision_dataset import create_synthetic_dataset

# Create training data
create_synthetic_dataset(
    output_dir='./data/vision_encoder_synthetic/train',
    num_samples=1000,
    split='train'
)

# Create validation data
create_synthetic_dataset(
    output_dir='./data/vision_encoder_synthetic/val',
    num_samples=200,
    split='val'
)

# Create test data
create_synthetic_dataset(
    output_dir='./data/vision_encoder_synthetic/test',
    num_samples=200,
    split='test'
)
"
```

## Training

### Single GPU Training

For testing or small datasets:

```bash
python scripts/train_vision_encoder.py \
    --data_dir ./data/vision_encoder/train \
    --val_data_dir ./data/vision_encoder/val \
    --output_dir ./models/vision_encoder \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --max_steps 50000 \
    --fp16 \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --use_wandb \
    --wandb_project sap-llm-vision-encoder \
    --wandb_run_name vision-encoder-baseline
```

### Multi-GPU Training with FSDP

Recommended for production training:

```bash
# 4 GPUs
torchrun --nproc_per_node=4 scripts/train_vision_encoder.py \
    --data_dir ./data/vision_encoder/train \
    --val_data_dir ./data/vision_encoder/val \
    --output_dir ./models/vision_encoder \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --max_steps 50000 \
    --fp16 \
    --use_fsdp \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 5000 \
    --use_wandb \
    --wandb_project sap-llm-vision-encoder \
    --wandb_run_name vision-encoder-fsdp-4gpu

# 8 GPUs
torchrun --nproc_per_node=8 scripts/train_vision_encoder.py \
    --data_dir ./data/vision_encoder/train \
    --val_data_dir ./data/vision_encoder/val \
    --output_dir ./models/vision_encoder \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 5e-5 \
    --warmup_steps 1000 \
    --max_steps 50000 \
    --fp16 \
    --use_fsdp \
    --logging_steps 10 \
    --eval_steps 1000 \
    --save_steps 5000
```

### Training with Synthetic Data

For quick testing:

```bash
python scripts/train_vision_encoder.py \
    --data_dir ./data/vision_encoder_synthetic/train \
    --val_data_dir ./data/vision_encoder_synthetic/val \
    --output_dir ./models/vision_encoder_test \
    --create_synthetic \
    --synthetic_samples 500 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --max_steps 1000 \
    --fp16 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500
```

### Training Hyperparameters

**Recommended Configuration (from PLAN_02.md):**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size per GPU | 4 | Effective batch size = 4 × 4 GPUs × 8 accum = 128 |
| Gradient accumulation | 8 | Accumulate gradients over 8 steps |
| Learning rate | 5e-5 | Peak learning rate |
| Warmup steps | 1000 | Linear warmup from 0 to max LR |
| Max steps | 50,000 | Total training steps (~36h on 4xA100) |
| Weight decay | 0.01 | L2 regularization |
| Max grad norm | 1.0 | Gradient clipping threshold |
| Mixed precision | FP16 | Use automatic mixed precision |
| Sequence length | 512 | Maximum token sequence length |

## Evaluation

### Run Evaluation

```bash
python scripts/evaluate_vision_encoder.py \
    --model_path ./models/vision_encoder/best \
    --data_dir ./data/vision_encoder/test \
    --output_dir ./evaluation_results \
    --batch_size 8 \
    --benchmark \
    --benchmark_iterations 100
```

### Evaluation Metrics

The evaluation script provides:

1. **Document Type Classification Metrics**
   - Accuracy
   - Precision, Recall, F1 (weighted average)
   - Confusion matrix
   - Per-class classification report

2. **PO Subtype Classification Metrics**
   - Accuracy
   - Precision, Recall, F1 (weighted average)
   - Confusion matrix
   - Per-class classification report

3. **Token Classification Metrics**
   - Accuracy
   - Precision, Recall, F1 (weighted average)
   - Field-level extraction performance

4. **Latency Benchmarking**
   - Mean, min, max latency
   - P50, P95, P99 percentiles

### Success Criteria

According to PLAN_02.md, the trained model should achieve:

- ✅ **Document classification accuracy:** ≥95%
- ✅ **Field detection F1 score:** ≥92%
- ✅ **Inference latency:** <50ms per page (on A10 GPU)
- ✅ **Model size:** ~440MB checkpoint

## Model Export

### Export to ONNX

For optimized inference:

```bash
python -c "
import torch
from transformers import LayoutLMv3Config
from sap_llm.models.vision_encoder import MultiTaskLayoutLMv3, export_to_onnx

# Load model
config = LayoutLMv3Config.from_pretrained('microsoft/layoutlmv3-base')
model = MultiTaskLayoutLMv3(config=config)
model.load_state_dict(torch.load('./models/vision_encoder/best/pytorch_model.bin'))

# Export to ONNX
export_to_onnx(
    model=model,
    output_path='./models/vision_encoder/model.onnx',
    opset_version=14
)
"
```

The training script can also automatically export to ONNX:

```bash
python scripts/train_vision_encoder.py \
    --data_dir ./data/vision_encoder/train \
    --output_dir ./models/vision_encoder \
    ... \
    --export_onnx
```

## Monitoring Training

### Weights & Biases

Enable W&B logging for experiment tracking:

```bash
# Login to W&B
wandb login

# Run training with W&B
python scripts/train_vision_encoder.py \
    --data_dir ./data/vision_encoder/train \
    --output_dir ./models/vision_encoder \
    --use_wandb \
    --wandb_project sap-llm-vision-encoder \
    --wandb_run_name experiment-001 \
    ...
```

**Tracked Metrics:**
- Training loss (total, doc_type, po_subtype, token)
- Validation loss
- Learning rate
- Document type accuracy
- PO subtype accuracy
- Gradient norms
- Training steps/epoch

### TensorBoard (Alternative)

If you prefer TensorBoard, you can modify the training script to use `torch.utils.tensorboard.SummaryWriter`.

## Troubleshooting

### Out of Memory (OOM) Errors

**Solutions:**
1. Reduce batch size: `--batch_size 2`
2. Increase gradient accumulation: `--gradient_accumulation_steps 16`
3. Reduce sequence length: `--max_length 256`
4. Enable FSDP: `--use_fsdp`
5. Use CPU offloading with DeepSpeed

### Slow Training

**Solutions:**
1. Enable mixed precision: `--fp16`
2. Increase number of data workers: `--num_workers 8`
3. Use faster storage (SSD) for data
4. Enable FSDP for multi-GPU: `--use_fsdp`
5. Check GPU utilization: `nvidia-smi dmon`

### Poor Convergence

**Solutions:**
1. Increase learning rate: `--learning_rate 1e-4`
2. Increase warmup steps: `--warmup_steps 2000`
3. Reduce weight decay: `--weight_decay 0.001`
4. Check data quality and labels
5. Try different batch sizes

### Data Loading Errors

**Solutions:**
1. Verify data directory structure
2. Check annotation JSON format
3. Ensure images are readable (PIL.Image.open)
4. Verify bounding box coordinates (0-1000 scale)
5. Check that all required fields are present

## Advanced Topics

### Custom Field Labels

To add custom SAP field labels, edit `sap_llm/training/vision_dataset.py`:

```python
SAP_FIELD_LABELS = {
    "O": 0,
    "B-CUSTOM_FIELD": 181,
    "I-CUSTOM_FIELD": 182,
    # Add more fields...
}
```

Then update the training command:

```bash
python scripts/train_vision_encoder.py \
    ... \
    --num_token_labels 183  # Updated count
```

### Resume Training

To resume from a checkpoint:

```bash
python scripts/train_vision_encoder.py \
    --data_dir ./data/vision_encoder/train \
    --output_dir ./models/vision_encoder \
    --resume_from ./models/vision_encoder/checkpoint-25000 \
    ...
```

### Fine-tuning on Domain-Specific Data

To fine-tune the pretrained model on your domain-specific data:

```bash
# First, train on general SAP documents
python scripts/train_vision_encoder.py \
    --data_dir ./data/general_sap_docs/train \
    --output_dir ./models/vision_encoder_general \
    --max_steps 50000 \
    ...

# Then, fine-tune on domain-specific documents
python scripts/train_vision_encoder.py \
    --data_dir ./data/domain_specific/train \
    --output_dir ./models/vision_encoder_domain \
    --resume_from ./models/vision_encoder_general/best \
    --max_steps 10000 \
    --learning_rate 2e-5 \  # Lower LR for fine-tuning
    ...
```

## Checkpoints and Model Management

### Checkpoint Structure

```
models/vision_encoder/
├── checkpoint-5000/
│   ├── pytorch_model.bin
│   ├── optimizer.pt
│   ├── scheduler.pt
│   └── training_state.json
├── checkpoint-10000/
│   └── ...
├── best/
│   ├── pytorch_model.bin
│   └── training_state.json
└── final/
    ├── pytorch_model.bin
    ├── model.onnx  # If --export_onnx was used
    └── training_state.json
```

### Loading Trained Model

```python
from transformers import LayoutLMv3Config
from sap_llm.models.vision_encoder import MultiTaskLayoutLMv3

# Load config
config = LayoutLMv3Config.from_pretrained("microsoft/layoutlmv3-base")

# Create model
model = MultiTaskLayoutLMv3(
    config=config,
    num_doc_types=15,
    num_po_subtypes=35,
    num_token_labels=181,
)

# Load weights
model.load_state_dict(torch.load("./models/vision_encoder/best/pytorch_model.bin"))
model.eval()

# Use for inference
outputs = model(**inputs)
```

## Performance Optimization

### Inference Optimization

1. **Use ONNX Runtime:**
   ```python
   import onnxruntime
   session = onnxruntime.InferenceSession("model.onnx")
   outputs = session.run(None, inputs)
   ```

2. **Quantization:**
   ```python
   from torch.quantization import quantize_dynamic
   quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

3. **TensorRT (NVIDIA GPUs):**
   Convert ONNX model to TensorRT for maximum performance on NVIDIA GPUs.

### Batch Inference

For processing multiple documents:

```python
# Batch multiple documents together
batch_inputs = processor(
    images=[img1, img2, img3, img4],
    words=[words1, words2, words3, words4],
    boxes=[boxes1, boxes2, boxes3, boxes4],
    padding=True,
    return_tensors="pt"
)

outputs = model(**batch_inputs)
```

## References

- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the training logs
3. Check W&B experiments for insights
4. Open an issue in the project repository

## Citation

If you use this vision encoder in your research, please cite:

```bibtex
@software{sap_llm_vision_encoder,
  title={SAP_LLM Vision Encoder},
  author={SAP_LLM Team},
  year={2025},
  description={Multi-task LayoutLMv3-based vision encoder for SAP document understanding}
}
```
