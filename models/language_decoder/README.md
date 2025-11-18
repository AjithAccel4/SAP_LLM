# Language Decoder Model Checkpoints

This directory contains trained Language Decoder model weights and checkpoints.

## Directory Structure

```
models/language_decoder/
├── README.md                 # This file
├── best/                     # Best checkpoint (highest F1)
│   ├── pytorch_model.bin
│   ├── adapter_config.json   # LoRA configuration
│   ├── adapter_model.bin     # LoRA weights
│   ├── custom_layers.pt      # Cross-attention + vision projection
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── training_state.pt     # Training metadata
├── final/                    # Final checkpoint
├── checkpoint-1000/          # Intermediate checkpoints
├── checkpoint-2000/
└── ...
```

## Model Specifications

- **Base Model**: LLaMA-2-7B
- **Total Parameters**: ~7B
- **Trainable Parameters**: ~4.2M (LoRA only)
- **LoRA Configuration**: r=16, alpha=32
- **Cross-Attention Layers**: 4 layers at positions [8, 16, 24, 31]
- **Vision Projection**: 768 → 4096

## Target Metrics

- **Field Extraction F1**: ≥92%
- **Schema Compliance**: ≥99%
- **Required Field Completeness**: ≥95%
- **Inference Latency (P95)**: <800ms per document

## Usage

### Loading a Checkpoint

```python
from sap_llm.models.language_decoder_with_lora import LanguageDecoderWithLoRA

# Load best checkpoint
model = LanguageDecoderWithLoRA.load(
    model_path="models/language_decoder/best",
    device="cuda",
    precision="fp16",
)

# Extract fields from document
schema = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "date": {"type": "string"},
        "total": {"type": "number"},
    },
    "required": ["invoice_number", "date"],
}

extracted_data = model.extract_fields(
    ocr_text="INVOICE\nInvoice Number: INV-2024-001\nDate: 2024-01-15\nTotal: $1,234.56",
    doc_type="invoice",
    schema=schema,
    use_self_correction=True,
)

print(extracted_data)
# {
#   "invoice_number": "INV-2024-001",
#   "date": "2024-01-15",
#   "total": 1234.56
# }
```

## Training Information

Trained using:
- **Dataset**: 500K labeled documents
- **Training Time**: ~48 hours on 4x A100 GPUs
- **Batch Size**: 4 per GPU × 8 gradient accumulation = 32 effective
- **Learning Rate**: 1e-4 (phase 1), 5e-6 (phase 2)
- **Epochs**: 3 total (2 decoder-only, 1 full model)

## Checkpoint Selection

- **best/**: Use this for production inference (highest validation F1)
- **final/**: Use this for further fine-tuning
- **checkpoint-N/**: Intermediate checkpoints for analysis

## Model Card

See `MODEL_CARD.md` for detailed model documentation including:
- Training data
- Evaluation results
- Limitations
- Intended use cases
- Ethical considerations
