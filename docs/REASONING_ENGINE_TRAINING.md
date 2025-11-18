
# Reasoning Engine Training Guide

## Overview

This guide covers the complete training pipeline for the **Reasoning Engine** - a Mixtral-8x7B based model fine-tuned for autonomous SAP routing and decision-making.

## Architecture

### Model: Mixtral-8x7B-v0.1

- **Total Parameters:** 47B (6B active per token)
- **Architecture:** Mixture of Experts (8 experts, 2 active per token)
- **Quantization:** 4-bit NF4 for training (QLoRA)
- **LoRA Config:**
  - Rank (r): 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj
  - Dropout: 0.05
  - Trainable parameters: ~94M (0.2% of total)

### Training Approach

1. **Supervised Fine-Tuning (SFT)**
   - 200K routing examples with chain-of-thought reasoning
   - 10K training steps
   - Learning rate: 2e-5

2. **RLHF with PPO**
   - Reward model based on SAP API success rate
   - 5K PPO iterations
   - Learning rate: 1e-6

## Training Pipeline

### Stage 1: Data Preparation

Creates 200K+ SAP routing examples with:
- Chain-of-thought reasoning traces
- PMG context (similar document routings)
- Success/failure feedback
- Field transformations

```bash
python scripts/train_reasoning_engine.py --stage data_prep \
  --num-examples 200000 \
  --transaction-logs data/sap_transactions.jsonl \
  --api-schemas data/sap_api_schemas.json \
  --pmg-data data/pmg_routing_history.json
```

**Output:**
- `data/training/reasoning_engine/train_routing_examples.jsonl` (160K)
- `data/training/reasoning_engine/val_routing_examples.jsonl` (20K)
- `data/training/reasoning_engine/test_routing_examples.jsonl` (20K)
- `data/training/reasoning_engine/preference_pairs.jsonl` (for RLHF)

### Stage 2: Supervised Fine-Tuning (SFT)

Fine-tunes Mixtral-8x7B with QLoRA on routing examples.

```bash
python scripts/train_reasoning_engine.py --stage sft \
  --model-name mistralai/Mixtral-8x7B-v0.1 \
  --sft-epochs 3 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --sft-learning-rate 2e-5 \
  --lora-r 16 \
  --lora-alpha 32
```

**Training Details:**
- Batch size: 2 per device (effective: 16 with accumulation)
- Epochs: 3
- Steps: ~30K (on 160K examples)
- GPU Memory: ~40GB (with 4-bit quantization + gradient checkpointing)
- Training time: ~24-36 hours on A100 80GB

**Output:**
- `models/reasoning_engine/final/` - Trained model + LoRA adapters

### Stage 3: RLHF/PPO Training

Optimizes routing decisions using reinforcement learning.

**Reward Function:**
```
Reward = 0.7 × API_Success + 0.2 × Business_Rules + 0.1 × Confidence
```

- **API Success:** +1.0 for correct endpoint, -1.0 for wrong
- **Business Rules:** +0.25 per satisfied rule (max 1.0)
- **Confidence:** Calibration reward/penalty

```bash
python scripts/train_reasoning_engine.py --stage rlhf \
  --ppo-iterations 5000 \
  --rlhf-learning-rate 1e-6 \
  --kl-penalty 0.05 \
  --api-success-weight 0.7 \
  --business-rule-weight 0.2 \
  --confidence-weight 0.1
```

**Training Details:**
- PPO iterations: 5000
- Batch size: 8 routing problems per iteration
- Mini-batch size: 2
- PPO epochs per iteration: 4
- Training time: ~12-18 hours on A100 80GB

**Output:**
- `models/reasoning_engine_rlhf/final/` - RLHF-optimized model

### Stage 4: Evaluation

Comprehensive evaluation against success criteria.

```bash
python scripts/train_reasoning_engine.py --stage evaluate \
  --evaluate-rlhf \
  --eval-samples 1000
```

**Metrics:**
- ✅ **Routing Accuracy:** ≥97% (overall correctness)
- ✅ **API Selection Accuracy:** 100% (CRITICAL - no wrong endpoints)
- ✅ **Payload Accuracy:** ≥99% (field-level correctness)
- ✅ **Inference Latency:** <500ms per decision

**Output:**
- `evaluation_results/reasoning_engine/EVALUATION_REPORT.md`
- `evaluation_results/reasoning_engine/metrics.json`
- `evaluation_results/reasoning_engine/accuracy_metrics.png`
- `evaluation_results/reasoning_engine/accuracy_by_doc_type.png`

## Complete Training Pipeline

Run all stages sequentially:

```bash
python scripts/train_reasoning_engine.py --stage all \
  --num-examples 200000 \
  --sft-epochs 3 \
  --ppo-iterations 5000
```

This will:
1. Prepare 200K training examples
2. Run SFT for 3 epochs
3. Run RLHF for 5K iterations
4. Evaluate final model

**Total Training Time:** ~48-72 hours on A100 80GB

## Hardware Requirements

### Minimum (Training)
- **GPU:** 1× A100 80GB or 2× A100 40GB
- **RAM:** 128GB
- **Storage:** 500GB SSD (for checkpoints)

### Recommended (Training)
- **GPU:** 2× A100 80GB
- **RAM:** 256GB
- **Storage:** 1TB NVMe SSD

### Minimum (Inference)
- **GPU:** 1× V100 16GB (with 8-bit quantization)
- **RAM:** 32GB
- **Latency:** ~300ms per decision

## Model Capabilities

### 1. SAP Endpoint Selection
```python
decision = model.decide_routing(
    adc_json=extracted_data,
    doc_type="PURCHASE_ORDER",
    api_schemas=sap_apis,
    similar_cases=pmg_context,
)
# Output: {"endpoint": "API_PURCHASEORDER_PROCESS_SRV", "confidence": 0.98}
```

### 2. Payload Generation
Automatically generates SAP OData V2 compliant payloads with:
- Field transformations
- Default value injection
- Schema validation

### 3. Confidence Scoring
- **High (>0.9):** Clear routing with strong PMG support
- **Medium (0.7-0.9):** Ambiguous but resolvable
- **Low (<0.7):** Novel case requiring human review

### 4. Fallback Strategy
When confidence is low or decision is ambiguous:
- Uses rule-based fallback routing
- Escalates to human operator
- Logs for continuous learning

### 5. Explanation Generation
Provides chain-of-thought reasoning for every decision:
```
Step 1: Document Analysis
  - Type: PURCHASE_ORDER
  - Supplier: ACME Corp (SUP1234)

Step 2: Historical Context
  - Similar cases: 15 found
  - Success rate: 98.5%

Step 3: Decision
  - Route to: API_PURCHASEORDER_PROCESS_SRV
  - Confidence: 0.98
```

## Inference Examples

See `examples/reasoning_engine_inference.py` for complete examples:

1. **Purchase Order Routing**
2. **Exception Handling**
3. **Batch Processing**
4. **Confidence Calibration**

Run examples:
```bash
python examples/reasoning_engine_inference.py
```

## Success Criteria Validation

After training, validate against production requirements:

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Routing Accuracy | ≥97% | Automated test on 1K examples |
| API Selection | 100% | Zero tolerance - manual audit on 500 examples |
| Payload Accuracy | ≥99% | Field-level validation on 1K examples |
| Inference Latency | <500ms | P95 latency on 10K requests |
| Explanation Quality | Human-validated | Manual review of 100 samples |

## Troubleshooting

### OOM (Out of Memory) Errors

1. **Reduce batch size:**
   ```bash
   --batch-size 1 --gradient-accumulation-steps 16
   ```

2. **Enable gradient checkpointing:**
   ```bash
   --gradient-checkpointing
   ```

3. **Use 4-bit quantization:**
   Already enabled by default in QLoRA config

### Low Accuracy

1. **Increase training data:**
   ```bash
   --num-examples 500000
   ```

2. **More SFT epochs:**
   ```bash
   --sft-epochs 5
   ```

3. **Adjust RLHF rewards:**
   ```bash
   --api-success-weight 0.8  # Prioritize endpoint correctness
   ```

### Slow Inference

1. **Use int8 quantization:**
   ```bash
   --eval-precision int8
   ```

2. **Reduce max_length:**
   ```bash
   --max-length 2048
   ```

3. **Use FlashAttention:** (requires installation)
   ```bash
   pip install flash-attn --no-build-isolation
   ```

## File Structure

```
SAP_LLM/
├── sap_llm/
│   ├── models/
│   │   └── reasoning_engine.py          # Core model with QLoRA
│   ├── training/
│   │   ├── data_preparation.py          # Dataset builder
│   │   ├── sft_trainer.py               # SFT trainer
│   │   └── reasoning_rlhf_trainer.py    # RLHF/PPO trainer
│   └── evaluation/
│       └── reasoning_evaluator.py       # Comprehensive evaluator
├── scripts/
│   └── train_reasoning_engine.py        # Main training script
├── examples/
│   └── reasoning_engine_inference.py    # Inference examples
├── models/
│   ├── reasoning_engine/                # SFT checkpoints
│   └── reasoning_engine_rlhf/           # RLHF checkpoints
├── data/
│   └── training/reasoning_engine/       # Training data
└── evaluation_results/
    └── reasoning_engine/                # Evaluation reports
```

## Next Steps

After successful training:

1. **Integration Testing**
   - Test with real SAP system
   - Validate API calls
   - Monitor error rates

2. **A/B Testing**
   - Compare against rule-based routing
   - Measure success rate improvement
   - Track cost savings

3. **Continuous Learning**
   - Collect production feedback
   - Retrain monthly with new data
   - Track drift and performance

4. **Production Deployment**
   - Deploy to inference endpoint
   - Set up monitoring and alerts
   - Configure autoscaling

## References

- PLAN_02.md Phase 4.8 (Reasoning Engine specification)
- Mixtral Paper: https://arxiv.org/abs/2401.04088
- QLoRA: https://arxiv.org/abs/2305.14314
- PPO: https://arxiv.org/abs/1707.06347

## Support

For issues or questions:
- Check troubleshooting section above
- Review evaluation reports for detailed metrics
- Consult PLAN_02.md for architecture details
