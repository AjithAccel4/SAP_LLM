# SAP_LLM Examples

Comprehensive examples demonstrating all SAP_LLM capabilities.

## Overview

This directory contains production-ready examples for:

- **Training**: Distributed training, RLHF, fine-tuning
- **Optimization**: Quantization, pruning, distillation, TensorRT
- **APOP**: Event-driven workflows, autonomous agents
- **Deployment**: Kubernetes, Docker, cloud deployment

## Quick Start

### Prerequisites

```bash
# Install SAP_LLM
pip install -e .

# Install optional dependencies
pip install kafka-python  # For APOP CloudEvents
pip install tensorrt  # For TensorRT optimization
pip install auto-gptq  # For GPTQ quantization
```

### Running Examples

```bash
# RLHF Training
python examples/training/train_with_rlhf.py

# Model Quantization
python examples/optimization/quantize_model.py

# APOP Workflow
python examples/apop/run_workflow.py
```

## Examples by Category

### Training

#### `training/train_with_rlhf.py`

Complete RLHF (Reinforcement Learning from Human Feedback) training pipeline.

**Features:**
- Preference dataset collection from PMG
- Reward model training with margin ranking loss
- PPO (Proximal Policy Optimization)
- Integration with Process Memory Graph

**Requirements:**
- GPU: 8x H100 80GB (or 8x A100 80GB)
- Memory: 512GB RAM
- Storage: 500GB for checkpoints

**Usage:**
```bash
python examples/training/train_with_rlhf.py
```

**Expected Results:**
- Reward model accuracy: 85-90%
- PPO convergence: ~1000 iterations
- Performance improvement: 10-15% over baseline
- Training time: 24-48 hours on 8x H100

**Key Concepts:**
- **Preference Pairs**: Human feedback comparing two model outputs
- **Reward Model**: Learns to predict human preferences
- **PPO**: Optimizes policy while staying close to reference model
- **KL Penalty**: Prevents policy from deviating too far

### Optimization

#### `optimization/quantize_model.py`

Model quantization and optimization for efficient inference.

**Features:**
- INT8 quantization (2× compression)
- GPTQ-4 quantization (4× compression)
- TensorRT optimization (2-5× speedup)
- Knowledge distillation (72B ’ 7B)

**Requirements:**
- GPU: NVIDIA GPU with Tensor Cores (V100, A100, H100)
- CUDA: 11.8+
- TensorRT: 9.0+

**Usage:**
```bash
# Interactive mode
python examples/optimization/quantize_model.py

# Select specific optimization:
# 1. INT8 Quantization
# 2. GPTQ-4 Quantization
# 3. TensorRT Optimization
# 4. Knowledge Distillation
```

**Expected Results:**

| Method | Size | Latency | Accuracy |
|--------|------|---------|----------|
| FP16 (baseline) | 144GB | 200ms | 100% |
| INT8 | 72GB | 100ms | 98% |
| GPTQ-4 | 36GB | 80ms | 95% |
| TensorRT FP16 | 144GB | 80ms | 100% |
| TensorRT INT8 | 72GB | 50ms | 98% |
| Distilled 7B | 14GB | 20ms | 90% |

**Key Concepts:**
- **Quantization**: Reducing precision (FP16 ’ INT8/INT4)
- **Calibration**: Using dataset to find optimal quantization parameters
- **TensorRT**: NVIDIA's inference optimizer with kernel fusion
- **Distillation**: Transferring knowledge from large to small model

### APOP (Agentic Process Orchestration)

#### `apop/run_workflow.py`

Event-driven document processing workflows with autonomous agents.

**Features:**
- CloudEvents v1.0 compliant event bus
- Kafka backend for distributed messaging
- Autonomous stage agents (preprocessing, classification, extraction, etc.)
- Workflow replay and debugging
- Dead letter queue for failed events

**Requirements:**
- Kafka: 3.0+ (running on localhost:9092)
- Zookeeper: 3.6+ (for Kafka)
- Neo4j: 5.0+ (for PMG)
- Qdrant: 1.7+ (for vector embeddings)

**Usage:**
```bash
# Start Kafka (using Docker)
docker-compose up -d kafka zookeeper

# Run single document processing
python examples/apop/run_workflow.py
# Select: 1 (Single Document Processing)

# Run batch processing
python examples/apop/run_workflow.py
# Select: 2 (Batch Processing)

# Replay workflow for debugging
python examples/apop/run_workflow.py
# Select: 3 (Workflow Replay)
```

**Workflow Stages:**

1. **Document Received** ’ Preprocessing Agent
2. **Preprocessed** (OCR) ’ Classification Agent
3. **Classified** (invoice/PO/etc) ’ Extraction Agent
4. **Extracted** (fields) ’ Quality Control Agent
5. **Quality Checked** ’ Business Rules Agent
6. **Rules Validated** ’ Post-Processing Agent
7. **Completed** ’ Archive/SAP System

**Expected Performance:**
- Throughput: 1000 documents/hour per agent instance
- Latency: 5-10 seconds end-to-end
- Scalability: Horizontal scaling per agent type
- Reliability: 99.9% success rate with retry

**Key Concepts:**
- **CloudEvents**: Standard format for event data
- **Event Bus**: Kafka for pub/sub messaging
- **Autonomous Agents**: Self-contained processing stages
- **Correlation ID**: Track document through entire workflow
- **Dead Letter Queue**: Handle failed events

## Advanced Examples

### Deployment

#### Kubernetes with Helm

```bash
# Install Helm chart
helm install sap-llm ./helm/sap-llm \
  --namespace sap-llm \
  --create-namespace \
  --values ./helm/sap-llm/values.yaml
```

See `helm/sap-llm/README.md` for full deployment guide.

#### Docker Compose

```bash
# Build and run all services
docker-compose up -d

# Scale agents
docker-compose up -d --scale preprocessing-agent=5
```

### Integration with SAP Systems

#### SAP S/4HANA Integration

```python
from sap_llm.integrations.sap import SAPS4HANAConnector

# Connect to SAP S/4HANA
connector = SAPS4HANAConnector(
    host="sap.example.com",
    client="100",
    user="SAP_USER",
    password="PASSWORD",
)

# Post invoice to SAP
connector.post_invoice(extracted_fields)
```

#### SAP Ariba Integration

```python
from sap_llm.integrations.ariba import AribaConnector

# Connect to SAP Ariba
connector = AribaConnector(
    api_key="ARIBA_API_KEY",
    realm="your-realm",
)

# Create purchase order
connector.create_purchase_order(extracted_fields)
```

## Performance Tuning

### GPU Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use DeepSpeed ZeRO-3
from sap_llm.training.trainer import DistributedTrainer

trainer = DistributedTrainer(
    model=model,
    use_deepspeed=True,
    deepspeed_config="./configs/deepspeed_zero3.json"
)
```

### Inference Optimization

```python
# Load quantized model
from sap_llm.optimization.quantization import ModelQuantizer

quantized_model = ModelQuantizer.load_quantized_model(
    "./models/qwen2.5-vl-72b-gptq4",
    model_class=Qwen2VLForConditionalGeneration
)

# Use TensorRT for inference
from sap_llm.optimization.tensorrt_converter import TensorRTConverter

engine = TensorRTConverter.load_engine("./models/qwen2.5-vl-72b-tensorrt-fp16")
inference = TensorRTInference(engine)
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

**Symptoms:** CUDA OOM error during training/inference

**Solutions:**
```python
# Reduce batch size
config.batch_size = 4  # Instead of 16

# Enable gradient accumulation
config.gradient_accumulation_steps = 4

# Use mixed precision
config.fp16 = True  # or config.bf16 = True
```

#### Kafka Connection Failed

**Symptoms:** `NoBrokersAvailable` error

**Solutions:**
```bash
# Check Kafka is running
docker ps | grep kafka

# Restart Kafka
docker-compose restart kafka

# Update Kafka brokers in config
bus = CloudEventsBus(kafka_brokers="kafka1:9092,kafka2:9092,kafka3:9092")
```

#### Slow Inference

**Symptoms:** High latency (>500ms per document)

**Solutions:**
1. Use quantization (INT8 or GPTQ-4)
2. Enable TensorRT optimization
3. Batch multiple documents together
4. Use smaller model (distilled 7B)

## Best Practices

### Training

- **Use gradient checkpointing** for large models
- **Enable mixed precision** (BF16 on H100, FP16 on V100/A100)
- **Monitor GPU utilization** (should be >80%)
- **Save checkpoints frequently** (every 100 steps)
- **Use Weights & Biases** for experiment tracking

### Optimization

- **Always calibrate** quantized models with representative data
- **Benchmark before deploying** to production
- **Test accuracy** on validation set after optimization
- **Use TensorRT** for GPU inference
- **Use ONNX Runtime** for CPU inference

### APOP

- **Use correlation IDs** for workflow tracking
- **Enable DLQ** for failed events
- **Monitor Kafka lag** to detect processing bottlenecks
- **Scale agents independently** based on workload
- **Implement idempotent handlers** for retry safety

## Contributing

To add new examples:

1. Create example in appropriate category directory
2. Add docstring with description and requirements
3. Include expected results and performance metrics
4. Update this README
5. Test example works end-to-end
6. Submit pull request

## Support

- **Documentation**: https://docs.sap-llm.example.com
- **Issues**: https://github.com/AjithAccel4/SAP_LLM/issues
- **Slack**: https://sap-llm.slack.com

## License

See [LICENSE](../LICENSE) for details.
