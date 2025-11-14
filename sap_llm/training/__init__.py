"""
Model Training Module for SAP_LLM

Supports:
- Distributed training with FSDP (Fully Sharded Data Parallel)
- DeepSpeed ZeRO-3 optimization
- Mixed precision training (BF16, FP8)
- LoRA/QLoRA fine-tuning
- RLHF with PPO
- Continuous learning from PMG feedback
"""

from .trainer import DistributedTrainer
from .rlhf_trainer import RLHFTrainer
from .lora_trainer import LoRATrainer

__all__ = [
    "DistributedTrainer",
    "RLHFTrainer",
    "LoRATrainer",
]
