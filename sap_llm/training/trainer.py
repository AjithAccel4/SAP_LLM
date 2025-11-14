"""
Distributed model training with FSDP and DeepSpeed.

Supports training Qwen2.5-VL-72B on 16x H100 GPUs with:
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed ZeRO-3
- Mixed precision (BF16)
- Gradient checkpointing
- Model checkpointing and resumption
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logger.warning("DeepSpeed not available")

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    logger.warning("FSDP not available (requires PyTorch 2.0+)")


class DistributedTrainer:
    """
    Distributed trainer for SAP_LLM models.

    Features:
    - Multi-GPU training with FSDP or DeepSpeed
    - Mixed precision training (BF16)
    - Gradient accumulation
    - Learning rate scheduling
    - Checkpoint management
    - Weights & Biases integration
    """

    def __init__(self,
                 model: nn.Module,
                 train_dataset: Any,
                 val_dataset: Any,
                 output_dir: str,
                 config: Dict[str, Any],
                 use_deepspeed: bool = True):
        """
        Initialize distributed trainer.

        Args:
            model: Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory for checkpoints and logs
            config: Training configuration
            use_deepspeed: Use DeepSpeed if available
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Initialize distributed training
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.global_rank = int(os.environ.get("RANK", 0))

        if self.world_size > 1:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend="nccl")

        # Setup training backend
        self.use_deepspeed = use_deepspeed and DEEPSPEED_AVAILABLE
        if self.use_deepspeed:
            self._setup_deepspeed()
        elif FSDP_AVAILABLE and self.world_size > 1:
            self._setup_fsdp()
        else:
            self.model = self.model.cuda()

        # Initialize optimizer and scheduler (will be set up by DeepSpeed or manually)
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Weights & Biases logging
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb and self.global_rank == 0:
            try:
                import wandb
                wandb.init(
                    project=config.get("wandb_project", "sap-llm-training"),
                    name=config.get("run_name", f"training_{datetime.now().strftime('%Y%m%d_%H%M')}"),
                    config=config
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not available, skipping W&B logging")
                self.use_wandb = False

        logger.info(f"DistributedTrainer initialized: world_size={self.world_size}, rank={self.global_rank}")

    def _setup_deepspeed(self):
        """Setup DeepSpeed distributed training."""
        logger.info("Setting up DeepSpeed training")

        # DeepSpeed configuration
        ds_config = {
            "train_batch_size": self.config.get("batch_size", 32) * self.world_size,
            "train_micro_batch_size_per_gpu": self.config.get("batch_size", 32),
            "gradient_accumulation_steps": self.config.get("gradient_accumulation_steps", 1),
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.get("learning_rate", 5e-5),
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": self.config.get("weight_decay", 0.01)
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.config.get("learning_rate", 5e-5),
                    "warmup_num_steps": self.config.get("warmup_steps", 1000),
                    "total_num_steps": self.config.get("max_steps", 100000)
                }
            },
            "fp16": {
                "enabled": False
            },
            "bf16": {
                "enabled": self.config.get("use_bf16", True)
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "gather_16bit_weights_on_model_save": True
            },
            "gradient_clipping": self.config.get("max_grad_norm", 1.0),
            "steps_per_print": 10,
            "wall_clock_breakdown": False
        }

        # Save DeepSpeed config
        ds_config_path = self.output_dir / "ds_config.json"
        with open(ds_config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)

        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            config=ds_config
        )

    def _setup_fsdp(self):
        """Setup FSDP distributed training."""
        logger.info("Setting up FSDP training")

        # FSDP configuration
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Wrap model with FSDP
        self.model = FSDP(
            self.model,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01)
        )

        # Setup scheduler
        from torch.optim.lr_scheduler import OneCycleLR
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config.get("learning_rate", 5e-5),
            total_steps=self.config.get("max_steps", 100000),
            pct_start=0.1  # 10% warmup
        )

    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        Train model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            resume_from: Checkpoint path to resume from
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        # Create data loaders
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True
        ) if self.world_size > 1 else None

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.get("batch_size", 32),
            sampler=train_sampler,
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.get("batch_size", 32),
            num_workers=self.config.get("num_workers", 4),
            pin_memory=True
        )

        logger.info(f"Starting training: {num_epochs} epochs, {len(train_loader)} steps/epoch")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            if train_sampler:
                train_sampler.set_epoch(epoch)

            # Train epoch
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._validate(val_loader)

            # Log metrics
            if self.global_rank == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}"
                )

                if self.use_wandb:
                    self.wandb.log({
                        "epoch": epoch,
                        **{f"train/{k}": v for k, v in train_metrics.items()},
                        **{f"val/{k}": v for k, v in val_metrics.items()}
                    })

            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint("best_model")

            if (epoch + 1) % self.config.get("save_every", 5) == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")

        logger.info("Training complete!")

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        if self.use_deepspeed:
            self.model_engine.train()
        else:
            self.model.train()

        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if not self.use_deepspeed:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            # Forward pass
            if self.use_deepspeed:
                loss = self.model_engine(**batch).loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss

            # Backward pass
            if self.use_deepspeed:
                self.model_engine.backward(loss)
                self.model_engine.step()
            else:
                loss.backward()
                if self.config.get("max_grad_norm"):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["max_grad_norm"]
                    )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Log progress
            if batch_idx % 10 == 0 and self.global_rank == 0:
                logger.info(
                    f"Epoch {self.epoch}, Step {batch_idx}/{len(train_loader)}: "
                    f"loss={loss.item():.4f}"
                )

        return {"loss": total_loss / num_batches}

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        if self.use_deepspeed:
            self.model_engine.eval()
        else:
            self.model.eval()

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                if not self.use_deepspeed:
                    batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                             for k, v in batch.items()}

                if self.use_deepspeed:
                    loss = self.model_engine(**batch).loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss

                total_loss += loss.item()
                num_batches += 1

        return {"loss": total_loss / num_batches}

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        if self.global_rank != 0:
            return

        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.use_deepspeed:
            self.model_engine.save_checkpoint(str(checkpoint_dir))
        else:
            torch.save({
                'epoch': self.epoch,
                'global_step': self.global_step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_loss': self.best_val_loss,
            }, checkpoint_dir / "checkpoint.pt")

        # Save config
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Saved checkpoint: {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        if self.use_deepspeed:
            _, client_state = self.model_engine.load_checkpoint(str(checkpoint_path))
            if client_state:
                self.epoch = client_state.get('epoch', 0)
                self.global_step = client_state.get('global_step', 0)
                self.best_val_loss = client_state.get('best_val_loss', float('inf'))
        else:
            checkpoint = torch.load(checkpoint_path / "checkpoint.pt")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Loaded checkpoint from: {checkpoint_path}")


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = {
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "warmup_steps": 1000,
        "max_steps": 100000,
        "use_bf16": True,
        "num_workers": 4,
        "save_every": 5,
        "use_wandb": False,
    }

    # This would be your actual model and datasets
    # model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
    # train_dataset = SAP_LLM_Dataset("./data/processed/train")
    # val_dataset = SAP_LLM_Dataset("./data/processed/val")

    # trainer = DistributedTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     output_dir="./models/checkpoints",
    #     config=config,
    #     use_deepspeed=True
    # )

    # trainer.train(num_epochs=10)

    print("Distributed trainer module loaded successfully")
