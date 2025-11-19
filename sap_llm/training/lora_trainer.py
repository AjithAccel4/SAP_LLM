"""
LoRA (Low-Rank Adaptation) Trainer for Efficient Fine-Tuning.

Enables parameter-efficient fine-tuning by training small adapter layers
instead of the full model.
"""

import logging
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT library not available. LoRA training will be disabled.")

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    LoRA trainer for efficient model fine-tuning.

    LoRA (Low-Rank Adaptation) trains small adapter matrices instead of
    full model parameters, reducing:
    - Training time (10-100x faster)
    - Memory usage (3-10x less)
    - Storage (100-1000x smaller)
    """

    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[list] = None
    ):
        """
        Initialize LoRA trainer.

        Args:
            lora_r: LoRA rank (typical: 8-64)
            lora_alpha: LoRA alpha (typically 2x rank)
            lora_dropout: Dropout for LoRA layers
            target_modules: Modules to apply LoRA to
        """
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library required for LoRA training. "
                "Install with: pip install peft"
            )

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]

        logger.info(
            f"LoRATrainer initialized "
            f"(r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})"
        )

    def prepare_model_for_lora(
        self,
        model: torch.nn.Module,
        task_type: str = "SEQ_CLS"
    ) -> torch.nn.Module:
        """
        Prepare model for LoRA training.

        Args:
            model: Base model
            task_type: Task type for LoRA

        Returns:
            Model with LoRA adapters
        """
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=task_type,
            inference_mode=False
        )

        # Apply LoRA to model
        model = get_peft_model(model, lora_config)

        # Count parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())

        logger.info(
            f"LoRA model prepared: "
            f"{trainable_params:,} trainable parameters "
            f"({100 * trainable_params / total_params:.2f}% of total)"
        )

        return model

    def train_with_lora(
        self,
        base_model: torch.nn.Module,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> torch.nn.Module:
        """
        Train model with LoRA.

        Args:
            base_model: Base model to fine-tune
            training_data: Training dataset
            validation_data: Optional validation dataset
            config: Training configuration

        Returns:
            Fine-tuned model with LoRA adapters
        """
        logger.info("Starting LoRA training...")

        # Default config
        default_config = {
            "learning_rate": 3e-4,
            "batch_size": 32,
            "num_epochs": 3,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0
        }
        config = {**default_config, **(config or {})}

        # Prepare model
        model = self.prepare_model_for_lora(base_model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Prepare data loaders
        train_loader = DataLoader(
            training_data,
            batch_size=config["batch_size"],
            shuffle=True
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        # Training loop
        model.train()
        for epoch in range(config["num_epochs"]):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                # Backward pass
                loss = loss / config["gradient_accumulation_steps"]
                loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config["max_grad_norm"]
                    )

                    optimizer.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * config["gradient_accumulation_steps"]
                num_batches += 1

                if batch_idx % 100 == 0:
                    logger.info(
                        f"Epoch {epoch+1}/{config['num_epochs']}, "
                        f"Batch {batch_idx}/{len(train_loader)}, "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")

            # Validation
            if validation_data is not None:
                val_loss = self._validate(model, validation_data, config, device)
                logger.info(f"Validation loss: {val_loss:.4f}")

        logger.info("LoRA training completed")
        return model

    def _validate(
        self,
        model: torch.nn.Module,
        validation_data: Dataset,
        config: Dict[str, Any],
        device: torch.device
    ) -> float:
        """Run validation."""
        model.eval()
        val_loader = DataLoader(
            validation_data,
            batch_size=config["batch_size"],
            shuffle=False
        )

        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                total_loss += loss.item()
                num_batches += 1

        model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0

    def merge_and_unload(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Merge LoRA weights into base model.

        Args:
            model: Model with LoRA adapters

        Returns:
            Base model with merged weights
        """
        if hasattr(model, 'merge_and_unload'):
            return model.merge_and_unload()
        return model
