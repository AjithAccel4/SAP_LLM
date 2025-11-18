#!/usr/bin/env python3
"""
Training script for Vision Encoder (LayoutLMv3-based).

This script trains the multi-task vision encoder for SAP_LLM with:
- Document type classification (15 classes)
- PO subtype classification (35 classes)
- Token classification for field extraction (180+ fields)

Training configuration:
- Model: LayoutLMv3-base (300M parameters)
- Hardware: 4x A100 80GB or 8x A10 40GB
- Batch size: 4 per GPU with gradient accumulation (8 steps)
- Learning rate: 5e-5 with 1000 warmup steps
- Max steps: 50,000 (~36 hours on 4x A100)
- Mixed precision: fp16

Usage:
    # Single GPU
    python scripts/train_vision_encoder.py --data_dir ./data/vision_encoder --output_dir ./models/vision_encoder

    # Multi-GPU with FSDP
    torchrun --nproc_per_node=4 scripts/train_vision_encoder.py \
        --data_dir ./data/vision_encoder \
        --output_dir ./models/vision_encoder \
        --use_fsdp

    # Multi-GPU with DeepSpeed
    deepspeed scripts/train_vision_encoder.py \
        --data_dir ./data/vision_encoder \
        --output_dir ./models/vision_encoder \
        --deepspeed \
        --deepspeed_config configs/deepspeed_config.json
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import LayoutLMv3Config, LayoutLMv3Processor
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sap_llm.models.vision_encoder import MultiTaskLayoutLMv3, export_to_onnx, benchmark_model
from sap_llm.training.vision_dataset import VisionEncoderDataset, create_synthetic_dataset
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)

# FSDP imports
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullStateDictConfig,
        StateDictConfig,
    )
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    logger.warning("FSDP not available (requires PyTorch 2.0+)")

# Weights & Biases import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Vision Encoder")

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing training data",
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default=None,
        help="Directory containing validation data (default: data_dir/../val)",
    )
    parser.add_argument(
        "--create_synthetic",
        action="store_true",
        help="Create synthetic dataset for testing",
    )
    parser.add_argument(
        "--synthetic_samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to create",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/layoutlmv3-base",
        help="Base model name or path",
    )
    parser.add_argument(
        "--num_doc_types",
        type=int,
        default=15,
        help="Number of document types",
    )
    parser.add_argument(
        "--num_po_subtypes",
        type=int,
        default=35,
        help="Number of PO subtypes",
    )
    parser.add_argument(
        "--num_token_labels",
        type=int,
        default=181,
        help="Number of token classification labels",
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models/vision_encoder",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides max_steps if set)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 mixed precision",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    # Distributed training arguments
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        help="Use FSDP for distributed training",
    )
    parser.add_argument(
        "--deepspeed",
        action="store_true",
        help="Use DeepSpeed for distributed training",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="DeepSpeed configuration file",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sap-llm-vision-encoder",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=5000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Checkpoint to resume from",
    )

    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export model to ONNX after training",
    )

    args = parser.parse_args()
    return args


def setup_distributed():
    """Setup distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def create_optimizer(model, args):
    """Create optimizer with weight decay."""
    # Separate parameters for weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    return optimizer


def create_scheduler(optimizer, args, num_training_steps):
    """Create learning rate scheduler."""
    from torch.optim.lr_scheduler import OneCycleLR

    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=num_training_steps,
        pct_start=args.warmup_steps / num_training_steps,
        anneal_strategy="linear",
    )

    return scheduler


def evaluate(model, dataloader, device, rank):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_doc_type_loss = 0
    total_po_subtype_loss = 0
    total_token_loss = 0
    num_batches = 0

    # Metrics
    doc_type_correct = 0
    po_subtype_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=rank != 0):
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)

            # Accumulate losses
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
            if outputs["doc_type_loss"] is not None:
                total_doc_type_loss += outputs["doc_type_loss"].item()
            if outputs["po_subtype_loss"] is not None:
                total_po_subtype_loss += outputs["po_subtype_loss"].item()
            if outputs["token_loss"] is not None:
                total_token_loss += outputs["token_loss"].item()

            # Calculate accuracy
            doc_type_preds = torch.argmax(outputs["doc_type_logits"], dim=-1)
            doc_type_correct += (doc_type_preds == batch["doc_type_labels"]).sum().item()

            po_subtype_preds = torch.argmax(outputs["po_subtype_logits"], dim=-1)
            po_subtype_correct += (po_subtype_preds == batch["po_subtype_labels"]).sum().item()

            total_samples += batch["doc_type_labels"].size(0)
            num_batches += 1

    # Calculate averages
    metrics = {
        "eval/loss": total_loss / num_batches if num_batches > 0 else 0,
        "eval/doc_type_loss": total_doc_type_loss / num_batches if num_batches > 0 else 0,
        "eval/po_subtype_loss": total_po_subtype_loss / num_batches if num_batches > 0 else 0,
        "eval/token_loss": total_token_loss / num_batches if num_batches > 0 else 0,
        "eval/doc_type_accuracy": doc_type_correct / total_samples if total_samples > 0 else 0,
        "eval/po_subtype_accuracy": po_subtype_correct / total_samples if total_samples > 0 else 0,
    }

    return metrics


def save_checkpoint(model, optimizer, scheduler, global_step, args, output_dir):
    """Save model checkpoint."""
    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving checkpoint to {checkpoint_dir}")

    # Save model
    if isinstance(model, FSDP):
        # FSDP save
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state_dict = model.state_dict()
            if torch.distributed.get_rank() == 0:
                torch.save(model_state_dict, checkpoint_dir / "pytorch_model.bin")
    else:
        # Regular save
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), checkpoint_dir / "pytorch_model.bin")

    # Save optimizer and scheduler
    if torch.distributed.get_rank() == 0:
        torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

        # Save training state
        training_state = {
            "global_step": global_step,
            "args": vars(args),
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)

    logger.info(f"Checkpoint saved: {checkpoint_dir}")


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Rank: {rank}, World size: {world_size}, Device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B
    if args.use_wandb and WANDB_AVAILABLE and rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Create synthetic dataset if requested
    if args.create_synthetic:
        logger.info("Creating synthetic dataset...")
        train_dir = Path(args.data_dir)
        val_dir = Path(args.val_data_dir) if args.val_data_dir else train_dir.parent / "val"

        create_synthetic_dataset(
            output_dir=train_dir,
            num_samples=args.synthetic_samples,
            split="train",
        )
        create_synthetic_dataset(
            output_dir=val_dir,
            num_samples=args.synthetic_samples // 5,
            split="val",
        )
        logger.info("Synthetic dataset created")

    # Load processor
    logger.info(f"Loading processor: {args.model_name}")
    processor = LayoutLMv3Processor.from_pretrained(args.model_name, apply_ocr=False)

    # Create datasets
    logger.info(f"Loading training data from {args.data_dir}")
    train_dataset = VisionEncoderDataset(
        data_dir=args.data_dir,
        processor=processor,
        max_length=args.max_length,
        mode="train",
    )

    val_data_dir = args.val_data_dir or str(Path(args.data_dir).parent / "val")
    logger.info(f"Loading validation data from {val_data_dir}")
    val_dataset = VisionEncoderDataset(
        data_dir=val_data_dir,
        processor=processor,
        max_length=args.max_length,
        mode="val",
    )

    # Create data loaders
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Calculate training steps
    if args.num_epochs:
        num_training_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation_steps
    else:
        num_training_steps = args.max_steps

    logger.info(f"Total training steps: {num_training_steps}")

    # Create model
    logger.info(f"Loading model: {args.model_name}")
    config = LayoutLMv3Config.from_pretrained(args.model_name)
    model = MultiTaskLayoutLMv3(
        config=config,
        num_doc_types=args.num_doc_types,
        num_po_subtypes=args.num_po_subtypes,
        num_token_labels=args.num_token_labels,
    )

    # Load pretrained weights
    from transformers import LayoutLMv3Model
    pretrained_model = LayoutLMv3Model.from_pretrained(args.model_name)
    model.layoutlmv3.load_state_dict(pretrained_model.state_dict())
    logger.info("Loaded pretrained LayoutLMv3 weights")

    # Move model to device
    model = model.to(device)

    # Wrap with FSDP if requested
    if args.use_fsdp and FSDP_AVAILABLE and world_size > 1:
        logger.info("Wrapping model with FSDP")
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16 if args.fp16 else torch.float32,
            reduce_dtype=torch.float16 if args.fp16 else torch.float32,
            buffer_dtype=torch.float16 if args.fp16 else torch.float32,
        )

        model = FSDP(
            model,
            mixed_precision=mixed_precision,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=torch.cuda.current_device(),
        )

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, num_training_steps)

    # Create gradient scaler for FP16
    scaler = torch.cuda.amp.GradScaler() if args.fp16 and not args.use_fsdp else None

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_eval_loss = float("inf")

    model.train()

    for epoch in range(args.num_epochs if args.num_epochs else 9999):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        epoch_loss = 0
        epoch_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=rank != 0)

        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            if args.fp16 and not args.use_fsdp:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs["loss"] / args.gradient_accumulation_steps
            else:
                outputs = model(**batch)
                loss = outputs["loss"] / args.gradient_accumulation_steps

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            epoch_steps += 1

            # Update weights
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % args.logging_steps == 0 and rank == 0:
                    avg_loss = epoch_loss / epoch_steps
                    lr = scheduler.get_last_lr()[0]

                    log_msg = f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                    logger.info(log_msg)
                    pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

                    if args.use_wandb and WANDB_AVAILABLE:
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        })

                # Evaluation
                if global_step % args.eval_steps == 0:
                    logger.info("Running evaluation...")
                    eval_metrics = evaluate(model, val_loader, device, rank)

                    if rank == 0:
                        logger.info(f"Eval metrics: {eval_metrics}")

                        if args.use_wandb and WANDB_AVAILABLE:
                            wandb.log({**eval_metrics, "train/global_step": global_step})

                        # Save best model
                        if eval_metrics["eval/loss"] < best_eval_loss:
                            best_eval_loss = eval_metrics["eval/loss"]
                            save_checkpoint(model, optimizer, scheduler, global_step, args, output_dir / "best")

                    model.train()

                # Save checkpoint
                if global_step % args.save_steps == 0 and rank == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step, args, output_dir)

                # Check if max steps reached
                if global_step >= num_training_steps:
                    break

        if global_step >= num_training_steps:
            break

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_metrics = evaluate(model, val_loader, device, rank)

    if rank == 0:
        logger.info(f"Final eval metrics: {eval_metrics}")

        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({**eval_metrics, "train/global_step": global_step})

    # Save final checkpoint
    if rank == 0:
        save_checkpoint(model, optimizer, scheduler, global_step, args, output_dir / "final")

        # Export to ONNX
        if args.export_onnx:
            logger.info("Exporting model to ONNX...")
            model_to_export = model.module if hasattr(model, "module") else model
            onnx_path = output_dir / "final" / "model.onnx"
            export_to_onnx(model_to_export, str(onnx_path))
            logger.info(f"Model exported to {onnx_path}")

    # Cleanup
    if args.use_wandb and WANDB_AVAILABLE and rank == 0:
        wandb.finish()

    cleanup_distributed()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
