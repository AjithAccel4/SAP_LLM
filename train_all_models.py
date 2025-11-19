#!/usr/bin/env python3
"""
Complete Model Training Orchestration Script for SAP_LLM.

This script orchestrates the training of all production models:
1. Vision Encoder (LayoutLMv3 fine-tuning)
2. Language Decoder (LLaMA-2 fine-tuning with LoRA)
3. Reasoning Engine (Mixtral RLHF training)

Based on 2025 best practices for enterprise LLM training.

Usage:
    python train_all_models.py --config config/training_config.yaml
    python train_all_models.py --stage vision  # Train only vision encoder
    python train_all_models.py --stage language  # Train only language decoder
    python train_all_models.py --stage reasoning  # Train only reasoning engine
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from sap_llm.training.trainer import Trainer
from sap_llm.training.sft_trainer import SFTTrainer
from sap_llm.training.rlhf_trainer import RLHFTrainer
from sap_llm.training.lora_trainer import LoRATrainer
from sap_llm.training.data_preparation import prepare_training_data
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates end-to-end model training pipeline.

    Features:
    - Multi-stage training (vision, language, reasoning)
    - Automatic checkpoint management
    - Distributed training support
    - Resource monitoring
    - Training resumption from checkpoints
    - Validation and evaluation
    """

    def __init__(self, config_path: str):
        """Initialize training orchestrator."""
        self.config_path = config_path
        self.config = self._load_config()

        # Setup directories
        self.output_dir = Path(self.config.get("output_dir", "outputs/models"))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"

        for dir_path in [self.output_dir, self.checkpoint_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Training state
        self.current_stage = None
        self.training_history = []

        logger.info(f"Training Orchestrator initialized")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            logger.info("Using default configuration")
            return self._get_default_config()

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            "output_dir": "outputs/models",
            "data_dir": "data/processed",
            "num_gpus": torch.cuda.device_count(),
            "mixed_precision": "bf16",

            # Vision Encoder (LayoutLMv3)
            "vision_encoder": {
                "enabled": True,
                "model_name": "microsoft/layoutlmv3-base",
                "num_epochs": 10,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "gradient_accumulation_steps": 4,
                "warmup_steps": 1000,
                "max_steps": 50000,
                "save_steps": 5000,
                "eval_steps": 1000,
            },

            # Language Decoder (LLaMA-2 + LoRA)
            "language_decoder": {
                "enabled": True,
                "model_name": "meta-llama/Llama-2-7b-hf",
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "num_epochs": 5,
                "batch_size": 4,
                "learning_rate": 2e-4,
                "gradient_accumulation_steps": 8,
                "max_length": 2048,
                "warmup_steps": 500,
                "save_steps": 2500,
                "eval_steps": 500,
                "use_flash_attention": True,
                "quantization": "int4",  # QLoRA
            },

            # Reasoning Engine (Mixtral RLHF)
            "reasoning_engine": {
                "enabled": True,
                "model_name": "mistralai/Mixtral-8x7B-v0.1",
                "num_epochs": 3,
                "batch_size": 2,
                "learning_rate": 1e-5,
                "gradient_accumulation_steps": 16,
                "ppo_epochs": 4,
                "kl_coef": 0.1,
                "clip_range": 0.2,
                "value_clip_range": 0.2,
                "use_lora": True,
                "lora_r": 16,
                "save_steps": 1000,
            }
        }

    def train_vision_encoder(self) -> Dict[str, Any]:
        """
        Train Vision Encoder (LayoutLMv3).

        Fine-tunes LayoutLMv3 for document image understanding.
        """
        logger.info("=" * 80)
        logger.info("STAGE 1: Training Vision Encoder (LayoutLMv3)")
        logger.info("=" * 80)

        self.current_stage = "vision_encoder"
        config = self.config["vision_encoder"]

        if not config.get("enabled", True):
            logger.info("Vision encoder training disabled in config")
            return {"status": "skipped"}

        # Prepare training data
        logger.info("Preparing training data...")
        train_dataset, eval_dataset = prepare_training_data(
            data_dir=self.config["data_dir"],
            task="document_understanding",
            model_type="vision"
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")

        # Initialize trainer
        trainer = SFTTrainer(
            model_name=config["model_name"],
            output_dir=str(self.checkpoint_dir / "vision_encoder"),
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"],
            save_steps=config["save_steps"],
            eval_steps=config["eval_steps"],
            fp16=self.config["mixed_precision"] == "fp16",
            bf16=self.config["mixed_precision"] == "bf16",
        )

        # Train
        logger.info("Starting vision encoder training...")
        start_time = datetime.now()

        results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Vision encoder training completed in {duration:.2f}s")

        # Save model
        output_path = self.output_dir / "vision_encoder_final"
        trainer.save_model(str(output_path))
        logger.info(f"Model saved to {output_path}")

        results["duration"] = duration
        results["stage"] = "vision_encoder"
        self.training_history.append(results)

        return results

    def train_language_decoder(self) -> Dict[str, Any]:
        """
        Train Language Decoder (LLaMA-2 with LoRA).

        Fine-tunes LLaMA-2 for structured JSON generation.
        Uses QLoRA for memory efficiency.
        """
        logger.info("=" * 80)
        logger.info("STAGE 2: Training Language Decoder (LLaMA-2 + LoRA)")
        logger.info("=" * 80)

        self.current_stage = "language_decoder"
        config = self.config["language_decoder"]

        if not config.get("enabled", True):
            logger.info("Language decoder training disabled in config")
            return {"status": "skipped"}

        # Prepare training data
        logger.info("Preparing training data...")
        train_dataset, eval_dataset = prepare_training_data(
            data_dir=self.config["data_dir"],
            task="field_extraction",
            model_type="language"
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")

        # Initialize LoRA trainer
        trainer = LoRATrainer(
            model_name=config["model_name"],
            output_dir=str(self.checkpoint_dir / "language_decoder"),
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=["q_proj", "v_proj"],  # LLaMA attention layers
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            max_length=config["max_length"],
            warmup_steps=config["warmup_steps"],
            save_steps=config["save_steps"],
            eval_steps=config["eval_steps"],
            load_in_4bit=config["quantization"] == "int4",
            load_in_8bit=config["quantization"] == "int8",
            use_flash_attention_2=config.get("use_flash_attention", False),
        )

        # Train
        logger.info("Starting language decoder training with LoRA...")
        start_time = datetime.now()

        results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Language decoder training completed in {duration:.2f}s")

        # Save model and LoRA adapters
        output_path = self.output_dir / "language_decoder_final"
        trainer.save_model(str(output_path))
        logger.info(f"Model and LoRA adapters saved to {output_path}")

        results["duration"] = duration
        results["stage"] = "language_decoder"
        self.training_history.append(results)

        return results

    def train_reasoning_engine(self) -> Dict[str, Any]:
        """
        Train Reasoning Engine (Mixtral with RLHF).

        Fine-tunes Mixtral for multi-step reasoning using RLHF.
        """
        logger.info("=" * 80)
        logger.info("STAGE 3: Training Reasoning Engine (Mixtral + RLHF)")
        logger.info("=" * 80)

        self.current_stage = "reasoning_engine"
        config = self.config["reasoning_engine"]

        if not config.get("enabled", True):
            logger.info("Reasoning engine training disabled in config")
            return {"status": "skipped"}

        # Prepare training data
        logger.info("Preparing training data...")
        train_dataset, eval_dataset = prepare_training_data(
            data_dir=self.config["data_dir"],
            task="reasoning",
            model_type="reasoning"
        )

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Evaluation samples: {len(eval_dataset)}")

        # Initialize RLHF trainer
        trainer = RLHFTrainer(
            model_name=config["model_name"],
            output_dir=str(self.checkpoint_dir / "reasoning_engine"),
            num_train_epochs=config["num_epochs"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"],
            ppo_epochs=config["ppo_epochs"],
            kl_coef=config["kl_coef"],
            clip_range=config["clip_range"],
            value_clip_range=config["value_clip_range"],
            use_lora=config.get("use_lora", True),
            lora_r=config.get("lora_r", 16),
            save_steps=config["save_steps"],
        )

        # Train
        logger.info("Starting reasoning engine training with RLHF...")
        start_time = datetime.now()

        results = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Reasoning engine training completed in {duration:.2f}s")

        # Save model
        output_path = self.output_dir / "reasoning_engine_final"
        trainer.save_model(str(output_path))
        logger.info(f"Model saved to {output_path}")

        results["duration"] = duration
        results["stage"] = "reasoning_engine"
        self.training_history.append(results)

        return results

    def train_all(self) -> Dict[str, Any]:
        """Train all models in sequence."""
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE MODEL TRAINING PIPELINE")
        logger.info("=" * 80)

        overall_start = datetime.now()

        # Stage 1: Vision Encoder
        vision_results = self.train_vision_encoder()

        # Stage 2: Language Decoder
        language_results = self.train_language_decoder()

        # Stage 3: Reasoning Engine
        reasoning_results = self.train_reasoning_engine()

        overall_duration = (datetime.now() - overall_start).total_seconds()

        # Summary
        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total duration: {overall_duration:.2f}s ({overall_duration/3600:.2f}h)")
        logger.info(f"Vision Encoder: {vision_results.get('status', 'completed')}")
        logger.info(f"Language Decoder: {language_results.get('status', 'completed')}")
        logger.info(f"Reasoning Engine: {reasoning_results.get('status', 'completed')}")

        return {
            "overall_duration": overall_duration,
            "vision_encoder": vision_results,
            "language_decoder": language_results,
            "reasoning_engine": reasoning_results,
            "training_history": self.training_history
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train SAP_LLM models")
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["vision", "language", "reasoning", "all"],
        default="all",
        help="Training stage to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )

    # Initialize orchestrator
    orchestrator = TrainingOrchestrator(args.config)

    if args.output_dir:
        orchestrator.output_dir = Path(args.output_dir)
        orchestrator.checkpoint_dir = orchestrator.output_dir / "checkpoints"
        orchestrator.logs_dir = orchestrator.output_dir / "logs"

        for dir_path in [orchestrator.output_dir, orchestrator.checkpoint_dir, orchestrator.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    # Run training
    try:
        if args.stage == "vision":
            results = orchestrator.train_vision_encoder()
        elif args.stage == "language":
            results = orchestrator.train_language_decoder()
        elif args.stage == "reasoning":
            results = orchestrator.train_reasoning_engine()
        else:  # all
            results = orchestrator.train_all()

        logger.info("Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
