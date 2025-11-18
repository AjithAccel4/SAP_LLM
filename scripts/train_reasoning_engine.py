#!/usr/bin/env python3
"""
Main Training Script for Reasoning Engine.

Complete pipeline:
1. Data Preparation (200K SAP routing examples)
2. Supervised Fine-Tuning (SFT) with QLoRA
3. RLHF/PPO Training with reward model
4. Evaluation and accuracy reporting

Usage:
    python scripts/train_reasoning_engine.py --stage all
    python scripts/train_reasoning_engine.py --stage data_prep
    python scripts/train_reasoning_engine.py --stage sft
    python scripts/train_reasoning_engine.py --stage rlhf
    python scripts/train_reasoning_engine.py --stage evaluate
"""

import argparse
import logging
from pathlib import Path

from sap_llm.training.data_preparation import SAPRoutingDatasetBuilder
from sap_llm.training.sft_trainer import ReasoningSFTTrainer, SFTConfig
from sap_llm.training.reasoning_rlhf_trainer import ReasoningRLHFTrainer, RLHFConfig
from sap_llm.evaluation.reasoning_evaluator import ReasoningEngineEvaluator
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


def stage_1_data_preparation(args):
    """Stage 1: Prepare training data."""
    logger.info("="*60)
    logger.info("STAGE 1: DATA PREPARATION")
    logger.info("="*60)

    builder = SAPRoutingDatasetBuilder(
        transaction_log_path=args.transaction_logs,
        api_schemas_path=args.api_schemas,
        pmg_data_path=args.pmg_data,
        output_dir=args.data_output_dir,
    )

    # Build dataset
    logger.info(f"Building dataset with {args.num_examples} examples...")
    train, val, test = builder.build_dataset(
        num_examples=args.num_examples,
        train_split=0.8,
        val_split=0.1,
    )

    # Create preference pairs for RLHF
    logger.info("Creating preference pairs for RLHF...")
    preference_pairs = builder.create_preference_pairs(train)

    logger.info("✅ Data preparation complete!")
    logger.info(f"  Train: {len(train)} examples")
    logger.info(f"  Val: {len(val)} examples")
    logger.info(f"  Test: {len(test)} examples")
    logger.info(f"  Preference pairs: {len(preference_pairs)}")


def stage_2_supervised_fine_tuning(args):
    """Stage 2: Supervised Fine-Tuning with QLoRA."""
    logger.info("="*60)
    logger.info("STAGE 2: SUPERVISED FINE-TUNING (SFT)")
    logger.info("="*60)

    config = SFTConfig(
        model_name=args.model_name,
        output_dir=args.sft_output_dir,
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.sft_learning_rate,
        max_length=args.max_length,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    trainer = ReasoningSFTTrainer(config)

    # Train
    logger.info("Starting SFT training...")
    trainer.train(
        train_data_file=f"{args.data_output_dir}/train_routing_examples.jsonl",
        val_data_file=f"{args.data_output_dir}/val_routing_examples.jsonl",
    )

    # Evaluate
    logger.info("Evaluating SFT model...")
    metrics = trainer.evaluate(
        test_data_file=f"{args.data_output_dir}/test_routing_examples.jsonl",
    )

    logger.info("✅ SFT training complete!")
    logger.info(f"  Endpoint Accuracy: {metrics['endpoint_accuracy'] * 100:.2f}%")
    logger.info(f"  Payload Accuracy: {metrics['payload_accuracy'] * 100:.2f}%")


def stage_3_rlhf_training(args):
    """Stage 3: RLHF/PPO Training."""
    logger.info("="*60)
    logger.info("STAGE 3: RLHF/PPO TRAINING")
    logger.info("="*60)

    config = RLHFConfig(
        model_path=f"{args.sft_output_dir}/final",
        reward_model_path=args.reward_model_path,
        output_dir=args.rlhf_output_dir,
        ppo_iterations=args.ppo_iterations,
        learning_rate=args.rlhf_learning_rate,
        kl_penalty=args.kl_penalty,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        api_success_weight=args.api_success_weight,
        business_rule_weight=args.business_rule_weight,
        confidence_weight=args.confidence_weight,
    )

    trainer = ReasoningRLHFTrainer(config)

    # Train
    logger.info("Starting RLHF training...")
    trainer.train(
        train_data_file=f"{args.data_output_dir}/train_routing_examples.jsonl",
        num_iterations=args.ppo_iterations,
    )

    logger.info("✅ RLHF training complete!")


def stage_4_evaluation(args):
    """Stage 4: Comprehensive Evaluation."""
    logger.info("="*60)
    logger.info("STAGE 4: EVALUATION")
    logger.info("="*60)

    evaluator = ReasoningEngineEvaluator(
        model_path=f"{args.rlhf_output_dir}/final" if args.evaluate_rlhf else f"{args.sft_output_dir}/final",
        device="cuda" if args.cuda else "cpu",
        precision=args.eval_precision,
    )

    # Evaluate
    logger.info("Running comprehensive evaluation...")
    metrics = evaluator.evaluate(
        test_data_file=f"{args.data_output_dir}/test_routing_examples.jsonl",
        num_samples=args.eval_samples,
        output_dir=args.eval_output_dir,
    )

    logger.info("✅ Evaluation complete!")
    logger.info(f"  Results saved to: {args.eval_output_dir}")

    # Check success criteria
    logger.info("\n" + "="*60)
    logger.info("SUCCESS CRITERIA VALIDATION")
    logger.info("="*60)

    criteria = [
        ("Routing Accuracy", metrics.routing_accuracy, 0.97, "≥97%"),
        ("API Selection Accuracy", metrics.api_selection_accuracy, 1.0, "100%"),
        ("Payload Accuracy", metrics.payload_accuracy, 0.99, "≥99%"),
        ("Inference Latency", metrics.avg_inference_latency_ms, 500, "<500ms"),
    ]

    all_pass = True
    for name, value, threshold, threshold_str in criteria:
        if name == "Inference Latency":
            passed = value < threshold
            value_str = f"{value:.2f}ms"
        else:
            passed = value >= threshold
            value_str = f"{value * 100:.2f}%"

        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{name}: {value_str} (target: {threshold_str}) - {status}")

        if not passed:
            all_pass = False

    logger.info("="*60)

    if all_pass:
        logger.info("🎉 ALL SUCCESS CRITERIA MET - READY FOR PRODUCTION!")
    else:
        logger.warning("⚠️  SOME CRITERIA NOT MET - ADDITIONAL TRAINING REQUIRED")

    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train Reasoning Engine")

    # Stage selection
    parser.add_argument(
        "--stage",
        choices=["all", "data_prep", "sft", "rlhf", "evaluate"],
        default="all",
        help="Training stage to run",
    )

    # Data preparation
    parser.add_argument("--transaction-logs", default="data/sap_transactions.jsonl")
    parser.add_argument("--api-schemas", default="data/sap_api_schemas.json")
    parser.add_argument("--pmg-data", default="data/pmg_routing_history.json")
    parser.add_argument("--data-output-dir", default="data/training/reasoning_engine")
    parser.add_argument("--num-examples", type=int, default=200000)

    # Model
    parser.add_argument("--model-name", default="mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--max-length", type=int, default=4096)

    # SFT training
    parser.add_argument("--sft-output-dir", default="./models/reasoning_engine")
    parser.add_argument("--sft-epochs", type=int, default=3)
    parser.add_argument("--sft-learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # RLHF training
    parser.add_argument("--rlhf-output-dir", default="./models/reasoning_engine_rlhf")
    parser.add_argument("--reward-model-path", default="./models/reasoning_reward_model.pt")
    parser.add_argument("--ppo-iterations", type=int, default=5000)
    parser.add_argument("--rlhf-learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-penalty", type=float, default=0.05)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--api-success-weight", type=float, default=0.7)
    parser.add_argument("--business-rule-weight", type=float, default=0.2)
    parser.add_argument("--confidence-weight", type=float, default=0.1)

    # Evaluation
    parser.add_argument("--eval-output-dir", default="./evaluation_results/reasoning_engine")
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--eval-precision", default="int8")
    parser.add_argument("--evaluate-rlhf", action="store_true", help="Evaluate RLHF model instead of SFT")

    # General
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run stages
    if args.stage in ["all", "data_prep"]:
        stage_1_data_preparation(args)

    if args.stage in ["all", "sft"]:
        stage_2_supervised_fine_tuning(args)

    if args.stage in ["all", "rlhf"]:
        stage_3_rlhf_training(args)

    if args.stage in ["all", "evaluate"]:
        stage_4_evaluation(args)

    logger.info("\n" + "="*60)
    logger.info("🎉 TRAINING PIPELINE COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
