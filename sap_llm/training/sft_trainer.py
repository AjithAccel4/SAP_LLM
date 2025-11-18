"""
Supervised Fine-Tuning (SFT) Trainer for Reasoning Engine.

Trains Mixtral-8x7B with QLoRA on SAP routing examples using:
- Chain-of-thought reasoning traces
- PMG context integration
- Routing decision accuracy
"""

import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel

from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SFTConfig:
    """SFT training configuration."""
    model_name: str = "mistralai/Mixtral-8x7B-v0.1"
    output_dir: str = "./models/reasoning_engine"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_length: int = 4096
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = True
    gradient_checkpointing: bool = True
    # LoRA config
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class RoutingDataset(Dataset):
    """
    Dataset for routing examples.

    Formats examples as:
    Input: Prompt with ADC + PMG context + API schemas
    Output: Routing decision + reasoning + payload
    """

    def __init__(
        self,
        data_file: Path,
        tokenizer: Any,
        max_length: int = 4096,
    ):
        """
        Initialize dataset.

        Args:
            data_file: Path to training data (JSONL)
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load examples
        self.examples = []
        with open(data_file) as f:
            for line in f:
                self.examples.append(json.loads(line))

        logger.info(f"Loaded {len(self.examples)} examples from {data_file}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training example."""
        example = self.examples[idx]

        # Create prompt
        prompt = self._create_prompt(example)

        # Create target (reasoning + decision)
        target = self._create_target(example)

        # Combine
        full_text = f"{prompt}{target}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Create labels (mask prompt, only train on target)
        labels = encoding["input_ids"].clone()

        # Tokenize prompt alone to find where to mask
        prompt_encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
        )
        prompt_length = len(prompt_encoding["input_ids"])

        # Mask prompt tokens
        labels[0, :prompt_length] = -100

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }

    def _create_prompt(self, example: Dict[str, Any]) -> str:
        """Create chain-of-thought prompt."""
        adc_json = example["adc_json"]
        doc_type = example["doc_type"]
        api_schemas = example["api_schemas"]
        similar_cases = example["similar_cases"]

        prompt = f"""You are an SAP routing expert. Given a document, determine the correct SAP API endpoint and generate the payload.

**Document Information:**
Type: {doc_type}
Supplier: {adc_json.get('supplier_name', 'Unknown')}
Company Code: {adc_json.get('company_code', 'Unknown')}
Total Amount: {adc_json.get('total_amount', 0)} {adc_json.get('currency', 'USD')}

**Extracted Data (ADC):**
{json.dumps(adc_json, indent=2)[:1500]}

**Available SAP APIs:**
{json.dumps([api['name'] for api in api_schemas], indent=2)}

**Similar Past Routings:**
{json.dumps(similar_cases[:3], indent=2)[:1000] if similar_cases else 'No similar cases found'}

**Task:**
1. Analyze the document type and extracted data
2. Consider similar past routing decisions
3. Select the appropriate SAP API endpoint
4. Explain your reasoning step-by-step
5. Provide a confidence score
6. Generate the SAP payload

**Reasoning and Decision:**
"""
        return prompt

    def _create_target(self, example: Dict[str, Any]) -> str:
        """Create target output (reasoning + decision)."""
        target_endpoint = example["target_endpoint"]
        target_payload = example["target_payload"]
        reasoning_trace = example["reasoning_trace"]
        confidence = example["confidence"]

        target = f"""{reasoning_trace}

**Final Decision (JSON):**
{{
  "endpoint": "{target_endpoint}",
  "method": "POST",
  "confidence": {confidence:.2f},
  "payload": {json.dumps(target_payload, indent=2)}
}}
"""
        return target


class ReasoningSFTTrainer:
    """
    Supervised fine-tuning trainer for Reasoning Engine.

    Training approach:
    1. Load Mixtral-8x7B with 4-bit quantization
    2. Apply QLoRA (4-bit quantized LoRA)
    3. Fine-tune on routing examples (10K steps)
    4. Evaluate routing accuracy
    """

    def __init__(self, config: SFTConfig):
        """
        Initialize SFT trainer.

        Args:
            config: Training configuration
        """
        self.config = config

        # Initialize model with QLoRA
        logger.info("Loading Mixtral-8x7B with QLoRA...")
        self.model = ReasoningEngine(
            model_name=config.model_name,
            device="cuda",
            precision="int4",
            use_lora=True,
            lora_config={
                "r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_dropout": config.lora_dropout,
                "bias": "none",
            },
        )

        self.tokenizer = self.model.tokenizer

        # Enable gradient checkpointing for memory efficiency
        if config.gradient_checkpointing:
            self.model.model.gradient_checkpointing_enable()

        logger.info("Model initialized with QLoRA")

    def train(
        self,
        train_data_file: str,
        val_data_file: str,
    ) -> None:
        """
        Train the reasoning engine.

        Args:
            train_data_file: Path to training data
            val_data_file: Path to validation data
        """
        logger.info("Starting SFT training...")

        # Create datasets
        train_dataset = RoutingDataset(
            Path(train_data_file),
            self.tokenizer,
            self.config.max_length,
        )

        val_dataset = RoutingDataset(
            Path(val_data_file),
            self.tokenizer,
            self.config.max_length,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            fp16=self.config.fp16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim="paged_adamw_8bit",  # Memory-efficient optimizer
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Create trainer
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()

        # Save final model
        logger.info(f"Saving final model to {self.config.output_dir}/final")
        trainer.save_model(f"{self.config.output_dir}/final")
        self.tokenizer.save_pretrained(f"{self.config.output_dir}/final")

        logger.info("Training complete!")

    def evaluate(
        self,
        test_data_file: str,
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_data_file: Path to test data

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating model...")

        # Load test data
        test_examples = []
        with open(test_data_file) as f:
            for line in f:
                test_examples.append(json.loads(line))

        # Metrics
        total = len(test_examples)
        correct_endpoint = 0
        correct_payload = 0
        total_confidence = 0.0

        # Evaluate
        self.model.model.eval()

        for example in tqdm(test_examples[:1000], desc="Evaluating"):  # Evaluate on 1000 examples
            # Create prompt
            prompt = self._create_eval_prompt(example)

            # Generate
            with torch.no_grad():
                response = self.model.generate(prompt, max_new_tokens=500)

            # Parse response
            decision = self._parse_response(response)

            # Check correctness
            if decision.get("endpoint") == example["target_endpoint"]:
                correct_endpoint += 1

            # Check payload (simplified - just check if key fields match)
            if self._check_payload_correctness(decision.get("payload", {}), example["target_payload"]):
                correct_payload += 1

            # Confidence
            total_confidence += decision.get("confidence", 0.0)

        # Compute metrics
        endpoint_accuracy = correct_endpoint / min(total, 1000)
        payload_accuracy = correct_payload / min(total, 1000)
        avg_confidence = total_confidence / min(total, 1000)

        metrics = {
            "endpoint_accuracy": endpoint_accuracy,
            "payload_accuracy": payload_accuracy,
            "avg_confidence": avg_confidence,
        }

        logger.info(f"Evaluation results:")
        logger.info(f"  Endpoint Accuracy: {endpoint_accuracy * 100:.2f}%")
        logger.info(f"  Payload Accuracy: {payload_accuracy * 100:.2f}%")
        logger.info(f"  Avg Confidence: {avg_confidence:.2f}")

        return metrics

    def _create_eval_prompt(self, example: Dict[str, Any]) -> str:
        """Create prompt for evaluation."""
        dataset = RoutingDataset.__new__(RoutingDataset)
        dataset.tokenizer = self.tokenizer
        return dataset._create_prompt(example)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response."""
        # Try to extract JSON
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {}

    def _check_payload_correctness(self, pred_payload: Dict[str, Any], target_payload: Dict[str, Any]) -> bool:
        """Check if predicted payload matches target."""
        # Simplified check - just verify key fields
        if not pred_payload or not target_payload:
            return False

        # Check if main fields are present
        pred_data = pred_payload.get("d", pred_payload)
        target_data = target_payload.get("d", target_payload)

        # Check at least 50% of target fields are present and correct
        matching_fields = 0
        total_fields = len(target_data)

        for key, value in target_data.items():
            if key in pred_data and pred_data[key] == value:
                matching_fields += 1

        return (matching_fields / max(total_fields, 1)) >= 0.5


def main():
    """Main training script."""
    # Configuration
    config = SFTConfig(
        model_name="mistralai/Mixtral-8x7B-v0.1",
        output_dir="./models/reasoning_engine",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Small batch due to model size
        gradient_accumulation_steps=8,  # Effective batch size = 16
        learning_rate=2e-5,
        max_length=4096,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        fp16=True,
        gradient_checkpointing=True,
    )

    # Initialize trainer
    trainer = ReasoningSFTTrainer(config)

    # Train
    trainer.train(
        train_data_file="data/training/reasoning_engine/train_routing_examples.jsonl",
        val_data_file="data/training/reasoning_engine/val_routing_examples.jsonl",
    )

    # Evaluate
    metrics = trainer.evaluate(
        test_data_file="data/training/reasoning_engine/test_routing_examples.jsonl",
    )

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Endpoint Accuracy: {metrics['endpoint_accuracy'] * 100:.2f}%")
    print(f"Payload Accuracy: {metrics['payload_accuracy'] * 100:.2f}%")
    print(f"Avg Confidence: {metrics['avg_confidence']:.2f}")
    print("="*50)


if __name__ == "__main__":
    main()
