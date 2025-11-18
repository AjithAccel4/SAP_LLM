"""
RLHF Trainer for Reasoning Engine.

Implements Proximal Policy Optimization (PPO) with reward model for:
- SAP API success rate optimization
- Business rule compliance
- Routing decision quality improvement

Reward Signal Components:
1. SAP API success rate (70%)
2. Business rule compliance (20%)
3. Confidence calibration (10%)
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from tqdm import tqdm

from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RLHFConfig:
    """RLHF training configuration."""
    model_path: str  # Path to SFT checkpoint
    reward_model_path: str
    output_dir: str = "./models/reasoning_engine_rlhf"
    ppo_iterations: int = 5000
    learning_rate: float = 1e-6
    kl_penalty: float = 0.05
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    batch_size: int = 8
    mini_batch_size: int = 2
    ppo_epochs: int = 4
    max_length: int = 4096
    # Reward weights
    api_success_weight: float = 0.7
    business_rule_weight: float = 0.2
    confidence_weight: float = 0.1


class RoutingRewardModel(nn.Module):
    """
    Reward model for SAP routing decisions.

    Reward components:
    1. API Success Rate: +1.0 for successful routing, -1.0 for failed
    2. Business Rule Compliance: +0.5 for each satisfied rule
    3. Confidence Calibration: Penalize over/under confidence
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int = 4096,
    ):
        """
        Initialize reward model.

        Args:
            base_model: Base Mixtral model
            hidden_size: Hidden dimension
        """
        super().__init__()

        self.base_model = base_model

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reward score.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Reward scores (batch_size,)
        """
        # Get base model outputs
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Use last hidden state
        hidden_states = outputs.hidden_states[-1]

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_hidden / sum_mask

        # Compute reward
        reward = self.reward_head(pooled)

        return reward.squeeze(-1)


class ReasoningRLHFTrainer:
    """
    RLHF trainer for Reasoning Engine using PPO.

    Training loop:
    1. Sample batch of routing problems
    2. Generate solutions with current policy
    3. Compute rewards (API success + business rules)
    4. Update policy with PPO
    """

    def __init__(
        self,
        config: RLHFConfig,
        sap_api_simulator: Optional[Any] = None,
    ):
        """
        Initialize RLHF trainer.

        Args:
            config: Training configuration
            sap_api_simulator: Simulator for SAP API calls (for reward computation)
        """
        self.config = config
        self.sap_api_simulator = sap_api_simulator

        # Load SFT model as policy
        logger.info(f"Loading policy model from {config.model_path}")
        self.policy = ReasoningEngine.load(
            config.model_path,
            device="cuda",
            precision="int4",
        )

        # Enable training mode for LoRA layers
        for name, param in self.policy.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        # Create reference model (frozen copy)
        logger.info("Creating reference model...")
        self.reference_policy = ReasoningEngine.load(
            config.model_path,
            device="cuda",
            precision="int4",
        )
        self.reference_policy.model.eval()
        for param in self.reference_policy.model.parameters():
            param.requires_grad = False

        # Load or create reward model
        logger.info("Initializing reward model...")
        if Path(config.reward_model_path).exists():
            self.reward_model = self._load_reward_model()
        else:
            logger.warning("Reward model not found, creating new one")
            self.reward_model = RoutingRewardModel(
                base_model=self.reference_policy.model,
                hidden_size=4096,
            )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Value head for advantage estimation
        self.value_head = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        ).cuda()

        self.value_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=config.learning_rate * 10,
        )

        logger.info("RLHF Trainer initialized")

    def train(
        self,
        train_data_file: str,
        num_iterations: Optional[int] = None,
    ) -> None:
        """
        Run RLHF training.

        Args:
            train_data_file: Path to training data
            num_iterations: Number of PPO iterations (default: config.ppo_iterations)
        """
        if num_iterations is None:
            num_iterations = self.config.ppo_iterations

        logger.info(f"Starting RLHF training for {num_iterations} iterations")

        # Load training examples
        train_examples = []
        with open(train_data_file) as f:
            for line in f:
                train_examples.append(json.loads(line))

        logger.info(f"Loaded {len(train_examples)} training examples")

        # Training loop
        for iteration in range(num_iterations):
            # Sample batch
            batch = np.random.choice(train_examples, size=self.config.batch_size, replace=False)

            # Generate rollouts
            rollouts = self._generate_rollouts(batch)

            # Compute rewards
            self._compute_rewards(rollouts)

            # PPO update
            metrics = self._ppo_update(rollouts)

            # Logging
            if iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}/{num_iterations}: "
                    f"reward={metrics['avg_reward']:.3f}, "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"kl={metrics['kl_div']:.4f}"
                )

            # Save checkpoint
            if (iteration + 1) % 500 == 0:
                self._save_checkpoint(f"iter_{iteration + 1}")

        logger.info("RLHF training complete!")

        # Save final model
        self._save_checkpoint("final")

    def _generate_rollouts(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate rollouts for batch.

        Args:
            batch: Batch of routing problems

        Returns:
            List of rollouts with states, actions, log_probs
        """
        rollouts = []

        self.policy.model.eval()

        for example in batch:
            # Create prompt
            prompt = self._create_prompt(example)

            # Tokenize
            inputs = self.policy.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
            ).to(self.policy.device)

            # Generate with log probs
            with torch.no_grad():
                outputs = self.policy.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Get generated tokens
            generated_ids = outputs.sequences[0, inputs.input_ids.shape[1]:]

            # Compute log probs
            log_probs = self._compute_log_probs(outputs.scores, generated_ids)

            # Decode response
            response = self.policy.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Parse decision
            decision = self._parse_decision(response)

            rollout = {
                "example": example,
                "prompt": prompt,
                "inputs": inputs,
                "generated_ids": generated_ids,
                "log_probs": log_probs,
                "response": response,
                "decision": decision,
            }

            rollouts.append(rollout)

        return rollouts

    def _compute_log_probs(self, scores: Tuple[torch.Tensor], generated_ids: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for generated tokens."""
        log_probs = []

        for i, score in enumerate(scores):
            if i >= len(generated_ids):
                break

            # Get log prob for generated token
            log_prob = F.log_softmax(score[0], dim=-1)
            token_log_prob = log_prob[generated_ids[i]]
            log_probs.append(token_log_prob)

        return torch.stack(log_probs)

    def _compute_rewards(self, rollouts: List[Dict[str, Any]]) -> None:
        """
        Compute rewards for rollouts.

        Reward = w1 * API_success + w2 * business_rules + w3 * confidence_calibration
        """
        for rollout in rollouts:
            decision = rollout["decision"]
            example = rollout["example"]

            # Component 1: API Success (simulated)
            api_success_reward = self._compute_api_success_reward(decision, example)

            # Component 2: Business Rule Compliance
            business_rule_reward = self._compute_business_rule_reward(decision, example)

            # Component 3: Confidence Calibration
            confidence_reward = self._compute_confidence_reward(decision, example)

            # Combined reward
            total_reward = (
                self.config.api_success_weight * api_success_reward +
                self.config.business_rule_weight * business_rule_reward +
                self.config.confidence_weight * confidence_reward
            )

            rollout["reward"] = total_reward
            rollout["reward_components"] = {
                "api_success": api_success_reward,
                "business_rules": business_rule_reward,
                "confidence": confidence_reward,
            }

    def _compute_api_success_reward(self, decision: Dict[str, Any], example: Dict[str, Any]) -> float:
        """Compute reward for API routing correctness."""
        # Check if endpoint matches
        if decision.get("endpoint") == example["target_endpoint"]:
            # Correct endpoint: +1.0
            return 1.0
        else:
            # Wrong endpoint: -1.0 (critical error)
            return -1.0

    def _compute_business_rule_reward(self, decision: Dict[str, Any], example: Dict[str, Any]) -> float:
        """Compute reward for business rule compliance."""
        reward = 0.0

        # Check payload correctness
        payload = decision.get("payload", {})
        target_payload = example["target_payload"]

        # Rule 1: All required fields present
        payload_data = payload.get("d", payload)
        target_data = target_payload.get("d", target_payload)

        required_fields = ["CompanyCode", "DocumentDate"]
        for field in required_fields:
            if field in payload_data and field in target_data:
                if payload_data[field] == target_data[field]:
                    reward += 0.25  # +0.25 for each correct required field

        # Rule 2: Currency matches
        if payload_data.get("Currency") == target_data.get("Currency"):
            reward += 0.25

        # Rule 3: Amount matches
        if payload_data.get("TotalAmount") == target_data.get("TotalAmount"):
            reward += 0.25

        return reward

    def _compute_confidence_reward(self, decision: Dict[str, Any], example: Dict[str, Any]) -> float:
        """Compute reward for confidence calibration."""
        confidence = decision.get("confidence", 0.5)

        # Check if decision is correct
        correct = decision.get("endpoint") == example["target_endpoint"]

        if correct:
            # Reward high confidence for correct decisions
            return confidence
        else:
            # Penalize high confidence for wrong decisions
            return -(confidence)

    def _ppo_update(self, rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Perform PPO update.

        Args:
            rollouts: Rollouts with rewards

        Returns:
            Training metrics
        """
        self.policy.model.train()

        # Compute advantages
        for rollout in rollouts:
            # Simple advantage: just use reward (no value baseline for simplicity)
            rollout["advantage"] = rollout["reward"]
            rollout["return"] = rollout["reward"]

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl_div = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            # Shuffle rollouts
            np.random.shuffle(rollouts)

            # Mini-batches
            for i in range(0, len(rollouts), self.config.mini_batch_size):
                mini_batch = rollouts[i:i + self.config.mini_batch_size]

                # Compute policy loss
                policy_loss = 0.0
                kl_div = 0.0
                entropy = 0.0

                for rollout in mini_batch:
                    # Recompute log probs with current policy
                    outputs = self.policy.model(
                        **rollout["inputs"],
                        output_hidden_states=True,
                    )

                    # Get log probs for generated tokens
                    logits = outputs.logits[0, -len(rollout["generated_ids"]):]
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Get log probs for actual tokens
                    new_log_probs = []
                    for j, token_id in enumerate(rollout["generated_ids"]):
                        if j < len(log_probs):
                            new_log_probs.append(log_probs[j, token_id])

                    if not new_log_probs:
                        continue

                    new_log_probs = torch.stack(new_log_probs)
                    old_log_probs = rollout["log_probs"]

                    # Ensure same length
                    min_len = min(len(new_log_probs), len(old_log_probs))
                    new_log_probs = new_log_probs[:min_len]
                    old_log_probs = old_log_probs[:min_len]

                    # Compute ratio
                    ratio = torch.exp(new_log_probs - old_log_probs)

                    # Advantage
                    advantage = torch.tensor(rollout["advantage"], device=ratio.device)

                    # Clipped surrogate
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.config.clip_epsilon,
                        1 + self.config.clip_epsilon,
                    ) * advantage

                    policy_loss -= torch.min(surr1, surr2).mean()

                    # KL divergence
                    kl_div += (old_log_probs - new_log_probs).mean()

                    # Entropy
                    probs = torch.exp(new_log_probs)
                    entropy += -(probs * new_log_probs).mean()

                # Average over mini-batch
                policy_loss /= len(mini_batch)
                kl_div /= len(mini_batch)
                entropy /= len(mini_batch)

                # Total loss
                loss = policy_loss + self.config.kl_penalty * kl_div - self.config.entropy_coef * entropy

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.model.parameters() if p.requires_grad],
                    1.0,
                )
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_kl_div += kl_div.item()
                total_entropy += entropy.item()
                num_updates += 1

        # Compute average metrics
        avg_reward = np.mean([r["reward"] for r in rollouts])

        return {
            "avg_reward": avg_reward,
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "kl_div": total_kl_div / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
        }

    def _create_prompt(self, example: Dict[str, Any]) -> str:
        """Create prompt from example."""
        adc_json = example["adc_json"]
        doc_type = example["doc_type"]
        api_schemas = example["api_schemas"]
        similar_cases = example.get("similar_cases", [])

        return self.policy.create_routing_prompt(
            adc_json=adc_json,
            doc_type=doc_type,
            api_schemas=api_schemas,
            similar_cases=similar_cases,
        )

    def _parse_decision(self, response: str) -> Dict[str, Any]:
        """Parse decision from model response."""
        import re

        # Try to extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {}

    def _load_reward_model(self) -> RoutingRewardModel:
        """Load reward model from checkpoint."""
        model = RoutingRewardModel(
            base_model=self.reference_policy.model,
            hidden_size=4096,
        )

        checkpoint = torch.load(self.config.reward_model_path)
        model.load_state_dict(checkpoint)

        return model.cuda()

    def _save_checkpoint(self, name: str) -> None:
        """Save training checkpoint."""
        output_dir = Path(self.config.output_dir) / name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save policy
        self.policy.save(str(output_dir))

        logger.info(f"Saved checkpoint to {output_dir}")


def main():
    """Main RLHF training script."""
    config = RLHFConfig(
        model_path="./models/reasoning_engine/final",
        reward_model_path="./models/reasoning_reward_model.pt",
        output_dir="./models/reasoning_engine_rlhf",
        ppo_iterations=5000,
        learning_rate=1e-6,
        kl_penalty=0.05,
        batch_size=8,
        mini_batch_size=2,
    )

    trainer = ReasoningRLHFTrainer(config)

    trainer.train(
        train_data_file="data/training/reasoning_engine/train_routing_examples.jsonl",
    )

    print("\n" + "="*50)
    print("RLHF TRAINING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
