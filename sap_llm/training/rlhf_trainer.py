"""
RLHF (Reinforcement Learning from Human Feedback) Trainer.

Implements:
- Reward model training on preference pairs
- PPO (Proximal Policy Optimization) for policy improvement
- Integration with Process Memory Graph for feedback collection
- Continuous learning loop

Based on InstructGPT/ChatGPT methodology.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """RLHF training configuration."""
    reward_model_path: str
    ppo_iterations: int = 1000
    learning_rate: float = 1e-6
    kl_penalty: float = 0.05
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95  # GAE lambda
    batch_size: int = 16
    epochs_per_iteration: int = 4


class RewardModel(nn.Module):
    """
    Reward model for RLHF.

    Takes document + extraction pair and outputs scalar reward score.
    Trained on human feedback (preferences, corrections).
    """

    def __init__(self, base_model, hidden_size: int = 4096):
        """
        Initialize reward model.

        Args:
            base_model: Base language model (frozen or fine-tuned)
            hidden_size: Hidden size for reward head
        """
        super().__init__()

        self.base_model = base_model

        # Freeze base model parameters (optional)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze last N layers for fine-tuning
        num_layers_to_unfreeze = 2
        for layer in list(self.base_model.model.layers)[-num_layers_to_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Reward head: hidden_size -> 1 (scalar reward)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                pixel_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: compute reward score.

        Args:
            input_ids: Tokenized extraction output
            attention_mask: Attention mask
            pixel_values: Optional document image

        Returns:
            Reward scores (batch_size,)
        """
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True
        )

        # Use last hidden state (CLS token or mean pooling)
        hidden_states = outputs.hidden_states[-1]

        # Mean pooling over sequence length
        pooled = hidden_states.mean(dim=1)  # (batch_size, hidden_size)

        # Compute reward
        reward = self.reward_head(pooled)  # (batch_size, 1)

        return reward.squeeze(-1)  # (batch_size,)


class RLHFTrainer:
    """
    RLHF trainer with reward model and PPO.

    Training pipeline:
    1. Collect preference pairs from PMG
    2. Train reward model on preferences
    3. Use reward model for PPO policy improvement
    4. Deploy improved model
    """

    def __init__(self,
                 policy_model: nn.Module,
                 reward_model: RewardModel,
                 config: RLHFConfig,
                 pmg_client: Optional[Any] = None):
        """
        Initialize RLHF trainer.

        Args:
            policy_model: Policy model to improve (Qwen2.5-VL)
            reward_model: Trained reward model
            config: RLHF configuration
            pmg_client: Process Memory Graph client for feedback
        """
        self.policy = policy_model
        self.reward_model = reward_model
        self.config = config
        self.pmg = pmg_client

        # Create reference model (frozen copy of initial policy)
        self.reference_model = self._create_reference_model()

        # Optimizer for policy
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Move models to GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self.policy.to(self.device)
        self.reward_model = self.reward_model.to(self.device)
        self.reference_model = self.reference_model.to(self.device)

        logger.info("RLHFTrainer initialized")

    def _create_reference_model(self) -> nn.Module:
        """Create frozen reference model (copy of initial policy)."""
        import copy
        reference = copy.deepcopy(self.policy)
        reference.eval()
        for param in reference.parameters():
            param.requires_grad = False
        return reference

    def train_reward_model(self,
                            preference_dataset: List[Dict[str, Any]],
                            num_epochs: int = 3,
                            batch_size: int = 8) -> Dict[str, float]:
        """
        Train reward model on preference pairs.

        Preference dataset format:
        [
            {
                "chosen": {"input_ids": ..., "attention_mask": ..., "pixel_values": ...},
                "rejected": {"input_ids": ..., "attention_mask": ..., "pixel_values": ...},
                "margin": 1.0  # reward(chosen) should be > reward(rejected) + margin
            },
            ...
        ]

        Args:
            preference_dataset: List of preference pairs
            num_epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training metrics
        """
        logger.info(f"Training reward model on {len(preference_dataset)} preference pairs")

        self.reward_model.train()
        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=5e-5)
        loss_fn = nn.MarginRankingLoss(margin=1.0)

        total_loss = 0
        num_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0

            for i in range(0, len(preference_dataset), batch_size):
                batch = preference_dataset[i:i+batch_size]

                # Prepare batch
                chosen_rewards = []
                rejected_rewards = []

                for pair in batch:
                    # Process chosen
                    chosen_inputs = {k: v.to(self.device) for k, v in pair["chosen"].items()}
                    r_chosen = self.reward_model(**chosen_inputs)
                    chosen_rewards.append(r_chosen)

                    # Process rejected
                    rejected_inputs = {k: v.to(self.device) for k, v in pair["rejected"].items()}
                    r_rejected = self.reward_model(**rejected_inputs)
                    rejected_rewards.append(r_rejected)

                chosen_rewards = torch.stack(chosen_rewards)
                rejected_rewards = torch.stack(rejected_rewards)

                # Ranking loss: reward(chosen) > reward(rejected)
                target = torch.ones_like(chosen_rewards)
                loss = loss_fn(chosen_rewards, rejected_rewards, target)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_epoch_loss = epoch_loss / (len(preference_dataset) / batch_size)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

            total_loss += epoch_loss

        # Save reward model
        torch.save(
            self.reward_model.state_dict(),
            self.config.reward_model_path
        )

        return {
            "total_loss": total_loss / num_batches,
            "num_batches": num_batches
        }

    def ppo_step(self, rollouts: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Perform one PPO update step.

        Rollouts format:
        [
            {
                "document_id": "doc_123",
                "states": torch.Tensor,  # Document representations
                "actions": torch.Tensor,  # Generated tokens
                "log_probs": torch.Tensor,  # Log probabilities
                "rewards": torch.Tensor,  # Rewards from reward model
                "values": torch.Tensor,  # Value estimates
            },
            ...
        ]

        Args:
            rollouts: List of rollout data

        Returns:
            PPO metrics
        """
        self.policy.train()

        # Compute advantages using GAE
        for rollout in rollouts:
            advantages = self._compute_gae(
                rollout["rewards"],
                rollout["values"],
                gamma=self.config.gamma,
                lam=self.config.lam
            )
            rollout["advantages"] = advantages
            rollout["returns"] = advantages + rollout["values"]

        # PPO epochs
        metrics = {"policy_loss": 0, "value_loss": 0, "kl_div": 0, "entropy": 0}

        for epoch in range(self.config.epochs_per_iteration):
            for rollout in rollouts:
                # Recompute log probs and values with current policy
                outputs = self.policy(
                    **rollout["inputs"],
                    output_hidden_states=True
                )

                new_log_probs = self._compute_log_probs(outputs, rollout["actions"])
                new_values = self._compute_values(outputs.hidden_states[-1])

                # Compute ratio: π_new / π_old
                ratio = torch.exp(new_log_probs - rollout["log_probs"])

                # Clipped surrogate objective
                surr1 = ratio * rollout["advantages"]
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * rollout["advantages"]

                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(new_values, rollout["returns"])

                # KL divergence from reference model
                with torch.no_grad():
                    ref_outputs = self.reference_model(**rollout["inputs"])
                    ref_log_probs = self._compute_log_probs(ref_outputs, rollout["actions"])

                kl_div = (rollout["log_probs"] - ref_log_probs).mean()

                # Entropy bonus (encourage exploration)
                entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()

                # Total loss
                loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.kl_penalty * kl_div -
                    self.config.entropy_coef * entropy
                )

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()

                # Track metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["kl_div"] += kl_div.item()
                metrics["entropy"] += entropy.item()

        # Average metrics
        num_updates = len(rollouts) * self.config.epochs_per_iteration
        for key in metrics:
            metrics[key] /= num_updates

        return metrics

    def _compute_gae(self,
                     rewards: torch.Tensor,
                     values: torch.Tensor,
                     gamma: float,
                     lam: float) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Rewards (T,)
            values: Value estimates (T+1,)
            gamma: Discount factor
            lam: GAE lambda

        Returns:
            Advantages (T,)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)

        last_gae = 0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t+1] - values[t]
            advantages[t] = last_gae = delta + gamma * lam * last_gae

        return advantages

    def _compute_log_probs(self, outputs: Any, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities of actions."""
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        # Gather log probs for taken actions
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        return action_log_probs

    def _compute_values(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute value estimates from hidden states."""
        # Simple value head (could be more sophisticated)
        pooled = hidden_states.mean(dim=1)
        values = torch.zeros(pooled.size(0), device=pooled.device)  # Placeholder

        return values

    def train(self, num_iterations: int = 1000):
        """
        Full RLHF training loop.

        Args:
            num_iterations: Number of PPO iterations
        """
        logger.info(f"Starting RLHF training: {num_iterations} iterations")

        for iteration in range(num_iterations):
            # 1. Generate rollouts
            rollouts = self._generate_rollouts(num_samples=self.config.batch_size)

            # 2. Compute rewards using reward model
            self._compute_rollout_rewards(rollouts)

            # 3. PPO update
            metrics = self.ppo_step(rollouts)

            # 4. Log metrics
            if iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}/{num_iterations}: "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"kl_div={metrics['kl_div']:.4f}"
                )

            # 5. Save checkpoint
            if (iteration + 1) % 100 == 0:
                self._save_checkpoint(f"ppo_iter_{iteration+1}")

        logger.info("RLHF training complete")

    def _generate_rollouts(self, num_samples: int) -> List[Dict[str, Any]]:
        """Generate rollouts using current policy."""
        # Placeholder - would sample documents and generate extractions
        rollouts = []
        return rollouts

    def _compute_rollout_rewards(self, rollouts: List[Dict[str, Any]]):
        """Compute rewards for rollouts using reward model."""
        self.reward_model.eval()

        with torch.no_grad():
            for rollout in rollouts:
                rewards = self.reward_model(**rollout["inputs"])
                rollout["rewards"] = rewards

    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint_path = f"./checkpoints/rlhf_{name}.pt"
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, checkpoint_path)

        logger.info(f"Saved checkpoint: {checkpoint_path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = RLHFConfig(
        reward_model_path="./models/reward_model_v1.pt",
        ppo_iterations=1000,
        learning_rate=1e-6,
        kl_penalty=0.05
    )

    # This would be your actual models
    # policy_model = Qwen2VLForConditionalGeneration.from_pretrained(...)
    # reward_model = RewardModel(base_model, hidden_size=4096)

    # trainer = RLHFTrainer(
    #     policy_model=policy_model,
    #     reward_model=reward_model,
    #     config=config
    # )

    # Train reward model
    # preference_dataset = collect_preferences_from_pmg()
    # trainer.train_reward_model(preference_dataset)

    # Run PPO
    # trainer.train(num_iterations=1000)

    print("RLHF trainer module loaded successfully")
