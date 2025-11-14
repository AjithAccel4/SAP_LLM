"""
Knowledge distillation for SAP_LLM.

Transfer knowledge from large teacher model to smaller student model.

Distillation methods:
- Response-based distillation (output probabilities)
- Feature-based distillation (intermediate representations)
- Relation-based distillation (layer relationships)

Typical results:
- 72B teacher ’ 7B student: 90-95% performance retention
- 72B teacher ’ 3B student: 85-90% performance retention
- Inference: 10× faster, 10× smaller

Based on:
- DistilBERT: https://arxiv.org/abs/1910.01108
- TinyBERT: https://arxiv.org/abs/1909.10351
- MiniLM: https://arxiv.org/abs/2002.10957
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Knowledge distillation configuration."""
    temperature: float = 3.0  # Softmax temperature (higher = softer probabilities)
    alpha: float = 0.5  # Weight for distillation loss vs student loss
    distill_layer_mapping: Optional[Dict[int, int]] = None  # Teacher layer ’ student layer
    feature_distill: bool = True  # Enable feature-based distillation
    response_distill: bool = True  # Enable response-based distillation
    attention_distill: bool = False  # Enable attention distillation
    cosine_distill: bool = False  # Use cosine similarity for features


class KnowledgeDistiller:
    """
    Knowledge distillation trainer.

    Trains smaller student model to mimic larger teacher model.

    Distillation loss:
        L = ± × L_distill + (1-±) × L_student
        L_distill = KL(softmax(teacher/T), softmax(student/T))
        L_student = CrossEntropy(student_logits, true_labels)
    """

    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 config: DistillationConfig):
        """
        Initialize distiller.

        Args:
            teacher_model: Large pretrained model (72B)
            student_model: Small model to train (7B/3B)
            config: Distillation configuration
        """
        self.teacher = teacher_model
        self.student = student_model
        self.config = config

        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.device = next(teacher_model.parameters()).device

        # Move models to device
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)

        logger.info(f"KnowledgeDistiller initialized: T={config.temperature}, ±={config.alpha}")

    def distill(self,
                train_loader: Any,
                optimizer: torch.optim.Optimizer,
                num_epochs: int = 10,
                save_path: Optional[str] = None) -> nn.Module:
        """
        Distill knowledge from teacher to student.

        Args:
            train_loader: Training data
            optimizer: Optimizer for student
            num_epochs: Number of training epochs
            save_path: Path to save distilled model

        Returns:
            Distilled student model
        """
        logger.info(f"Starting knowledge distillation: {num_epochs} epochs")

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in train_loader:
                loss = self._distillation_step(batch, optimizer)
                epoch_loss += loss
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, loss: {avg_loss:.4f}")

            # Save checkpoint
            if save_path and (epoch + 1) % 5 == 0:
                self._save_checkpoint(save_path, epoch + 1)

        # Save final model
        if save_path:
            self._save_distilled_model(save_path)

        logger.info("Knowledge distillation complete")

        return self.student

    def _distillation_step(self,
                           batch: Dict[str, torch.Tensor],
                           optimizer: torch.optim.Optimizer) -> float:
        """
        Single distillation training step.

        Args:
            batch: Training batch
            optimizer: Optimizer

        Returns:
            Loss value
        """
        # Prepare inputs
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
        labels = batch.get('labels', None)
        if labels is not None:
            labels = labels.to(self.device)

        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                **inputs,
                output_hidden_states=True,
                output_attentions=self.config.attention_distill
            )

        # Student forward pass
        student_outputs = self.student(
            **inputs,
            output_hidden_states=True,
            output_attentions=self.config.attention_distill
        )

        # Compute distillation losses
        total_loss = 0

        # 1. Response-based distillation (logits)
        if self.config.response_distill:
            response_loss = self._compute_response_loss(
                teacher_outputs.logits,
                student_outputs.logits
            )
            total_loss += self.config.alpha * response_loss

        # 2. Feature-based distillation (hidden states)
        if self.config.feature_distill:
            feature_loss = self._compute_feature_loss(
                teacher_outputs.hidden_states,
                student_outputs.hidden_states
            )
            total_loss += 0.1 * feature_loss  # Smaller weight for feature loss

        # 3. Attention distillation
        if self.config.attention_distill:
            attention_loss = self._compute_attention_loss(
                teacher_outputs.attentions,
                student_outputs.attentions
            )
            total_loss += 0.1 * attention_loss

        # 4. Student loss (ground truth)
        if labels is not None:
            student_loss = F.cross_entropy(
                student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            total_loss += (1 - self.config.alpha) * student_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        optimizer.step()

        return total_loss.item()

    def _compute_response_loss(self,
                                teacher_logits: torch.Tensor,
                                student_logits: torch.Tensor) -> torch.Tensor:
        """
        Response-based distillation loss (KL divergence).

        Args:
            teacher_logits: Teacher logits (batch, seq_len, vocab_size)
            student_logits: Student logits (batch, seq_len, vocab_size)

        Returns:
            KL divergence loss
        """
        T = self.config.temperature

        # Soften probabilities with temperature
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)

        # KL divergence: D_KL(teacher || student)
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        )

        # Scale by T^2 (as per Hinton et al.)
        kl_loss = kl_loss * (T ** 2)

        return kl_loss

    def _compute_feature_loss(self,
                               teacher_hiddens: Tuple[torch.Tensor],
                               student_hiddens: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Feature-based distillation loss (MSE on hidden states).

        Args:
            teacher_hiddens: Teacher hidden states (num_layers tuples)
            student_hiddens: Student hidden states (num_layers tuples)

        Returns:
            MSE loss between aligned hidden states
        """
        # Map teacher layers to student layers
        # e.g., 32 teacher layers ’ 12 student layers (every 3rd layer)
        num_teacher_layers = len(teacher_hiddens)
        num_student_layers = len(student_hiddens)

        if self.config.distill_layer_mapping is None:
            # Auto-create layer mapping (evenly spaced)
            stride = num_teacher_layers // num_student_layers
            layer_mapping = {
                i * stride: i
                for i in range(num_student_layers)
            }
        else:
            layer_mapping = self.config.distill_layer_mapping

        # Compute MSE for aligned layers
        feature_loss = 0
        num_aligned_layers = 0

        for teacher_idx, student_idx in layer_mapping.items():
            if teacher_idx >= num_teacher_layers or student_idx >= num_student_layers:
                continue

            teacher_hidden = teacher_hiddens[teacher_idx]
            student_hidden = student_hiddens[student_idx]

            # Project student hidden to teacher dimension if needed
            if teacher_hidden.size(-1) != student_hidden.size(-1):
                # Use linear projection
                student_hidden = self._project_hidden(
                    student_hidden,
                    teacher_hidden.size(-1)
                )

            # Compute loss
            if self.config.cosine_distill:
                # Cosine similarity loss
                loss = 1 - F.cosine_similarity(
                    teacher_hidden.view(-1, teacher_hidden.size(-1)),
                    student_hidden.view(-1, student_hidden.size(-1)),
                    dim=-1
                ).mean()
            else:
                # MSE loss
                loss = F.mse_loss(student_hidden, teacher_hidden)

            feature_loss += loss
            num_aligned_layers += 1

        if num_aligned_layers > 0:
            feature_loss /= num_aligned_layers

        return feature_loss

    def _compute_attention_loss(self,
                                 teacher_attentions: Tuple[torch.Tensor],
                                 student_attentions: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Attention distillation loss (MSE on attention maps).

        Args:
            teacher_attentions: Teacher attention maps
            student_attentions: Student attention maps

        Returns:
            MSE loss between attention maps
        """
        num_teacher_layers = len(teacher_attentions)
        num_student_layers = len(student_attentions)

        # Map layers
        stride = num_teacher_layers // num_student_layers

        attention_loss = 0
        num_layers = 0

        for i in range(num_student_layers):
            teacher_idx = i * stride
            student_attention = student_attentions[i]
            teacher_attention = teacher_attentions[teacher_idx]

            # Average over heads if needed
            if student_attention.size(1) != teacher_attention.size(1):
                # Different number of heads - average teacher heads
                teacher_attention = teacher_attention.mean(dim=1, keepdim=True)
                student_attention = student_attention.mean(dim=1, keepdim=True)

            # MSE loss
            loss = F.mse_loss(student_attention, teacher_attention)
            attention_loss += loss
            num_layers += 1

        if num_layers > 0:
            attention_loss /= num_layers

        return attention_loss

    def _project_hidden(self,
                        hidden: torch.Tensor,
                        target_dim: int) -> torch.Tensor:
        """
        Project hidden state to target dimension.

        Args:
            hidden: Hidden state (batch, seq_len, hidden_dim)
            target_dim: Target dimension

        Returns:
            Projected hidden state
        """
        # Simple linear projection
        # In practice, this projection layer should be learned
        batch_size, seq_len, hidden_dim = hidden.shape

        # Create projection matrix (or use cached one)
        if not hasattr(self, '_projection_matrices'):
            self._projection_matrices = {}

        key = f"{hidden_dim}_{target_dim}"
        if key not in self._projection_matrices:
            projection = nn.Linear(hidden_dim, target_dim).to(self.device)
            self._projection_matrices[key] = projection
        else:
            projection = self._projection_matrices[key]

        # Project
        hidden_projected = projection(hidden)

        return hidden_projected

    def _save_checkpoint(self, save_path: str, epoch: int):
        """
        Save distillation checkpoint.

        Args:
            save_path: Save directory
            epoch: Current epoch
        """
        from pathlib import Path

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "student_state_dict": self.student.state_dict(),
            "config": self.config,
        }, checkpoint_path)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_distilled_model(self, save_path: str):
        """
        Save final distilled student model.

        Args:
            save_path: Save directory
        """
        from pathlib import Path
        import json

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = save_dir / "distilled_student.pt"
        torch.save(self.student.state_dict(), model_path)

        # Save config
        config_path = save_dir / "distillation_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "temperature": self.config.temperature,
                "alpha": self.config.alpha,
                "feature_distill": self.config.feature_distill,
                "response_distill": self.config.response_distill,
                "attention_distill": self.config.attention_distill,
            }, f, indent=2)

        logger.info(f"Distilled model saved to {save_path}")

    @staticmethod
    def load_distilled_model(load_path: str, student_model: nn.Module) -> nn.Module:
        """
        Load distilled student model.

        Args:
            load_path: Load directory
            student_model: Student model instance

        Returns:
            Loaded student model
        """
        from pathlib import Path

        load_dir = Path(load_path)
        model_path = load_dir / "distilled_student.pt"

        student_model.load_state_dict(torch.load(model_path))

        logger.info(f"Distilled model loaded from {load_path}")

        return student_model

    def evaluate(self,
                 eval_loader: Any,
                 metric_fn: Optional[Any] = None) -> Dict[str, float]:
        """
        Evaluate distilled student vs teacher.

        Args:
            eval_loader: Evaluation data
            metric_fn: Custom metric function

        Returns:
            Evaluation metrics
        """
        logger.info("Evaluating distilled student vs teacher")

        self.teacher.eval()
        self.student.eval()

        teacher_losses = []
        student_losses = []

        with torch.no_grad():
            for batch in eval_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch.get('labels')

                # Teacher forward
                teacher_outputs = self.teacher(**inputs)
                if labels is not None:
                    teacher_loss = F.cross_entropy(
                        teacher_outputs.logits.view(-1, teacher_outputs.logits.size(-1)),
                        labels.view(-1).to(self.device),
                        ignore_index=-100
                    )
                    teacher_losses.append(teacher_loss.item())

                # Student forward
                student_outputs = self.student(**inputs)
                if labels is not None:
                    student_loss = F.cross_entropy(
                        student_outputs.logits.view(-1, student_outputs.logits.size(-1)),
                        labels.view(-1).to(self.device),
                        ignore_index=-100
                    )
                    student_losses.append(student_loss.item())

        results = {
            "teacher_loss": sum(teacher_losses) / len(teacher_losses) if teacher_losses else 0,
            "student_loss": sum(student_losses) / len(student_losses) if student_losses else 0,
        }

        # Performance retention
        if results["teacher_loss"] > 0:
            results["performance_retention"] = 1 - (
                (results["student_loss"] - results["teacher_loss"]) / results["teacher_loss"]
            )

        logger.info(f"Evaluation results: {results}")

        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = DistillationConfig(
        temperature=3.0,
        alpha=0.5,
        feature_distill=True,
        response_distill=True,
        attention_distill=False,
    )

    print(f"Distillation config: {config}")

    # Example models
    # teacher_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-72B")
    # student_model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B")

    # train_loader = create_data_loader(...)
    # optimizer = torch.optim.AdamW(student_model.parameters(), lr=5e-5)

    # distiller = KnowledgeDistiller(teacher_model, student_model, config)
    # distilled_student = distiller.distill(
    #     train_loader=train_loader,
    #     optimizer=optimizer,
    #     num_epochs=10,
    #     save_path="./models/qwen2.5-vl-7b-distilled"
    # )

    # # Evaluate
    # eval_loader = create_eval_loader(...)
    # results = distiller.evaluate(eval_loader)
    # print(f"Performance retention: {results['performance_retention']:.2%}")

    print("Knowledge distillation module loaded successfully")
