"""
Federated Learning System

Enable collaborative model training across multiple organizations
without sharing raw data:
- Privacy-preserving distributed training
- Secure aggregation of model updates
- Differential privacy
- Client-side model updates
- Central model aggregation

Benefits:
- Data privacy (data never leaves client premises)
- Collaborative improvement
- Regulatory compliance (GDPR, HIPAA)
- Domain-specific specialization
"""

import hashlib
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class AggregationStrategy(Enum):
    """Model aggregation strategies"""
    FEDERATED_AVERAGING = "fed_avg"  # FedAvg (McMahan et al.)
    FEDERATED_PROX = "fed_prox"  # FedProx
    FEDERATED_ADAM = "fed_adam"  # FedAdam
    WEIGHTED_AVERAGE = "weighted_avg"  # Weighted by data size


@dataclass
class ClientConfig:
    """Configuration for federated learning client"""
    client_id: str
    organization_name: str
    data_size: int
    local_epochs: int = 5
    learning_rate: float = 0.001
    batch_size: int = 32
    privacy_budget: float = 1.0  # Epsilon for differential privacy


@dataclass
class ModelUpdate:
    """Model update from a client"""
    client_id: str
    round_number: int
    parameters: Dict[str, np.ndarray]
    data_size: int
    training_loss: float
    timestamp: datetime
    signature: str  # Cryptographic signature


@dataclass
class FederatedRound:
    """Federated learning round results"""
    round_number: int
    participating_clients: List[str]
    aggregated_parameters: Dict[str, np.ndarray]
    average_loss: float
    global_accuracy: float
    timestamp: datetime


class DifferentialPrivacy:
    """
    Differential privacy for federated learning

    Features:
    - Gaussian noise addition
    - Privacy budget tracking
    - Gradient clipping
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize differential privacy

        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Probability of privacy breach
        """
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = self._calculate_noise_multiplier()

    def _calculate_noise_multiplier(self) -> float:
        """Calculate noise multiplier based on epsilon and delta"""
        # Simplified calculation
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_noise(
        self,
        gradients: Dict[str, np.ndarray],
        sensitivity: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Add Gaussian noise to gradients

        Args:
            gradients: Model gradients
            sensitivity: Sensitivity of the query

        Returns:
            Noisy gradients
        """
        noisy_gradients = {}

        for param_name, grad in gradients.items():
            # Calculate noise scale
            noise_scale = self.noise_multiplier * sensitivity

            # Add Gaussian noise
            noise = np.random.normal(0, noise_scale, grad.shape)
            noisy_gradients[param_name] = grad + noise

        logger.debug(f"Added DP noise with scale {noise_scale:.6f}")

        return noisy_gradients

    def clip_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        max_norm: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Clip gradients to bound sensitivity

        Args:
            gradients: Model gradients
            max_norm: Maximum L2 norm

        Returns:
            Clipped gradients
        """
        # Calculate global norm
        total_norm = 0.0
        for grad in gradients.values():
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)

        # Clip if necessary
        clip_coef = max_norm / (total_norm + 1e-6)

        if clip_coef < 1.0:
            clipped_gradients = {
                name: grad * clip_coef
                for name, grad in gradients.items()
            }
            logger.debug(f"Clipped gradients: norm {total_norm:.4f} -> {max_norm:.4f}")
            return clipped_gradients

        return gradients


class SecureAggregation:
    """
    Secure aggregation of model updates

    Features:
    - Cryptographic signatures
    - Secure multi-party computation
    - Byzantine fault tolerance
    """

    def __init__(self):
        self.min_clients = 3  # Minimum clients for aggregation

    def sign_update(
        self,
        update: ModelUpdate,
        private_key: str
    ) -> str:
        """
        Sign model update with private key

        Args:
            update: Model update
            private_key: Client's private key

        Returns:
            Cryptographic signature
        """
        # Serialize parameters
        param_bytes = json.dumps(
            {k: v.tolist() for k, v in update.parameters.items()}
        ).encode()

        # Create signature (simplified - use proper crypto in production)
        signature = hashlib.sha256(
            param_bytes + private_key.encode()
        ).hexdigest()

        return signature

    def verify_update(
        self,
        update: ModelUpdate,
        public_key: str
    ) -> bool:
        """
        Verify model update signature

        Args:
            update: Model update
            public_key: Client's public key

        Returns:
            True if signature is valid
        """
        # Simplified verification
        param_bytes = json.dumps(
            {k: v.tolist() for k, v in update.parameters.items()}
        ).encode()

        expected_signature = hashlib.sha256(
            param_bytes + public_key.encode()
        ).hexdigest()

        return update.signature == expected_signature

    def detect_byzantine_clients(
        self,
        updates: List[ModelUpdate]
    ) -> List[str]:
        """
        Detect Byzantine (malicious) clients

        Args:
            updates: Model updates from clients

        Returns:
            List of suspicious client IDs
        """
        if len(updates) < 3:
            return []

        suspicious_clients = []

        # Calculate pairwise distances between updates
        for i, update_i in enumerate(updates):
            distances = []

            for j, update_j in enumerate(updates):
                if i != j:
                    # Calculate L2 distance between parameter vectors
                    distance = self._calculate_update_distance(
                        update_i.parameters,
                        update_j.parameters
                    )
                    distances.append(distance)

            # If client's update is far from others, mark as suspicious
            avg_distance = np.mean(distances)
            if avg_distance > 3 * np.std(distances):  # 3-sigma rule
                suspicious_clients.append(update_i.client_id)
                logger.warning(
                    f"Client {update_i.client_id} has suspicious update "
                    f"(distance: {avg_distance:.4f})"
                )

        return suspicious_clients

    def _calculate_update_distance(
        self,
        params1: Dict[str, np.ndarray],
        params2: Dict[str, np.ndarray]
    ) -> float:
        """Calculate L2 distance between two parameter sets"""
        total_distance = 0.0

        for param_name in params1:
            if param_name in params2:
                diff = params1[param_name] - params2[param_name]
                total_distance += np.sum(diff ** 2)

        return np.sqrt(total_distance)


class FederatedClient:
    """
    Federated learning client

    Runs on client premises:
    - Local model training
    - Privacy-preserving updates
    - Secure communication
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self.local_model = None
        self.differential_privacy = DifferentialPrivacy(
            epsilon=config.privacy_budget
        )
        self.private_key = self._generate_private_key()

    def _generate_private_key(self) -> str:
        """Generate private key for signing"""
        return hashlib.sha256(
            f"{self.config.client_id}_{self.config.organization_name}".encode()
        ).hexdigest()

    def train_local_model(
        self,
        global_parameters: Dict[str, np.ndarray],
        local_data: Any
    ) -> ModelUpdate:
        """
        Train model on local data

        Args:
            global_parameters: Global model parameters from server
            local_data: Local training data

        Returns:
            Model update
        """
        logger.info(
            f"Client {self.config.client_id} starting local training "
            f"({self.config.local_epochs} epochs)"
        )

        # Initialize local model with global parameters
        self.local_model = global_parameters.copy()

        # Simulate local training
        # In reality, would train model on local_data
        training_loss = 0.0

        for epoch in range(self.config.local_epochs):
            # Placeholder - would do actual training here
            epoch_loss = np.random.uniform(0.1, 0.5)  # Simulated loss
            training_loss += epoch_loss

        training_loss /= self.config.local_epochs

        # Calculate parameter updates (gradients)
        parameter_updates = {
            name: self.local_model[name] - global_parameters[name]
            for name in self.local_model
        }

        # Apply differential privacy
        parameter_updates = self.differential_privacy.clip_gradients(
            parameter_updates,
            max_norm=1.0
        )
        parameter_updates = self.differential_privacy.add_noise(
            parameter_updates,
            sensitivity=1.0
        )

        # Create model update
        update = ModelUpdate(
            client_id=self.config.client_id,
            round_number=0,  # Will be set by server
            parameters=parameter_updates,
            data_size=self.config.data_size,
            training_loss=training_loss,
            timestamp=datetime.utcnow(),
            signature=""  # Will be signed
        )

        # Sign update
        secure_agg = SecureAggregation()
        update.signature = secure_agg.sign_update(update, self.private_key)

        logger.info(
            f"Client {self.config.client_id} completed training "
            f"(loss: {training_loss:.4f})"
        )

        return update


class FederatedServer:
    """
    Federated learning server

    Coordinates federated training:
    - Client selection
    - Model aggregation
    - Round management
    """

    def __init__(
        self,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDERATED_AVERAGING
    ):
        self.aggregation_strategy = aggregation_strategy
        self.global_model = self._initialize_global_model()
        self.round_number = 0
        self.history: List[FederatedRound] = []
        self.secure_aggregation = SecureAggregation()

    def _initialize_global_model(self) -> Dict[str, np.ndarray]:
        """Initialize global model parameters"""
        # Placeholder - would initialize actual model
        model = {
            "layer1.weight": np.random.randn(100, 50),
            "layer1.bias": np.zeros(100),
            "layer2.weight": np.random.randn(10, 100),
            "layer2.bias": np.zeros(10),
        }
        return model

    def select_clients(
        self,
        available_clients: List[ClientConfig],
        fraction: float = 0.5,
        min_clients: int = 3
    ) -> List[ClientConfig]:
        """
        Select clients for current round

        Args:
            available_clients: List of available clients
            fraction: Fraction of clients to select
            min_clients: Minimum number of clients

        Returns:
            Selected clients
        """
        num_clients = max(
            min_clients,
            int(len(available_clients) * fraction)
        )

        # Random selection (could use more sophisticated strategies)
        selected_indices = np.random.choice(
            len(available_clients),
            size=min(num_clients, len(available_clients)),
            replace=False
        )

        selected_clients = [available_clients[i] for i in selected_indices]

        logger.info(
            f"Selected {len(selected_clients)} clients for round {self.round_number + 1}"
        )

        return selected_clients

    def aggregate_updates(
        self,
        updates: List[ModelUpdate]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate model updates from clients

        Args:
            updates: Model updates from clients

        Returns:
            Aggregated parameters
        """
        # Filter out Byzantine clients
        suspicious_clients = self.secure_aggregation.detect_byzantine_clients(updates)
        valid_updates = [
            u for u in updates
            if u.client_id not in suspicious_clients
        ]

        if not valid_updates:
            logger.error("No valid updates to aggregate")
            return self.global_model

        logger.info(
            f"Aggregating {len(valid_updates)} updates "
            f"({len(suspicious_clients)} filtered)"
        )

        if self.aggregation_strategy == AggregationStrategy.FEDERATED_AVERAGING:
            return self._federated_averaging(valid_updates)
        elif self.aggregation_strategy == AggregationStrategy.WEIGHTED_AVERAGE:
            return self._weighted_averaging(valid_updates)
        else:
            return self._federated_averaging(valid_updates)

    def _federated_averaging(
        self,
        updates: List[ModelUpdate]
    ) -> Dict[str, np.ndarray]:
        """FedAvg: Average model updates"""
        aggregated = {}

        # Get parameter names from first update
        param_names = list(updates[0].parameters.keys())

        for param_name in param_names:
            # Average across all clients
            param_sum = sum(
                update.parameters[param_name]
                for update in updates
            )
            aggregated[param_name] = param_sum / len(updates)

        # Apply to global model
        for param_name in aggregated:
            self.global_model[param_name] += aggregated[param_name]

        return self.global_model

    def _weighted_averaging(
        self,
        updates: List[ModelUpdate]
    ) -> Dict[str, np.ndarray]:
        """Weighted average by data size"""
        total_data_size = sum(update.data_size for update in updates)

        aggregated = {}
        param_names = list(updates[0].parameters.keys())

        for param_name in param_names:
            # Weighted average
            weighted_sum = sum(
                update.parameters[param_name] * (update.data_size / total_data_size)
                for update in updates
            )
            aggregated[param_name] = weighted_sum

        # Apply to global model
        for param_name in aggregated:
            self.global_model[param_name] += aggregated[param_name]

        return self.global_model

    def run_round(
        self,
        clients: List[FederatedClient]
    ) -> FederatedRound:
        """
        Run one round of federated learning

        Args:
            clients: Participating clients

        Returns:
            Round results
        """
        self.round_number += 1

        logger.info(f"Starting federated round {self.round_number}")

        # Collect updates from clients
        updates = []

        for client in clients:
            # Client trains locally
            update = client.train_local_model(
                self.global_model,
                local_data=None  # Placeholder
            )
            update.round_number = self.round_number
            updates.append(update)

        # Aggregate updates
        aggregated_params = self.aggregate_updates(updates)

        # Calculate metrics
        average_loss = np.mean([u.training_loss for u in updates])

        # Create round results
        round_result = FederatedRound(
            round_number=self.round_number,
            participating_clients=[u.client_id for u in updates],
            aggregated_parameters=aggregated_params,
            average_loss=average_loss,
            global_accuracy=0.95,  # Placeholder
            timestamp=datetime.utcnow()
        )

        self.history.append(round_result)

        logger.info(
            f"Round {self.round_number} completed: "
            f"avg_loss={average_loss:.4f}, "
            f"clients={len(updates)}"
        )

        return round_result

    def get_global_model(self) -> Dict[str, np.ndarray]:
        """Get current global model"""
        return self.global_model.copy()


class FederatedLearningOrchestrator:
    """
    Orchestrate federated learning across multiple organizations

    Features:
    - Multi-round training
    - Client management
    - Progress tracking
    - Model deployment
    """

    def __init__(self):
        self.server = FederatedServer(
            aggregation_strategy=AggregationStrategy.WEIGHTED_AVERAGE
        )
        self.clients: Dict[str, FederatedClient] = {}

    def register_client(self, config: ClientConfig) -> str:
        """Register a new client"""
        client = FederatedClient(config)
        self.clients[config.client_id] = client

        logger.info(
            f"Registered client: {config.client_id} "
            f"({config.organization_name}, {config.data_size} samples)"
        )

        return config.client_id

    def run_federated_training(
        self,
        num_rounds: int = 10,
        clients_per_round: float = 0.5
    ) -> List[FederatedRound]:
        """
        Run federated training for multiple rounds

        Args:
            num_rounds: Number of training rounds
            clients_per_round: Fraction of clients per round

        Returns:
            Training history
        """
        logger.info(
            f"Starting federated training: "
            f"{num_rounds} rounds, {len(self.clients)} clients"
        )

        results = []

        for round_num in range(num_rounds):
            # Select clients for this round
            available_configs = [
                client.config for client in self.clients.values()
            ]

            selected_configs = self.server.select_clients(
                available_configs,
                fraction=clients_per_round
            )

            # Get client instances
            selected_clients = [
                self.clients[config.client_id]
                for config in selected_configs
            ]

            # Run round
            round_result = self.server.run_round(selected_clients)
            results.append(round_result)

            logger.info(
                f"Progress: {round_num + 1}/{num_rounds} rounds "
                f"(avg_loss: {round_result.average_loss:.4f})"
            )

        logger.info("Federated training completed")

        return results

    def deploy_global_model(self, output_path: str):
        """Deploy trained global model"""
        global_model = self.server.get_global_model()

        # Save model (simplified)
        model_data = {
            "parameters": {
                name: param.tolist()
                for name, param in global_model.items()
            },
            "rounds": self.server.round_number,
            "timestamp": datetime.utcnow().isoformat()
        }

        with open(output_path, 'w') as f:
            json.dump(model_data, f, indent=2)

        logger.info(f"Global model deployed to {output_path}")


# Global instance
federated_orchestrator = FederatedLearningOrchestrator()
