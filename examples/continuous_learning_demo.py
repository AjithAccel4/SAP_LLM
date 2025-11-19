"""
Continuous Learning Pipeline - Complete Demo

This demo shows all features of the continuous learning system:
1. Model registration and versioning
2. Drift detection
3. Automated retraining
4. A/B testing
5. Champion promotion
6. Rollback
"""

import logging
import torch
import torch.nn as nn
from pathlib import Path

from sap_llm.training.continuous_learner import ContinuousLearner, LearningConfig
from sap_llm.models.registry import ModelRegistry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Simple model for demonstration."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def demo_model_registry():
    """Demonstrate model registry features."""
    print("\n" + "="*60)
    print("DEMO 1: Model Registry & Versioning")
    print("="*60)

    registry = ModelRegistry(db_path="./demo_registry.db")
    model = SimpleModel()

    # Register first version
    print("\n1. Registering model v1.0.0...")
    id_v1 = registry.register_model(
        model=model,
        name="demo_model",
        model_type="classifier",
        metrics={"accuracy": 0.90}
    )
    print(f"   ✓ Registered: {id_v1}")

    # Register second version
    print("\n2. Registering model v1.0.1 (auto-increment)...")
    id_v2 = registry.register_model(
        model=model,
        name="demo_model",
        model_type="classifier",
        metrics={"accuracy": 0.92}
    )
    print(f"   ✓ Registered: {id_v2}")

    # Promote to champion
    print("\n3. Promoting v1.0.0 to champion...")
    registry.promote_to_champion(id_v1, reason="Initial deployment")
    champion = registry.get_champion("classifier")
    print(f"   ✓ Champion: {champion['id']} (version {champion['version']})")

    # Replace champion
    print("\n4. Promoting v1.0.1 to champion (better accuracy)...")
    registry.promote_to_champion(id_v2, reason="Improved accuracy")
    champion = registry.get_champion("classifier")
    print(f"   ✓ New Champion: {champion['id']} (version {champion['version']})")

    # Show statistics
    print("\n5. Registry Statistics:")
    stats = registry.get_statistics()
    print(f"   - Total models: {stats['total_models']}")
    print(f"   - Champions: {stats['by_status'].get('champion', 0)}")
    print(f"   - Archived: {stats['by_status'].get('archived', 0)}")

    return registry


def demo_ab_testing(registry):
    """Demonstrate A/B testing."""
    print("\n" + "="*60)
    print("DEMO 2: A/B Testing Framework")
    print("="*60)

    from sap_llm.training.ab_testing import ABTestingManager

    ab_testing = ABTestingManager(
        model_registry=registry,
        default_traffic_split=0.1
    )

    # Get champion and create challenger
    champion = registry.get_champion("classifier")

    model = SimpleModel()
    challenger_id = registry.register_model(
        model=model,
        name="demo_model",
        model_type="classifier",
        metrics={"accuracy": 0.94}
    )

    print(f"\n1. Creating A/B test...")
    print(f"   Champion: {champion['id']} (accuracy: 90%)")
    print(f"   Challenger: {challenger_id} (accuracy: 94%)")

    test_id = ab_testing.create_ab_test(
        champion_id=champion["id"],
        challenger_id=challenger_id,
        traffic_split=0.1
    )
    print(f"   ✓ A/B Test Created: {test_id}")

    print(f"\n2. Simulating predictions (90/10 split)...")
    # Simulate 1000 predictions
    champion_preds = 0
    challenger_preds = 0

    for i in range(1000):
        model_id = ab_testing.route_prediction(test_id)

        # Simulate prediction with different accuracy
        if model_id == champion["id"]:
            champion_preds += 1
            # 90% accuracy
            correct = i % 10 != 0
        else:
            challenger_preds += 1
            # 94% accuracy
            correct = i % 100 < 94

        ab_testing.record_prediction(
            test_id=test_id,
            model_id=model_id,
            document_id=f"doc_{i}",
            prediction={"result": "class_a"},
            ground_truth={"result": "class_a" if correct else "class_b"},
            latency_ms=100.0
        )

    print(f"   ✓ Champion predictions: {champion_preds}")
    print(f"   ✓ Challenger predictions: {challenger_preds}")

    print(f"\n3. Evaluating A/B test...")
    result = ab_testing.evaluate_ab_test(test_id, min_samples=50)

    print(f"   Champion Accuracy: {result.champion_metrics['accuracy']:.2%}")
    print(f"   Challenger Accuracy: {result.challenger_metrics['accuracy']:.2%}")
    print(f"   P-value: {result.p_value:.4f}")
    print(f"   Winner: {result.winner}")
    print(f"   Recommendation: {result.recommendation}")

    return ab_testing, test_id


def demo_champion_promotion(registry, ab_testing, test_id):
    """Demonstrate champion promotion."""
    print("\n" + "="*60)
    print("DEMO 3: Automated Champion Promotion")
    print("="*60)

    from sap_llm.training.champion_promoter import ChampionPromoter

    promoter = ChampionPromoter(
        model_registry=registry,
        ab_testing=ab_testing,
        min_improvement=0.02
    )

    print(f"\n1. Evaluating A/B test for promotion...")
    result = promoter.evaluate_and_promote(
        test_id=test_id,
        auto_promote=True
    )

    print(f"   Decision: {result['decision']}")
    print(f"   Promoted: {result.get('promoted', False)}")

    if result.get('promoted'):
        champion = registry.get_champion("classifier")
        print(f"   ✓ New Champion: {champion['id']}")

    return promoter


def demo_rollback(registry):
    """Demonstrate rollback."""
    print("\n" + "="*60)
    print("DEMO 4: Rollback Capability")
    print("="*60)

    print(f"\n1. Current champion:")
    champion = registry.get_champion("classifier")
    print(f"   {champion['id']} (version {champion['version']})")

    print(f"\n2. Initiating rollback...")
    previous_id = registry.rollback_to_previous_champion(
        model_type="classifier",
        reason="Demo rollback"
    )

    print(f"   ✓ Rolled back to: {previous_id}")

    champion = registry.get_champion("classifier")
    print(f"   ✓ Current champion: {champion['id']}")


def demo_drift_detection():
    """Demonstrate drift detection."""
    print("\n" + "="*60)
    print("DEMO 5: Drift Detection")
    print("="*60)

    from sap_llm.training.drift_detector import DriftDetector

    detector = DriftDetector(psi_threshold=0.25)

    # Simulate baseline data
    baseline_data = [
        {"prediction": {"doc_type": "invoice"}} for _ in range(100)
    ]

    # Simulate drifted data
    current_data = [
        {"prediction": {"doc_type": "invoice" if i < 60 else "purchase_order"}}
        for i in range(100)
    ]

    print(f"\n1. Detecting drift...")
    print(f"   Baseline: 100% invoices")
    print(f"   Current: 60% invoices, 40% purchase orders")

    drift_report = detector.detect_data_drift(baseline_data, current_data)

    print(f"\n2. Drift Report:")
    print(f"   PSI Score: {drift_report.psi_score:.4f}")
    print(f"   Drift Detected: {drift_report.drift_detected}")
    print(f"   Severity: {drift_report.severity}")
    print(f"   Drift Types: {', '.join(drift_report.drift_types)}")


def demo_continuous_learner():
    """Demonstrate full continuous learner."""
    print("\n" + "="*60)
    print("DEMO 6: Continuous Learning Pipeline")
    print("="*60)

    # Configure
    config = LearningConfig(
        drift_threshold_psi=0.25,
        min_improvement_threshold=0.02,
        enable_auto_retraining=True,
        enable_auto_promotion=True,
        enable_auto_rollback=True
    )

    print(f"\n1. Initializing Continuous Learner...")
    learner = ContinuousLearner(config=config)
    print(f"   ✓ All components initialized")

    print(f"\n2. Configuration:")
    print(f"   - Drift threshold: PSI > {config.drift_threshold_psi}")
    print(f"   - Min improvement: {config.min_improvement_threshold:.0%}")
    print(f"   - Auto-retraining: {config.enable_auto_retraining}")
    print(f"   - Auto-promotion: {config.enable_auto_promotion}")
    print(f"   - Auto-rollback: {config.enable_auto_rollback}")

    print(f"\n3. Statistics:")
    stats = learner.get_statistics()
    print(f"   - Learning cycles: {stats['learning_cycles_run']}")
    print(f"   - Drift detected: {stats['drift_detected_count']}")
    print(f"   - Retraining triggered: {stats['retraining_triggered']}")
    print(f"   - Promotions: {stats['promotions']}")
    print(f"   - Rollbacks: {stats['rollbacks']}")

    print(f"\n4. To start continuous loop:")
    print(f"   learner.start_continuous_learning()  # Runs indefinitely")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + " Continuous Learning Pipeline - Complete Demo ".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    try:
        # Demo 1: Model Registry
        registry = demo_model_registry()

        # Demo 2: A/B Testing
        ab_testing, test_id = demo_ab_testing(registry)

        # Demo 3: Champion Promotion
        promoter = demo_champion_promotion(registry, ab_testing, test_id)

        # Demo 4: Rollback
        demo_rollback(registry)

        # Demo 5: Drift Detection
        demo_drift_detection()

        # Demo 6: Continuous Learner
        demo_continuous_learner()

        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        import os
        if os.path.exists("demo_registry.db"):
            os.remove("demo_registry.db")


if __name__ == "__main__":
    main()
