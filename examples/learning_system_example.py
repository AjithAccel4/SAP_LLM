"""
Comprehensive Example: SAP_LLM Auto-Learning and Continuous Improvement

This example demonstrates the complete learning ecosystem including:
- Online learning from user corrections
- Feedback loop with A/B testing
- Adaptive learning with drift detection
- Knowledge augmentation
- Self-improvement pipeline
"""

import asyncio
from datetime import datetime, time

from sap_llm.learning.adaptive_learning import AdaptiveLearningEngine
from sap_llm.learning.feedback_loop import FeedbackLoopSystem, FeedbackType
from sap_llm.learning.knowledge_augmentation import KnowledgeAugmentationEngine
from sap_llm.learning.online_learning import OnlineLearningEngine
from sap_llm.learning.self_improvement import SelfImprovementPipeline
from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.pmg.learning import ContinuousLearner


def example_1_online_learning():
    """Example 1: Real-time online learning from corrections."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Online Learning from User Corrections")
    print("=" * 80)

    # Initialize PMG and online learner
    pmg = ProcessMemoryGraph()
    online_learner = OnlineLearningEngine(
        pmg=pmg,
        uncertainty_threshold=0.7,
        active_learning_enabled=True,
    )

    # Scenario: User corrects a misclassified invoice
    doc_type = "SUPPLIER_INVOICE"
    features = {
        "total_amount": 1250.50,
        "supplier_id": "SUP12345",
        "invoice_number": "INV-2024-001",
        "currency": "USD",
    }

    # Model predicted PURCHASE_ORDER, but user corrected to SUPPLIER_INVOICE
    result = online_learner.update_from_correction(
        doc_type=doc_type,
        features=features,
        correct_label="SUPPLIER_INVOICE",
        predicted_label="PURCHASE_ORDER",
        confidence=0.65,
    )

    print(f"\nCorrection applied:")
    print(f"  - Document type: {doc_type}")
    print(f"  - Update count: {result['update_count']}")
    print(f"  - Was incorrect: {result['was_incorrect']}")

    # Make a prediction to check uncertainty
    print("\n--- Active Learning: Identifying Uncertain Predictions ---")

    new_features = {
        "total_amount": 500.00,
        "supplier_id": "SUP99999",
        "invoice_number": "INV-2024-999",
        "currency": "EUR",
    }

    prediction, confidence, is_uncertain = online_learner.identify_uncertain_predictions(
        doc_type=doc_type,
        features=new_features,
    )

    print(f"\nPrediction: {prediction}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Uncertain (needs review): {is_uncertain}")

    # Get uncertain predictions for human review
    uncertain_queue = online_learner.get_uncertain_predictions(limit=10)
    print(f"\nUncertain predictions queue: {len(uncertain_queue)} items")


def example_2_feedback_loop():
    """Example 2: Feedback loop with A/B testing."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Feedback Loop with A/B Testing")
    print("=" * 80)

    pmg = ProcessMemoryGraph()
    feedback_system = FeedbackLoopSystem(
        pmg=pmg,
        confidence_threshold=0.8,
        ab_test_enabled=True,
    )

    # Collect user feedback
    print("\n--- Collecting User Feedback ---")

    feedback_result = feedback_system.collect_feedback(
        feedback_type=FeedbackType.CORRECTION,
        doc_id="doc_12345",
        doc_type="INVOICE",
        user_id="user_001",
        prediction="1250.00",
        correct_value="1250.50",
        confidence=0.92,
    )

    print(f"Feedback collected:")
    print(f"  - Feedback ID: {feedback_result['feedback_id']}")
    print(f"  - Priority: {feedback_result['priority']}")
    print(f"  - Retrain needed: {feedback_result['retrain_needed']}")

    # Request feedback for low-confidence prediction
    print("\n--- Proactive Feedback Request ---")

    request = feedback_system.request_feedback(
        doc_id="doc_67890",
        doc_type="INVOICE",
        prediction={"amount": 999.99},
        confidence=0.65,
    )

    print(f"Feedback requested: {request['feedback_requested']}")
    if request['feedback_requested']:
        print(f"  - Request type: {request['request']['request_type']}")
        print(f"  - Priority fields: {request['request']['priority_fields']}")

    # Start A/B test
    print("\n--- Starting A/B Test ---")

    ab_test = feedback_system.start_ab_test(
        test_name="invoice_v2_test",
        doc_type="INVOICE",
        champion_version="v1.0.0",
        challenger_version="v2.0.0",
        duration_hours=24,
        success_metric="accuracy",
        min_improvement=0.02,
    )

    print(f"A/B test started:")
    print(f"  - Test ID: {ab_test['test_id']}")
    print(f"  - Champion: {ab_test['champion_version']}")
    print(f"  - Challenger: {ab_test['challenger_version']}")
    print(f"  - Duration: {ab_test['end_time'] - ab_test['start_time']}")

    # Route traffic
    version, variant = feedback_system.route_ab_test("INVOICE", "doc_12345")
    print(f"\nRouting decision: {variant} (version: {version})")

    # Get feedback statistics
    stats = feedback_system.get_feedback_statistics()
    print(f"\nFeedback statistics:")
    print(f"  - Total feedback: {stats['total_feedback']}")
    print(f"  - Unprocessed: {stats['unprocessed']}")
    print(f"  - Active A/B tests: {stats['active_ab_tests']}")


def example_3_adaptive_learning():
    """Example 3: Adaptive learning with drift detection."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Adaptive Learning and Drift Detection")
    print("=" * 80)

    pmg = ProcessMemoryGraph()
    adaptive_engine = AdaptiveLearningEngine(
        pmg=pmg,
        drift_threshold=0.25,
        refresh_threshold=0.15,
        min_accuracy=0.85,
    )

    doc_type = "PURCHASE_ORDER"

    # Track predictions
    print("\n--- Tracking Predictions ---")

    predictions = [
        ("PO", "PO", 0.95),
        ("PO", "PO", 0.88),
        ("PO", "INVOICE", 0.72),  # Incorrect
        ("PO", "PO", 0.91),
    ]

    for pred, actual, conf in predictions:
        adaptive_engine.track_prediction(
            doc_type=doc_type,
            prediction=pred,
            actual=actual,
            confidence=conf,
            features={"amount": 1000.0, "supplier": "SUP123"},
        )

    # Set baseline
    adaptive_engine.set_baseline_performance(doc_type, accuracy=0.95)

    # Check performance degradation
    print("\n--- Checking Performance Degradation ---")

    degraded, metrics = adaptive_engine.check_performance_degradation(doc_type)
    print(f"Performance degraded: {degraded}")
    print(f"Current metrics: {metrics}")

    # Check drift
    print("\n--- Checking for Drift ---")

    # Set baseline distribution
    baseline_amounts = [1000.0, 1500.0, 2000.0, 2500.0]
    adaptive_engine.set_baseline_distribution(doc_type, "amount", baseline_amounts)

    # Add some current samples (with drift)
    for amount in [5000.0, 6000.0, 7000.0]:  # Much higher amounts
        adaptive_engine.drift_detector.add_sample(f"{doc_type}:amount", amount)

    drift_results = adaptive_engine.check_drift(doc_type)
    print(f"Drift detected: {drift_results['drift_detected']}")
    if drift_results['drift_detected']:
        print(f"Features with drift: {drift_results['features_with_drift']}")

    # Get performance summary
    print("\n--- Performance Summary ---")

    summary = adaptive_engine.get_performance_summary(doc_type)
    print(f"Summary for {doc_type}:")
    for key, value in summary[doc_type].items():
        print(f"  - {key}: {value}")


def example_4_knowledge_augmentation():
    """Example 4: Knowledge augmentation from production data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Knowledge Augmentation")
    print("=" * 80)

    pmg = ProcessMemoryGraph()
    knowledge_engine = KnowledgeAugmentationEngine(pmg=pmg)

    # Extract patterns
    print("\n--- Extracting Patterns ---")

    invoice_numbers = [
        "INV-2024-001",
        "INV-2024-002",
        "INV-2024-003",
        "INV-2025-001",
    ]

    patterns = knowledge_engine.pattern_extractor.extract_patterns(
        field_name="invoice_number",
        values=invoice_numbers,
    )

    print(f"Extracted {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"  - Type: {pattern['type']}, Pattern: {pattern.get('pattern')}, "
              f"Confidence: {pattern.get('confidence', 0):.2f}")

    # Build dictionaries
    print("\n--- Building Dictionaries ---")

    supplier_names = ["ACME Corp", "Tech Solutions Inc", "Global Supplies Ltd"]
    supplier_ids = ["SUP001", "SUP002", "SUP003"]

    supplier_dict = knowledge_engine.dictionary_builder.build_supplier_dictionary(
        supplier_names=supplier_names,
        supplier_ids=supplier_ids,
    )

    print(f"Built supplier dictionary: {len(supplier_dict)} suppliers")

    # Learn field mappings
    print("\n--- Learning Field Mappings ---")

    mapping_examples = [
        ("total_amount", "1250.50"),
        ("total_amt", "1250.50"),
        ("amount_total", "1250.50"),
    ]

    knowledge_engine.field_mapping_learner.learn_mapping(
        doc_type="INVOICE",
        source_field="total_amount",
        target_field="SAP_TOTAL_AMT",
        examples=mapping_examples,
    )

    # Discover validation rules
    print("\n--- Discovering Validation Rules ---")

    amounts = [100.0, 250.0, 500.0, 1000.0, 9999.0]  # Valid amounts
    valid_flags = [True, True, True, True, False]  # Last one invalid

    rules = knowledge_engine.validation_rule_learner.discover_rules(
        field_name="total_amount",
        values=amounts,
        valid_flags=valid_flags,
    )

    print(f"Discovered {len(rules)} validation rules:")
    for rule in rules:
        print(f"  - Type: {rule['type']}, Confidence: {rule.get('confidence', 0):.2f}")


def example_5_self_improvement():
    """Example 5: Automated self-improvement pipeline."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Self-Improvement Pipeline")
    print("=" * 80)

    # Initialize all components
    pmg = ProcessMemoryGraph()

    online_learner = OnlineLearningEngine(pmg=pmg)
    feedback_system = FeedbackLoopSystem(pmg=pmg)
    adaptive_engine = AdaptiveLearningEngine(pmg=pmg)
    knowledge_engine = KnowledgeAugmentationEngine(pmg=pmg)

    # Create self-improvement pipeline
    pipeline = SelfImprovementPipeline(
        pmg=pmg,
        online_learner=online_learner,
        feedback_system=feedback_system,
        adaptive_engine=adaptive_engine,
        knowledge_augmentation=knowledge_engine,
        schedule_time=time(2, 0),  # 2 AM
    )

    # Run nightly improvement (normally scheduled)
    print("\n--- Running Nightly Improvement Cycle ---")

    stats = pipeline.run_nightly_improvement()

    print(f"\nImprovement cycle completed:")
    print(f"  - Duration: {stats['duration_seconds']:.1f} seconds")
    print(f"  - Jobs executed: {stats['jobs_executed']}")
    print(f"  - Models deployed: {stats['models_deployed']}")
    print(f"  - Training samples: {stats.get('training_samples_generated', 0)}")

    if stats.get('deployments'):
        print("\n  Deployments:")
        for deploy in stats['deployments']:
            print(f"    - {deploy['doc_type']}: {deploy['version']} "
                  f"({deploy['deployment_type']})")


def example_6_integrated_ecosystem():
    """Example 6: Complete integrated learning ecosystem."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Integrated Learning Ecosystem")
    print("=" * 80)

    # Initialize all components
    pmg = ProcessMemoryGraph()

    online_learner = OnlineLearningEngine(
        pmg=pmg,
        uncertainty_threshold=0.7,
        active_learning_enabled=True,
    )

    feedback_system = FeedbackLoopSystem(
        pmg=pmg,
        confidence_threshold=0.8,
        ab_test_enabled=True,
    )

    adaptive_engine = AdaptiveLearningEngine(
        pmg=pmg,
        drift_threshold=0.25,
        min_accuracy=0.85,
    )

    knowledge_engine = KnowledgeAugmentationEngine(pmg=pmg)

    # Create continuous learner with integrations
    continuous_learner = ContinuousLearner(pmg=pmg)

    continuous_learner.integrate_with_learning_ecosystem(
        online_learner=online_learner,
        feedback_system=feedback_system,
        adaptive_engine=adaptive_engine,
        knowledge_augmentation=knowledge_engine,
    )

    # Run continuous improvement cycle
    print("\n--- Running Continuous Improvement Cycle ---")

    cycle_stats = continuous_learner.continuous_improvement_cycle(
        doc_types=["PURCHASE_ORDER", "SUPPLIER_INVOICE"]
    )

    print(f"\nCycle completed:")
    print(f"  - Timestamp: {cycle_stats['timestamp']}")
    print(f"  - Retraining triggered: {cycle_stats['retraining_triggered']}")

    for doc_type, stats in cycle_stats['learning_stats'].items():
        print(f"\n  {doc_type}:")
        print(f"    - Samples: {stats.get('num_samples', 0)}")
        print(f"    - Drift detected: {stats.get('drift_detected', False)}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("SAP_LLM AUTO-LEARNING AND CONTINUOUS IMPROVEMENT")
    print("Comprehensive Examples")
    print("=" * 80)

    # Run examples
    example_1_online_learning()
    example_2_feedback_loop()
    example_3_adaptive_learning()
    example_4_knowledge_augmentation()
    example_5_self_improvement()
    example_6_integrated_ecosystem()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
