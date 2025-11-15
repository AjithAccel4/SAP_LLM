# SAP_LLM Auto-Learning and Continuous Improvement System

## Overview

The SAP_LLM learning system is a comprehensive, production-ready framework that enables truly adaptive AI through continuous learning, real-time feedback, and automated self-improvement. The system operates autonomously to improve model performance without human intervention.

## Architecture

The learning system consists of six integrated components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Self-Improvement Pipeline                    │
│  (Orchestrates nightly retraining and model deployment)         │
└────────────┬────────────────────────────────────────────────────┘
             │
     ┌───────┴───────┐
     │               │
     ▼               ▼
┌─────────────┐ ┌──────────────────┐ ┌──────────────────────┐
│   Online    │ │   Feedback Loop  │ │  Adaptive Learning   │
│  Learning   │ │     System       │ │      Engine          │
│  Engine     │ │                  │ │                      │
│             │ │ - User feedback  │ │ - Drift detection    │
│ - Real-time │ │ - A/B testing    │ │ - Performance        │
│   updates   │ │ - Model versions │ │   monitoring         │
│ - Active    │ │ - Auto retrain   │ │ - Tenant learning    │
│   learning  │ │                  │ │                      │
└─────────────┘ └──────────────────┘ └──────────────────────┘
     │               │                        │
     └───────────────┴────────────────────────┘
                     │
                     ▼
     ┌───────────────────────────────────────┐
     │   Knowledge Augmentation Engine       │
     │                                       │
     │ - Pattern extraction                 │
     │ - Dictionary building                │
     │ - Field mapping learning             │
     │ - Validation rule discovery          │
     │ - Training data generation           │
     └───────────────────────────────────────┘
                     │
                     ▼
     ┌───────────────────────────────────────┐
     │      Process Memory Graph (PMG)       │
     │   (Stores all learning artifacts)     │
     └───────────────────────────────────────┘
```

## Components

### 1. Online Learning Engine (`online_learning.py`)

Real-time model updates from user corrections and predictions.

**Features:**
- **Incremental Learning**: Update models without full retraining using SGD-based classifiers
- **Active Learning**: Identify uncertain predictions for human review
- **Transfer Learning**: Bootstrap new document types from similar models
- **Few-Shot Learning**: Learn from minimal examples (2-10 samples)
- **Uncertainty Quantification**: Track prediction confidence

**Key Methods:**
```python
# Update from user correction
online_learner.update_from_correction(
    doc_type="INVOICE",
    features={"amount": 1250.50, "supplier": "SUP123"},
    correct_label="SUPPLIER_INVOICE",
    predicted_label="PURCHASE_ORDER",
    confidence=0.65
)

# Identify uncertain predictions
prediction, confidence, is_uncertain = online_learner.identify_uncertain_predictions(
    doc_type="INVOICE",
    features=features
)

# Transfer learning
online_learner.transfer_learning(
    source_doc_type="PURCHASE_ORDER",
    target_doc_type="SALES_ORDER",
    similarity_threshold=0.7
)
```

### 2. Feedback Loop System (`feedback_loop.py`)

Comprehensive user feedback collection and A/B testing framework.

**Features:**
- **Multi-Channel Feedback**: Thumbs up/down, corrections, comments, validations
- **Confidence-Based Requests**: Automatically request feedback for uncertain predictions
- **A/B Testing**: Test model improvements before full deployment
- **Model Versioning**: Track and rollback model versions
- **Automatic Retraining**: Trigger retraining based on feedback volume and performance

**Feedback Types:**
- `THUMBS_UP` / `THUMBS_DOWN`: Quick sentiment feedback
- `CORRECTION`: User provides correct value
- `COMMENT`: Textual feedback
- `VALIDATION`: User validates prediction

**Key Methods:**
```python
# Collect feedback
feedback_system.collect_feedback(
    feedback_type=FeedbackType.CORRECTION,
    doc_id="doc_12345",
    doc_type="INVOICE",
    user_id="user_001",
    prediction="1250.00",
    correct_value="1250.50",
    confidence=0.92
)

# Start A/B test
ab_test = feedback_system.start_ab_test(
    test_name="invoice_v2",
    doc_type="INVOICE",
    champion_version="v1.0.0",
    challenger_version="v2.0.0",
    duration_hours=24,
    min_improvement=0.02
)

# Route traffic
version, variant = feedback_system.route_ab_test("INVOICE", "doc_12345")
```

### 3. Adaptive Learning Engine (`adaptive_learning.py`)

Real-time performance monitoring and drift detection.

**Features:**
- **Real-Time Metrics**: Track accuracy, F1, precision, recall per document type
- **Concept Drift Detection**: Detect when model assumptions become invalid
- **Distribution Shift Monitoring**: Identify changes in data distribution
- **Automatic Refresh**: Trigger model updates when performance degrades
- **Tenant-Specific Learning**: Per-customer model customization

**Drift Detection Methods:**
- **Kolmogorov-Smirnov Test**: Statistical test for distribution differences
- **Population Stability Index (PSI)**: Industry-standard drift metric
- **Chi-Square Test**: Alternative distribution comparison

**Key Methods:**
```python
# Track predictions
adaptive_engine.track_prediction(
    doc_type="INVOICE",
    prediction="SUPPLIER_INVOICE",
    actual="SUPPLIER_INVOICE",
    confidence=0.95,
    features={"amount": 1000.0}
)

# Check performance degradation
degraded, metrics = adaptive_engine.check_performance_degradation("INVOICE")

# Detect drift
drift_results = adaptive_engine.check_drift("INVOICE", method='psi')

# Get performance summary
summary = adaptive_engine.get_performance_summary()
```

### 4. Knowledge Augmentation Engine (`knowledge_augmentation.py`)

Extract knowledge from production data to improve models.

**Features:**
- **Pattern Extraction**: Auto-discover regex patterns from successful extractions
- **Dictionary Building**: Build custom dictionaries (suppliers, products, etc.)
- **Field Mapping Learning**: Learn how fields map between systems
- **Validation Rule Discovery**: Automatically discover business rules
- **Training Data Generation**: Generate synthetic data for augmentation

**Pattern Types:**
- Date patterns (YYYY-MM-DD, DD/MM/YYYY, etc.)
- Number patterns (amounts with currency, decimals)
- Code patterns (PO numbers, SKUs, part numbers)
- Email and phone patterns

**Key Methods:**
```python
# Extract patterns
patterns = knowledge_engine.pattern_extractor.extract_patterns(
    field_name="invoice_number",
    values=["INV-2024-001", "INV-2024-002", ...]
)

# Build dictionaries
supplier_dict = knowledge_engine.dictionary_builder.build_supplier_dictionary(
    supplier_names=["ACME Corp", ...],
    supplier_ids=["SUP001", ...]
)

# Learn field mappings
knowledge_engine.field_mapping_learner.learn_mapping(
    doc_type="INVOICE",
    source_field="total_amount",
    target_field="SAP_TOTAL_AMT",
    examples=[("1250.50", "1250.50"), ...]
)

# Discover validation rules
rules = knowledge_engine.validation_rule_learner.discover_rules(
    field_name="total_amount",
    values=[100.0, 250.0, 500.0],
    valid_flags=[True, True, True]
)
```

### 5. Self-Improvement Pipeline (`self_improvement.py`)

Orchestrates automated model improvement without human intervention.

**Features:**
- **Nightly Retraining**: Scheduled model updates (default: 2 AM)
- **Synthetic Data Generation**: Augment training data for edge cases
- **Curriculum Learning**: Train from easy to hard examples
- **Ensemble Learning**: Combine multiple model versions
- **Meta-Learning**: Fast adaptation to new document types
- **Automatic Deployment**: Deploy improved models with A/B testing

**Training Strategies:**
- `FULL_RETRAIN`: Complete model retraining
- `INCREMENTAL`: Incremental updates
- `TRANSFER`: Transfer from similar document type
- `FEW_SHOT`: Learn from few examples
- `ENSEMBLE`: Combine multiple models
- `META`: Meta-learning for fast adaptation

**Key Methods:**
```python
# Run nightly improvement
pipeline = SelfImprovementPipeline(
    pmg=pmg,
    online_learner=online_learner,
    feedback_system=feedback_system,
    adaptive_engine=adaptive_engine,
    knowledge_augmentation=knowledge_engine,
    schedule_time=time(2, 0)  # 2 AM
)

stats = pipeline.run_nightly_improvement()

# Check improvement history
history = pipeline.get_improvement_history(days=30)
```

### 6. Continuous Learner (Enhanced `pmg/learning.py`)

Integrates all components for seamless operation.

**Key Methods:**
```python
# Initialize and integrate
continuous_learner = ContinuousLearner(pmg=pmg)

continuous_learner.integrate_with_learning_ecosystem(
    online_learner=online_learner,
    feedback_system=feedback_system,
    adaptive_engine=adaptive_engine,
    knowledge_augmentation=knowledge_engine
)

# Run improvement cycle
stats = continuous_learner.continuous_improvement_cycle(
    doc_types=["PURCHASE_ORDER", "SUPPLIER_INVOICE"]
)
```

## Production Deployment

### 1. Basic Setup

```python
from sap_llm.learning import (
    AdaptiveLearningEngine,
    FeedbackLoopSystem,
    KnowledgeAugmentationEngine,
    OnlineLearningEngine,
    SelfImprovementPipeline,
)
from sap_llm.pmg import ProcessMemoryGraph, ContinuousLearner

# Initialize PMG
pmg = ProcessMemoryGraph(
    endpoint=os.getenv("COSMOS_ENDPOINT"),
    key=os.getenv("COSMOS_KEY")
)

# Initialize learning components
online_learner = OnlineLearningEngine(pmg=pmg)
feedback_system = FeedbackLoopSystem(pmg=pmg)
adaptive_engine = AdaptiveLearningEngine(pmg=pmg)
knowledge_engine = KnowledgeAugmentationEngine(pmg=pmg)

# Create pipeline
pipeline = SelfImprovementPipeline(
    pmg=pmg,
    online_learner=online_learner,
    feedback_system=feedback_system,
    adaptive_engine=adaptive_engine,
    knowledge_augmentation=knowledge_engine,
)
```

### 2. Real-Time Learning Integration

```python
# In your document processing pipeline

# 1. Make prediction
prediction, confidence = model.predict(document)

# 2. Track prediction (for monitoring)
adaptive_engine.track_prediction(
    doc_type=doc_type,
    prediction=prediction,
    actual=None,  # Will be updated later
    confidence=confidence,
    features=extracted_features
)

# 3. Request feedback if uncertain
if confidence < 0.8:
    feedback_system.request_feedback(
        doc_id=doc_id,
        doc_type=doc_type,
        prediction=prediction,
        confidence=confidence
    )

# 4. Process user correction
if user_provides_correction:
    online_learner.update_from_correction(
        doc_type=doc_type,
        features=extracted_features,
        correct_label=user_correction,
        predicted_label=prediction,
        confidence=confidence
    )
```

### 3. Scheduled Improvement

```python
# Schedule nightly improvement (using scheduler like APScheduler)
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import time

scheduler = BackgroundScheduler()

def nightly_improvement_job():
    """Run nightly self-improvement."""
    stats = pipeline.run_nightly_improvement()

    # Send notifications if needed
    if stats['models_deployed'] > 0:
        send_notification(f"Deployed {stats['models_deployed']} improved models")

# Schedule for 2 AM daily
scheduler.add_job(
    nightly_improvement_job,
    'cron',
    hour=2,
    minute=0
)

scheduler.start()
```

### 4. Monitoring and Alerts

```python
# Get alerts
alerts = adaptive_engine.get_alerts(limit=100)

for alert in alerts:
    if alert['alert_type'] == 'model_refresh_needed':
        print(f"Alert: {alert['doc_type']} needs refresh - {alert['reason']}")

        # Optionally trigger immediate improvement
        # instead of waiting for nightly run

# Get performance summary
summary = adaptive_engine.get_performance_summary()

for doc_type, metrics in summary.items():
    if metrics['needs_refresh']:
        print(f"Warning: {doc_type} performance degraded")
        print(f"  Current: {metrics['metrics']['accuracy']:.3f}")
        print(f"  Baseline: {metrics['baseline']:.3f}")
```

## Configuration

### Environment Variables

```bash
# Cosmos DB (for PMG)
export COSMOS_ENDPOINT="https://your-account.gremlin.cosmos.azure.com:443/"
export COSMOS_KEY="your-cosmos-key"

# Model storage
export MODEL_STORE_PATH="/path/to/model/storage"

# Learning parameters
export LEARNING_CONFIDENCE_THRESHOLD="0.8"
export LEARNING_DRIFT_THRESHOLD="0.25"
export LEARNING_MIN_ACCURACY="0.85"
```

### Learning Configuration

```python
# Fine-tune learning parameters
online_learner = OnlineLearningEngine(
    pmg=pmg,
    uncertainty_threshold=0.7,  # Request feedback below this
    active_learning_enabled=True,
)

feedback_system = FeedbackLoopSystem(
    pmg=pmg,
    confidence_threshold=0.8,  # Request feedback below this
    drift_threshold=0.15,  # Retrain if drift > 15%
    min_samples_for_retrain=100,  # Min feedback samples
    ab_test_enabled=True,  # Enable A/B testing
    ab_test_split=0.5,  # 50/50 traffic split
)

adaptive_engine = AdaptiveLearningEngine(
    pmg=pmg,
    performance_window=1000,  # Track last 1000 predictions
    drift_window=500,  # Drift detection window
    drift_threshold=0.25,  # PSI threshold
    refresh_threshold=0.15,  # Refresh if accuracy drops 15%
    min_accuracy=0.85,  # Minimum acceptable accuracy
)
```

## Performance Metrics

The system tracks comprehensive metrics:

- **Accuracy**: Overall prediction accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of actual positives identified
- **Confidence**: Average prediction confidence
- **Drift Score**: Distribution shift magnitude (PSI)
- **Update Count**: Number of model updates
- **Feedback Volume**: Amount of user feedback

## Best Practices

### 1. Gradual Rollout

Always use A/B testing for model updates:

```python
# Start with small improvement threshold
ab_test = feedback_system.start_ab_test(
    champion_version="v1.0.0",
    challenger_version="v1.1.0",
    min_improvement=0.01,  # 1% improvement required
    duration_hours=24
)
```

### 2. Monitor Drift Continuously

Set up continuous drift monitoring:

```python
# Check drift daily
drift_results = adaptive_engine.check_drift(
    doc_type="INVOICE",
    method='psi'  # Population Stability Index
)

if drift_results['drift_detected']:
    # Trigger immediate retraining
    pipeline.run_nightly_improvement()
```

### 3. Collect Quality Feedback

Prioritize high-value feedback:

```python
# Only request feedback for important predictions
if confidence < 0.8 and document_amount > 10000:
    feedback_system.request_feedback(...)
```

### 4. Use Synthetic Data Carefully

Generate synthetic data for edge cases:

```python
# Generate synthetic samples for rare cases
synthetic = pipeline.synthetic_generator.generate_synthetic_samples(
    doc_type="RARE_INVOICE_TYPE",
    base_samples=production_samples,
    num_samples=1000,
    strategies=["edge_case_generation"]  # Focus on edge cases
)
```

### 5. Version Control Models

Track all model versions:

```python
# Create version with metadata
version = feedback_system.create_model_version(
    doc_type="INVOICE",
    metrics={"accuracy": 0.95, "f1": 0.93},
    parent_version="v1.0.0"
)

# Rollback if needed
if production_issues:
    feedback_system.rollback_version("INVOICE")
```

## Troubleshooting

### Issue: Models not improving

**Solution:**
1. Check if enough feedback is being collected
2. Verify drift detection thresholds aren't too high
3. Ensure training data quality is high
4. Review synthetic data generation parameters

### Issue: Too many uncertain predictions

**Solution:**
1. Lower uncertainty threshold
2. Increase training data volume
3. Use transfer learning from similar document types
4. Review feature extraction quality

### Issue: Performance degradation after update

**Solution:**
1. Immediately rollback to previous version
2. Review A/B test results
3. Check for data quality issues
4. Increase minimum improvement threshold

## API Reference

See individual module documentation:
- `/home/user/SAP_LLM/sap_llm/learning/online_learning.py`
- `/home/user/SAP_LLM/sap_llm/learning/feedback_loop.py`
- `/home/user/SAP_LLM/sap_llm/learning/adaptive_learning.py`
- `/home/user/SAP_LLM/sap_llm/learning/knowledge_augmentation.py`
- `/home/user/SAP_LLM/sap_llm/learning/self_improvement.py`

## Examples

See comprehensive examples:
- `/home/user/SAP_LLM/examples/learning_system_example.py`

## Support

For issues or questions:
1. Check logs in `/tmp/sap_llm/improvement_stats/`
2. Review performance metrics in adaptive engine
3. Examine A/B test results in feedback system
4. Check PMG for historical data

## License

Copyright 2024 SAP_LLM. All rights reserved.
