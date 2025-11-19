# Advanced Self-Correction System

## Overview

The Advanced Self-Correction System provides autonomous error detection, multi-strategy correction, pattern learning, and human escalation for the SAP_LLM document processing pipeline.

### Key Features

- **Multi-Method Error Detection**: Confidence-based, business rules, historical inconsistencies, and anomaly detection
- **Multiple Correction Strategies**: Rule-based, re-extraction, context enhancement, and human-in-the-loop
- **Pattern Learning**: Learns from corrections to improve future accuracy
- **Smart Escalation**: Intelligently escalates to human review only when needed
- **Comprehensive Analytics**: Tracks performance and generates detailed reports

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  UnifiedExtractorModel                      │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │        SelfCorrectionEngine                        │   │
│  │                                                     │   │
│  │  ┌────────────────┐  ┌──────────────────────────┐ │   │
│  │  │ ErrorDetector  │  │ Correction Strategies:   │ │   │
│  │  │                │  │  - RuleBasedCorrection   │ │   │
│  │  │  - Confidence  │  │  - RerunHigherConf       │ │   │
│  │  │  - Rules       │  │  - ContextEnhancement    │ │   │
│  │  │  - History     │  │  - HumanInTheLoop        │ │   │
│  │  │  - Anomalies   │  └──────────────────────────┘ │   │
│  │  └────────────────┘                               │   │
│  │                                                     │   │
│  │  ┌────────────────┐  ┌──────────────────────────┐ │   │
│  │  │ PatternLearner │  │  EscalationManager       │ │   │
│  │  └────────────────┘  └──────────────────────────┘ │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │        CorrectionAnalytics                          │   │
│  └────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. ErrorDetector

Detects errors using multiple methods:

**Confidence-Based Detection**
- Identifies fields with low confidence scores
- Different thresholds for critical vs. optional fields
- Weighted scoring for business-critical fields

**Business Rule Violations**
- Total amount matches line items
- Subtotal + tax = total
- Due date after invoice date
- Required fields present

**Historical Inconsistencies**
- Compares with similar historical documents
- Identifies deviations from vendor patterns
- Detects unusual field values

**Anomaly Detection**
- Negative amounts
- Extreme values
- Future dates
- Statistical outliers

#### Usage

```python
from sap_llm.correction import ErrorDetector

detector = ErrorDetector(pmg=process_memory_graph)

error_report = detector.detect_errors(
    prediction=extracted_data,
    context={
        "document_type": "INVOICE",
        "vendor": "ACME Corp"
    }
)

if error_report.has_errors:
    print(f"Detected {len(error_report.errors)} error type(s)")
    for error in error_report.errors:
        print(f"  - {error.type} (severity: {error.severity})")
```

### 2. Correction Strategies

#### RuleBasedCorrectionStrategy

Applies deterministic business rules to fix obvious errors.

**Fixes:**
- Calculation errors (recalculate totals)
- Format issues (standardize dates/amounts)
- Negative amounts (remove sign)
- Derived fields (recompute from dependencies)

**Example:**
```python
from sap_llm.correction.strategies import RuleBasedCorrectionStrategy

strategy = RuleBasedCorrectionStrategy()
result = strategy.correct(prediction, error, context)

if result.success:
    print(f"Fixed {len(result.fields_corrected)} field(s)")
```

#### RerunWithHigherConfidenceStrategy

Re-runs extraction with better models or parameters.

**Approach:**
- Uses higher-quality models
- Focuses on specific problem fields
- Lower temperature for more deterministic results
- Enhanced visual features

**Example:**
```python
from sap_llm.correction.strategies import RerunWithHigherConfidenceStrategy

strategy = RerunWithHigherConfidenceStrategy(
    language_decoder=language_decoder,
    vision_encoder=vision_encoder
)
result = strategy.correct(prediction, error, context)
```

#### ContextEnhancementStrategy

Adds historical context from PMG to improve extraction.

**Approach:**
- Queries similar historical documents
- Extracts common patterns
- Applies vendor-specific knowledge
- Uses field co-occurrence patterns

**Example:**
```python
from sap_llm.correction.strategies import ContextEnhancementStrategy

strategy = ContextEnhancementStrategy(pmg=process_memory_graph)
result = strategy.correct(prediction, error, context)
```

#### HumanInTheLoopStrategy

Escalates to human review when automatic correction fails.

**Features:**
- Priority-based task creation
- SLA tracking
- Review queue management
- Notification system

**Example:**
```python
from sap_llm.correction.strategies import HumanInTheLoopStrategy

strategy = HumanInTheLoopStrategy(review_queue=review_queue)
result = strategy.correct(prediction, error, context)

if result.requires_human:
    print(f"Escalated to human: task_id={result.task_id}")
```

### 3. SelfCorrectionEngine

Orchestrates the complete correction workflow.

**Workflow:**
1. Detect errors using ErrorDetector
2. Try correction strategies in order of effectiveness
3. Re-validate after each correction
4. Learn from successful corrections
5. Escalate if all strategies fail

**Configuration:**
```python
from sap_llm.correction import SelfCorrectionEngine

engine = SelfCorrectionEngine(
    pmg=process_memory_graph,
    language_decoder=language_decoder,
    vision_encoder=vision_encoder,
    max_attempts=3,
    confidence_threshold=0.80,
    pattern_learner=pattern_learner
)

result = engine.correct_prediction(
    prediction=extracted_data,
    context={
        "document": image,
        "document_type": "INVOICE",
        "ocr_text": ocr_text
    },
    enable_learning=True
)

metadata = result["correction_metadata"]
print(f"Attempts: {metadata['total_attempts']}")
print(f"Success: {metadata['success']}")
print(f"Strategies tried: {metadata['strategies_tried']}")
```

### 4. ErrorPatternLearner

Learns from corrections to improve future performance.

**Capabilities:**
- Identifies correction patterns
- Tracks strategy effectiveness
- Stores successful correction examples
- Recommends best strategies for error types

**Example:**
```python
from sap_llm.correction import ErrorPatternLearner

learner = ErrorPatternLearner(
    pmg=process_memory_graph,
    storage_path="./data/correction_patterns"
)

# After a successful correction
learner.learn_from_correction(
    original_prediction=original,
    corrected_prediction=corrected,
    correction_strategy="rule_based",
    context=context
)

# Get recommendations
best_strategy = learner.suggest_correction_strategy(
    error_type="calculation_error",
    context=context
)

# View effectiveness
effectiveness = learner.get_strategy_effectiveness()
for strategy, stats in effectiveness["strategies"].items():
    print(f"{strategy}: {stats['success_rate']:.1%} success rate")
```

### 5. EscalationManager

Manages escalation to human review.

**Escalation Criteria:**
1. Low confidence after multiple attempts
2. High/critical severity errors
3. Business-critical fields with low confidence
4. Max attempts reached with unresolved errors
5. Uncorrected business rule violations

**SLA Tracking:**
- Urgent: 2 hours
- High: 8 hours
- Normal: 24 hours
- Low: 72 hours

**Example:**
```python
from sap_llm.correction import EscalationManager

manager = EscalationManager(
    review_queue=review_queue,
    pattern_learner=pattern_learner,
    confidence_threshold=0.70,
    max_auto_attempts=3
)

if manager.should_escalate(prediction, error_report, attempts=3, context=context):
    task_id = manager.escalate_to_human(
        prediction=prediction,
        error_report=error_report,
        context=context,
        priority="high"
    )
    print(f"Escalated: {task_id}")

# Process human feedback
manager.process_human_feedback(
    task_id=task_id,
    corrected_prediction=human_corrected,
    reviewer_id="reviewer_001",
    reviewer_notes="Fixed vendor ID"
)
```

### 6. CorrectionAnalytics

Tracks and reports on correction performance.

**Metrics:**
- Total corrections attempted
- Success rate
- Average attempts per correction
- Human escalation rate
- Strategy effectiveness
- Common error patterns
- Correction trends over time

**Example:**
```python
from sap_llm.correction import CorrectionAnalytics

analytics = CorrectionAnalytics(
    correction_engine=engine,
    pattern_learner=learner,
    escalation_manager=manager,
    storage_path="./data/correction_analytics"
)

# Generate report
report = analytics.generate_correction_report(
    period_days=7,
    include_details=True
)

print(f"Success Rate: {report['success_rate']:.1%}")
print(f"Escalation Rate: {report['escalation_rate']:.1%}")
print(f"Avg Attempts: {report['avg_attempts']:.1f}")

# Export report
analytics.export_report(
    report=report,
    output_path="./reports/correction_report.html",
    format="html"
)
```

## Integration with UnifiedModel

The advanced self-correction system is automatically integrated into the `UnifiedExtractorModel`:

```python
from sap_llm.models.unified_model import UnifiedExtractorModel

model = UnifiedExtractorModel(config=config)

# Process document (correction happens automatically)
result = model.process_document(
    image=image,
    ocr_text=ocr_text,
    words=words,
    boxes=boxes,
    schemas=schemas,
    api_schemas=api_schemas,
    pmg_context=pmg_context,
    enable_self_correction=True  # Enable advanced correction
)

# Check correction results
if "advanced_self_correction" in result:
    metadata = result["advanced_self_correction"]
    print(f"Correction attempted: {metadata['total_attempts']} times")
    print(f"Success: {metadata['success']}")

    if metadata.get("required_human_review"):
        print(f"Escalated to human: {metadata['human_review_task_id']}")

# Generate analytics report
report = model.get_correction_report(period_days=7)
print(f"\nWeekly Correction Report:")
print(f"  Total: {report['total_corrections']}")
print(f"  Success Rate: {report['success_rate']:.1%}")
print(f"  Escalation Rate: {report['escalation_rate']:.1%}")

# Get pending human reviews
pending = model.get_pending_human_reviews(priority="high")
print(f"\nPending high-priority reviews: {len(pending)}")
```

## Performance Metrics

### Target Metrics
- **Error Detection Accuracy**: >95%
- **Automatic Correction Rate**: >80%
- **Human Escalation Rate**: <5%
- **Correction Time**: <10 seconds per attempt

### Monitoring

```python
# Get real-time statistics
stats = engine.get_correction_stats()
print(f"Total corrections: {stats['total_corrections']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Escalation rate: {stats['escalation_rate']:.1%}")

# Get escalation statistics
escalation_stats = manager.get_escalation_stats()
print(f"Total escalations: {escalation_stats['total_escalations']}")
print(f"Completed reviews: {escalation_stats['completed_reviews']}")
print(f"SLA breaches: {escalation_stats['sla_breaches']}")

# Get pending tasks with SLA info
pending_tasks = manager.get_pending_tasks(include_sla_info=True)
for task in pending_tasks:
    print(f"Task {task['id']}: {task['sla_remaining_hours']:.1f}h remaining")
```

## Example: Complete Workflow

Here's a complete example showing the entire self-correction workflow:

```python
from sap_llm.models.unified_model import UnifiedExtractorModel
from pathlib import Path

# Initialize model
model = UnifiedExtractorModel(config=config)

# Process document
image = load_image("invoice.pdf")
ocr_text, words, boxes = extract_ocr(image)

result = model.process_document(
    image=image,
    ocr_text=ocr_text,
    words=words,
    boxes=boxes,
    schemas=document_schemas,
    api_schemas=sap_api_schemas,
    pmg_context=pmg_context,
    enable_self_correction=True
)

# Check results
print(f"Document Type: {result['doc_type']}")
print(f"Quality Score: {result['quality_score']:.2f}")

if "advanced_self_correction" in result:
    correction = result["advanced_self_correction"]

    print(f"\nCorrection Summary:")
    print(f"  Attempted: {correction['total_attempts']} time(s)")
    print(f"  Success: {correction['success']}")
    print(f"  Strategies: {', '.join(correction['strategies_tried'])}")

    if correction.get("required_human_review"):
        print(f"  ⚠️  Escalated to human review")
        print(f"  Task ID: {correction['human_review_task_id']}")
    else:
        print(f"  ✓ Automatically corrected")
        print(f"  Final confidence: {correction['final_confidence']:.2f}")

# Generate weekly report
report = model.get_correction_report(
    period_days=7,
    export_path="./reports/weekly_correction_report.html",
    export_format="html"
)

print(f"\nWeekly Correction Report:")
print(f"  Total corrections: {report['total_corrections']}")
print(f"  Success rate: {report['success_rate']:.1%}")
print(f"  Avg attempts: {report.get('avg_attempts', 0):.1f}")
print(f"  Escalation rate: {report['escalation_rate']:.1%}")
```

## Best Practices

### 1. Enable Learning
Always enable pattern learning to improve performance over time:
```python
result = engine.correct_prediction(prediction, context, enable_learning=True)
```

### 2. Provide Rich Context
Include as much context as possible for better corrections:
```python
context = {
    "document": image,
    "document_type": doc_type,
    "ocr_text": ocr_text,
    "words": words,
    "boxes": boxes,
    "vendor": vendor_name,
    "pmg": pmg_context
}
```

### 3. Monitor Performance
Regularly review correction analytics:
```python
report = analytics.generate_correction_report(period_days=7)
# Review strategy effectiveness
# Identify common error patterns
# Adjust thresholds if needed
```

### 4. Process Human Feedback Promptly
Process human corrections to improve learning:
```python
manager.process_human_feedback(
    task_id=task_id,
    corrected_prediction=corrected,
    reviewer_id=reviewer_id,
    reviewer_notes=notes
)
```

### 5. Tune Thresholds
Adjust thresholds based on your use case:
```python
engine = SelfCorrectionEngine(
    max_attempts=3,           # More attempts for higher accuracy
    confidence_threshold=0.85  # Higher threshold for critical applications
)
```

## Troubleshooting

### High Escalation Rate

If escalation rate is too high (>10%):
1. Lower confidence threshold
2. Add more correction strategies
3. Improve pattern learning with more historical data
4. Review error patterns to identify systematic issues

### Low Success Rate

If automatic correction success rate is low (<70%):
1. Ensure language decoder and vision encoder are available
2. Check that PMG has sufficient historical data
3. Review strategy effectiveness and adjust order
4. Increase max_attempts

### Slow Performance

If corrections are taking too long (>10s):
1. Reduce max_attempts
2. Disable expensive strategies (re-extraction)
3. Use faster models for re-extraction
4. Cache PMG queries

## API Reference

See individual module documentation for detailed API reference:
- `sap_llm.correction.error_detector`
- `sap_llm.correction.strategies`
- `sap_llm.correction.correction_engine`
- `sap_llm.correction.pattern_learner`
- `sap_llm.correction.escalation`
- `sap_llm.correction.analytics`

## Future Enhancements

Planned improvements:
- [ ] Active learning from human corrections
- [ ] Confidence calibration
- [ ] Multi-model ensemble corrections
- [ ] Real-time anomaly detection
- [ ] A/B testing for strategies
- [ ] Automated threshold optimization
