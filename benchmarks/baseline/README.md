# Enterprise Performance Baseline Results

This directory contains baseline performance results for regression detection.

## Purpose
These baseline results serve as the reference point for detecting performance regressions in CI/CD pipelines.

## Industry Standards Applied

### Resource Thresholds (AWS Well-Architected Framework):
- **CPU Usage**: <70% (allows 30% headroom for traffic surges)
- **Memory Usage**: <70% (prevents OOM during peak load)
- **GPU Utilization**: >80% (target during active processing)
- **Disk Usage**: <85% (critical threshold)

### Latency Targets:
- **P95 Latency**: <600ms for document processing
- **P99 Latency**: <800ms (allows for tail latency)
- Web API standard is 200ms, but document processing is more intensive

### Throughput Targets:
- **Sustained Throughput**: ≥100,000 documents/minute
- **Scaling Efficiency**: ≥85% at 4 workers

### Accuracy Targets:
- **Classification**: ≥99%
- **Extraction F1**: ≥97%
- **Routing**: ≥99.5%

### Error Rates (Industry Standard):
- **Error Rate**: <1% (indicates reliability issues if exceeded)
- **Failure Rate**: <0.1% (critical threshold)

## Regression Detection

The `check_regressions.py` script compares current results against these baselines:

### Thresholds:
- **10% degradation** triggers a warning
- **20% degradation** fails the CI/CD pipeline

### Statistical Significance:
- Uses threshold-based detection (simpler than E-Divisive Means but effective)
- Tracks trends over time to identify gradual degradation

## Usage

```bash
# Compare current results against baseline
python benchmarks/scripts/check_regressions.py \
    --current benchmarks/results \
    --baseline benchmarks/baseline \
    --threshold 0.10
```

## Baseline Update Policy

Baselines should be updated when:
1. Major architectural changes are made
2. Performance improvements are validated
3. Target SLAs are revised

**Never update baselines to mask regressions!**

## Files

- `baseline_latency.json` - Latency benchmark baseline
- `baseline_throughput.json` - Throughput benchmark baseline
- `baseline_accuracy.json` - Accuracy metrics baseline
- `baseline_resources.json` - Resource usage baseline
- `README.md` - This file

## Compliance

These baselines conform to:
- AWS Well-Architected Framework (2025)
- Industry standard SLA targets
- Enterprise performance testing best practices
