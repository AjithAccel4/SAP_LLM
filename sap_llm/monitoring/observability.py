"""
Advanced Monitoring & Observability System

Implements comprehensive observability with:
- Prometheus metrics (RED metrics: Rate, Errors, Duration)
- Distributed tracing (OpenTelemetry)
- Structured logging (JSON format)
- Custom business metrics
- Anomaly detection
- SLO tracking

Target: Full visibility into system performance and behavior
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from functools import wraps
import json

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest
)
from opentelemetry import trace, metrics as otel_metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """
    Prometheus metrics collector for SAP_LLM

    Tracks:
    - Request rates and latencies
    - Error rates and types
    - Cache performance
    - Model inference time
    - Resource utilization
    """

    def __init__(self):
        self.registry = CollectorRegistry()

        # Request metrics (RED)
        self.requests_total = Counter(
            'sap_llm_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'sap_llm_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )

        self.errors_total = Counter(
            'sap_llm_errors_total',
            'Total number of errors',
            ['error_type', 'stage'],
            registry=self.registry
        )

        # Cache metrics
        self.cache_hits = Counter(
            'sap_llm_cache_hits_total',
            'Total cache hits',
            ['tier'],
            registry=self.registry
        )

        self.cache_misses = Counter(
            'sap_llm_cache_misses_total',
            'Total cache misses',
            registry=self.registry
        )

        self.cache_hit_rate = Gauge(
            'sap_llm_cache_hit_rate',
            'Current cache hit rate',
            ['tier'],
            registry=self.registry
        )

        # Model metrics
        self.model_inference_duration = Histogram(
            'sap_llm_model_inference_duration_seconds',
            'Model inference duration',
            ['model_type', 'stage'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry
        )

        self.model_accuracy = Gauge(
            'sap_llm_model_accuracy',
            'Model accuracy score',
            ['stage'],
            registry=self.registry
        )

        # Pipeline stage metrics
        self.stage_duration = Histogram(
            'sap_llm_stage_duration_seconds',
            'Pipeline stage duration',
            ['stage'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )

        self.stage_success_rate = Gauge(
            'sap_llm_stage_success_rate',
            'Stage success rate',
            ['stage'],
            registry=self.registry
        )

        # Resource metrics
        self.gpu_utilization = Gauge(
            'sap_llm_gpu_utilization_percent',
            'GPU utilization percentage',
            ['gpu_id'],
            registry=self.registry
        )

        self.gpu_memory_used = Gauge(
            'sap_llm_gpu_memory_used_bytes',
            'GPU memory used in bytes',
            ['gpu_id'],
            registry=self.registry
        )

        self.cpu_usage = Gauge(
            'sap_llm_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'sap_llm_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )

        # Business metrics
        self.documents_processed = Counter(
            'sap_llm_documents_processed_total',
            'Total documents processed',
            ['doc_type', 'status'],
            registry=self.registry
        )

        self.processing_cost = Counter(
            'sap_llm_processing_cost_dollars',
            'Total processing cost in dollars',
            registry=self.registry
        )

        self.sla_violations = Counter(
            'sap_llm_sla_violations_total',
            'Total SLA violations',
            ['metric'],
            registry=self.registry
        )

    def record_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Record HTTP request metrics"""
        self.requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()

        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_cache_hit(self, tier: str):
        """Record cache hit"""
        self.cache_hits.labels(tier=tier).inc()

    def record_cache_miss(self):
        """Record cache miss"""
        self.cache_misses.inc()

    def update_cache_hit_rate(self, tier: str, rate: float):
        """Update cache hit rate gauge"""
        self.cache_hit_rate.labels(tier=tier).set(rate)

    def record_model_inference(
        self,
        model_type: str,
        stage: str,
        duration: float
    ):
        """Record model inference time"""
        self.model_inference_duration.labels(
            model_type=model_type,
            stage=stage
        ).observe(duration)

    def record_error(self, error_type: str, stage: str):
        """Record error"""
        self.errors_total.labels(
            error_type=error_type,
            stage=stage
        ).inc()

    def record_document_processed(
        self,
        doc_type: str,
        status: str,
        cost: float
    ):
        """Record document processing"""
        self.documents_processed.labels(
            doc_type=doc_type,
            status=status
        ).inc()

        self.processing_cost.inc(cost)

    def record_sla_violation(self, metric: str):
        """Record SLA violation"""
        self.sla_violations.labels(metric=metric).inc()

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in exposition format"""
        return generate_latest(self.registry)


class DistributedTracing:
    """
    OpenTelemetry distributed tracing

    Provides:
    - End-to-end request tracing across 8 pipeline stages
    - Parent-child span relationships
    - Trace context propagation
    - Span attributes and events
    """

    def __init__(self, service_name: str = "sap-llm"):
        # Initialize tracer provider
        trace.set_tracer_provider(TracerProvider())

        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint="http://jaeger:4317",
            insecure=True
        )

        # Add span processor
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(otlp_exporter)
        )

        self.tracer = trace.get_tracer(service_name)

    def trace_request(self, name: str):
        """Decorator to trace a request"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(name) as span:
                    # Add attributes
                    span.set_attribute("function", func.__name__)

                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("status", "error")
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        span.record_exception(e)
                        raise

            return wrapper
        return decorator

    def create_span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None
    ):
        """Create a new span"""
        span = self.tracer.start_span(name)

        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        return span


class AnomalyDetector:
    """
    ML-based anomaly detection for metrics

    Detects:
    - Sudden latency spikes
    - Error rate increases
    - Throughput drops
    - Resource utilization anomalies
    """

    def __init__(self):
        self.baseline_metrics = {}
        self.threshold_multiplier = 3.0  # 3 standard deviations

    async def detect_anomalies(
        self,
        metrics: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in current metrics

        Args:
            metrics: Current metric values

        Returns:
            List of detected anomalies
        """
        anomalies = []

        for metric_name, current_value in metrics.items():
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]

                # Z-score anomaly detection
                z_score = abs(
                    (current_value - baseline['mean']) / baseline['std']
                )

                if z_score > self.threshold_multiplier:
                    anomalies.append({
                        'metric': metric_name,
                        'current_value': current_value,
                        'baseline_mean': baseline['mean'],
                        'z_score': z_score,
                        'severity': self._calculate_severity(z_score),
                        'timestamp': datetime.now().isoformat()
                    })

        return anomalies

    def update_baseline(self, metric_name: str, value: float):
        """Update baseline statistics for a metric"""
        if metric_name not in self.baseline_metrics:
            self.baseline_metrics[metric_name] = {
                'values': [],
                'mean': 0,
                'std': 0
            }

        baseline = self.baseline_metrics[metric_name]
        baseline['values'].append(value)

        # Keep last 1000 values
        if len(baseline['values']) > 1000:
            baseline['values'].pop(0)

        # Recalculate statistics
        import numpy as np
        baseline['mean'] = np.mean(baseline['values'])
        baseline['std'] = np.std(baseline['values']) or 1.0

    def _calculate_severity(self, z_score: float) -> str:
        """Calculate anomaly severity"""
        if z_score > 5.0:
            return "CRITICAL"
        elif z_score > 4.0:
            return "HIGH"
        elif z_score > 3.0:
            return "MEDIUM"
        else:
            return "LOW"


class SLOTracker:
    """
    Service Level Objective (SLO) tracking

    Tracks:
    - Availability SLO (99.99%)
    - Latency SLO (P95 < 100ms)
    - Accuracy SLO (>95%)
    - Error rate SLO (<1%)
    """

    def __init__(self):
        self.slos = {
            'availability': {
                'target': 0.9999,
                'current': 1.0,
                'window': timedelta(days=30)
            },
            'latency_p95': {
                'target': 0.100,  # 100ms
                'current': 0.030,
                'window': timedelta(hours=1)
            },
            'accuracy': {
                'target': 0.95,
                'current': 0.97,
                'window': timedelta(days=7)
            },
            'error_rate': {
                'target': 0.01,  # <1%
                'current': 0.002,
                'window': timedelta(hours=1)
            }
        }

        self.violations = []

    def check_slos(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check if SLOs are being met

        Args:
            metrics: Current metrics

        Returns:
            List of SLO violations
        """
        violations = []

        # Check availability SLO
        if 'availability' in metrics:
            if metrics['availability'] < self.slos['availability']['target']:
                violations.append({
                    'slo': 'availability',
                    'target': self.slos['availability']['target'],
                    'current': metrics['availability'],
                    'severity': 'CRITICAL'
                })

        # Check latency SLO
        if 'latency_p95' in metrics:
            if metrics['latency_p95'] > self.slos['latency_p95']['target']:
                violations.append({
                    'slo': 'latency_p95',
                    'target': self.slos['latency_p95']['target'],
                    'current': metrics['latency_p95'],
                    'severity': 'HIGH'
                })

        # Check accuracy SLO
        if 'accuracy' in metrics:
            if metrics['accuracy'] < self.slos['accuracy']['target']:
                violations.append({
                    'slo': 'accuracy',
                    'target': self.slos['accuracy']['target'],
                    'current': metrics['accuracy'],
                    'severity': 'HIGH'
                })

        # Check error rate SLO
        if 'error_rate' in metrics:
            if metrics['error_rate'] > self.slos['error_rate']['target']:
                violations.append({
                    'slo': 'error_rate',
                    'target': self.slos['error_rate']['target'],
                    'current': metrics['error_rate'],
                    'severity': 'MEDIUM'
                })

        return violations

    def get_error_budget(self, slo_name: str) -> Dict[str, Any]:
        """
        Calculate error budget for SLO

        Error budget = (1 - SLO target) × time window
        """
        if slo_name not in self.slos:
            return {}

        slo = self.slos[slo_name]

        # Calculate error budget
        error_budget = (1 - slo['target']) * 100  # Percentage

        # Calculate consumption
        if slo_name == 'availability':
            consumed = (1 - slo['current']) * 100
        elif slo_name == 'accuracy':
            consumed = (slo['target'] - slo['current']) * 100
        else:
            consumed = slo['current'] * 100

        remaining = max(0, error_budget - consumed)

        return {
            'slo': slo_name,
            'target': slo['target'],
            'error_budget': error_budget,
            'consumed': consumed,
            'remaining': remaining,
            'window': str(slo['window'])
        }


class ObservabilityManager:
    """
    Unified observability manager

    Integrates:
    - Metrics collection
    - Distributed tracing
    - Anomaly detection
    - SLO tracking
    """

    def __init__(self):
        self.metrics = MetricsCollector()
        self.tracing = DistributedTracing()
        self.anomaly_detector = AnomalyDetector()
        self.slo_tracker = SLOTracker()

        # Start background monitoring
        asyncio.create_task(self._continuous_monitoring())

    async def _continuous_monitoring(self):
        """Continuously monitor system health"""
        while True:
            await asyncio.sleep(60)  # Check every minute

            try:
                # Collect current metrics
                current_metrics = await self._collect_current_metrics()

                # Detect anomalies
                anomalies = await self.anomaly_detector.detect_anomalies(
                    current_metrics
                )

                if anomalies:
                    logger.warning(f"Detected {len(anomalies)} anomalies")
                    for anomaly in anomalies:
                        logger.warning(f"Anomaly: {json.dumps(anomaly)}")

                # Check SLOs
                violations = self.slo_tracker.check_slos(current_metrics)

                if violations:
                    logger.error(f"SLO violations: {len(violations)}")
                    for violation in violations:
                        logger.error(f"SLO violation: {json.dumps(violation)}")
                        self.metrics.record_sla_violation(violation['slo'])

                # Update baselines
                for metric_name, value in current_metrics.items():
                    self.anomaly_detector.update_baseline(metric_name, value)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")

    async def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        # This would query actual metrics from the system
        # Placeholder implementation
        return {
            'latency_p95': 0.030,
            'error_rate': 0.002,
            'availability': 0.9999,
            'accuracy': 0.974
        }

    def trace_request(self, name: str):
        """Decorator for tracing requests"""
        return self.tracing.trace_request(name)

    def get_prometheus_metrics(self) -> bytes:
        """Get Prometheus metrics"""
        return self.metrics.get_metrics()

    def get_slo_status(self) -> Dict[str, Any]:
        """Get current SLO status"""
        return {
            slo_name: self.slo_tracker.get_error_budget(slo_name)
            for slo_name in self.slo_tracker.slos.keys()
        }


# Global observability instance
observability = ObservabilityManager()
