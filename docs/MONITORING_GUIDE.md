# SAP_LLM Monitoring and Observability Guide

## Table of Contents

- [Overview](#overview)
- [Monitoring Architecture](#monitoring-architecture)
- [Prometheus Setup](#prometheus-setup)
- [Grafana Dashboards](#grafana-dashboards)
- [Alert Rules](#alert-rules)
- [Service Level Objectives (SLOs)](#service-level-objectives-slos)
- [Log Aggregation](#log-aggregation)
- [Distributed Tracing](#distributed-tracing)
- [Custom Metrics](#custom-metrics)
- [Runbooks](#runbooks)
- [Troubleshooting](#troubleshooting)

## Overview

This guide provides comprehensive monitoring and observability setup for SAP_LLM, ensuring system reliability, performance, and quick incident response.

### Monitoring Goals

- **Availability**: 99.9% uptime SLO
- **Performance**: P95 latency < 1.5 seconds
- **Reliability**: Error rate < 0.1%
- **Cost**: Cost per document < $0.005
- **Data Quality**: Schema compliance > 99%

### Observability Pillars

```
┌──────────────────────────────────────────────────────┐
│                 SAP_LLM Observability                │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Metrics   │  │    Logs     │  │   Traces    │  │
│  │             │  │             │  │             │  │
│  │ Prometheus  │  │ ELK / Loki  │  │   Jaeger    │  │
│  │   Grafana   │  │  Grafana    │  │   Tempo     │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
│         ▲                ▲                 ▲         │
│         │                │                 │         │
│         └────────────────┴─────────────────┘         │
│                          │                           │
│                   ┌──────▼──────┐                    │
│                   │ Alertmanager│                    │
│                   │  PagerDuty  │                    │
│                   └─────────────┘                    │
└──────────────────────────────────────────────────────┘
```

## Monitoring Architecture

### Component Overview

#### 1. Metrics (Prometheus + Grafana)

**Purpose**: Real-time metrics collection and visualization

**Components**:
- Prometheus Server: Metrics collection and storage
- Grafana: Visualization and dashboards
- Alertmanager: Alert routing and notification
- Node Exporter: System metrics
- cAdvisor: Container metrics

**Deployment**:
```yaml
# Docker Compose (included in docker-compose.yml)
services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./deployments/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./deployments/monitoring/grafana:/etc/grafana/provisioning
```

#### 2. Logs (ELK Stack or Loki)

**Purpose**: Centralized log aggregation and analysis

**Options**:

**Option A: ELK Stack**
- Elasticsearch: Log storage and indexing
- Logstash: Log processing pipeline
- Kibana: Log visualization

**Option B: Grafana Loki** (Recommended)
- Loki: Log aggregation
- Promtail: Log shipping
- Grafana: Unified visualization with metrics

#### 3. Traces (Jaeger or Tempo)

**Purpose**: Distributed tracing for request flow analysis

**Components**:
- Jaeger/Tempo: Trace collection and storage
- OpenTelemetry: Instrumentation library

## Prometheus Setup

### Installation and Configuration

#### Prometheus Configuration

Create `/home/user/SAP_LLM/deployments/monitoring/prometheus.yml`:

```yaml
# Prometheus Configuration for SAP_LLM

global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'sap-llm-prod'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Load rules
rule_files:
  - "/etc/prometheus/rules/*.yml"

# Scrape configurations
scrape_configs:
  # SAP_LLM API Server
  - job_name: 'sap-llm-api'
    static_configs:
      - targets:
          - 'sap-llm-api:8000'
    metrics_path: '/metrics'
    scrape_interval: 10s

  # SAP_LLM SHWL Service
  - job_name: 'sap-llm-shwl'
    static_configs:
      - targets:
          - 'sap-llm-shwl:8001'
    metrics_path: '/metrics'

  # MongoDB Exporter
  - job_name: 'mongodb'
    static_configs:
      - targets:
          - 'mongodb-exporter:9216'

  # Redis Exporter
  - job_name: 'redis'
    static_configs:
      - targets:
          - 'redis-exporter:9121'

  # Node Exporter (System Metrics)
  - job_name: 'node'
    static_configs:
      - targets:
          - 'node-exporter:9100'

  # cAdvisor (Container Metrics)
  - job_name: 'cadvisor'
    static_configs:
      - targets:
          - 'cadvisor:8080'

  # Kubernetes Pods (if using Kubernetes)
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - sap-llm
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__

# Remote write (for long-term storage)
remote_write:
  - url: "https://prometheus-remote-write.azure.com/api/v1/write"
    queue_config:
      capacity: 10000
      max_shards: 50
      min_shards: 1
      max_samples_per_send: 5000
      batch_send_deadline: 5s
```

#### Alert Rules

Create `/home/user/SAP_LLM/deployments/monitoring/rules/alerts.yml`:

```yaml
# SAP_LLM Alert Rules

groups:
  - name: sap_llm_alerts
    interval: 30s
    rules:
      # High-Severity Alerts

      - alert: ServiceDown
        expr: up{job=~"sap-llm.*"} == 0
        for: 1m
        labels:
          severity: critical
          component: api
        annotations:
          summary: "SAP_LLM service is down"
          description: "{{ $labels.job }} has been down for more than 1 minute."
          runbook: "https://runbooks.sap-llm.com/service-down"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
          component: api
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes."
          runbook: "https://runbooks.sap-llm.com/high-error-rate"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.5
        for: 10m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value | humanizeDuration }} over the last 10 minutes."
          runbook: "https://runbooks.sap-llm.com/high-latency"

      - alert: DatabaseConnectionFailure
        expr: mongodb_up == 0
        for: 2m
        labels:
          severity: critical
          component: database
        annotations:
          summary: "MongoDB connection failure"
          description: "Cannot connect to MongoDB for more than 2 minutes."
          runbook: "https://runbooks.sap-llm.com/database-connection"

      - alert: ModelLoadingFailure
        expr: model_loading_errors_total > 0
        for: 1m
        labels:
          severity: critical
          component: model
        annotations:
          summary: "Model loading failure"
          description: "Failed to load model: {{ $labels.model_name }}"
          runbook: "https://runbooks.sap-llm.com/model-loading"

      # Medium-Severity Alerts

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 10m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}."
          runbook: "https://runbooks.sap-llm.com/high-memory"

      - alert: HighCPUUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 90
        for: 15m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "High CPU usage"
          description: "CPU usage is {{ $value }}% on {{ $labels.instance }}."
          runbook: "https://runbooks.sap-llm.com/high-cpu"

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) < 0.1
        for: 5m
        labels:
          severity: warning
          component: infrastructure
        annotations:
          summary: "Low disk space"
          description: "Disk space is {{ $value | humanizePercentage }} available on {{ $labels.instance }}."
          runbook: "https://runbooks.sap-llm.com/low-disk"

      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: warning
          component: kubernetes
        annotations:
          summary: "Pod crash looping"
          description: "Pod {{ $labels.namespace }}/{{ $labels.pod }} is crash looping."
          runbook: "https://runbooks.sap-llm.com/pod-crash"

      # Low-Severity Alerts

      - alert: SlowDatabaseQueries
        expr: mongodb_query_duration_seconds{quantile="0.95"} > 1.0
        for: 10m
        labels:
          severity: info
          component: database
        annotations:
          summary: "Slow database queries"
          description: "P95 MongoDB query duration is {{ $value | humanizeDuration }}."
          runbook: "https://runbooks.sap-llm.com/slow-queries"

      - alert: CacheMissRateHigh
        expr: rate(cache_misses_total[5m]) / rate(cache_requests_total[5m]) > 0.5
        for: 15m
        labels:
          severity: info
          component: cache
        annotations:
          summary: "High cache miss rate"
          description: "Cache miss rate is {{ $value | humanizePercentage }}."
          runbook: "https://runbooks.sap-llm.com/cache-miss"

  - name: sap_llm_slo_alerts
    interval: 1m
    rules:
      # SLO-based alerts

      - alert: AvailabilitySLOBreach
        expr: avg_over_time(up{job=~"sap-llm.*"}[30m]) < 0.999
        for: 5m
        labels:
          severity: critical
          slo: availability
        annotations:
          summary: "Availability SLO breach"
          description: "Availability is {{ $value | humanizePercentage }}, below 99.9% SLO."
          runbook: "https://runbooks.sap-llm.com/slo-availability"

      - alert: LatencySLOBreach
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.5
        for: 10m
        labels:
          severity: warning
          slo: latency
        annotations:
          summary: "Latency SLO breach"
          description: "P95 latency is {{ $value | humanizeDuration }}, above 1.5s SLO."
          runbook: "https://runbooks.sap-llm.com/slo-latency"

      - alert: ErrorRateSLOBreach
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.001
        for: 10m
        labels:
          severity: critical
          slo: error_rate
        annotations:
          summary: "Error rate SLO breach"
          description: "Error rate is {{ $value | humanizePercentage }}, above 0.1% SLO."
          runbook: "https://runbooks.sap-llm.com/slo-error-rate"
```

### Key Metrics to Monitor

#### Application Metrics

```python
# In sap_llm/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Request metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
)

# Document processing metrics
documents_processed = Counter(
    'documents_processed_total',
    'Total documents processed',
    ['document_type', 'stage', 'status']
)

document_processing_duration = Histogram(
    'document_processing_duration_seconds',
    'Document processing duration',
    ['document_type', 'stage'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# Model metrics
model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration',
    ['model_name'],
    buckets=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
)

model_loading_errors = Counter(
    'model_loading_errors_total',
    'Model loading errors',
    ['model_name']
)

# Database metrics
database_query_duration = Histogram(
    'database_query_duration_seconds',
    'Database query duration',
    ['operation', 'collection'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

database_connection_errors = Counter(
    'database_connection_errors_total',
    'Database connection errors',
    ['database']
)

# Cache metrics
cache_requests = Counter(
    'cache_requests_total',
    'Total cache requests',
    ['cache_level', 'operation']
)

cache_hits = Counter(
    'cache_hits_total',
    'Cache hits',
    ['cache_level']
)

cache_misses = Counter(
    'cache_misses_total',
    'Cache misses',
    ['cache_level']
)

# Business metrics
extraction_accuracy = Gauge(
    'extraction_accuracy',
    'Field extraction accuracy',
    ['document_type', 'field']
)

classification_confidence = Histogram(
    'classification_confidence',
    'Classification confidence scores',
    ['document_type'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

validation_errors = Counter(
    'validation_errors_total',
    'Validation errors',
    ['document_type', 'validation_type']
)

# SHWL metrics
exceptions_clustered = Counter(
    'shwl_exceptions_clustered_total',
    'Total exceptions clustered',
    ['cluster_id']
)

rules_generated = Counter(
    'shwl_rules_generated_total',
    'Total rules generated',
    ['approval_status']
)

# Cost metrics
inference_cost = Counter(
    'inference_cost_usd',
    'Inference cost in USD',
    ['model_name']
)

processing_cost = Counter(
    'processing_cost_usd',
    'Total processing cost per document',
    ['document_type']
)
```

#### System Metrics (Automatically Collected)

- **CPU**: `node_cpu_seconds_total`
- **Memory**: `node_memory_*`
- **Disk**: `node_filesystem_*`
- **Network**: `node_network_*`
- **Containers**: `container_*` (from cAdvisor)

## Grafana Dashboards

### Dashboard Setup

#### Auto-Provisioning Dashboards

Create `/home/user/SAP_LLM/deployments/monitoring/grafana/dashboards/dashboard.yaml`:

```yaml
apiVersion: 1

providers:
  - name: 'SAP_LLM Dashboards'
    orgId: 1
    folder: 'SAP_LLM'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

#### Datasource Configuration

Create `/home/user/SAP_LLM/deployments/monitoring/grafana/datasources/prometheus.yaml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: true

  - name: Jaeger
    type: jaeger
    access: proxy
    url: http://jaeger:16686
    editable: true
```

### Dashboard 1: System Overview

**Purpose**: High-level system health and performance

**Panels**:
1. **Service Status** (Stat)
   ```promql
   up{job=~"sap-llm.*"}
   ```

2. **Request Rate** (Graph)
   ```promql
   sum(rate(http_requests_total[5m])) by (job)
   ```

3. **Error Rate** (Graph)
   ```promql
   sum(rate(http_requests_total{status=~"5.."}[5m])) by (job)
   ```

4. **P95 Latency** (Graph)
   ```promql
   histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint))
   ```

5. **CPU Usage** (Graph)
   ```promql
   100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
   ```

6. **Memory Usage** (Graph)
   ```promql
   (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100
   ```

### Dashboard 2: Document Processing

**Purpose**: Monitor document processing pipeline

**Panels**:
1. **Documents Processed** (Counter)
   ```promql
   sum(increase(documents_processed_total[1h])) by (document_type)
   ```

2. **Processing Duration by Stage** (Graph)
   ```promql
   histogram_quantile(0.95, sum(rate(document_processing_duration_seconds_bucket[5m])) by (le, stage))
   ```

3. **Documents by Stage** (Pie Chart)
   ```promql
   sum(documents_processed_total) by (stage)
   ```

4. **Success Rate by Document Type** (Bar Gauge)
   ```promql
   sum(rate(documents_processed_total{status="success"}[5m])) by (document_type) /
   sum(rate(documents_processed_total[5m])) by (document_type)
   ```

5. **Extraction Accuracy** (Gauge)
   ```promql
   avg(extraction_accuracy) by (document_type)
   ```

6. **Validation Errors** (Time Series)
   ```promql
   sum(rate(validation_errors_total[5m])) by (validation_type)
   ```

### Dashboard 3: Model Performance

**Purpose**: Monitor ML model performance and resource usage

**Panels**:
1. **Inference Duration by Model** (Heatmap)
   ```promql
   sum(rate(model_inference_duration_seconds_bucket[5m])) by (le, model_name)
   ```

2. **Model Loading Status** (Stat)
   ```promql
   sum(model_loading_errors_total) by (model_name)
   ```

3. **GPU Utilization** (Graph)
   ```promql
   dcgm_gpu_utilization
   ```

4. **GPU Memory Usage** (Graph)
   ```promql
   dcgm_fb_used / dcgm_fb_total * 100
   ```

5. **Classification Confidence Distribution** (Histogram)
   ```promql
   sum(rate(classification_confidence_bucket[5m])) by (le)
   ```

6. **Inference Cost** (Counter)
   ```promql
   sum(increase(inference_cost_usd[1h])) by (model_name)
   ```

### Dashboard 4: Database & Cache

**Purpose**: Monitor database and cache performance

**Panels**:
1. **MongoDB Connection Status** (Stat)
   ```promql
   mongodb_up
   ```

2. **MongoDB Query Duration** (Graph)
   ```promql
   histogram_quantile(0.95, sum(rate(database_query_duration_seconds_bucket[5m])) by (le, operation))
   ```

3. **Redis Cache Hit Rate** (Gauge)
   ```promql
   sum(rate(cache_hits_total[5m])) /
   (sum(rate(cache_hits_total[5m])) + sum(rate(cache_misses_total[5m])))
   ```

4. **Cache Requests by Level** (Stacked Graph)
   ```promql
   sum(rate(cache_requests_total[5m])) by (cache_level)
   ```

5. **MongoDB Collections Size** (Table)
   ```promql
   mongodb_database_collection_size_bytes
   ```

6. **Database Connection Pool** (Graph)
   ```promql
   mongodb_connections{state=~"current|available"}
   ```

### Dashboard 5: SHWL (Self-Healing)

**Purpose**: Monitor self-healing workflow loop

**Panels**:
1. **Exceptions Detected** (Counter)
   ```promql
   sum(increase(shwl_exceptions_clustered_total[1h]))
   ```

2. **Rules Generated** (Time Series)
   ```promql
   sum(rate(shwl_rules_generated_total[5m])) by (approval_status)
   ```

3. **Exception Clusters** (Pie Chart)
   ```promql
   sum(shwl_exceptions_clustered_total) by (cluster_id)
   ```

4. **Auto-Approval Rate** (Gauge)
   ```promql
   sum(rate(shwl_rules_generated_total{approval_status="auto_approved"}[5m])) /
   sum(rate(shwl_rules_generated_total[5m]))
   ```

### Dashboard 6: Business Metrics

**Purpose**: Business-level KPIs and SLOs

**Panels**:
1. **Daily Documents Processed** (Stat)
   ```promql
   sum(increase(documents_processed_total[24h]))
   ```

2. **Revenue Impact** (Graph)
   ```promql
   sum(increase(documents_processed_total{status="success"}[1h])) * 0.005
   ```

3. **Cost per Document** (Gauge)
   ```promql
   sum(increase(processing_cost_usd[1h])) /
   sum(increase(documents_processed_total[1h]))
   ```

4. **SLO Compliance** (Bar Gauge)
   - Availability: `avg_over_time(up{job=~"sap-llm.*"}[30d])`
   - Latency: `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[30d])) by (le))`
   - Error Rate: `sum(rate(http_requests_total{status=~"5.."}[30d])) / sum(rate(http_requests_total[30d]))`

5. **Schema Compliance Rate** (Time Series)
   ```promql
   sum(rate(documents_processed_total{status="success"}[5m])) /
   sum(rate(documents_processed_total[5m]))
   ```

## Alert Rules

### Alertmanager Configuration

Create `/home/user/SAP_LLM/deployments/monitoring/alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m
  slack_api_url: '${SLACK_WEBHOOK_URL}'
  pagerduty_url: 'https://events.pagerduty.com/v2/enqueue'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

  routes:
    # Critical alerts -> PagerDuty
    - match:
        severity: critical
      receiver: 'pagerduty-critical'
      continue: true

    # Critical alerts -> Slack
    - match:
        severity: critical
      receiver: 'slack-critical'

    # Warning alerts -> Slack
    - match:
        severity: warning
      receiver: 'slack-warnings'

    # Info alerts -> Slack (low priority)
    - match:
        severity: info
      receiver: 'slack-info'

receivers:
  - name: 'default'
    slack_configs:
      - channel: '#sap-llm-alerts'
        title: 'SAP_LLM Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        description: '{{ .CommonAnnotations.summary }}'
        details:
          firing: '{{ .Alerts.Firing | len }}'
          resolved: '{{ .Alerts.Resolved | len }}'

  - name: 'slack-critical'
    slack_configs:
      - channel: '#sap-llm-critical'
        color: 'danger'
        title: ' CRITICAL: {{ .CommonAnnotations.summary }}'
        text: |
          {{ range .Alerts }}
          *Alert:* {{ .Labels.alertname }}
          *Severity:* {{ .Labels.severity }}
          *Summary:* {{ .Annotations.summary }}
          *Description:* {{ .Annotations.description }}
          *Runbook:* {{ .Annotations.runbook }}
          {{ end }}

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#sap-llm-alerts'
        color: 'warning'
        title: ' WARNING: {{ .CommonAnnotations.summary }}'

  - name: 'slack-info'
    slack_configs:
      - channel: '#sap-llm-info'
        color: 'good'
        title: ' INFO: {{ .CommonAnnotations.summary }}'

inhibit_rules:
  # Inhibit warning if critical firing
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

### Alert Notification Channels

**Channels**:
1. **PagerDuty**: Critical alerts requiring immediate action
2. **Slack**: All alerts for team visibility
3. **Email**: Daily summary reports
4. **Webhook**: Integration with ticketing systems (Jira, ServiceNow)

## Service Level Objectives (SLOs)

### SLO Definitions

#### 1. Availability SLO

**Target**: 99.9% uptime (43.2 minutes downtime/month)

**Measurement**:
```promql
# Error budget: 0.1% = 43.2 minutes/month
# Current availability
avg_over_time(up{job="sap-llm-api"}[30d])

# Error budget remaining
(1 - 0.999) - (1 - avg_over_time(up{job="sap-llm-api"}[30d]))
```

**Burn Rate Alerts**:
```yaml
# Fast burn (2% error budget in 1 hour)
- alert: HighAvailabilityBurnRate
  expr: (1 - avg_over_time(up{job="sap-llm-api"}[1h])) > 0.02
  labels:
    severity: critical

# Slow burn (10% error budget in 24 hours)
- alert: ModerateAvailabilityBurnRate
  expr: (1 - avg_over_time(up{job="sap-llm-api"}[24h])) > 0.10
  labels:
    severity: warning
```

#### 2. Latency SLO

**Target**: 95% of requests < 1.5 seconds

**Measurement**:
```promql
# Current P95 latency
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# Percentage of requests meeting SLO
sum(rate(http_request_duration_seconds_bucket{le="1.5"}[5m])) /
sum(rate(http_request_duration_seconds_count[5m]))
```

#### 3. Error Rate SLO

**Target**: < 0.1% error rate

**Measurement**:
```promql
# Current error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m]))

# Error budget remaining
0.001 - (sum(rate(http_requests_total{status=~"5.."}[30d])) / sum(rate(http_requests_total[30d])))
```

#### 4. Data Quality SLO

**Target**: > 99% schema compliance

**Measurement**:
```promql
# Schema compliance rate
sum(rate(documents_processed_total{status="success"}[5m])) /
sum(rate(documents_processed_total[5m]))
```

### SLO Dashboard

Create a dedicated SLO dashboard showing:
- Current SLO compliance
- Error budget remaining
- Burn rate trends
- Historical performance

## Log Aggregation

### Loki Setup (Recommended)

#### Loki Configuration

```yaml
# /home/user/SAP_LLM/deployments/monitoring/loki/config.yaml

auth_enabled: false

server:
  http_listen_port: 3100

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2024-01-01
      store: boltdb-shipper
      object_store: s3
      schema: v11
      index:
        prefix: loki_index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/index
    cache_location: /loki/cache
    shared_store: s3

  aws:
    s3: s3://${AWS_REGION}/${S3_BUCKET}
    s3forcepathstyle: true

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  ingestion_rate_mb: 10
  ingestion_burst_size_mb: 20

chunk_store_config:
  max_look_back_period: 0s

table_manager:
  retention_deletes_enabled: true
  retention_period: 2160h  # 90 days
```

#### Promtail Configuration

```yaml
# /home/user/SAP_LLM/deployments/monitoring/promtail/config.yaml

server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # SAP_LLM application logs
  - job_name: sap-llm
    static_configs:
      - targets:
          - localhost
        labels:
          job: sap-llm
          __path__: /app/logs/*.log

    pipeline_stages:
      # Parse JSON logs
      - json:
          expressions:
            timestamp: timestamp
            level: level
            message: message
            logger: logger
            request_id: request_id

      # Extract labels
      - labels:
          level:
          logger:
          request_id:

      # Set timestamp
      - timestamp:
          source: timestamp
          format: RFC3339

  # Kubernetes pod logs
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - sap-llm

    pipeline_stages:
      - docker: {}

    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
```

### Log Queries (LogQL)

Common log queries:

```logql
# Error logs in last hour
{job="sap-llm"} |= "ERROR" | json | level="ERROR"

# Specific request trace
{job="sap-llm"} | json | request_id="req-12345"

# Slow requests
{job="sap-llm"} | json | duration > 1.5

# Database errors
{job="sap-llm"} |= "database" |= "error" | json

# Top error messages
topk(10, sum by (message) (count_over_time({job="sap-llm"} |= "ERROR" [1h])))

# Error rate by endpoint
sum by (endpoint) (rate({job="sap-llm"} |= "ERROR" | json [5m]))
```

### Structured Logging

**Python Logging Configuration**:

```python
# sap_llm/logging/config.py

import logging
import json
import sys
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add extra fields
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id

        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id

        if hasattr(record, 'document_id'):
            log_data['document_id'] = record.document_id

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_data)

def setup_logging(level=logging.INFO):
    """Setup structured JSON logging"""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    logging.basicConfig(
        level=level,
        handlers=[handler]
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
```

## Distributed Tracing

### OpenTelemetry Instrumentation

```python
# sap_llm/tracing/setup.py

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.pymongo import PymongoInstrumentor

def setup_tracing(service_name="sap-llm-api"):
    """Setup OpenTelemetry tracing"""

    # Create resource
    resource = Resource(attributes={
        "service.name": service_name,
        "service.version": "1.0.0",
        "deployment.environment": "production",
    })

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )

    # Add span processor
    tracer_provider.add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )

    # Set global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Instrument libraries
    FastAPIInstrumentor.instrument()
    RequestsInstrumentor().instrument()
    PymongoInstrumentor().instrument()

    return trace.get_tracer(__name__)
```

### Trace Context Propagation

```python
# In your API handlers

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

@app.post("/api/v1/process")
async def process_document(document: Document):
    # Start a span
    with tracer.start_as_current_span("process_document") as span:
        # Add attributes
        span.set_attribute("document.id", document.id)
        span.set_attribute("document.type", document.type)

        # Process through pipeline
        with tracer.start_as_current_span("classification"):
            result = await classifier.classify(document)
            span.set_attribute("classification.result", result)

        with tracer.start_as_current_span("extraction"):
            data = await extractor.extract(document, result)

        return data
```

## Custom Metrics

### Application-Level Metrics

```python
# sap_llm/monitoring/custom_metrics.py

from prometheus_client import Counter, Histogram, Gauge
import functools
import time

def track_processing_time(stage_name):
    """Decorator to track processing time"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                document_processing_duration.labels(
                    document_type=kwargs.get('document_type', 'unknown'),
                    stage=stage_name
                ).observe(duration)

                documents_processed.labels(
                    document_type=kwargs.get('document_type', 'unknown'),
                    stage=stage_name,
                    status=status
                ).inc()

        return wrapper
    return decorator

# Usage
@track_processing_time("classification")
async def classify_document(document):
    # Classification logic
    pass
```

## Runbooks

### Runbook Template

Each alert should have a corresponding runbook:

**Example: High Error Rate Runbook**

```markdown
# Runbook: High Error Rate

## Alert Details
- **Alert Name**: HighErrorRate
- **Severity**: Critical
- **Threshold**: Error rate > 1% for 5 minutes

## Symptoms
- Users experiencing 5xx errors
- Error rate spike in Grafana dashboard
- PagerDuty alert fired

## Impact
- Service degradation
- User-facing errors
- Potential SLO breach

## Investigation Steps

1. **Check Error Distribution**
   ```promql
   sum by (endpoint, status) (rate(http_requests_total{status=~"5.."}[5m]))
   ```

2. **Review Recent Logs**
   ```logql
   {job="sap-llm"} |= "ERROR" | json | line_format "{{.message}}"
   ```

3. **Check Dependencies**
   - MongoDB: `mongodb_up`
   - Redis: `redis_up`
   - Cosmos DB: Check Azure portal

4. **Review Recent Deployments**
   ```bash
   kubectl rollout history deployment/sap-llm-api -n sap-llm
   ```

## Resolution Steps

### If Database Connection Error:
1. Check database connectivity
2. Review connection pool settings
3. Restart application pods if needed

### If Model Loading Error:
1. Verify model files exist
2. Check GPU availability
3. Review model configuration

### If Deployment Issue:
1. Rollback to previous version
   ```bash
   kubectl rollout undo deployment/sap-llm-api -n sap-llm
   ```

## Escalation
- **L1**: On-call engineer (initial response)
- **L2**: Platform team (infrastructure issues)
- **L3**: ML team (model-related issues)
- **L4**: CTO (critical incidents)

## Post-Incident
- [ ] Create post-mortem document
- [ ] Update runbook based on findings
- [ ] Implement preventive measures
```

## Troubleshooting

### Common Issues

#### 1. Prometheus Not Scraping Metrics

**Symptoms**: Missing data in Grafana

**Check**:
```bash
# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets

# Check service endpoint
curl http://sap-llm-api:8000/metrics
```

**Fix**:
- Verify service is exposing metrics endpoint
- Check Prometheus configuration
- Verify network connectivity

#### 2. High Cardinality Metrics

**Symptoms**: Prometheus running out of memory

**Check**:
```bash
# Check metric cardinality
curl http://prometheus:9090/api/v1/label/__name__/values
```

**Fix**:
- Remove or aggregate high-cardinality labels
- Increase retention time limits
- Scale Prometheus resources

#### 3. Alert Fatigue

**Symptoms**: Too many false positive alerts

**Fix**:
- Adjust alert thresholds
- Increase `for` duration
- Add inhibition rules
- Implement proper alert routing

---

**Document Control**:
- **Version**: 1.0
- **Last Updated**: 2025-11-14
- **Owner**: SRE Team

---
*End of Monitoring Guide*
