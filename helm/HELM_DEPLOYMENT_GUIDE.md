# SAP LLM Helm Deployment Guide

Complete guide for deploying SAP LLM to Kubernetes using Helm across multiple environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Environment-Specific Deployments](#environment-specific-deployments)
4. [Configuration](#configuration)
5. [Security](#security)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Scaling and Performance](#scaling-and-performance)
8. [Backup and Disaster Recovery](#backup-and-disaster-recovery)
9. [Troubleshooting](#troubleshooting)
10. [Upgrade Strategy](#upgrade-strategy)

## Prerequisites

### Required Tools

- Kubernetes cluster v1.24+
- Helm v3.10+
- kubectl v1.24+
- GPU support (NVIDIA GPU Operator for GPU environments)
- cert-manager (for TLS certificates)
- ingress-nginx controller
- Prometheus Operator (for monitoring)

### Cluster Requirements

#### Development
- 1 node with 4 CPU, 16GB RAM
- No GPU required
- 100GB storage

#### Staging
- 3 nodes with 8 CPU, 32GB RAM, 1 GPU (A100) per node
- 1TB storage

#### Production
- 10+ nodes with 16 CPU, 64GB RAM, 1 GPU (H100) per node
- 10TB+ storage
- Multi-zone deployment recommended

### Installation

```bash
# Add required Helm repositories
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add neo4j https://helm.neo4j.com/neo4j
helm repo add jetstack https://charts.jetstack.io
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install cert-manager (if not already installed)
kubectl create namespace cert-manager
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --version v1.13.0 \
  --set installCRDs=true

# Install ingress-nginx (if not already installed)
kubectl create namespace ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --set controller.service.type=LoadBalancer

# Install Prometheus Operator (if not already installed)
kubectl create namespace monitoring
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

## Quick Start

### Development Deployment

```bash
# Create namespace
kubectl create namespace sap-llm-dev

# Create secrets
kubectl create secret generic sap-llm-redis-secret \
  --from-literal=redis-password=dev-redis-password \
  -n sap-llm-dev

kubectl create secret generic sap-llm-neo4j-secret \
  --from-literal=neo4j-password=dev-neo4j-password \
  -n sap-llm-dev

kubectl create secret generic sap-llm-postgres-secret \
  --from-literal=username=postgres \
  --from-literal=password=dev-postgres-password \
  -n sap-llm-dev

kubectl create secret generic sap-llm-minio-secret \
  --from-literal=access-key=minioadmin \
  --from-literal=secret-key=minioadmin \
  -n sap-llm-dev

# Deploy with dev values
helm install sap-llm ./sap-llm \
  --namespace sap-llm-dev \
  --values sap-llm/values-dev.yaml \
  --create-namespace \
  --timeout 10m

# Check deployment status
kubectl get pods -n sap-llm-dev
helm status sap-llm -n sap-llm-dev

# Port-forward to access application
kubectl port-forward -n sap-llm-dev svc/sap-llm 8000:8000
```

### Staging Deployment

```bash
# Create namespace
kubectl create namespace sap-llm-staging

# Create secrets from external secret manager (recommended)
# Example using AWS Secrets Manager
kubectl create secret generic sap-llm-redis-secret \
  --from-literal=redis-password=$(aws secretsmanager get-secret-value \
    --secret-id staging/sap-llm/redis --query SecretString --output text) \
  -n sap-llm-staging

# Or use sealed-secrets for GitOps
# kubeseal --format=yaml < secrets.yaml > sealed-secrets.yaml

# Deploy with staging values
helm install sap-llm ./sap-llm \
  --namespace sap-llm-staging \
  --values sap-llm/values-staging.yaml \
  --create-namespace \
  --timeout 15m

# Verify deployment
kubectl get all -n sap-llm-staging
kubectl get ingress -n sap-llm-staging
```

### Production Deployment

```bash
# Create namespace with labels
kubectl create namespace sap-llm-prod
kubectl label namespace sap-llm-prod environment=production

# Create secrets using external secrets operator (recommended)
cat <<EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: sap-llm-secrets
  namespace: sap-llm-prod
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: sap-llm-redis-secret
  data:
  - secretKey: redis-password
    remoteRef:
      key: prod/sap-llm/redis
      property: password
EOF

# Deploy with production values
helm install sap-llm ./sap-llm \
  --namespace sap-llm-prod \
  --values sap-llm/values-production.yaml \
  --create-namespace \
  --timeout 20m \
  --wait

# Verify critical components
kubectl get pods,svc,ingress,pdb,hpa -n sap-llm-prod
kubectl get servicemonitor -n sap-llm-prod
```

## Environment-Specific Deployments

### Development Environment

**Purpose**: Local development and testing

**Key Features**:
- Single replica
- No GPU required
- Minimal resource requests
- Relaxed security
- Full debug logging
- No persistence for caches
- Embedded dependencies

**Access**:
```bash
# Port-forward for local access
kubectl port-forward -n sap-llm-dev svc/sap-llm 8000:8000

# Test the API
curl http://localhost:8000/health
```

**Customization**:
```bash
# Override specific values
helm upgrade sap-llm ./sap-llm \
  --namespace sap-llm-dev \
  --values sap-llm/values-dev.yaml \
  --set replicaCount=2 \
  --set config.logging.level=TRACE
```

### Staging Environment

**Purpose**: Pre-production testing and validation

**Key Features**:
- 2 replicas with autoscaling
- A100 GPU support
- Production-like configuration
- Full monitoring enabled
- Canary deployment support
- Moderate resource allocation

**Domain Setup**:
```bash
# Update DNS to point to ingress
kubectl get ingress -n sap-llm-staging

# Test HTTPS endpoint
curl https://sap-llm-staging.example.com/health
```

**Testing**:
```bash
# Run smoke tests
kubectl run -it --rm test-pod \
  --image=curlimages/curl \
  --restart=Never \
  -n sap-llm-staging \
  -- curl http://sap-llm:8000/health

# Check metrics
kubectl port-forward -n sap-llm-staging svc/sap-llm 9090:9090
curl http://localhost:9090/metrics
```

### Production Environment

**Purpose**: Live production workloads

**Key Features**:
- 5+ replicas with aggressive autoscaling
- H100 GPU support
- Maximum security hardening
- Full observability stack
- Multi-zone distribution
- Comprehensive backup strategy
- Resource quotas and limits

**High Availability Verification**:
```bash
# Check pod distribution across zones
kubectl get pods -n sap-llm-prod -o wide

# Verify PodDisruptionBudget
kubectl get pdb -n sap-llm-prod

# Test failover
kubectl drain <node-name> --ignore-daemonsets
# Verify pods are rescheduled
kubectl get pods -n sap-llm-prod
```

**Production Checklist**:
- [ ] Secrets managed by external system (AWS Secrets Manager, Vault)
- [ ] TLS certificates configured
- [ ] Monitoring alerts configured
- [ ] Backup schedule verified
- [ ] Resource quotas in place
- [ ] Network policies active
- [ ] Security scanning completed
- [ ] Load testing performed
- [ ] Disaster recovery plan documented
- [ ] On-call rotation established

## Configuration

### Custom Values Files

Create environment-specific overrides:

```yaml
# my-custom-values.yaml
global:
  environment: staging
  region: eu-west-1

image:
  tag: "v1.2.3"

resources:
  requests:
    cpu: "6"
    memory: "24Gi"

ingress:
  hosts:
    - host: sap-llm.mycompany.com
      paths:
        - path: /
          pathType: Prefix
```

Deploy with multiple values files:
```bash
helm upgrade sap-llm ./sap-llm \
  --namespace sap-llm-staging \
  --values sap-llm/values-staging.yaml \
  --values my-custom-values.yaml
```

### Model Configuration

Configure model download source:

```yaml
config:
  model:
    name: "qwen25vl-sap-llm-v1"
    downloadSource: "s3://my-bucket/models/qwen25vl"  # S3
    # downloadSource: "gs://my-bucket/models/qwen25vl"  # GCS
    # downloadSource: "hf://Qwen/Qwen2-VL-7B-Instruct"  # Hugging Face
    # downloadSource: "https://example.com/model.tar.gz"  # HTTP
```

### Feature Flags

Enable/disable features per environment:

```yaml
features:
  pmg_enabled: true           # Process Memory Graph
  rlhf_enabled: true          # RLHF training
  shwl_enabled: true          # Self-Healing Workflow Loop
  apop_enabled: true          # APOP autonomous agents
  multilingual_enabled: false # Multi-language support
  online_learning_enabled: false  # Online learning
```

### Resource Tuning

Optimize resources based on workload:

```yaml
# GPU-optimized configuration
resources:
  requests:
    nvidia.com/gpu: "1"
    cpu: "8"
    memory: "32Gi"
  limits:
    nvidia.com/gpu: "1"
    cpu: "16"
    memory: "64Gi"

gpu:
  enabled: true
  type: "nvidia-h100-80gb"
  runtimeClass: nvidia

# CPU-only configuration (inference on CPU)
resources:
  requests:
    cpu: "16"
    memory: "64Gi"
  limits:
    cpu: "32"
    memory: "128Gi"

gpu:
  enabled: false
```

## Security

### Secrets Management

#### Option 1: External Secrets Operator (Recommended for Production)

```bash
# Install External Secrets Operator
helm install external-secrets \
  external-secrets/external-secrets \
  -n external-secrets-system \
  --create-namespace

# Create SecretStore
cat <<EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets-manager
  namespace: sap-llm-prod
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-east-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets
EOF

# Create ExternalSecret
cat <<EOF | kubectl apply -f -
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: sap-llm-secrets
  namespace: sap-llm-prod
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: SecretStore
  target:
    name: sap-llm-redis-secret
    creationPolicy: Owner
  dataFrom:
  - extract:
      key: prod/sap-llm/credentials
EOF
```

#### Option 2: Sealed Secrets (GitOps-friendly)

```bash
# Install sealed-secrets
helm install sealed-secrets \
  sealed-secrets/sealed-secrets \
  -n kube-system

# Create sealed secret
kubectl create secret generic sap-llm-secrets \
  --from-literal=redis-password=supersecret \
  --dry-run=client -o yaml | \
  kubeseal -o yaml > sealed-secrets.yaml

# Commit to git
git add sealed-secrets.yaml
git commit -m "Add sealed secrets"
```

#### Option 3: HashiCorp Vault

```bash
# Install Vault
helm install vault hashicorp/vault \
  --set "injector.enabled=true"

# Configure Vault integration
kubectl exec -it vault-0 -- vault auth enable kubernetes
kubectl exec -it vault-0 -- vault write auth/kubernetes/config \
    kubernetes_host="https://$KUBERNETES_PORT_443_TCP_ADDR:443"
```

### Network Policies

Verify network policies are enforced:

```bash
# Test network isolation
kubectl run test-pod --image=busybox -n default -- sleep 3600
kubectl exec -it test-pod -n default -- wget -O- http://sap-llm.sap-llm-prod:8000/health
# Should fail if network policy is working

# Test from allowed namespace
kubectl run test-pod --image=busybox -n ingress-nginx -- sleep 3600
kubectl exec -it test-pod -n ingress-nginx -- wget -O- http://sap-llm.sap-llm-prod:8000/health
# Should succeed
```

### RBAC

Verify service account permissions:

```bash
# Check service account
kubectl get sa -n sap-llm-prod

# Verify role bindings
kubectl get rolebindings -n sap-llm-prod

# Test permissions
kubectl auth can-i get secrets \
  --as=system:serviceaccount:sap-llm-prod:sap-llm-prod \
  -n sap-llm-prod
```

### Pod Security

Check security contexts:

```bash
# Verify pod security
kubectl get pods -n sap-llm-prod -o jsonpath='{.items[0].spec.securityContext}'

# Check container security
kubectl get pods -n sap-llm-prod -o jsonpath='{.items[0].spec.containers[0].securityContext}'
```

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus metrics:

```bash
# Port-forward to Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# Query metrics
curl http://localhost:9090/api/v1/query?query=sap_llm_requests_total

# Check ServiceMonitor
kubectl get servicemonitor -n sap-llm-prod
```

### Grafana Dashboards

Access Grafana:

```bash
# Get Grafana password
kubectl get secret -n monitoring prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 --decode

# Port-forward to Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Open http://localhost:3000
# Login with admin / <password>
```

Import SAP LLM dashboard:
- Dashboard ID: Create custom dashboard for SAP LLM metrics
- Key metrics:
  - Request rate and latency
  - Error rate
  - Extraction accuracy (F1 score)
  - GPU utilization
  - Model inference time
  - Queue depth

### Distributed Tracing

Enable Jaeger for distributed tracing:

```bash
# Install Jaeger
kubectl create namespace observability
helm install jaeger jaegertracing/jaeger \
  --namespace observability \
  --set cassandra.persistence.enabled=true

# Port-forward to Jaeger UI
kubectl port-forward -n observability svc/jaeger-query 16686:16686

# Open http://localhost:16686
```

### Logging

Centralized logging with ELK/Loki:

```bash
# Install Loki stack
helm install loki grafana/loki-stack \
  --namespace monitoring \
  --set promtail.enabled=true \
  --set grafana.enabled=true

# Query logs
kubectl logs -n sap-llm-prod -l app=sap-llm --tail=100 -f

# Filter by level
kubectl logs -n sap-llm-prod -l app=sap-llm | jq 'select(.level=="ERROR")'
```

### Alerts

Configure alerting:

```yaml
# prometheus-rules.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: sap-llm-alerts
  namespace: sap-llm-prod
spec:
  groups:
  - name: sap-llm
    interval: 30s
    rules:
    - alert: HighErrorRate
      expr: rate(sap_llm_requests_total{status=~"5.."}[5m]) > 0.01
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "High error rate detected"
        description: "Error rate is {{ $value }} errors per second"

    - alert: HighLatency
      expr: histogram_quantile(0.95, sap_llm_request_duration_seconds_bucket) > 2
      for: 5m
      labels:
        severity: warning
      annotations:
        summary: "High latency detected"
        description: "95th percentile latency is {{ $value }} seconds"
```

Apply alerts:
```bash
kubectl apply -f prometheus-rules.yaml
```

## Scaling and Performance

### Horizontal Pod Autoscaling

The HPA is configured to scale based on:
- CPU utilization (70%)
- Memory utilization (75%)
- GPU utilization (80%)

Verify HPA:
```bash
# Check HPA status
kubectl get hpa -n sap-llm-prod

# Describe HPA for details
kubectl describe hpa sap-llm -n sap-llm-prod

# Watch HPA in action
kubectl get hpa -n sap-llm-prod -w
```

Custom metrics scaling:
```bash
# Scale based on request queue depth
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sap-llm-custom
  namespace: sap-llm-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sap-llm
  minReplicas: 5
  maxReplicas: 100
  metrics:
  - type: Pods
    pods:
      metric:
        name: sap_llm_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
EOF
```

### Vertical Pod Autoscaling

Install VPA:
```bash
git clone https://github.com/kubernetes/autoscaler.git
cd autoscaler/vertical-pod-autoscaler
./hack/vpa-up.sh
```

Create VPA:
```bash
kubectl apply -f - <<EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: sap-llm-vpa
  namespace: sap-llm-prod
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sap-llm
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: sap-llm
      minAllowed:
        cpu: 4
        memory: 16Gi
      maxAllowed:
        cpu: 32
        memory: 128Gi
EOF
```

### Cluster Autoscaling

Configure cluster autoscaler for GPU nodes:

```yaml
# cluster-autoscaler-values.yaml
autoDiscovery:
  clusterName: sap-llm-cluster

extraArgs:
  balance-similar-node-groups: true
  skip-nodes-with-system-pods: false

nodeSelector:
  node-role.kubernetes.io/master: ""

tolerations:
  - effect: NoSchedule
    key: node-role.kubernetes.io/master
```

Install:
```bash
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --values cluster-autoscaler-values.yaml
```

### Performance Tuning

GPU optimization:
```yaml
# Enable GPU sharing (MIG)
gpu:
  enabled: true
  type: "nvidia-a100-40gb"
  runtimeClass: nvidia
  migStrategy: "mixed"  # Enable MIG for better utilization

resources:
  limits:
    nvidia.com/mig-1g.5gb: "1"  # Use MIG slice
```

Batch processing optimization:
```yaml
config:
  triton:
    enabled: true
    instances: 4
    max_batch_size: 32
    preferred_batch_sizes: [8, 16, 32]
    max_queue_delay_microseconds: 100
    dynamic_batching: true
```

## Backup and Disaster Recovery

### Velero Backup

Install Velero:
```bash
# Install Velero CLI
wget https://github.com/vmware-tanzu/velero/releases/download/v1.12.0/velero-v1.12.0-linux-amd64.tar.gz
tar -xvf velero-v1.12.0-linux-amd64.tar.gz
sudo mv velero-v1.12.0-linux-amd64/velero /usr/local/bin/

# Install Velero server
velero install \
  --provider aws \
  --plugins velero/velero-plugin-for-aws:v1.8.0 \
  --bucket sap-llm-backups \
  --secret-file ./credentials-velero \
  --use-volume-snapshots=true \
  --backup-location-config region=us-east-1
```

Create backup schedule:
```bash
# Backup entire namespace daily
velero schedule create sap-llm-daily \
  --schedule="0 1 * * *" \
  --include-namespaces sap-llm-prod \
  --ttl 720h0m0s

# Backup just PVCs hourly
velero schedule create sap-llm-pvc-hourly \
  --schedule="0 * * * *" \
  --include-namespaces sap-llm-prod \
  --include-resources pvc,pv \
  --ttl 168h0m0s
```

Test restore:
```bash
# Create backup
velero backup create sap-llm-test --include-namespaces sap-llm-prod

# Simulate disaster
kubectl delete namespace sap-llm-prod

# Restore
velero restore create --from-backup sap-llm-test

# Verify
kubectl get all -n sap-llm-prod
```

### Database Backups

PostgreSQL backup:
```bash
# Manual backup
kubectl exec -n sap-llm-prod sap-llm-postgresql-0 -- \
  pg_dump -U postgres sap_llm | gzip > sap-llm-db-$(date +%Y%m%d).sql.gz

# Automated with CronJob
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgres-backup
  namespace: sap-llm-prod
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:14
            command:
            - /bin/sh
            - -c
            - |
              pg_dump -h sap-llm-postgresql -U postgres sap_llm | \
              gzip | \
              aws s3 cp - s3://sap-llm-backups/postgres/backup-\$(date +%Y%m%d-%H%M%S).sql.gz
            env:
            - name: PGPASSWORD
              valueFrom:
                secretKeyRef:
                  name: sap-llm-postgres-secret
                  key: password
          restartPolicy: OnFailure
EOF
```

Neo4j backup:
```bash
# Create backup
kubectl exec -n sap-llm-prod sap-llm-neo4j-0 -- \
  neo4j-admin backup --backup-dir=/backups --name=graph-backup

# Copy to S3
kubectl cp sap-llm-prod/sap-llm-neo4j-0:/backups ./neo4j-backup
aws s3 sync ./neo4j-backup s3://sap-llm-backups/neo4j/
```

## Troubleshooting

### Common Issues

#### Pods not starting

```bash
# Check pod status
kubectl get pods -n sap-llm-prod

# Describe pod for events
kubectl describe pod <pod-name> -n sap-llm-prod

# Check logs
kubectl logs <pod-name> -n sap-llm-prod
kubectl logs <pod-name> -n sap-llm-prod --previous  # Previous container

# Check init containers
kubectl logs <pod-name> -n sap-llm-prod -c model-downloader
```

Common causes:
- Image pull errors: Check `imagePullSecrets`
- Resource constraints: Check node resources
- Init container failures: Check model download
- Volume mount issues: Check PVC status

#### GPU not available

```bash
# Check GPU nodes
kubectl get nodes -l accelerator=nvidia-h100

# Verify NVIDIA device plugin
kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset

# Check GPU resources
kubectl describe node <gpu-node-name> | grep nvidia.com/gpu

# Verify runtime class
kubectl get runtimeclass nvidia
```

#### High latency

```bash
# Check HPA status
kubectl get hpa -n sap-llm-prod

# Check resource usage
kubectl top pods -n sap-llm-prod

# Check for throttling
kubectl get pods -n sap-llm-prod -o jsonpath='{.items[*].status.containerStatuses[*].state}'

# Review metrics
kubectl port-forward -n sap-llm-prod svc/sap-llm 9090:9090
curl http://localhost:9090/metrics | grep sap_llm_request_duration
```

#### Database connection issues

```bash
# Check database pods
kubectl get pods -n sap-llm-prod -l app=postgresql

# Test connectivity
kubectl run -it --rm debug \
  --image=postgres:14 \
  --restart=Never \
  -n sap-llm-prod \
  -- psql -h sap-llm-postgresql -U postgres -d sap_llm

# Check secrets
kubectl get secret sap-llm-postgres-secret -n sap-llm-prod -o yaml
```

### Debug Mode

Enable debug logging:
```bash
helm upgrade sap-llm ./sap-llm \
  --namespace sap-llm-prod \
  --reuse-values \
  --set config.logging.level=DEBUG

# Check logs
kubectl logs -n sap-llm-prod -l app=sap-llm -f
```

### Performance Profiling

```bash
# CPU profiling
kubectl exec -it <pod-name> -n sap-llm-prod -- \
  curl http://localhost:8000/debug/pprof/profile?seconds=30 > cpu.prof

# Memory profiling
kubectl exec -it <pod-name> -n sap-llm-prod -- \
  curl http://localhost:8000/debug/pprof/heap > mem.prof

# Analyze
go tool pprof cpu.prof
```

## Upgrade Strategy

### Rolling Update (Zero Downtime)

```bash
# Update image version
helm upgrade sap-llm ./sap-llm \
  --namespace sap-llm-prod \
  --reuse-values \
  --set image.tag=v1.2.0 \
  --wait \
  --timeout 15m

# Monitor rollout
kubectl rollout status deployment/sap-llm -n sap-llm-prod

# Check pods
kubectl get pods -n sap-llm-prod -w
```

### Blue-Green Deployment

```bash
# Deploy green version
helm install sap-llm-green ./sap-llm \
  --namespace sap-llm-prod \
  --values sap-llm/values-production.yaml \
  --set image.tag=v1.2.0 \
  --set fullnameOverride=sap-llm-green

# Test green deployment
kubectl run test -it --rm --image=curlimages/curl -n sap-llm-prod -- \
  curl http://sap-llm-green:8000/health

# Switch traffic (update ingress)
kubectl patch ingress sap-llm -n sap-llm-prod \
  --type=json \
  -p='[{"op": "replace", "path": "/spec/rules/0/http/paths/0/backend/service/name", "value": "sap-llm-green"}]'

# Monitor metrics
# If successful, delete blue
helm uninstall sap-llm -n sap-llm-prod

# If issues, rollback
kubectl patch ingress sap-llm -n sap-llm-prod \
  --type=json \
  -p='[{"op": "replace", "path": "/spec/rules/0/http/paths/0/backend/service/name", "value": "sap-llm"}]'
```

### Canary Deployment

Enable canary:
```yaml
# values-production-canary.yaml
canary:
  enabled: true
  weight: 10  # 10% traffic to canary
  analysis:
    interval: 1m
    threshold: 5
    metrics:
      - name: request-success-rate
        threshold: 99
      - name: request-duration
        threshold: 2000
```

Deploy canary:
```bash
helm upgrade sap-llm ./sap-llm \
  --namespace sap-llm-prod \
  --values sap-llm/values-production.yaml \
  --values values-production-canary.yaml \
  --set image.tag=v1.2.0
```

### Rollback

```bash
# Check release history
helm history sap-llm -n sap-llm-prod

# Rollback to previous version
helm rollback sap-llm -n sap-llm-prod

# Rollback to specific revision
helm rollback sap-llm 3 -n sap-llm-prod

# Verify
kubectl get pods -n sap-llm-prod
helm status sap-llm -n sap-llm-prod
```

## Advanced Topics

### Multi-Region Deployment

Deploy to multiple regions with global load balancing:

```bash
# Deploy to us-east-1
helm install sap-llm ./sap-llm \
  --namespace sap-llm-prod \
  --values sap-llm/values-production.yaml \
  --set global.region=us-east-1 \
  --kube-context us-east-1-cluster

# Deploy to eu-west-1
helm install sap-llm ./sap-llm \
  --namespace sap-llm-prod \
  --values sap-llm/values-production.yaml \
  --set global.region=eu-west-1 \
  --kube-context eu-west-1-cluster

# Configure Route53 health checks and routing
```

### GitOps with ArgoCD

```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: sap-llm
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/AjithAccel4/SAP_LLM
    targetRevision: main
    path: helm/sap-llm
    helm:
      valueFiles:
        - values-production.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: sap-llm-prod
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

### Cost Optimization

```bash
# Use spot instances for non-critical workloads
kubectl label nodes <node-name> node-lifecycle=spot

# Configure pod affinity for spot nodes
cat <<EOF | kubectl apply -f -
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 50
      preference:
        matchExpressions:
        - key: node-lifecycle
          operator: In
          values:
          - spot
EOF

# Use cluster autoscaler with cost optimization
helm upgrade cluster-autoscaler autoscaler/cluster-autoscaler \
  --set extraArgs.expander=least-waste
```

## Appendix

### Helm Commands Reference

```bash
# List releases
helm list -n sap-llm-prod

# Get values
helm get values sap-llm -n sap-llm-prod

# Get manifest
helm get manifest sap-llm -n sap-llm-prod

# Dry run
helm upgrade sap-llm ./sap-llm \
  --namespace sap-llm-prod \
  --values sap-llm/values-production.yaml \
  --dry-run --debug

# Template
helm template sap-llm ./sap-llm \
  --namespace sap-llm-prod \
  --values sap-llm/values-production.yaml

# Package
helm package ./sap-llm
```

### Kubectl Commands Reference

```bash
# Get all resources
kubectl get all -n sap-llm-prod

# Get events
kubectl get events -n sap-llm-prod --sort-by='.lastTimestamp'

# Resource usage
kubectl top pods -n sap-llm-prod
kubectl top nodes

# Exec into pod
kubectl exec -it <pod-name> -n sap-llm-prod -- /bin/bash

# Copy files
kubectl cp <pod-name>:/path/to/file ./local-file -n sap-llm-prod

# Delete stuck resources
kubectl delete pod <pod-name> -n sap-llm-prod --grace-period=0 --force
```

## Support and Community

- GitHub Issues: https://github.com/AjithAccel4/SAP_LLM/issues
- Documentation: https://github.com/AjithAccel4/SAP_LLM/wiki
- Slack Channel: #sap-llm
- Email: team@sap-llm.example.com

## License

This deployment guide is part of the SAP_LLM project. See LICENSE file for details.
