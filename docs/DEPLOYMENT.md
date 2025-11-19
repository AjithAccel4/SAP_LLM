# SAP_LLM Deployment Guide

This guide provides detailed instructions for deploying SAP_LLM in various environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Docker Compose)](#quick-start-docker-compose)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: 8 cores
- RAM: 32 GB
- GPU: NVIDIA A10 (24GB VRAM) or equivalent
- Storage: 200 GB SSD

**Recommended:**
- CPU: 16+ cores
- RAM: 64+ GB
- GPU: NVIDIA A100 (40GB VRAM)
- Storage: 500 GB NVMe SSD

### Software Requirements

- Docker 24.0+
- Docker Compose 2.0+
- Kubernetes 1.27+ (for K8s deployment)
- kubectl (for K8s deployment)
- NVIDIA GPU Driver 525+
- NVIDIA Container Toolkit

### External Services

- **Azure Cosmos DB**: For Process Memory Graph
- **Redis**: For caching (can use Docker container)
- **MongoDB**: For Knowledge Base (can use Docker container)

## Quick Start (Docker Compose)

### 1. Clone Repository

```bash
git clone https://github.com/your-org/SAP_LLM.git
cd SAP_LLM
```

### 2. Configure Environment

Create `.env` file from template:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Azure Cosmos DB
COSMOS_ENDPOINT=https://your-cosmos.documents.azure.com:443/
COSMOS_KEY=your-cosmos-key-here

# Redis
REDIS_PASSWORD=your-redis-password

# MongoDB
MONGO_USERNAME=admin
MONGO_PASSWORD=your-mongo-password

# Grafana
GRAFANA_PASSWORD=your-grafana-password
```

### 3. Download Models

Download pre-trained models (or use your own):

```bash
# Create models directory
mkdir -p models

# Download vision encoder
# (Replace with actual model download commands)

# Download language decoder
# (Replace with actual model download commands)

# Download reasoning engine
# (Replace with actual model download commands)
```

### 4. Build and Start

```bash
# Build Docker image
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f sap-llm-api
```

### 5. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check readiness
curl http://localhost:8000/ready

# View Grafana dashboard
open http://localhost:3000  # admin/admin123
```

### 6. Test API

```bash
# Upload test document
curl -X POST http://localhost:8000/v1/extract \
  -H "X-API-Key: dev_key_12345" \
  -F "file=@test_document.pdf"
```

## Kubernetes Deployment

### 1. Prepare Cluster

Ensure you have a Kubernetes cluster with:
- GPU nodes (with NVIDIA device plugin)
- Sufficient resources
- LoadBalancer or Ingress controller

```bash
# Check cluster
kubectl cluster-info

# Check GPU nodes
kubectl get nodes -l nvidia.com/gpu=true
```

### 2. Create Namespace and Secrets

```bash
# Create namespace
kubectl apply -f deployments/kubernetes/namespace.yaml

# Create secrets
kubectl create secret generic sap-llm-secrets \
  --from-literal=cosmos-endpoint=$COSMOS_ENDPOINT \
  --from-literal=cosmos-key=$COSMOS_KEY \
  --from-literal=redis-password=$REDIS_PASSWORD \
  --from-literal=mongo-username=$MONGO_USERNAME \
  --from-literal=mongo-password=$MONGO_PASSWORD \
  --namespace=sap-llm
```

### 3. Deploy Using Script

```bash
cd deployments

# Deploy everything
./deploy.sh all

# Or deploy step by step
./deploy.sh infrastructure
./deploy.sh databases
./deploy.sh application

# Check status
./deploy.sh status
```

### 4. Manual Deployment (Alternative)

```bash
cd deployments/kubernetes

# Apply manifests in order
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f redis-deployment.yaml
kubectl apply -f mongo-deployment.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml
```

### 5. Verify Deployment

```bash
# Check all resources
kubectl get all -n sap-llm

# Check pod logs
kubectl logs -f deployment/sap-llm-api -n sap-llm

# Check events
kubectl get events -n sap-llm --sort-by='.lastTimestamp'

# Port forward for testing
kubectl port-forward service/sap-llm-api-service 8000:80 -n sap-llm
```

### 6. Using Kustomize (Alternative)

```bash
cd deployments/kubernetes

# Preview changes
kubectl kustomize .

# Apply
kubectl apply -k .
```

## Configuration

### Application Configuration

Configuration is managed through `configs/default_config.yaml`:

```yaml
system:
  environment: production
  log_level: INFO
  workers: 4

models:
  vision_encoder:
    model_name: microsoft/layoutlmv3-base
    device: cuda
    precision: fp16
    batch_size: 16

  language_decoder:
    model_name: meta-llama/Llama-2-7b-hf
    device: cuda
    precision: int8
    max_length: 2048

  reasoning_engine:
    model_name: mistralai/Mixtral-8x7B-v0.1
    device: cuda
    precision: int8
    max_length: 4096
```

### Environment Variables

Key environment variables:

- `ENVIRONMENT`: deployment environment (development/production)
- `LOG_LEVEL`: logging level (DEBUG/INFO/WARNING/ERROR)
- `COSMOS_ENDPOINT`: Cosmos DB endpoint URL
- `COSMOS_KEY`: Cosmos DB access key
- `REDIS_HOST`: Redis host
- `REDIS_PORT`: Redis port
- `REDIS_PASSWORD`: Redis password
- `MONGO_URI`: MongoDB connection URI
- `CUDA_VISIBLE_DEVICES`: GPU device IDs
- `CORS_ALLOWED_ORIGINS`: Allowed origins for CORS (see CORS Security section below)

### CORS Security Configuration

#### Overview

Cross-Origin Resource Sharing (CORS) is a critical security mechanism that controls which domains can access your API. SAP_LLM includes comprehensive CORS validation to prevent common security vulnerabilities.

#### Security Requirements

**PRODUCTION REQUIREMENTS (Enforced Automatically):**

1. ✅ **No Wildcard Origins** - Wildcard (`*`) is strictly forbidden in production
2. ✅ **HTTPS Only** - All origins must use HTTPS (no HTTP allowed)
3. ✅ **Explicit Origins** - Must specify at least one allowed origin
4. ✅ **Valid URLs** - All origins must be properly formatted URLs with scheme and host

#### Configuration

Set the `CORS_ALLOWED_ORIGINS` environment variable with a comma-separated list of allowed origins:

```bash
# Development (HTTP allowed for localhost)
export CORS_ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8000"

# Staging
export CORS_ALLOWED_ORIGINS="https://staging-app.example.com,https://staging-api.example.com"

# Production (HTTPS required)
export CORS_ALLOWED_ORIGINS="https://app.qorsync.com,https://api.qorsync.com"
```

#### Docker Compose Configuration

Edit your `.env` file:

```bash
ENVIRONMENT=production
CORS_ALLOWED_ORIGINS=https://app.qorsync.com,https://api.qorsync.com
```

#### Kubernetes Configuration

Add to your ConfigMap or Secret:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sap-llm-config
  namespace: sap-llm
data:
  ENVIRONMENT: "production"
  CORS_ALLOWED_ORIGINS: "https://app.qorsync.com,https://api.qorsync.com"
```

Or use a Secret for sensitive configuration:

```bash
kubectl create secret generic sap-llm-secrets \
  --from-literal=CORS_ALLOWED_ORIGINS="https://app.qorsync.com,https://api.qorsync.com" \
  --namespace=sap-llm
```

#### Security Validation

The API performs automatic security validation on startup:

1. **Wildcard Check**: Rejects `*` in production
2. **HTTPS Validation**: Ensures all production origins use HTTPS
3. **URL Format**: Validates proper URL structure
4. **Empty Origins**: Requires at least one origin in production

**Validation Failures:**

If CORS configuration is insecure, the API will **fail to start** with a clear error message:

```
CRITICAL SECURITY ERROR: Invalid CORS Configuration
SECURITY VIOLATION: CORS wildcard (*) is not allowed in production.
Specify explicit allowed origins in CORS_ALLOWED_ORIGINS environment variable.
Example: CORS_ALLOWED_ORIGINS=https://app.example.com,https://api.example.com
```

#### Best Practices

**DO:**
- ✅ Use HTTPS for all production origins
- ✅ Keep the list minimal (only domains you own and trust)
- ✅ Use separate configurations for dev/staging/production
- ✅ Review CORS logs on startup for warnings
- ✅ Document all allowed origins in your deployment docs

**DON'T:**
- ❌ Use wildcard (`*`) in production
- ❌ Use HTTP in production
- ❌ Add untrusted third-party domains
- ❌ Allow more than 5 origins without review
- ❌ Use the same configuration across all environments

#### Common Scenarios

**Single Page Application (SPA):**
```bash
# Frontend hosted on app.example.com
CORS_ALLOWED_ORIGINS=https://app.example.com
```

**Multiple Domains:**
```bash
# Main app + admin panel
CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
```

**Subdomain Strategy:**
```bash
# Multiple subdomains (must list explicitly)
CORS_ALLOWED_ORIGINS=https://app.example.com,https://dashboard.example.com,https://reports.example.com
```

**Development with Local Frontend:**
```bash
# Local development (HTTP allowed in dev mode)
ENVIRONMENT=development
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

#### Troubleshooting

**Problem: CORS errors in browser console**

```
Access to fetch at 'https://api.example.com' from origin 'https://app.example.com'
has been blocked by CORS policy
```

**Solution:**
1. Check that your frontend domain is in `CORS_ALLOWED_ORIGINS`
2. Verify the URL matches exactly (including protocol and port)
3. Check API startup logs for CORS configuration
4. Ensure no trailing slashes in origin URLs

**Problem: API fails to start with "SECURITY VIOLATION"**

**Solution:**
```bash
# Check current configuration
echo $CORS_ALLOWED_ORIGINS

# Fix for production (use HTTPS)
export CORS_ALLOWED_ORIGINS="https://app.example.com"

# Restart API
docker-compose restart sap-llm-api
```

**Problem: Wildcard needed for development**

**Solution:**
```bash
# Use explicit localhost origins instead of wildcard
ENVIRONMENT=development
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# If you absolutely need wildcard for local dev:
ENVIRONMENT=development  # Must NOT be production
CORS_ALLOWED_ORIGINS=*   # Only works in non-production
```

#### Monitoring and Auditing

Monitor CORS configuration in logs:

```bash
# Check CORS configuration on startup
docker logs sap-llm-api | grep "CORS Configuration"

# Kubernetes
kubectl logs deployment/sap-llm-api -n sap-llm | grep "CORS Configuration"
```

Expected output:
```
======================================================================
CORS Configuration Initialized
======================================================================
Environment: production
Allowed Origins: ['https://app.qorsync.com', 'https://api.qorsync.com']
======================================================================
✓ CORS middleware configured successfully
```

#### Security Warnings

The system will log warnings for potential misconfigurations:

**Warning: Too Many Origins**
```
⚠️  SECURITY WARNING: 6 CORS origins configured.
Large numbers of allowed origins may indicate misconfiguration.
```

**Action**: Review your origins list and remove any unnecessary domains.

**Warning: Wildcard in Non-Production**
```
⚠️  SECURITY WARNING: CORS wildcard (*) detected in development environment.
This allows ANY domain to access your API.
```

**Action**: Replace with explicit origins before deploying to production.

#### Migration from Legacy Configuration

If upgrading from an older version using `API_CORS_ORIGINS`:

**Old (.env):**
```bash
API_CORS_ORIGINS=http://localhost:3000,http://localhost:8000
```

**New (.env):**
```bash
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

**Migration Steps:**
1. Update `.env` file with new variable name
2. Remove `API_CORS_ORIGINS` variable
3. Restart API service
4. Verify CORS configuration in startup logs

### Scaling Configuration

#### Horizontal Pod Autoscaler (HPA)

Edit `deployments/kubernetes/hpa.yaml`:

```yaml
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

#### Resource Limits

Edit `deployments/kubernetes/deployment.yaml`:

```yaml
resources:
  requests:
    cpu: "4"
    memory: "32Gi"
    nvidia.com/gpu: "1"
  limits:
    cpu: "8"
    memory: "64Gi"
    nvidia.com/gpu: "1"
```

## Monitoring

### Prometheus Metrics

SAP_LLM exposes Prometheus metrics at `/metrics`:

```bash
# View metrics
curl http://localhost:8000/metrics
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (Docker Compose) or via K8s ingress.

Default credentials: `admin/admin123`

Pre-configured dashboards:
- **SAP_LLM Overview**: High-level system metrics
- **Document Processing**: Pipeline performance
- **Model Performance**: Inference latency and throughput
- **Infrastructure**: Resource utilization

### Logs

#### Docker Compose

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f sap-llm-api

# With timestamps
docker-compose logs -f --timestamps sap-llm-api
```

#### Kubernetes

```bash
# All pods
kubectl logs -f -l app=sap-llm-api -n sap-llm

# Specific pod
kubectl logs -f <pod-name> -n sap-llm

# Previous container (if crashed)
kubectl logs --previous <pod-name> -n sap-llm
```

### Alerts

Configure alerts in `deployments/monitoring/prometheus.yml`:

```yaml
groups:
- name: sap_llm_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Available

**Symptom:** Models fail to load or fall back to CPU

**Solution:**
```bash
# Check GPU availability
nvidia-smi

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. Out of Memory (OOM)

**Symptom:** Pods/containers getting killed

**Solution:**
- Reduce batch size in config
- Enable INT8 quantization
- Increase memory limits
- Use smaller models

```yaml
# Reduce batch size
models:
  vision_encoder:
    batch_size: 8  # Reduced from 16

# Enable quantization
models:
  language_decoder:
    precision: int8  # Changed from fp16
```

#### 3. Cosmos DB Connection Failure

**Symptom:** PMG operations failing

**Solution:**
```bash
# Check credentials
echo $COSMOS_ENDPOINT
echo $COSMOS_KEY | head -c 20

# Test connection
curl -X GET "$COSMOS_ENDPOINT" \
  -H "Authorization: type=master&ver=1.0&sig=$COSMOS_KEY"

# Use mock mode for testing
environment:
  - PMG_MOCK_MODE=true
```

#### 4. Slow Processing

**Symptom:** High latency, requests timing out

**Solution:**
- Check GPU utilization: `nvidia-smi`
- Check CPU load: `top` or `htop`
- Review logs for bottlenecks
- Enable caching
- Increase worker count
- Scale horizontally

```yaml
# Enable caching
performance:
  caching:
    enabled: true
    ttl: 3600

# Increase workers
system:
  workers: 8
```

#### 5. Model Loading Failures

**Symptom:** Models fail to load at startup

**Solution:**
```bash
# Check model files exist
ls -lh models/

# Check permissions
chmod -R 755 models/

# Download models manually
python -c "from transformers import AutoModel; \
  AutoModel.from_pretrained('microsoft/layoutlmv3-base')"
```

### Debugging Commands

```bash
# Check pod details
kubectl describe pod <pod-name> -n sap-llm

# Get pod events
kubectl get events --field-selector involvedObject.name=<pod-name> -n sap-llm

# Execute commands in pod
kubectl exec -it <pod-name> -n sap-llm -- bash

# Check resource usage
kubectl top pods -n sap-llm
kubectl top nodes

# Check PVC status
kubectl get pvc -n sap-llm
kubectl describe pvc <pvc-name> -n sap-llm
```

### Performance Tuning

#### GPU Optimization

```yaml
# Use FP16 precision for faster inference
models:
  vision_encoder:
    precision: fp16

# Increase batch size (if memory allows)
models:
  vision_encoder:
    batch_size: 32

# Enable CUDA graphs (for supported models)
performance:
  cuda_graphs: true
```

#### Caching Strategy

```yaml
performance:
  caching:
    enabled: true
    ttl: 3600
    max_size: 10000
    strategy: lru
```

#### Batching Configuration

```yaml
performance:
  batching:
    enabled: true
    batch_size: 16
    timeout: 1.0
    max_queue_size: 100
```

## Production Checklist

Before going to production:

### Security
- [ ] **Configure CORS properly** - Set CORS_ALLOWED_ORIGINS with HTTPS origins only
- [ ] **Verify no CORS wildcards** - Ensure `*` is not in CORS_ALLOWED_ORIGINS
- [ ] **Test CORS configuration** - Verify frontend can access API, others cannot
- [ ] Set strong passwords for all services
- [ ] Configure SSL/TLS certificates
- [ ] Review security best practices
- [ ] Configure RBAC properly
- [ ] Set up network policies

### Data & Backup
- [ ] Set up backup strategy for Cosmos DB and MongoDB
- [ ] Test disaster recovery procedures

### Monitoring & Logging
- [ ] Configure log aggregation
- [ ] Set up alerting rules
- [ ] Review CORS startup logs for warnings

### Infrastructure
- [ ] Configure resource limits and quotas
- [ ] Enable pod disruption budgets
- [ ] Load test the system

### Documentation & Training
- [ ] Document runbooks
- [ ] Document all allowed CORS origins
- [ ] Train operations team

## Support

For issues or questions:

- GitHub Issues: https://github.com/your-org/SAP_LLM/issues
- Documentation: https://docs.your-org.com/sap-llm
- Email: support@your-org.com
