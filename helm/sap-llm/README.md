# SAP_LLM Helm Chart

Enterprise-grade Helm chart for deploying SAP_LLM document AI system with Process Memory Graph, RLHF, Self-Healing, and APOP capabilities.

## Features

- **Production-ready deployment** with auto-scaling and high availability
- **GPU support** for NVIDIA H100 GPUs
- **Process Memory Graph** (Neo4j + Qdrant) for continuous learning
- **RLHF** (Reinforcement Learning from Human Feedback) for model improvement
- **Self-Healing Workflow Loop** (SHWL) for automated error recovery
- **APOP** (Agentic Process Orchestration Protocol) with CloudEvents
- **Complete observability** with Prometheus, Grafana, and Jaeger
- **Security hardening** with RBAC, network policies, and secrets management
- **Canary deployments** for safe production rollouts

## Prerequisites

- Kubernetes 1.24+
- Helm 3.12+
- NVIDIA GPU Operator (for GPU support)
- Storage class for persistent volumes
- Cert-manager (for TLS certificates)

## Installation

### Quick Start

```bash
# Add the Helm repository (if published)
helm repo add sap-llm https://charts.sap-llm.example.com
helm repo update

# Install with default values
helm install sap-llm sap-llm/sap-llm \
  --namespace sap-llm \
  --create-namespace
```

### From Source

```bash
# Clone the repository
git clone https://github.com/AjithAccel4/SAP_LLM.git
cd SAP_LLM

# Install the chart
helm install sap-llm ./helm/sap-llm \
  --namespace sap-llm \
  --create-namespace \
  --values ./helm/sap-llm/values.yaml
```

### Production Installation

```bash
# Create secrets first
kubectl create secret generic sap-llm-redis-secret \
  --from-literal=redis-password=$(openssl rand -base64 32) \
  --namespace sap-llm

kubectl create secret generic sap-llm-neo4j-secret \
  --from-literal=neo4j-password=$(openssl rand -base64 32) \
  --namespace sap-llm

kubectl create secret generic sap-llm-postgres-secret \
  --from-literal=username=sap_llm \
  --from-literal=password=$(openssl rand -base64 32) \
  --namespace sap-llm

kubectl create secret generic sap-llm-minio-secret \
  --from-literal=access-key=$(openssl rand -base64 20) \
  --from-literal=secret-key=$(openssl rand -base64 40) \
  --namespace sap-llm

# Install with production values
helm install sap-llm ./helm/sap-llm \
  --namespace sap-llm \
  --create-namespace \
  --values ./helm/sap-llm/values.yaml \
  --set global.environment=production \
  --set replicaCount=10 \
  --set resources.requests.nvidia.com/gpu=1 \
  --set autoscaling.maxReplicas=50 \
  --set ingress.hosts[0].host=sap-llm.example.com \
  --wait \
  --timeout 15m
```

## Configuration

### Key Configuration Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `global.environment` | Environment (production, staging, development) | `production` |
| `replicaCount` | Number of replicas | `3` |
| `image.repository` | Container image repository | `ghcr.io/ajithaccel4/sap_llm` |
| `image.tag` | Container image tag | `1.0.0` |
| `gpu.enabled` | Enable GPU support | `true` |
| `resources.requests.nvidia.com/gpu` | Number of GPUs per pod | `1` |
| `autoscaling.enabled` | Enable HPA | `true` |
| `autoscaling.minReplicas` | Minimum replicas | `3` |
| `autoscaling.maxReplicas` | Maximum replicas | `50` |
| `config.pmg.enabled` | Enable Process Memory Graph | `true` |
| `config.rlhf.enabled` | Enable RLHF | `true` |
| `config.shwl.enabled` | Enable Self-Healing | `true` |
| `config.apop.enabled` | Enable APOP | `true` |

### Component Configuration

#### Process Memory Graph (PMG)

```yaml
config:
  pmg:
    enabled: true
    similarity_threshold: 0.85
    max_similar_documents: 10
    retraining_threshold: 500

neo4j:
  enabled: true
  core:
    numberOfServers: 3
  readReplica:
    numberOfServers: 2

qdrant:
  enabled: true
  replicaCount: 3
  persistence:
    size: 200Gi
```

#### RLHF Configuration

```yaml
config:
  rlhf:
    enabled: true
    reward_model_path: "/models/reward_model_v1"
    ppo_iterations: 1000
    learning_rate: 1e-6
    kl_penalty: 0.05
```

#### Self-Healing (SHWL)

```yaml
config:
  shwl:
    enabled: true
    anomaly_detection_interval: 300
    contamination_rate: 0.05
    recovery_strategies:
      - retry_with_preprocessing
      - route_to_manual_review
      - apply_tolerance_rules
```

#### APOP Orchestration

```yaml
config:
  apop:
    enabled: true
    event_bus_type: kafka
    agent_heartbeat_ttl: 30
    dynamic_routing: true

kafka:
  enabled: true
  replicaCount: 3
  persistence:
    size: 500Gi
```

### Resource Requirements

#### Development Environment

```yaml
replicaCount: 1
resources:
  requests:
    cpu: "2"
    memory: "8Gi"
    nvidia.com/gpu: "0"  # No GPU
autoscaling:
  enabled: false
```

#### Staging Environment

```yaml
replicaCount: 3
resources:
  requests:
    cpu: "4"
    memory: "16Gi"
    nvidia.com/gpu: "1"
autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
```

#### Production Environment

```yaml
replicaCount: 10
resources:
  requests:
    cpu: "8"
    memory: "32Gi"
    nvidia.com/gpu: "1"  # H100 80GB
  limits:
    nvidia.com/gpu: "1"
autoscaling:
  enabled: true
  minReplicas: 10
  maxReplicas: 50
```

## Upgrading

```bash
# Upgrade to new version
helm upgrade sap-llm ./helm/sap-llm \
  --namespace sap-llm \
  --values ./helm/sap-llm/values.yaml \
  --set image.tag=1.1.0 \
  --wait

# Rollback if needed
helm rollback sap-llm --namespace sap-llm
```

## Monitoring

The chart includes built-in monitoring with Prometheus and Grafana:

```bash
# Access Grafana dashboard
kubectl port-forward -n sap-llm svc/sap-llm-grafana 3000:80

# Access Prometheus
kubectl port-forward -n sap-llm svc/sap-llm-prometheus 9090:9090

# View metrics
curl http://localhost:9090/api/v1/query?query=sap_llm_extraction_f1_score
```

## Troubleshooting

### Pods not starting

```bash
# Check pod status
kubectl get pods -n sap-llm

# Check pod logs
kubectl logs -n sap-llm -l app=sap-llm --tail=100

# Check events
kubectl get events -n sap-llm --sort-by='.lastTimestamp'
```

### GPU not available

```bash
# Check GPU operator
kubectl get pods -n gpu-operator

# Check node labels
kubectl get nodes --show-labels | grep accelerator

# Verify GPU resources
kubectl describe node <node-name> | grep nvidia.com/gpu
```

### Performance issues

```bash
# Check HPA status
kubectl get hpa -n sap-llm

# Check resource usage
kubectl top pods -n sap-llm

# Check Prometheus metrics
kubectl port-forward -n sap-llm svc/sap-llm-prometheus 9090:9090
# Open http://localhost:9090
```

## Uninstallation

```bash
# Uninstall the release
helm uninstall sap-llm --namespace sap-llm

# Delete PVCs (optional - this deletes all data!)
kubectl delete pvc -n sap-llm -l app.kubernetes.io/name=sap-llm

# Delete namespace (optional)
kubectl delete namespace sap-llm
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

See [LICENSE](../../LICENSE) for details.

## Support

- Documentation: https://docs.sap-llm.example.com
- Issues: https://github.com/AjithAccel4/SAP_LLM/issues
- Slack: https://sap-llm.slack.com
