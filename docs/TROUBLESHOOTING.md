# SAP_LLM Troubleshooting Runbooks

Comprehensive troubleshooting guide for common issues and their resolutions.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [API Issues](#api-issues)
3. [Performance Issues](#performance-issues)
4. [Model/AI Issues](#modelai-issues)
5. [Database Issues](#database-issues)
6. [Cache Issues](#cache-issues)
7. [Security Issues](#security-issues)
8. [Deployment Issues](#deployment-issues)
9. [Monitoring Issues](#monitoring-issues)

---

## Quick Diagnostics

### Health Check Commands

```bash
# System health
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# Metrics
curl http://localhost:8000/metrics | grep sap_llm

# SLO status
curl http://localhost:8000/v1/slo

# Stats
curl http://localhost:8000/v1/stats
```

### Quick Status Check Script

```bash
#!/bin/bash
# quick-check.sh

echo "=== SAP_LLM Health Check ==="

# API health
API_HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')
echo "API Health: $API_HEALTH"

# Redis
REDIS_STATUS=$(redis-cli ping 2>/dev/null || echo "FAILED")
echo "Redis: $REDIS_STATUS"

# GPU availability
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null || echo "0")
echo "GPUs Available: $GPU_COUNT"

# Disk space
DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}')
echo "Disk Usage: $DISK_USAGE"

# Memory usage
MEM_USAGE=$(free | awk 'NR==2 {printf "%.0f%%", $3/$2 * 100}')
echo "Memory Usage: $MEM_USAGE"

# Pod status (if Kubernetes)
if command -v kubectl &> /dev/null; then
    echo "=== Pod Status ==="
    kubectl get pods -n sap-llm-production
fi
```

---

## API Issues

### Issue: API Returns 503 Service Unavailable

**Symptoms:**
- All requests return 503
- Health endpoint returns "not ready"

**Diagnostics:**
```bash
# Check readiness
curl http://localhost:8000/ready

# Check logs
kubectl logs -l app=sap-llm-api --tail=50

# Check models loaded
curl http://localhost:8000/ready | jq '.details'
```

**Common Causes & Solutions:**

**1. Models Not Loaded**
```bash
# Check model files exist
ls -lh /models/

# Check model loading errors in logs
grep "model" /var/log/sap_llm/api.log | grep -i error

# Solution: Restart service to reload models
kubectl rollout restart deployment/sap-llm-api
```

**2. Database Connection Failed**
```bash
# Test Cosmos DB connection
python3 << EOF
from azure.cosmos import CosmosClient
client = CosmosClient(url, credential)
database = client.get_database_client("sap_llm")
container = database.get_container_client("documents")
print("Connected!")
EOF

# Solution: Check connection string in secrets
kubectl get secret sap-llm-secrets -o json | jq '.data'
```

**3. Out of Memory**
```bash
# Check memory usage
kubectl top pods | grep sap-llm

# Check OOM kills
kubectl get events | grep OOMKilled

# Solution: Increase memory limits
kubectl edit deployment sap-llm-api
# Update: resources.limits.memory to 32Gi
```

---

### Issue: High Latency (P95 > 1000ms)

**Symptoms:**
- Slow API responses
- Timeouts
- Queue backlog

**Diagnostics:**
```bash
# Check current latency
curl http://localhost:8000/metrics | grep request_duration_seconds

# Check queue depth
curl http://localhost:8000/v1/stats | jq '.jobs.queued'

# Check resource usage
kubectl top pods
```

**Common Causes & Solutions:**

**1. Cache Miss Rate High**
```bash
# Check cache hit rate
curl http://localhost:8000/metrics | grep cache_hit_rate

# Expected: >0.85 (85%)
# If lower, investigate:

# Check Redis health
redis-cli info stats | grep hit_rate

# Solution: Increase cache size
kubectl edit configmap redis-config
# Update: maxmemory 16gb
```

**2. GPU Bottleneck**
```bash
# Check GPU utilization
nvidia-smi

# If utilization > 95%, scale up:
kubectl scale deployment sap-llm-gpu --replicas=5

# Or enable auto-scaling
kubectl autoscale deployment sap-llm-gpu \
  --min=2 --max=10 --cpu-percent=70
```

**3. Database Slow Queries**
```bash
# Check Cosmos DB metrics
az cosmosdb show -n sap-llm-cosmos -g sap-llm-rg

# Check query latency
curl http://localhost:8000/metrics | grep cosmos_query_duration

# Solution: Add indexes or increase RUs
az cosmosdb sql container update \
  -a sap-llm-cosmos -g sap-llm-rg \
  -d sap_llm -n documents \
  --throughput 10000
```

---

### Issue: Authentication Failures

**Symptoms:**
- 401 Unauthorized errors
- Invalid token errors
- "Token expired" messages

**Diagnostics:**
```bash
# Test token generation
python3 << EOF
from sap_llm.security import AuthenticationManager
auth = AuthenticationManager(secret_key="your-secret")
token = auth.generate_access_token("test_user", Role.USER)
print(f"Token: {token}")

# Verify token
payload = auth.verify_token(token)
print(f"Valid: {payload}")
EOF

# Check JWT secret
kubectl get secret sap-llm-secrets -o jsonpath='{.data.JWT_SECRET_KEY}' | base64 -d
```

**Common Causes & Solutions:**

**1. Token Expired**
```
Error: "Signature has expired"

# Tokens expire after 15 minutes
# Solution: Implement refresh token logic

curl -X POST http://localhost:8000/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "YOUR_REFRESH_TOKEN"}'
```

**2. Wrong Secret Key**
```
Error: "Signature verification failed"

# Check if secret key changed
# Solution: Update secret consistently

kubectl create secret generic sap-llm-secrets \
  --from-literal=JWT_SECRET_KEY="new-secret-key" \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pods
kubectl rollout restart deployment/sap-llm-api
```

**3. Clock Skew**
```
Error: "Token used before issued"

# Check system time
date
timedatectl

# Solution: Sync time
sudo ntpdate pool.ntp.org
# Or
sudo chronyc makestep
```

---

## Performance Issues

### Issue: Low Throughput (<100K docs/hour)

**Diagnostics:**
```bash
# Check processing rate
curl http://localhost:8000/metrics | grep documents_processed_total

# Calculate rate
watch 'curl -s http://localhost:8000/v1/stats | jq ".jobs.completed"'

# Check bottlenecks
kubectl top nodes
kubectl top pods
```

**Common Causes & Solutions:**

**1. Insufficient Replicas**
```bash
# Check current replicas
kubectl get deployment sap-llm-api -o jsonpath='{.spec.replicas}'

# Solution: Scale up
kubectl scale deployment sap-llm-api --replicas=10
kubectl scale deployment sap-llm-worker --replicas=20
kubectl scale deployment sap-llm-gpu --replicas=5
```

**2. CPU Throttling**
```bash
# Check CPU throttling
kubectl top pods | awk '{if ($3 > 90) print $0}'

# Solution: Increase CPU limits
kubectl edit deployment sap-llm-worker
# Update: resources.limits.cpu to "8000m"
```

**3. Network Latency**
```bash
# Test network latency to databases
ping cosmos-db.documents.azure.com
ping redis-cluster.cache.windows.net

# Solution: Use region co-location
# Deploy to same region as databases
```

---

### Issue: High Memory Usage

**Symptoms:**
- OOMKilled pods
- Memory usage > 90%
- Slow garbage collection

**Diagnostics:**
```bash
# Check memory usage
kubectl top pods --sort-by=memory

# Check OOM events
kubectl get events | grep OOMKilled

# Check memory leaks
kubectl exec -it sap-llm-api-xxx -- ps aux --sort=-%mem | head
```

**Common Causes & Solutions:**

**1. Memory Leak**
```bash
# Monitor memory over time
watch -n 5 'kubectl top pod sap-llm-api-xxx'

# Check for increasing trend
# Solution: Restart pod periodically
kubectl delete pod sap-llm-api-xxx

# Or set memory limits to trigger restart
kubectl edit deployment sap-llm-api
# Add: resources.limits.memory: "16Gi"
```

**2. Large Model Cache**
```bash
# Check model cache size
kubectl exec sap-llm-gpu-xxx -- du -sh /models/cache

# Solution: Clear cache
kubectl exec sap-llm-gpu-xxx -- rm -rf /models/cache/*

# Or increase memory
kubectl edit deployment sap-llm-gpu
# Update: resources.limits.memory: "32Gi"
```

**3. Too Many Concurrent Requests**
```bash
# Check concurrent requests
curl http://localhost:8000/v1/stats | jq '.jobs.processing'

# Solution: Add rate limiting
# Update configs/production_config.yaml:
# rate_limit_per_tenant: 100
```

---

## Model/AI Issues

### Issue: Low Model Accuracy

**Symptoms:**
- Accuracy < 95%
- High error rate
- User feedback indicates wrong extractions

**Diagnostics:**
```bash
# Check current accuracy
curl http://localhost:8000/metrics | grep model_accuracy

# Check extraction errors
curl http://localhost:8000/metrics | grep errors_total | grep extraction

# Get online learning status
curl http://localhost:8000/v1/learning/status
```

**Common Causes & Solutions:**

**1. Concept Drift Detected**
```bash
# Check drift detection
curl http://localhost:8000/v1/learning/status | jq '.performance'

# If drift detected:
# Solution: Trigger model update
curl -X POST http://localhost:8000/v1/learning/trigger-update

# Or retrain from scratch
python scripts/train_model.py --data /data/recent --epochs 10
```

**2. Wrong Model Variant**
```bash
# Check loaded model
kubectl logs sap-llm-gpu-xxx | grep "model loaded"

# Solution: Load correct model
kubectl edit configmap sap-llm-config
# Update: model_version: "v2.1-optimized"

kubectl rollout restart deployment/sap-llm-gpu
```

**3. Insufficient Training Data**
```bash
# Check training data size
curl http://localhost:8000/v1/learning/status | jq '.feedback.total'

# Need at least 1000 samples
# Solution: Collect more feedback
# Enable active learning:
curl -X POST http://localhost:8000/v1/learning/configure \
  -d '{"active_learning_enabled": true, "query_budget": 500}'
```

---

### Issue: Model Inference Timeout

**Symptoms:**
- "Model inference timeout" errors
- Requests taking > 30 seconds
- GPU not responding

**Diagnostics:**
```bash
# Check GPU status
nvidia-smi

# Check model inference time
curl http://localhost:8000/metrics | grep model_inference_duration

# Check GPU utilization
nvidia-smi dmon
```

**Common Causes & Solutions:**

**1. GPU Out of Memory**
```bash
# Check GPU memory
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# If OOM:
# Solution: Reduce batch size
kubectl edit configmap sap-llm-config
# Update: inference_batch_size: 8  # from 16

# Or use smaller model
# Update: model_size: "3B"  # from "7B"
```

**2. Model Not Loaded**
```bash
# Check model loading
kubectl logs sap-llm-gpu-xxx | grep "model"

# Solution: Reload model
kubectl delete pod sap-llm-gpu-xxx

# Or check model files
kubectl exec sap-llm-gpu-xxx -- ls -lh /models/
```

**3. GPU Driver Issue**
```bash
# Check GPU driver
nvidia-smi

# If driver error:
# Solution: Restart node (Kubernetes)
kubectl drain node-gpu-1 --ignore-daemonsets
kubectl uncordon node-gpu-1

# Or update driver
sudo apt-get update
sudo apt-get install --only-upgrade nvidia-driver-535
```

---

## Database Issues

### Issue: Cosmos DB Throttling

**Symptoms:**
- 429 "Too Many Requests" errors
- High RU consumption
- Slow database operations

**Diagnostics:**
```bash
# Check Cosmos DB metrics
az monitor metrics list \
  --resource /subscriptions/.../cosmosdb/sap-llm-cosmos \
  --metric TotalRequests

# Check RU consumption
curl http://localhost:8000/metrics | grep cosmos_ru_consumption
```

**Solutions:**

**1. Increase RU/s**
```bash
az cosmosdb sql container throughput update \
  -a sap-llm-cosmos \
  -g sap-llm-rg \
  -d sap_llm \
  -n documents \
  --throughput 20000  # Increase from 10000
```

**2. Enable Auto-scale**
```bash
az cosmosdb sql container throughput migrate \
  -a sap-llm-cosmos \
  -g sap-llm-rg \
  -d sap_llm \
  -n documents \
  -t autoscale \
  --max-throughput 20000
```

**3. Optimize Queries**
```python
# Add composite indexes
# In Cosmos DB portal, add index:
{
  "indexingMode": "consistent",
  "automatic": true,
  "includedPaths": [{
    "path": "/*"
  }],
  "excludedPaths": [{
    "path": "/\"_etag\"/?"
  }],
  "compositeIndexes": [[
    {"path": "/document_type", "order": "ascending"},
    {"path": "/timestamp", "order": "descending"}
  ]]
}
```

---

### Issue: Redis Connection Failures

**Symptoms:**
- "Connection refused" errors
- Cache misses at 100%
- Slow API responses

**Diagnostics:**
```bash
# Test Redis connection
redis-cli -h redis-cluster ping

# Check Redis status
redis-cli -h redis-cluster info replication

# Check connections
redis-cli -h redis-cluster info clients
```

**Solutions:**

**1. Redis Down**
```bash
# Check Redis status
kubectl get pods -l app=redis

# If pod crashed:
kubectl logs redis-xxx --previous

# Solution: Restart Redis
kubectl delete pod redis-xxx  # StatefulSet will recreate
```

**2. Max Connections Reached**
```bash
# Check current connections
redis-cli info clients | grep connected_clients

# Solution: Increase max connections
kubectl edit configmap redis-config
# Update: maxclients 10000

kubectl rollout restart statefulset/redis
```

**3. Network Partition**
```bash
# Test network connectivity
kubectl exec sap-llm-api-xxx -- nc -zv redis-service 6379

# If failed:
# Check NetworkPolicy
kubectl get networkpolicy
kubectl describe networkpolicy allow-redis

# Solution: Update NetworkPolicy
kubectl apply -f - << EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-redis
spec:
  podSelector:
    matchLabels:
      app: sap-llm-api
  policyTypes:
  - Egress
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
EOF
```

---

## Security Issues

### Issue: Unauthorized Access Attempts

**Symptoms:**
- High rate of 401 errors
- Multiple failed login attempts
- Security alerts

**Diagnostics:**
```bash
# Check failed authentication
curl http://localhost:8000/metrics | grep authentication_failures

# Check audit logs
kubectl logs sap-llm-api-xxx | grep "authentication.*failed"

# Get security events
curl http://localhost:8000/v1/security/audit | jq '.[] | select(.event_type=="authentication") | select(.success==false)'
```

**Solutions:**

**1. Brute Force Attack**
```bash
# Check failed attempts by IP
kubectl logs sap-llm-api-xxx | grep "authentication failed" | awk '{print $NF}' | sort | uniq -c | sort -rn

# Solution: Block IP
kubectl apply -f - << EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-malicious-ip
spec:
  podSelector:
    matchLabels:
      app: sap-llm-api
  policyTypes:
  - Ingress
  ingress:
  - from:
    - ipBlock:
        cidr: 0.0.0.0/0
        except:
        - 192.168.1.100/32  # Malicious IP
EOF

# Or enable rate limiting
kubectl edit configmap sap-llm-config
# Update: max_auth_attempts: 3
# Update: lockout_duration_minutes: 15
```

**2. Token Theft**
```bash
# Revoke compromised tokens
curl -X POST http://localhost:8000/auth/revoke \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -d '{"token": "COMPROMISED_TOKEN"}'

# Force re-authentication for user
curl -X POST http://localhost:8000/auth/logout-user \
  -d '{"user_id": "compromised_user_id"}'

# Rotate JWT secret
kubectl create secret generic sap-llm-secrets \
  --from-literal=JWT_SECRET_KEY="$(openssl rand -base64 32)" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl rollout restart deployment/sap-llm-api
```

---

## Deployment Issues

### Issue: Pod CrashLoopBackOff

**Diagnostics:**
```bash
# Check pod status
kubectl get pods | grep CrashLoop

# Check logs
kubectl logs sap-llm-api-xxx --previous

# Check events
kubectl describe pod sap-llm-api-xxx
```

**Common Causes & Solutions:**

**1. Missing Environment Variables**
```bash
# Check if all required env vars are set
kubectl exec sap-llm-api-xxx -- env | grep SAP_LLM

# Solution: Add missing variables
kubectl edit deployment sap-llm-api
# Add under spec.template.spec.containers.env
```

**2. Failed Health Check**
```bash
# Check liveness probe
kubectl describe pod sap-llm-api-xxx | grep Liveness

# Solution: Increase initialDelaySeconds
kubectl edit deployment sap-llm-api
# Update: livenessProbe.initialDelaySeconds: 120
```

**3. Image Pull Error**
```bash
# Check image pull
kubectl describe pod sap-llm-api-xxx | grep ImagePull

# Solution: Check registry credentials
kubectl get secret regcred -o yaml
# Or pull image manually to verify
docker pull your-registry.io/sap-llm:latest
```

---

## Monitoring Issues

### Issue: Metrics Not Appearing in Grafana

**Diagnostics:**
```bash
# Check Prometheus targets
kubectl port-forward svc/prometheus 9090:9090
# Open http://localhost:9090/targets

# Check metrics endpoint
curl http://sap-llm-api:8000/metrics

# Check Grafana data source
curl http://grafana:3000/api/datasources
```

**Solutions:**

**1. Prometheus Not Scraping**
```bash
# Check ServiceMonitor
kubectl get servicemonitor sap-llm-monitor -o yaml

# Solution: Create/update ServiceMonitor
kubectl apply -f - << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sap-llm-monitor
spec:
  selector:
    matchLabels:
      app: sap-llm-api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
EOF
```

**2. Grafana Can't Connect to Prometheus**
```bash
# Test connectivity
kubectl exec -it grafana-xxx -- wget -O- http://prometheus:9090/api/v1/status/config

# Solution: Update data source
# In Grafana UI, update Prometheus URL to: http://prometheus:9090
```

---

## Escalation Procedures

### When to Escalate

Escalate to senior engineer if:
- Issue not resolved within 30 minutes
- Multiple systems affected
- Data loss possible
- Security breach suspected
- SLA violation imminent

### Escalation Contacts

```
Level 1 (On-call Engineer): +1-xxx-xxx-xxxx
Level 2 (Senior Engineer): +1-xxx-xxx-xxxx
Level 3 (Engineering Manager): +1-xxx-xxx-xxxx
Security Team: security@company.com
```

### Incident Report Template

```markdown
## Incident Report

**Incident ID:** INC-YYYYMMDD-NNN
**Severity:** [P1-Critical | P2-High | P3-Medium | P4-Low]
**Date/Time:** YYYY-MM-DD HH:MM UTC
**Reported By:** [Name]

### Summary
[Brief description]

### Impact
- Services affected:
- Users affected:
- Duration:

### Timeline
- HH:MM - Incident detected
- HH:MM - Investigation started
- HH:MM - Root cause identified
- HH:MM - Fix applied
- HH:MM - Service restored

### Root Cause
[Detailed root cause analysis]

### Resolution
[Steps taken to resolve]

### Prevention
[Steps to prevent recurrence]
```

---

## Emergency Procedures

### Complete System Failure

```bash
#!/bin/bash
# emergency-restore.sh

echo "EMERGENCY: Restoring SAP_LLM system..."

# 1. Scale down all pods
kubectl scale deployment --all --replicas=0

# 2. Clear all caches
redis-cli FLUSHALL

# 3. Restore database from backup
az cosmosdb sql container restore ...

# 4. Reload models
kubectl cp /backup/models sap-llm-gpu-xxx:/models/

# 5. Scale up gradually
kubectl scale deployment sap-llm-api --replicas=1
sleep 60
kubectl scale deployment sap-llm-api --replicas=3
kubectl scale deployment sap-llm-worker --replicas=5
kubectl scale deployment sap-llm-gpu --replicas=2

# 6. Verify health
curl http://localhost:8000/health

echo "Emergency restore complete!"
```

---

## Summary

This runbook covers 90% of common issues. For issues not covered:

1. Check logs: `kubectl logs -l app=sap-llm --tail=100`
2. Check metrics: `http://localhost:8000/metrics`
3. Check documentation: `/docs`
4. Escalate if needed

**Remember:** Always document resolution steps for future reference!
