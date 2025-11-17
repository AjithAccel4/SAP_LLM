# Runbook: High Memory Usage

## Alert Details

**Alert Name:** MemoryUsageHigh
**Severity:** Critical
**Component:** Application
**Threshold:** Memory usage > 10GB per pod

## Symptoms

- Pod memory usage exceeds 10GB threshold
- AlertManager firing MemoryUsageHigh alert
- Pods approaching OOM (Out of Memory) state
- Potential pod restarts due to OOMKilled
- Application performance degradation
- Memory leak suspected

## Diagnosis Steps

### 1. Check Current Memory Usage

```bash
# Check pod memory usage
kubectl top pods -n sap-llm

# Check detailed memory metrics
kubectl exec -it deployment/sap-llm -n sap-llm -- cat /sys/fs/cgroup/memory/memory.usage_in_bytes | awk '{print $1/1024/1024/1024 " GB"}'

# Query Prometheus for memory trends
curl -s 'http://prometheus:9090/api/v1/query?query=container_memory_usage_bytes{namespace="sap-llm",container="sap-llm"}' | jq .

# View memory dashboard
open https://grafana.example.com/d/sap-llm-memory

# Check memory over time
curl -s 'http://prometheus:9090/api/v1/query_range?query=container_memory_usage_bytes{namespace="sap-llm"}&start=-1h&step=60s' | jq .
```

### 2. Identify Memory Consumers

```bash
# Check process memory usage inside pod
kubectl exec -it deployment/sap-llm -n sap-llm -- ps aux --sort=-%mem | head -20

# Check Python memory usage (if applicable)
kubectl exec -it deployment/sap-llm -n sap-llm -- python3 -c "
import resource
print(f'Memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024:.2f} GB')
"

# Check GPU memory usage
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Check for memory leaks in logs
kubectl logs -n sap-llm deployment/sap-llm | grep -i "memory\|leak\|oom"
```

### 3. Check Model and Cache Usage

```bash
# Check loaded models size
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /models/*

# Check cache size
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /tmp/cache

# Check Redis memory usage
kubectl exec -it deployment/redis -n sap-llm -- redis-cli INFO memory

# Check model instances loaded
kubectl logs -n sap-llm deployment/sap-llm | grep "model loaded\|model cached"
```

### 4. Check for Memory Leaks

```bash
# Check memory growth over time
curl -s 'http://prometheus:9090/api/v1/query?query=rate(container_memory_usage_bytes{namespace="sap-llm"}[1h])' | jq .

# Check pod uptime vs memory usage
kubectl get pods -n sap-llm -o custom-columns=NAME:.metadata.name,AGE:.metadata.creationTimestamp

# Check for memory patterns in logs
kubectl logs -n sap-llm deployment/sap-llm --tail=500 | grep "allocated\|freed\|gc"

# Enable memory profiling if available
kubectl exec -it deployment/sap-llm -n sap-llm -- curl http://localhost:8080/debug/pprof/heap > heap.prof
```

### 5. Review Recent Changes

```bash
# Check recent deployments
kubectl rollout history deployment/sap-llm -n sap-llm

# Check configuration changes
kubectl get configmap sap-llm-config -n sap-llm -o yaml | grep -A 5 "memory\|cache\|batch"

# Check resource limit changes
kubectl describe deployment sap-llm -n sap-llm | grep -A 5 "Limits\|Requests"

# Review recent code changes
git log --oneline --since="24 hours ago" -- src/
```

## Common Root Causes

### 1. Model Loading Issues (35% of cases)

**Symptoms:** Multiple model copies in memory, model not unloaded, large model size

**Resolution:**
```bash
# Check how many model instances are loaded
kubectl logs -n sap-llm deployment/sap-llm | grep -c "model loaded"

# Check model configuration
kubectl get configmap sap-llm-config -n sap-llm -o yaml | grep -A 10 "model"

# Reduce model instances
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "max_model_instances":"1",
  "model_cache_size":"2"
}}'

# Use model quantization if available
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "model_precision":"fp16",
  "use_quantization":"true"
}}'

# Clear model cache and restart
kubectl exec -it deployment/sap-llm -n sap-llm -- rm -rf /tmp/model_cache/*
kubectl rollout restart deployment/sap-llm -n sap-llm
```

### 2. Memory Leak in Application (30% of cases)

**Symptoms:** Steady memory growth, no memory release, eventual OOM

**Resolution:**
```bash
# Enable garbage collection logging
kubectl set env deployment/sap-llm -n sap-llm PYTHONUNBUFFERED=1

# Force garbage collection
kubectl exec -it deployment/sap-llm -n sap-llm -- python3 -c "import gc; gc.collect(); print('GC done')"

# Check for circular references
kubectl logs -n sap-llm deployment/sap-llm | grep "circular\|reference"

# Restart pods to clear memory
kubectl rollout restart deployment/sap-llm -n sap-llm

# If leak persists, collect heap dump
kubectl exec -it deployment/sap-llm -n sap-llm -- curl http://localhost:8080/debug/pprof/heap > /tmp/heap-$(date +%s).prof

# Schedule regular restarts as temporary fix
kubectl patch deployment sap-llm -n sap-llm -p '{"spec":{"template":{"metadata":{"annotations":{"restart-timestamp":"'$(date +%s)'"}}}}}'
```

### 3. Large Batch Processing (20% of cases)

**Symptoms:** Memory spikes during batch processing, correlation with load

**Resolution:**
```bash
# Check current batch sizes
kubectl logs -n sap-llm deployment/sap-llm --tail=100 | grep "batch_size"

# Reduce batch size
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "batch_size":"32",
  "max_batch_size":"64"
}}'

# Implement streaming processing
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "use_streaming":"true",
  "stream_batch_size":"16"
}}'

# Limit concurrent requests
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "max_concurrent_requests":"50"
}}'

# Restart to apply changes
kubectl rollout restart deployment/sap-llm -n sap-llm
```

### 4. Cache Bloat (15% of cases)

**Symptoms:** Large cache directories, Redis memory high, cache not expiring

**Resolution:**
```bash
# Check cache sizes
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /tmp/cache /tmp/model_cache

# Clear application cache
kubectl exec -it deployment/sap-llm -n sap-llm -- rm -rf /tmp/cache/*

# Check Redis memory
kubectl exec -it deployment/redis -n sap-llm -- redis-cli INFO memory

# Clear Redis cache
kubectl exec -it deployment/redis -n sap-llm -- redis-cli FLUSHDB

# Implement cache eviction policy
kubectl exec -it deployment/redis -n sap-llm -- redis-cli CONFIG SET maxmemory-policy allkeys-lru
kubectl exec -it deployment/redis -n sap-llm -- redis-cli CONFIG SET maxmemory 8gb

# Update cache TTLs
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "cache_ttl_seconds":"3600",
  "max_cache_size_mb":"4096"
}}'
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Assess Criticality:**
   ```bash
   # Check if pods are near OOM
   kubectl describe pods -n sap-llm | grep -A 5 "memory"

   # Check for OOMKilled pods
   kubectl get pods -n sap-llm -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}' | grep OOM

   # Check memory limits
   kubectl get pods -n sap-llm -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].resources.limits.memory}{"\n"}{end}'
   ```

2. **Quick Memory Release:**
   ```bash
   # Clear caches
   kubectl exec -it deployment/redis -n sap-llm -- redis-cli FLUSHDB
   kubectl exec -it deployment/sap-llm -n sap-llm -- rm -rf /tmp/cache/*

   # Force garbage collection
   kubectl exec -it deployment/sap-llm -n sap-llm -- curl -X POST http://localhost:8080/admin/gc

   # Reduce load temporarily
   kubectl scale deployment sap-llm-api -n sap-llm --replicas=2
   ```

3. **Prevent OOM:**
   ```bash
   # Increase memory limits temporarily
   kubectl patch deployment sap-llm -n sap-llm --type json -p='[
     {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "16Gi"}
   ]'

   # Restart high-memory pods
   kubectl delete pod -n sap-llm $(kubectl get pods -n sap-llm --sort-by='.status.containerStatuses[0].restartCount' -o jsonpath='{.items[0].metadata.name}')
   ```

### Short-term Fixes (5-30 minutes)

1. **If Model Issues:**
   - Reduce model instances
   - Enable model quantization
   - Implement model sharing
   - Use smaller model variants

2. **If Memory Leak:**
   - Restart affected pods
   - Implement periodic restarts
   - Enable memory profiling
   - Review recent code changes

3. **If Batch Processing:**
   - Reduce batch sizes
   - Implement streaming
   - Limit concurrency
   - Add backpressure

### Long-term Solutions (30+ minutes)

1. **Application Optimization:**
   - Fix memory leaks in code
   - Optimize data structures
   - Implement object pooling
   - Add memory profiling

2. **Architecture Improvements:**
   - Separate model serving
   - Implement tiered caching
   - Use memory-efficient formats
   - Add memory limits per component

3. **Capacity Planning:**
   - Right-size memory allocations
   - Plan for peak load
   - Implement auto-scaling
   - Monitor memory trends

## Escalation

### Level 1: On-Call Engineer (0-15 minutes)
- Follow this runbook
- Clear caches and restart pods
- Increase memory limits if needed
- Monitor for OOMKilled events

### Level 2: Application Engineer (15-30 minutes)
- If memory doesn't stabilize
- If memory leak suspected
- If configuration tuning needed
- If repeated OOM events

### Level 3: Architecture Team (30+ minutes)
- If code changes required
- If memory leak confirmed
- If architectural changes needed
- If persistent OOM issues

## Prevention

1. **Memory Monitoring:**
   - Set up memory trend alerts
   - Monitor memory growth rate
   - Track cache sizes
   - Alert on approaching limits

2. **Code Quality:**
   - Memory leak testing
   - Profiling in CI/CD
   - Code reviews for memory usage
   - Static analysis tools

3. **Configuration:**
   - Appropriate memory limits
   - Cache eviction policies
   - Batch size tuning
   - Resource quotas

4. **Testing:**
   - Load testing with memory monitoring
   - Stress testing
   - Memory leak detection tests
   - Long-running stability tests

## Related Runbooks

- [High Latency](./high-latency.md)
- [Low Throughput](./low-throughput.md)
- [Model Inference Failure](./model-inference-failure.md)
- [GPU Utilization Low](./gpu-utilization-low.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
