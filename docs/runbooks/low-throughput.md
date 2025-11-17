# Runbook: Low Throughput

## Alert Details

**Alert Name:** LowThroughput
**Severity:** Warning
**Component:** Pipeline
**Threshold:** Throughput < 100,000 envelopes/minute over 5 minutes

## Symptoms

- Processing throughput drops below 100k envelopes/min
- AlertManager firing LowThroughput alert
- Queue backlog increasing
- Processing lag growing
- Users experiencing delays in document availability

## Diagnosis Steps

### 1. Check Current Throughput Metrics

```bash
# Query Prometheus for current throughput
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_envelopes_processed_total[5m])*60' | jq .

# View throughput dashboard
open https://grafana.example.com/d/sap-llm-throughput

# Check throughput by pipeline stage
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_stage_processed_total[5m])*60' | jq .

# Check success rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_envelopes_success_total[5m])/rate(sap_llm_envelopes_processed_total[5m])' | jq .
```

### 2. Check Queue Depth and Backlog

```bash
# Check Redis queue depth
kubectl exec -it deployment/redis -n sap-llm -- redis-cli LLEN sap_llm:processing_queue

# Check dead letter queue
kubectl exec -it deployment/redis -n sap-llm -- redis-cli LLEN sap_llm:dlq

# Check processing lag
curl -s 'http://prometheus:9090/api/v1/query?query=sap_llm_queue_lag_seconds' | jq .

# Monitor queue growth rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_queue_depth[5m])' | jq .
```

### 3. Check Active Workers and Pods

```bash
# Check pod status
kubectl get pods -n sap-llm -l app=sap-llm

# Check number of replicas
kubectl get deployment sap-llm -n sap-llm -o jsonpath='{.spec.replicas}'

# Check active workers per pod
kubectl logs -n sap-llm deployment/sap-llm --tail=20 | grep "active_workers"

# Check pod readiness
kubectl get pods -n sap-llm -l app=sap-llm -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'
```

### 4. Check Resource Utilization

```bash
# Check CPU and memory usage
kubectl top pods -n sap-llm

# Check GPU utilization
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi

# Check if resources are saturated
kubectl describe nodes | grep -A 5 "Allocated resources"

# Check for throttling
kubectl exec -it deployment/sap-llm -n sap-llm -- cat /sys/fs/cgroup/cpu/cpu.stat | grep throttled
```

### 5. Check Processing Efficiency

```bash
# Check batch sizes being processed
kubectl logs -n sap-llm deployment/sap-llm --tail=100 | grep "batch_size"

# Check average processing time per envelope
kubectl logs -n sap-llm deployment/sap-llm | grep "envelope_duration_ms" | awk '{sum+=$NF; count++} END {print sum/count}'

# Check for processing errors
kubectl logs -n sap-llm deployment/sap-llm --tail=200 | grep -c ERROR

# Check retry rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_retries_total[5m])' | jq .
```

## Common Root Causes

### 1. Insufficient Worker Capacity (40% of cases)

**Symptoms:** All pods at capacity, high CPU/GPU usage, queue growing

**Resolution:**
```bash
# Check current replica count
kubectl get deployment sap-llm -n sap-llm

# Scale up immediately
kubectl scale deployment sap-llm -n sap-llm --replicas=10

# Verify scaling
kubectl get pods -n sap-llm -l app=sap-llm -w

# Check if HPA is configured
kubectl get hpa -n sap-llm

# Update HPA if needed
kubectl patch hpa sap-llm -n sap-llm --type merge -p '{"spec":{"maxReplicas":15,"minReplicas":5}}'

# Monitor throughput improvement
watch -n 5 'curl -s "http://prometheus:9090/api/v1/query?query=rate(sap_llm_envelopes_processed_total[1m])*60" | jq .'
```

### 2. Processing Bottleneck (25% of cases)

**Symptoms:** Specific stage slow, uneven stage processing, backpressure

**Resolution:**
```bash
# Identify bottleneck stage
kubectl logs -n sap-llm deployment/sap-llm | grep "stage_duration" | awk -F: '{print $1, $2}' | sort -k2 -n | tail -10

# Check stage-specific metrics
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_stage_processed_total[5m])' | jq .

# Optimize batch size for slow stage
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"extraction_batch_size":"64"}}'

# Increase parallelism for stage
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"stage_parallelism":"8"}}'

# Restart to apply changes
kubectl rollout restart deployment/sap-llm -n sap-llm
```

### 3. Database Performance Degradation (20% of cases)

**Symptoms:** Slow writes, lock contention, connection pool exhaustion

**Resolution:**
```bash
# Check database connections
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.serverStatus().connections"

# Check for long-running operations
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.currentOp({active: true, secs_running: {$gt: 5}})"

# Check write performance
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.serverStatus().opcounters"

# Increase connection pool
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"db_pool_size":"100"}}'

# Check for missing indexes on write-heavy collections
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "db.envelopes.getIndexes()"

# Kill slow queries if needed
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.killOp(<opid>)"
```

### 4. Configuration Issues (15% of cases)

**Symptoms:** Suboptimal batch sizes, low parallelism, conservative timeouts

**Resolution:**
```bash
# Review current configuration
kubectl get configmap sap-llm-config -n sap-llm -o yaml

# Check batch size settings
kubectl exec -it deployment/sap-llm -n sap-llm -- env | grep BATCH

# Optimize batch sizes
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "batch_size":"128",
  "max_batch_wait_ms":"100",
  "worker_threads":"16"
}}'

# Optimize parallel processing
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "stage_parallelism":"8",
  "max_concurrent_requests":"500"
}}'

# Restart to apply optimizations
kubectl rollout restart deployment/sap-llm -n sap-llm

# Monitor impact
watch -n 5 'curl -s "http://prometheus:9090/api/v1/query?query=rate(sap_llm_envelopes_processed_total[1m])*60" | jq .'
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Assess Throughput Gap:**
   ```bash
   # Check current vs expected throughput
   curl http://prometheus:9090/api/v1/query?query=rate(sap_llm_envelopes_processed_total[5m])*60

   # Check queue backlog
   kubectl exec -it deployment/redis -n sap-llm -- redis-cli LLEN sap_llm:processing_queue

   # Estimate time to clear backlog at current rate
   echo "Backlog will clear in: $((queue_depth / (current_throughput / 60))) minutes"
   ```

2. **Quick Capacity Boost:**
   ```bash
   # Scale up immediately
   kubectl scale deployment sap-llm -n sap-llm --replicas=10

   # Verify new pods are starting
   kubectl get pods -n sap-llm -l app=sap-llm -w

   # Check if nodes need scaling
   kubectl get nodes
   ```

3. **Clear Any Stuck Processing:**
   ```bash
   # Check for stuck workers
   kubectl logs -n sap-llm deployment/sap-llm | grep "stuck\|timeout\|hung"

   # Restart problematic pods
   kubectl delete pod -n sap-llm -l app=sap-llm --field-selector=status.phase!=Running

   # Clear any locks in Redis
   kubectl exec -it deployment/redis -n sap-llm -- redis-cli KEYS "sap_llm:lock:*" | xargs redis-cli DEL
   ```

### Short-term Fixes (5-30 minutes)

1. **If Capacity Constrained:**
   - Scale deployment to max capacity
   - Increase resource limits
   - Enable cluster autoscaling
   - Prioritize critical workloads

2. **If Bottleneck Identified:**
   - Increase parallelism for slow stage
   - Optimize batch sizes
   - Cache frequently accessed data
   - Offload heavy processing

3. **If Database Limited:**
   - Increase connection pool
   - Add read replicas
   - Optimize write patterns
   - Implement write batching

### Long-term Solutions (30+ minutes)

1. **Capacity Planning:**
   - Analyze traffic patterns
   - Right-size infrastructure
   - Implement predictive scaling
   - Plan for peak loads

2. **Performance Optimization:**
   - Profile each pipeline stage
   - Optimize data structures
   - Implement better caching
   - Reduce serialization overhead

3. **Architecture Improvements:**
   - Implement stream processing
   - Add more parallelism
   - Optimize data flow
   - Consider distributed processing

## Escalation

### Level 1: On-Call Engineer (0-15 minutes)
- Follow this runbook
- Scale up capacity
- Clear obvious bottlenecks
- Monitor throughput recovery

### Level 2: Performance Engineer (15-30 minutes)
- If throughput doesn't recover
- If bottleneck analysis needed
- If configuration tuning required
- If resource limits reached

### Level 3: Architecture Team (30+ minutes)
- If architectural changes needed
- If capacity planning required
- If systemic issues identified
- If SLA at risk

## Prevention

1. **Capacity Management:**
   - Configure HPA with appropriate targets
   - Set up cluster autoscaling
   - Monitor capacity trends
   - Plan for growth

2. **Performance Monitoring:**
   - Track throughput metrics per stage
   - Monitor queue depths
   - Set up throughput alerts
   - Review processing efficiency

3. **Load Testing:**
   - Regular load tests
   - Stress testing
   - Capacity validation
   - Performance regression testing

4. **Configuration:**
   - Optimize batch sizes
   - Tune parallelism
   - Configure connection pools
   - Review timeout settings

## Related Runbooks

- [High Latency](./high-latency.md)
- [High Memory](./high-memory.md)
- [Database Connection Failure](./database-connection-failure.md)
- [GPU Utilization Low](./gpu-utilization-low.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
