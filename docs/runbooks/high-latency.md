# Runbook: High Latency

## Alert Details

**Alert Name:** HighLatency
**Severity:** Warning
**Component:** Pipeline
**Threshold:** P95 latency > 600ms over 5 minutes

## Symptoms

- P95 latency exceeds 600ms threshold
- AlertManager firing HighLatency alert
- Users experiencing slow document processing
- Increased time-to-completion for requests
- Queue buildup in processing pipeline

## Diagnosis Steps

### 1. Check Current Latency Metrics

```bash
# Query Prometheus for P95 latency
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(sap_llm_request_duration_bucket[5m]))' | jq .

# View latency dashboard in Grafana
open https://grafana.example.com/d/sap-llm-latency

# Check breakdown by stage
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(sap_llm_stage_duration_bucket[5m]))' | jq .
```

### 2. Identify Slow Components

```bash
# Check which pipeline stage is slow
kubectl logs -n sap-llm deployment/sap-llm --tail=200 | grep "duration_ms" | sort -t: -k2 -n | tail -20

# Check database query performance
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.currentOp()"

# Check Redis response times
kubectl exec -it deployment/redis -n sap-llm -- redis-cli --latency-history
```

### 3. Check System Resources

```bash
# Check CPU usage
kubectl top pods -n sap-llm

# Check memory usage
kubectl top nodes

# Check disk I/O
kubectl exec -it deployment/sap-llm -n sap-llm -- iostat -x 1 5

# Check network latency
kubectl exec -it deployment/sap-llm -n sap-llm -- ping -c 10 mongodb
```

### 4. Review Request Load

```bash
# Check current request rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_requests_total[5m])' | jq .

# Check concurrent requests
curl -s 'http://prometheus:9090/api/v1/query?query=sap_llm_requests_in_progress' | jq .

# Check queue depth
kubectl exec -it deployment/sap-llm -n sap-llm -- redis-cli LLEN sap_llm:processing_queue
```

### 5. Analyze Request Patterns

```bash
# Check for large documents
kubectl logs -n sap-llm deployment/sap-llm --tail=100 | grep "document_size" | sort -t: -k2 -n | tail -10

# Check request complexity distribution
kubectl logs -n sap-llm deployment/sap-llm | grep "complexity_score"

# Review recent processing history
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "db.processing_history.find().sort({duration: -1}).limit(10)"
```

## Common Root Causes

### 1. Database Performance Issues (35% of cases)

**Symptoms:** Slow queries, high DB CPU, missing indexes

**Resolution:**
```bash
# Check slow queries
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "db.setProfilingLevel(2, {slowms: 100})"
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "db.system.profile.find().sort({millis: -1}).limit(10)"

# Check index usage
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "db.documents.getIndexes()"

# Create missing indexes if needed
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "db.documents.createIndex({created_at: -1, status: 1})"

# Check DB resource usage
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.serverStatus()"
```

### 2. Model Inference Slowdown (30% of cases)

**Symptoms:** GPU saturation, memory pressure, model loading delays

**Resolution:**
```bash
# Check GPU utilization
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi dmon -c 10

# Check model inference latency
kubectl logs -n sap-llm deployment/sap-llm | grep "inference_duration_ms"

# Check batch processing efficiency
curl -s 'http://prometheus:9090/api/v1/query?query=sap_llm_batch_size' | jq .

# Increase batch size if low utilization
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"batch_size":"32"}}'

# Restart to clear GPU memory
kubectl rollout restart deployment/sap-llm -n sap-llm
```

### 3. Network Latency (20% of cases)

**Symptoms:** High inter-service latency, packet loss, DNS issues

**Resolution:**
```bash
# Check inter-service latency
kubectl exec -it deployment/sap-llm -n sap-llm -- ping -c 10 mongodb
kubectl exec -it deployment/sap-llm -n sap-llm -- ping -c 10 redis

# Check DNS resolution time
kubectl exec -it deployment/sap-llm -n sap-llm -- time nslookup mongodb

# Check network policies
kubectl get networkpolicy -n sap-llm

# Test service endpoints
kubectl exec -it deployment/sap-llm -n sap-llm -- curl -w "@-" -o /dev/null -s http://sap-llm-api:8000/v1/health <<EOF
    time_namelookup:  %{time_namelookup}\n
    time_connect:  %{time_connect}\n
    time_total:  %{time_total}\n
EOF
```

### 4. Resource Contention (15% of cases)

**Symptoms:** CPU throttling, memory pressure, I/O wait

**Resolution:**
```bash
# Check resource limits
kubectl describe pod -n sap-llm -l app=sap-llm | grep -A 5 "Limits"

# Check throttling
kubectl exec -it deployment/sap-llm -n sap-llm -- cat /sys/fs/cgroup/cpu/cpu.stat

# Increase resource limits if needed
kubectl patch deployment sap-llm -n sap-llm --type json -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/cpu", "value": "4000m"},
  {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "16Gi"}
]'

# Scale horizontally if needed
kubectl scale deployment sap-llm -n sap-llm --replicas=5
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Assess Current State:**
   ```bash
   # Check P95 latency by component
   curl http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(sap_llm_request_duration_bucket[5m]))

   # Check if specific stage is slow
   kubectl logs -n sap-llm deployment/sap-llm --tail=50 | grep "stage_duration"
   ```

2. **Quick Health Check:**
   ```bash
   # Check all services are responsive
   kubectl exec -it deployment/sap-llm -n sap-llm -- curl -f http://sap-llm-api:8000/v1/health

   # Check resource availability
   kubectl top pods -n sap-llm
   kubectl top nodes
   ```

3. **Clear Processing Bottlenecks:**
   ```bash
   # Check queue depth
   kubectl exec -it deployment/redis -n sap-llm -- redis-cli LLEN sap_llm:processing_queue

   # Scale up if queue is backing up
   kubectl scale deployment sap-llm -n sap-llm --replicas=5
   ```

### Short-term Fixes (5-30 minutes)

1. **If Database Slowdown:**
   - Enable query profiling
   - Identify slow queries
   - Add missing indexes
   - Consider read replica for queries

2. **If Model Inference Slow:**
   - Increase batch size
   - Clear GPU memory
   - Scale GPU pods
   - Check for memory leaks

3. **If Network Issues:**
   - Test connectivity to dependencies
   - Check DNS resolution
   - Review network policies
   - Consider service mesh optimizations

### Long-term Solutions (30+ minutes)

1. **Performance Optimization:**
   - Profile application code
   - Optimize database queries
   - Implement caching strategy
   - Tune model inference parameters

2. **Capacity Planning:**
   - Analyze traffic patterns
   - Right-size resource allocations
   - Plan for peak load
   - Implement auto-scaling

3. **Architecture Improvements:**
   - Add caching layers
   - Implement request queuing
   - Optimize data flow
   - Consider async processing

## Escalation

### Level 1: On-Call Engineer (0-15 minutes)
- Follow this runbook
- Check obvious bottlenecks
- Apply quick fixes (scaling, cache clearing)
- Document findings

### Level 2: Performance Engineer (15-30 minutes)
- If latency doesn't improve with quick fixes
- If root cause requires deep analysis
- If database or model tuning needed
- If capacity limits reached

### Level 3: Architecture Team (30+ minutes)
- If systemic performance issues identified
- If architectural changes required
- If requires code optimization
- If affects SLA significantly

## Prevention

1. **Pre-deployment:**
   - Run load tests
   - Profile performance impact
   - Review database query plans
   - Validate resource allocations

2. **Monitoring:**
   - Set up latency percentile alerts (P50, P95, P99)
   - Monitor component-level latency
   - Track slow query patterns
   - Set up distributed tracing

3. **Capacity Management:**
   - Regular capacity reviews
   - Performance benchmarking
   - Load testing schedule
   - Auto-scaling policies

4. **Optimization:**
   - Regular query optimization
   - Index maintenance
   - Code profiling sessions
   - Caching strategy reviews

## Related Runbooks

- [Low Throughput](./low-throughput.md)
- [High Memory](./high-memory.md)
- [Database Connection Failure](./database-connection-failure.md)
- [Model Inference Failure](./model-inference-failure.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
