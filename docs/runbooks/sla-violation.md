# Runbook: SLA Violation

## Alert Details

**Alert Name:** SLAViolation
**Severity:** Critical
**Component:** System-wide
**Threshold:** SLA compliance < 99.5% over 15 minutes

## Symptoms

- SLA metrics below threshold
- AlertManager firing SLAViolation alert
- Customer complaints about service quality
- Multiple performance degradations
- Elevated error rates or latency
- Business impact escalations

## Diagnosis Steps

### 1. Check Current SLA Metrics

```bash
# Check overall SLA compliance
curl -s 'http://prometheus:9090/api/v1/query?query=sap_llm_sla_compliance_percent' | jq .

# Check availability SLA
curl -s 'http://prometheus:9090/api/v1/query?query=(sum(up{job="sap-llm"})/count(up{job="sap-llm"}))*100' | jq .

# Check latency SLA (95% requests < 1s)
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(sap_llm_request_duration_bucket[5m]))' | jq .

# Check error rate SLA (< 0.5%)
curl -s 'http://prometheus:9090/api/v1/query?query=(rate(sap_llm_errors_total[5m])/rate(sap_llm_requests_total[5m]))*100' | jq .

# View SLA dashboard
open https://grafana.example.com/d/sap-llm-sla
```

### 2. Identify Which SLA Component is Violated

```bash
# Check uptime/availability
curl -s 'http://prometheus:9090/api/v1/query?query=avg_over_time(up{job="sap-llm"}[15m])' | jq .

# Check latency percentiles
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(sap_llm_request_duration_bucket[15m]))' | jq .
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.99,rate(sap_llm_request_duration_bucket[15m]))' | jq .

# Check error rates by type
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_errors_total[15m]) by (error_type)' | jq .

# Check throughput SLA
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_requests_total[15m])*60' | jq .

# Check success rate
curl -s 'http://prometheus:9090/api/v1/query?query=(rate(sap_llm_requests_success_total[15m])/rate(sap_llm_requests_total[15m]))*100' | jq .
```

### 3. Check Service Health

```bash
# Check all pods status
kubectl get pods -n sap-llm

# Check pod readiness
kubectl get pods -n sap-llm -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'

# Check recent pod restarts
kubectl get pods -n sap-llm -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\n"}{end}'

# Check service endpoints
kubectl get endpoints -n sap-llm

# Test health endpoints
kubectl exec -it deployment/sap-llm -n sap-llm -- curl -f http://sap-llm-api:8000/v1/health
```

### 4. Check Recent Incidents

```bash
# Check active alerts
curl -s http://alertmanager:9093/api/v2/alerts | jq '.[] | select(.status.state=="active")'

# Check recent deployments
kubectl rollout history deployment/sap-llm -n sap-llm

# Check recent configuration changes
kubectl get configmap sap-llm-config -n sap-llm -o yaml --show-labels

# Check git history
git log --oneline --since="4 hours ago"

# Check for recent incidents in logs
kubectl logs -n sap-llm deployment/sap-llm --since=1h | grep -i "error\|critical\|fatal"
```

### 5. Check Dependent Services

```bash
# Check MongoDB status
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.serverStatus().ok"

# Check Redis status
kubectl exec -it deployment/redis -n sap-llm -- redis-cli ping

# Check external API connectivity
kubectl exec -it deployment/sap-llm -n sap-llm -- curl -w "@-" -o /dev/null -s https://sap-api.example.com/health <<EOF
    http_code: %{http_code}
    time_total: %{time_total}
EOF

# Check network connectivity
kubectl exec -it deployment/sap-llm -n sap-llm -- ping -c 5 mongodb
kubectl exec -it deployment/sap-llm -n sap-llm -- ping -c 5 redis
```

## Common Root Causes

### 1. Service Outage/Degradation (35% of cases)

**Symptoms:** Pods down, service unavailable, health checks failing

**Resolution:**
```bash
# Check pod status
kubectl get pods -n sap-llm

# Restart failed pods
kubectl delete pod -n sap-llm -l app=sap-llm --field-selector=status.phase=Failed

# Check for OOMKilled pods
kubectl get pods -n sap-llm -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].lastState.terminated.reason}{"\n"}{end}'

# Check resource constraints
kubectl describe nodes | grep -A 5 "Allocated resources"

# Scale up if capacity issue
kubectl scale deployment sap-llm -n sap-llm --replicas=5

# Check for image pull errors
kubectl describe pods -n sap-llm | grep -A 5 "Events"

# Rollback if recent deployment caused issue
kubectl rollout undo deployment/sap-llm -n sap-llm

# Verify service recovery
kubectl get pods -n sap-llm -w
watch -n 5 'curl -s http://prometheus:9090/api/v1/query?query=up{job="sap-llm"} | jq .'
```

### 2. High Latency (25% of cases)

**Symptoms:** Slow response times, P95/P99 latency violations, timeouts

**Resolution:**
```bash
# Check P95/P99 latency
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(sap_llm_request_duration_bucket[5m]))' | jq .

# Identify slow components
kubectl logs -n sap-llm deployment/sap-llm --tail=100 | grep "duration_ms" | sort -t: -k2 -n | tail -20

# Check database performance
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.currentOp({secs_running: {$gt: 1}})"

# Scale up to reduce load per instance
kubectl scale deployment sap-llm -n sap-llm --replicas=10

# Clear caches if stale
kubectl exec -it deployment/redis -n sap-llm -- redis-cli FLUSHDB

# Optimize batch sizes
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"batch_size":"32"}}'

# Follow High Latency runbook for detailed steps
# See: ./high-latency.md
```

### 3. High Error Rate (20% of cases)

**Symptoms:** Elevated error rates, validation failures, API errors

**Resolution:**
```bash
# Check error rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_errors_total[5m])/rate(sap_llm_requests_total[5m])' | jq .

# Check error types
kubectl logs -n sap-llm deployment/sap-llm --tail=200 | grep ERROR | awk '{print $NF}' | sort | uniq -c | sort -rn

# Check for validation errors
kubectl logs -n sap-llm deployment/sap-llm | grep "ValidationError"

# Check for model errors
kubectl logs -n sap-llm deployment/sap-llm | grep "ModelError\|InferenceError"

# Check external API errors
kubectl logs -n sap-llm deployment/sap-llm | grep "APIError\|ConnectionError"

# Rollback if recent change caused errors
kubectl rollout undo deployment/sap-llm -n sap-llm

# Follow High Error Rate runbook for detailed steps
# See: ./high-error-rate.md
```

### 4. Dependent Service Issues (20% of cases)

**Symptoms:** Database errors, Redis failures, external API timeouts

**Resolution:**
```bash
# Check MongoDB health
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.serverStatus()"

# Check MongoDB replication lag
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "rs.status()"

# Check Redis health
kubectl exec -it deployment/redis -n sap-llm -- redis-cli INFO server
kubectl exec -it deployment/redis -n sap-llm -- redis-cli INFO stats

# Test external API
kubectl exec -it deployment/sap-llm -n sap-llm -- curl -v https://sap-api.example.com/health

# Check network policies
kubectl get networkpolicy -n sap-llm

# Restart dependent services if needed
kubectl rollout restart deployment/mongodb -n sap-llm
kubectl rollout restart deployment/redis -n sap-llm

# Implement circuit breaker if external API is down
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "circuit_breaker_enabled":"true",
  "circuit_breaker_threshold":"50"
}}'

# Restart to apply changes
kubectl rollout restart deployment/sap-llm -n sap-llm
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Assess SLA Impact:**
   ```bash
   # Calculate current SLA compliance
   curl http://prometheus:9090/api/v1/query?query=sap_llm_sla_compliance_percent

   # Estimate time to restore SLA
   # Calculate remaining downtime budget

   # Notify stakeholders if severe
   # Update status page
   ```

2. **Quick Triage:**
   ```bash
   # Check what's broken
   kubectl get pods -n sap-llm
   curl http://prometheus:9090/api/v2/alerts | jq '.[] | select(.status.state=="active")'

   # Restart failed services
   kubectl delete pod -n sap-llm -l app=sap-llm --field-selector=status.phase!=Running

   # Scale up if capacity issue
   kubectl scale deployment sap-llm -n sap-llm --replicas=10
   ```

3. **Begin Mitigation:**
   ```bash
   # Route to healthy instances
   kubectl label pods -n sap-llm -l app=sap-llm,status=healthy routing=enabled

   # Reduce non-critical load
   # Enable rate limiting if needed

   # Monitor SLA recovery
   watch -n 10 'curl -s http://prometheus:9090/api/v1/query?query=sap_llm_sla_compliance_percent | jq .'
   ```

### Short-term Fixes (5-30 minutes)

1. **If Availability Issue:**
   - Restart failed pods
   - Scale up replicas
   - Fix infrastructure issues
   - Route around failed nodes

2. **If Latency Issue:**
   - Scale horizontally
   - Optimize slow queries
   - Clear caches
   - Reduce batch sizes

3. **If Error Rate Issue:**
   - Identify error types
   - Rollback bad changes
   - Fix validation issues
   - Handle dependent service failures

### Long-term Solutions (30+ minutes)

1. **Root Cause Analysis:**
   - Identify incident trigger
   - Document timeline
   - Analyze contributing factors
   - Create prevention plan

2. **Improve Resilience:**
   - Add redundancy
   - Implement circuit breakers
   - Add retry logic
   - Improve error handling

3. **Prevent Recurrence:**
   - Update deployment process
   - Add pre-deployment checks
   - Improve monitoring
   - Update runbooks

## Escalation

### Level 1: On-Call Engineer (0-5 minutes)
- Follow this runbook
- Begin immediate mitigation
- Assess SLA impact
- Notify stakeholders

### Level 2: Team Lead (5-15 minutes)
- If SLA not recovering
- If customer impact severe
- If root cause unclear
- Coordinate escalation

### Level 3: Director/VP (15+ minutes)
- If extended SLA violation
- If major customer impact
- If requires executive decision
- If public statement needed

**Critical Communication:**
- Update status page immediately
- Notify key customers
- Post in incident channel
- Keep stakeholders informed

## Prevention

1. **Monitoring:**
   - Real-time SLA tracking
   - Predictive alerting
   - Trend analysis
   - Anomaly detection

2. **Testing:**
   - Load testing
   - Chaos engineering
   - Failover testing
   - SLA validation

3. **Process:**
   - Gradual rollouts
   - Automated rollbacks
   - Pre-deployment validation
   - Change management

4. **Architecture:**
   - Multi-region deployment
   - Redundancy
   - Circuit breakers
   - Graceful degradation

5. **Documentation:**
   - Incident postmortems
   - Updated runbooks
   - SLA reports
   - Lessons learned

## SLA Definitions

### Availability SLA
- **Target:** 99.9% uptime
- **Measurement:** Ratio of successful health checks
- **Exclusions:** Planned maintenance, customer-caused issues

### Latency SLA
- **Target:** P95 < 1000ms, P99 < 2000ms
- **Measurement:** End-to-end request duration
- **Exclusions:** Requests > 10MB, batch operations

### Error Rate SLA
- **Target:** < 0.5% error rate
- **Measurement:** 5xx errors / total requests
- **Exclusions:** 4xx client errors, rate limit errors

### Throughput SLA
- **Target:** > 100,000 envelopes/minute
- **Measurement:** Successful completions per minute
- **Exclusions:** Self-imposed rate limits

## Related Runbooks

- [High Error Rate](./high-error-rate.md)
- [High Latency](./high-latency.md)
- [Low Throughput](./low-throughput.md)
- [Database Connection Failure](./database-connection-failure.md)
- [API Endpoint Down](./api-endpoint-down.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
