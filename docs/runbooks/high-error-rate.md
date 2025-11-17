# Runbook: High Error Rate

## Alert Details

**Alert Name:** HighErrorRate  
**Severity:** Critical  
**Component:** Pipeline  
**Threshold:** Error rate > 5% over 5 minutes  

## Symptoms

- Error rate exceeds 5% over a 5-minute period
- AlertManager firing HighErrorRate alert
- Increased number of failed document processing requests
- Users reporting processing failures

## Diagnosis Steps

### 1. Check Current Error Rate

```bash
# Query Prometheus for current error rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_errors_total[5m])' | jq .

# View error rate in Grafana
open https://grafana.example.com/d/sap-llm-errors
```

### 2. Identify Error Types

```bash
# Check error distribution by type
kubectl logs -n sap-llm deployment/sap-llm --tail=100 | grep ERROR

# Query error types from Prometheus
curl -s 'http://prometheus:9090/api/v1/query?query=sap_llm_errors_total' | jq .
```

### 3. Check Affected Pipeline Stages

```bash
# Check which stage is failing
kubectl logs -n sap-llm deployment/sap-llm | grep -E "Stage [0-9]" | grep ERROR

# Query errors by stage
curl -s 'http://prometheus:9090/api/v1/query?query=sap_llm_errors_total{stage!=""}' | jq .
```

### 4. Review Recent Changes

```bash
# Check recent deployments
kubectl rollout history deployment/sap-llm -n sap-llm

# View recent commits
git log --oneline -10

# Check recent config changes
kubectl get configmap sap-llm-config -n sap-llm -o yaml
```

### 5. Check External Dependencies

```bash
# Check MongoDB connectivity
kubectl exec -it deployment/sap-llm -n sap-llm -- mongosh mongodb://mongodb:27017 --eval "db.adminCommand('ping')"

# Check Redis connectivity
kubectl exec -it deployment/sap-llm -n sap-llm -- redis-cli -h redis ping

# Check API connectivity
curl -f http://sap-llm-api:8000/v1/health
```

## Common Root Causes

### 1. Validation Errors (40% of cases)

**Symptoms:** High rate of validation failures, specific document types affected

**Resolution:**
```bash
# Check validation errors
kubectl logs -n sap-llm deployment/sap-llm | grep "ValidationError"

# Review business rules
kubectl get configmap sap-llm-rules -n sap-llm -o yaml

# Rollback rule changes if needed
kubectl rollout undo deployment/sap-llm -n sap-llm
```

### 2. Model Inference Errors (30% of cases)

**Symptoms:** GPU/memory errors, model loading failures

**Resolution:**
```bash
# Check GPU status
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi

# Check memory usage
kubectl top pods -n sap-llm

# Restart pods if OOM
kubectl rollout restart deployment/sap-llm -n sap-llm
```

### 3. External API Errors (20% of cases)

**Symptoms:** SAP API connection failures, timeout errors

**Resolution:**
```bash
# Test SAP API connectivity
kubectl exec -it deployment/sap-llm -n sap-llm -- curl -v https://sap-api.example.com/health

# Check API credentials
kubectl get secret sap-api-credentials -n sap-llm -o yaml

# Review API rate limits
# Check SAP API dashboard for rate limit status
```

### 4. Data Quality Issues (10% of cases)

**Symptoms:** Parsing errors, OCR failures, corrupted documents

**Resolution:**
```bash
# Check recent document quality metrics
kubectl logs -n sap-llm deployment/sap-llm | grep "QualityScore"

# Review problematic documents
kubectl exec -it deployment/sap-llm -n sap-llm -- ls /tmp/failed_documents/

# Adjust quality thresholds if needed
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"quality_threshold":"0.85"}}'
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Assess Impact:**
   ```bash
   # Check current error rate
   kubectl logs -n sap-llm deployment/sap-llm --tail=100 | grep -c ERROR
   
   # Check affected users/documents
   curl http://prometheus:9090/api/v1/query?query=sap_llm_errors_total
   ```

2. **Verify System Health:**
   ```bash
   # Check all pods are running
   kubectl get pods -n sap-llm
   
   # Check resource usage
   kubectl top pods -n sap-llm
   ```

3. **Check for Quick Wins:**
   ```bash
   # Restart failed pods
   kubectl delete pod -n sap-llm -l app=sap-llm --field-selector=status.phase=Failed
   
   # Clear cache if needed
   kubectl exec -it deployment/redis -n sap-llm -- redis-cli FLUSHDB
   ```

### Short-term Fixes (5-30 minutes)

1. **If Validation Errors:**
   - Review recent business rule changes
   - Rollback problematic rules
   - Adjust validation thresholds

2. **If Model Errors:**
   - Restart pods to clear GPU memory
   - Scale up replicas if load is high
   - Check model files integrity

3. **If API Errors:**
   - Verify API credentials
   - Check network connectivity
   - Review rate limit status

### Long-term Solutions (30+ minutes)

1. **Root Cause Analysis:**
   - Analyze error patterns
   - Review error logs in detail
   - Identify systemic issues

2. **Preventive Measures:**
   - Update input validation
   - Improve error handling
   - Add retry logic
   - Update documentation

## Escalation

### Level 1: On-Call Engineer (0-15 minutes)
- Follow this runbook
- Attempt quick fixes
- Document findings

### Level 2: Team Lead (15-30 minutes)
- If error rate doesn't improve
- If critical business impact
- If root cause unclear

### Level 3: Architecture Team (30+ minutes)
- If systemic issues identified
- If requires code changes
- If affects multiple components

## Prevention

1. **Pre-deployment:**
   - Run full test suite
   - Validate configuration changes
   - Review business rule changes

2. **Monitoring:**
   - Set up proactive alerts
   - Monitor error trends
   - Regular health checks

3. **Documentation:**
   - Keep runbooks updated
   - Document known issues
   - Share lessons learned

## Related Runbooks

- [Model Inference Failure](./model-inference-failure.md)
- [High Latency](./high-latency.md)
- [Database Connection Failure](./database-connection-failure.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
