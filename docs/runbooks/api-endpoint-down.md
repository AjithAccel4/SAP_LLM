# Runbook: API Endpoint Down

## Alert Details

**Alert Name:** APIEndpointDown
**Severity:** Critical
**Component:** API Gateway
**Threshold:** Health check failures > 3 consecutive attempts

## Symptoms

- API health checks failing
- AlertManager firing APIEndpointDown alert
- 503 Service Unavailable responses
- Connection refused errors
- Timeout on API requests
- Users unable to access service
- Complete service outage

## Diagnosis Steps

### 1. Check API Endpoint Health

```bash
# Test API health endpoint
curl -v http://sap-llm-api:8000/v1/health

# Test from within cluster
kubectl exec -it deployment/sap-llm -n sap-llm -- curl -v http://sap-llm-api:8000/v1/health

# Check external endpoint
curl -v https://api.sap-llm.example.com/v1/health

# Check response time
curl -w "@-" -o /dev/null -s http://sap-llm-api:8000/v1/health <<EOF
    http_code: %{http_code}
    time_total: %{time_total}s
EOF

# Query Prometheus for API availability
curl -s 'http://prometheus:9090/api/v1/query?query=up{job="sap-llm-api"}' | jq .
```

### 2. Check API Pod Status

```bash
# Check API pods are running
kubectl get pods -n sap-llm -l app=sap-llm-api

# Check pod readiness and liveness
kubectl get pods -n sap-llm -l app=sap-llm-api -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.conditions[?(@.type=="Ready")].status}{"\t"}{.status.conditions[?(@.type=="ContainersReady")].status}{"\n"}{end}'

# Check for recent restarts
kubectl get pods -n sap-llm -l app=sap-llm-api -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.containerStatuses[0].restartCount}{"\n"}{end}'

# Check pod events
kubectl describe pods -n sap-llm -l app=sap-llm-api

# Check API logs
kubectl logs -n sap-llm deployment/sap-llm-api --tail=100
```

### 3. Check Service and Ingress

```bash
# Check service status
kubectl get service sap-llm-api -n sap-llm

# Check service endpoints
kubectl get endpoints sap-llm-api -n sap-llm

# Verify service is pointing to ready pods
kubectl describe service sap-llm-api -n sap-llm

# Check ingress configuration
kubectl get ingress -n sap-llm

# Check ingress controller
kubectl get pods -n ingress-nginx

# Test service from within cluster
kubectl run test-pod --rm -it --image=curlimages/curl -- curl -v http://sap-llm-api:8000/v1/health
```

### 4. Check Load Balancer and Network

```bash
# Check load balancer status
kubectl get service sap-llm-api -n sap-llm -o jsonpath='{.status.loadBalancer.ingress}'

# Check external IP/hostname
kubectl get service sap-llm-api -n sap-llm -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'

# Test external connectivity
curl -v https://api.sap-llm.example.com/v1/health

# Check network policies
kubectl get networkpolicy -n sap-llm

# Check DNS resolution
nslookup api.sap-llm.example.com
dig api.sap-llm.example.com

# Check SSL certificate
openssl s_client -connect api.sap-llm.example.com:443 -servername api.sap-llm.example.com </dev/null 2>/dev/null | openssl x509 -noout -dates
```

### 5. Check Application Health

```bash
# Check API process inside pod
kubectl exec -it deployment/sap-llm-api -n sap-llm -- ps aux | grep api

# Check port binding
kubectl exec -it deployment/sap-llm-api -n sap-llm -- netstat -tlnp | grep 8000

# Check resource usage
kubectl top pods -n sap-llm -l app=sap-llm-api

# Check application logs for errors
kubectl logs -n sap-llm deployment/sap-llm-api --tail=200 | grep -i "error\|exception\|fatal"

# Check for OOM events
kubectl get pods -n sap-llm -l app=sap-llm-api -o jsonpath='{.items[0].status.containerStatuses[0].lastState.terminated.reason}'
```

## Common Root Causes

### 1. Pods Not Running/Ready (40% of cases)

**Symptoms:** Pods in CrashLoopBackOff, Not Ready, or Failed state

**Resolution:**
```bash
# Check pod status
kubectl get pods -n sap-llm -l app=sap-llm-api

# Check pod logs for errors
kubectl logs -n sap-llm deployment/sap-llm-api --tail=200

# Check previous pod logs if crashed
kubectl logs -n sap-llm deployment/sap-llm-api --previous

# Check pod events
kubectl describe pod -n sap-llm -l app=sap-llm-api

# Check for image pull errors
kubectl describe pod -n sap-llm -l app=sap-llm-api | grep -A 5 "Failed to pull image"

# Check resource constraints
kubectl describe pod -n sap-llm -l app=sap-llm-api | grep -A 5 "Insufficient"

# Delete failed pods to recreate
kubectl delete pod -n sap-llm -l app=sap-llm-api --field-selector=status.phase=Failed

# If OOMKilled, increase memory limits
kubectl patch deployment sap-llm-api -n sap-llm --type json -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/memory", "value": "4Gi"}
]'

# If CrashLoopBackOff, check startup configuration
kubectl get configmap sap-llm-config -n sap-llm -o yaml

# Rollback if recent deployment caused issue
kubectl rollout undo deployment/sap-llm-api -n sap-llm

# Scale up to add more instances
kubectl scale deployment sap-llm-api -n sap-llm --replicas=5

# Force recreation of all pods
kubectl rollout restart deployment/sap-llm-api -n sap-llm
```

### 2. Service/Endpoint Misconfiguration (30% of cases)

**Symptoms:** Service has no endpoints, selector mismatch, port misconfiguration

**Resolution:**
```bash
# Check service configuration
kubectl get service sap-llm-api -n sap-llm -o yaml

# Check if service selector matches pod labels
kubectl get pods -n sap-llm -l app=sap-llm-api --show-labels
kubectl get service sap-llm-api -n sap-llm -o jsonpath='{.spec.selector}'

# Check endpoints
kubectl get endpoints sap-llm-api -n sap-llm

# If no endpoints, fix selector
kubectl patch service sap-llm-api -n sap-llm -p '{"spec":{"selector":{"app":"sap-llm-api"}}}'

# Check port configuration
kubectl get service sap-llm-api -n sap-llm -o jsonpath='{.spec.ports}'

# If port mismatch, fix it
kubectl patch service sap-llm-api -n sap-llm --type json -p='[
  {"op": "replace", "path": "/spec/ports/0/targetPort", "value": 8000}
]'

# Check if readiness probe is failing
kubectl describe pod -n sap-llm -l app=sap-llm-api | grep -A 10 "Readiness"

# Temporarily disable readiness probe to debug
kubectl patch deployment sap-llm-api -n sap-llm --type json -p='[
  {"op": "remove", "path": "/spec/template/spec/containers/0/readinessProbe"}
]'

# Test service from another pod
kubectl run test-curl --rm -it --image=curlimages/curl -- curl -v http://sap-llm-api:8000/v1/health

# Recreate service if corrupted
kubectl delete service sap-llm-api -n sap-llm
kubectl apply -f deployment/service.yaml
```

### 3. Ingress/Load Balancer Issues (20% of cases)

**Symptoms:** External requests failing, ingress not routing, LB not configured

**Resolution:**
```bash
# Check ingress status
kubectl get ingress -n sap-llm -o yaml

# Check ingress controller
kubectl get pods -n ingress-nginx

# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller --tail=100

# Verify ingress rules
kubectl describe ingress sap-llm -n sap-llm

# Check if ingress backend is healthy
kubectl get ingress sap-llm -n sap-llm -o jsonpath='{.status.loadBalancer.ingress}'

# Test backend service directly
kubectl exec -it deployment/sap-llm -n sap-llm -- curl http://sap-llm-api:8000/v1/health

# Check SSL/TLS configuration
kubectl get secret -n sap-llm | grep tls
kubectl describe ingress sap-llm -n sap-llm | grep -A 5 "TLS"

# Check certificate validity
openssl s_client -connect api.sap-llm.example.com:443 -servername api.sap-llm.example.com </dev/null 2>/dev/null | openssl x509 -noout -dates

# Restart ingress controller
kubectl rollout restart deployment/ingress-nginx-controller -n ingress-nginx

# Update ingress annotation if needed
kubectl annotate ingress sap-llm -n sap-llm nginx.ingress.kubernetes.io/rewrite-target=/ --overwrite

# Check load balancer health checks
# AWS ELB example:
aws elb describe-target-health --target-group-arn <arn>

# Recreate ingress
kubectl delete ingress sap-llm -n sap-llm
kubectl apply -f deployment/ingress.yaml
```

### 4. Network/Firewall Issues (10% of cases)

**Symptoms:** Connection timeouts, network policies blocking, DNS failures

**Resolution:**
```bash
# Check network policies
kubectl get networkpolicy -n sap-llm -o yaml

# Test connectivity from pod to pod
kubectl exec -it deployment/sap-llm -n sap-llm -- curl -v http://sap-llm-api:8000/v1/health

# Check DNS resolution
kubectl exec -it deployment/sap-llm -n sap-llm -- nslookup sap-llm-api
kubectl exec -it deployment/sap-llm -n sap-llm -- dig sap-llm-api.sap-llm.svc.cluster.local

# Test external DNS
nslookup api.sap-llm.example.com

# Check if network policy is blocking
kubectl describe networkpolicy -n sap-llm

# Temporarily remove network policies to test
kubectl delete networkpolicy --all -n sap-llm

# Check security groups (AWS example)
aws ec2 describe-security-groups --group-ids <sg-id>

# Check firewall rules on load balancer

# Verify pod can reach internet
kubectl exec -it deployment/sap-llm-api -n sap-llm -- curl -v https://www.google.com

# Check kube-proxy
kubectl get pods -n kube-system -l k8s-app=kube-proxy

# Restart kube-proxy if needed
kubectl delete pod -n kube-system -l k8s-app=kube-proxy

# Check iptables rules
kubectl debug node/<node-name> -it --image=busybox -- chroot /host iptables -L -n | grep sap-llm

# Restart CoreDNS if DNS issues
kubectl rollout restart deployment/coredns -n kube-system
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Verify Endpoint Status:**
   ```bash
   # Quick health check
   curl -v http://sap-llm-api:8000/v1/health

   # Check pod status
   kubectl get pods -n sap-llm -l app=sap-llm-api

   # Check service endpoints
   kubectl get endpoints sap-llm-api -n sap-llm
   ```

2. **Quick Recovery:**
   ```bash
   # Restart API pods
   kubectl rollout restart deployment/sap-llm-api -n sap-llm

   # Delete failed pods
   kubectl delete pod -n sap-llm -l app=sap-llm-api --field-selector=status.phase=Failed

   # Scale up for redundancy
   kubectl scale deployment sap-llm-api -n sap-llm --replicas=5

   # Wait and verify
   sleep 30
   kubectl get pods -n sap-llm -l app=sap-llm-api
   curl http://sap-llm-api:8000/v1/health
   ```

3. **Notify Stakeholders:**
   ```bash
   # Update status page
   # Send notifications
   # Post in incident channel
   # Estimate time to recovery
   ```

### Short-term Fixes (5-30 minutes)

1. **If Pods Not Ready:**
   - Check logs for errors
   - Fix resource constraints
   - Rollback bad deployment
   - Scale up replicas

2. **If Service Issues:**
   - Fix service configuration
   - Update selectors
   - Correct port mappings
   - Recreate service

3. **If Ingress Issues:**
   - Fix ingress rules
   - Update certificates
   - Restart ingress controller
   - Check load balancer

4. **If Network Issues:**
   - Fix network policies
   - Resolve DNS issues
   - Update firewall rules
   - Restart network components

### Long-term Solutions (30+ minutes)

1. **Reliability Improvements:**
   - Implement health checks
   - Add redundancy
   - Multi-region deployment
   - Automated failover

2. **Monitoring:**
   - Enhanced endpoint monitoring
   - Synthetic monitoring
   - Alerting improvements
   - SLA tracking

3. **Process:**
   - Deployment safeguards
   - Automated rollback
   - Canary deployments
   - Blue-green deployments

## Escalation

### Level 1: On-Call Engineer (0-10 minutes)
- Follow this runbook
- Restart services
- Basic troubleshooting
- Initial communication

### Level 2: Platform Engineer (10-30 minutes)
- If infrastructure issues
- If network problems
- If complex troubleshooting needed
- If architectural issue suspected

### Level 3: Director/VP (30+ minutes)
- If extended outage
- If major customer impact
- If requires executive decision
- For public communications

**Communication Priority:**
- Update status page immediately
- Notify customers within 5 minutes
- Post updates every 15 minutes
- Send resolution notification

## Prevention

1. **High Availability:**
   - Multiple replicas (minimum 3)
   - Multi-zone deployment
   - Health checks configured
   - PodDisruptionBudget set

2. **Monitoring:**
   - Endpoint health monitoring
   - Synthetic checks
   - Response time tracking
   - Error rate alerts

3. **Deployment Safety:**
   - Rolling updates
   - Readiness gates
   - Automated rollback
   - Canary deployments

4. **Testing:**
   - Regular failover tests
   - Load testing
   - Chaos engineering
   - DR drills

5. **Documentation:**
   - Updated runbooks
   - Architecture diagrams
   - Incident postmortems
   - Known issues database

## Health Check Configuration

### Recommended Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /v1/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /v1/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3

startupProbe:
  httpGet:
    path: /v1/health
    port: 8000
  initialDelaySeconds: 0
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 30
```

## Related Runbooks

- [SLA Violation](./sla-violation.md)
- [High Error Rate](./high-error-rate.md)
- [Database Connection Failure](./database-connection-failure.md)
- [High Latency](./high-latency.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
