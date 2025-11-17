# Runbook: Database Connection Failure

## Alert Details

**Alert Name:** DatabaseConnectionFailure
**Severity:** Critical
**Component:** Database Layer
**Threshold:** DB connection failures > 5% over 5 minutes

## Symptoms

- Database connection errors in logs
- AlertManager firing DatabaseConnectionFailure alert
- Application unable to read/write data
- Connection pool exhaustion
- Connection timeouts
- Authentication failures
- Complete service disruption if database unavailable

## Diagnosis Steps

### 1. Check Database Connectivity

```bash
# Test basic connectivity
kubectl exec -it deployment/sap-llm -n sap-llm -- mongosh mongodb://mongodb:27017 --eval "db.adminCommand('ping')"

# Check if MongoDB pods are running
kubectl get pods -n sap-llm -l app=mongodb

# Check MongoDB service endpoints
kubectl get endpoints mongodb -n sap-llm

# Test from application pod
kubectl exec -it deployment/sap-llm -n sap-llm -- ping -c 5 mongodb

# Check DNS resolution
kubectl exec -it deployment/sap-llm -n sap-llm -- nslookup mongodb
```

### 2. Check Database Health

```bash
# Check MongoDB server status
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.serverStatus()"

# Check replication status
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "rs.status()"

# Check database logs
kubectl logs -n sap-llm deployment/mongodb --tail=100

# Check for authentication issues
kubectl logs -n sap-llm deployment/mongodb | grep -i "auth\|authentication\|unauthorized"

# Check connection count
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.serverStatus().connections"
```

### 3. Check Connection Pool Status

```bash
# Check current connections from application
kubectl logs -n sap-llm deployment/sap-llm --tail=200 | grep -i "connection\|pool"

# Query connection pool metrics
curl -s 'http://prometheus:9090/api/v1/query?query=sap_llm_db_connections_active' | jq .
curl -s 'http://prometheus:9090/api/v1/query?query=sap_llm_db_connections_idle' | jq .

# Check connection pool configuration
kubectl get configmap sap-llm-config -n sap-llm -o yaml | grep -A 5 "db_pool\|mongo"

# Check for connection leaks
kubectl logs -n sap-llm deployment/sap-llm | grep "connection not closed\|connection leak"
```

### 4. Check Network and Firewall

```bash
# Check network policies
kubectl get networkpolicy -n sap-llm

# Check if network policy blocks MongoDB
kubectl describe networkpolicy -n sap-llm

# Test network connectivity on MongoDB port
kubectl exec -it deployment/sap-llm -n sap-llm -- nc -zv mongodb 27017

# Check iptables rules (if applicable)
kubectl exec -it deployment/sap-llm -n sap-llm -- iptables -L -n

# Check for network issues
kubectl exec -it deployment/sap-llm -n sap-llm -- traceroute mongodb
```

### 5. Check Credentials and Authentication

```bash
# Check if secret exists
kubectl get secret mongodb-credentials -n sap-llm

# Verify secret is mounted
kubectl describe pod -n sap-llm -l app=sap-llm | grep -A 5 "Mounts"

# Check environment variables
kubectl exec -it deployment/sap-llm -n sap-llm -- env | grep -i "mongo\|db"

# Test authentication
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --username admin --password "$MONGO_PASSWORD" --authenticationDatabase admin --eval "db.adminCommand('ping')"

# Check user permissions
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.getUsers()"
```

## Common Root Causes

### 1. Connection Pool Exhaustion (40% of cases)

**Symptoms:** Max connections reached, waiting for available connections, timeouts

**Resolution:**
```bash
# Check current connection usage
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.serverStatus().connections"

# Check application connection pool
kubectl logs -n sap-llm deployment/sap-llm --tail=100 | grep "pool"

# Increase connection pool size
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "db_pool_size":"200",
  "db_pool_max":"500"
}}'

# Decrease connection timeout
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "db_connection_timeout_ms":"5000",
  "db_socket_timeout_ms":"30000"
}}'

# Enable connection reuse
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "db_pool_recycle_seconds":"3600"
}}'

# Restart to apply changes
kubectl rollout restart deployment/sap-llm -n sap-llm

# Increase MongoDB max connections if needed
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.adminCommand({setParameter: 1, maxIncomingConnections: 10000})"

# Monitor connection usage
watch -n 5 'kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.serverStatus().connections"'
```

### 2. MongoDB Service Down (30% of cases)

**Symptoms:** All connections failing, MongoDB pods not ready, pod restarts

**Resolution:**
```bash
# Check MongoDB pod status
kubectl get pods -n sap-llm -l app=mongodb

# Check pod events
kubectl describe pod -n sap-llm -l app=mongodb

# Check logs for errors
kubectl logs -n sap-llm deployment/mongodb --tail=200

# Check for OOMKilled
kubectl get pods -n sap-llm -l app=mongodb -o jsonpath='{.items[0].status.containerStatuses[0].lastState.terminated.reason}'

# Check disk space
kubectl exec -it deployment/mongodb -n sap-llm -- df -h

# Check MongoDB data directory
kubectl exec -it deployment/mongodb -n sap-llm -- du -sh /data/db

# Restart MongoDB if crashed
kubectl rollout restart deployment/mongodb -n sap-llm

# If data corruption suspected, restore from backup
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.repairDatabase()"

# Verify MongoDB is accepting connections
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.adminCommand('ping')"

# Scale up if single instance failed
kubectl scale deployment mongodb -n sap-llm --replicas=3
```

### 3. Network/Connectivity Issues (20% of cases)

**Symptoms:** Intermittent failures, DNS errors, timeout errors, packet loss

**Resolution:**
```bash
# Test connectivity
kubectl exec -it deployment/sap-llm -n sap-llm -- ping -c 10 mongodb

# Check DNS resolution
kubectl exec -it deployment/sap-llm -n sap-llm -- nslookup mongodb
kubectl exec -it deployment/sap-llm -n sap-llm -- dig mongodb.sap-llm.svc.cluster.local

# Test port connectivity
kubectl exec -it deployment/sap-llm -n sap-llm -- nc -zv mongodb 27017

# Check network policies
kubectl get networkpolicy -n sap-llm -o yaml

# Remove restrictive network policy if blocking
kubectl delete networkpolicy <policy-name> -n sap-llm

# Check service configuration
kubectl get service mongodb -n sap-llm -o yaml

# Verify endpoints are populated
kubectl get endpoints mongodb -n sap-llm

# Check for network segmentation issues
kubectl exec -it deployment/sap-llm -n sap-llm -- traceroute mongodb

# Test with fully qualified domain name
kubectl exec -it deployment/sap-llm -n sap-llm -- mongosh mongodb://mongodb.sap-llm.svc.cluster.local:27017 --eval "db.adminCommand('ping')"

# Restart CoreDNS if DNS issues
kubectl rollout restart deployment/coredns -n kube-system
```

### 4. Authentication/Authorization Issues (10% of cases)

**Symptoms:** Auth failed errors, permission denied, credential errors

**Resolution:**
```bash
# Check credentials secret
kubectl get secret mongodb-credentials -n sap-llm -o yaml

# Decode credentials
kubectl get secret mongodb-credentials -n sap-llm -o jsonpath='{.data.username}' | base64 -d
kubectl get secret mongodb-credentials -n sap-llm -o jsonpath='{.data.password}' | base64 -d

# Test authentication
MONGO_USER=$(kubectl get secret mongodb-credentials -n sap-llm -o jsonpath='{.data.username}' | base64 -d)
MONGO_PASS=$(kubectl get secret mongodb-credentials -n sap-llm -o jsonpath='{.data.password}' | base64 -d)
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --username "$MONGO_USER" --password "$MONGO_PASS" --eval "db.adminCommand('ping')"

# Check MongoDB users
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.getUsers()"

# Check user roles
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.getUser('$MONGO_USER')"

# Update user password if needed
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.changeUserPassword('$MONGO_USER', '<new-password>')"

# Update secret with new password
kubectl create secret generic mongodb-credentials -n sap-llm \
  --from-literal=username="$MONGO_USER" \
  --from-literal=password="<new-password>" \
  --dry-run=client -o yaml | kubectl apply -f -

# Grant necessary permissions
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.grantRolesToUser('$MONGO_USER', [{role: 'readWrite', db: 'sap_llm'}])"

# Restart application to pick up new credentials
kubectl rollout restart deployment/sap-llm -n sap-llm
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Verify Database Status:**
   ```bash
   # Check if MongoDB is running
   kubectl get pods -n sap-llm -l app=mongodb

   # Test basic connectivity
   kubectl exec -it deployment/sap-llm -n sap-llm -- mongosh mongodb://mongodb:27017 --eval "db.adminCommand('ping')"

   # Check error rate
   kubectl logs -n sap-llm deployment/sap-llm --tail=50 | grep -c "MongoDB\|connection"
   ```

2. **Quick Recovery Attempts:**
   ```bash
   # Restart application pods
   kubectl rollout restart deployment/sap-llm -n sap-llm

   # Clear connection pools
   kubectl delete pod -n sap-llm -l app=sap-llm

   # Test after restart
   sleep 30
   kubectl exec -it deployment/sap-llm -n sap-llm -- curl http://localhost:8080/v1/health
   ```

3. **Enable Fallback if Available:**
   ```bash
   # Switch to read replica if available
   kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
     "mongo_uri":"mongodb://mongodb-replica:27017/sap_llm"
   }}'

   # Enable circuit breaker
   kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
     "db_circuit_breaker_enabled":"true"
   }}'

   # Restart to apply changes
   kubectl rollout restart deployment/sap-llm -n sap-llm
   ```

### Short-term Fixes (5-30 minutes)

1. **If Connection Pool Issue:**
   - Increase pool size
   - Restart application pods
   - Fix connection leaks
   - Add connection monitoring

2. **If MongoDB Down:**
   - Restart MongoDB
   - Check disk space
   - Scale up replicas
   - Restore from backup if needed

3. **If Network Issue:**
   - Fix DNS resolution
   - Update network policies
   - Check service configuration
   - Test connectivity

4. **If Auth Issue:**
   - Verify credentials
   - Update secrets
   - Fix user permissions
   - Restart services

### Long-term Solutions (30+ minutes)

1. **High Availability:**
   - Implement MongoDB replica set
   - Add read replicas
   - Configure automatic failover
   - Multi-AZ deployment

2. **Connection Management:**
   - Optimize connection pooling
   - Implement connection retry logic
   - Add circuit breakers
   - Monitor connection health

3. **Monitoring:**
   - Enhanced database monitoring
   - Connection pool metrics
   - Replication lag tracking
   - Alert tuning

## Escalation

### Level 1: On-Call Engineer (0-15 minutes)
- Follow this runbook
- Restart services
- Fix obvious issues
- Test connectivity

### Level 2: Database Engineer (15-30 minutes)
- If MongoDB issues persist
- If replication issues
- If data corruption suspected
- If performance tuning needed

### Level 3: Infrastructure Team (30+ minutes)
- If infrastructure changes needed
- If network issues
- If multi-region failover needed
- If architectural changes required

## Prevention

1. **Connection Management:**
   - Proper connection pooling
   - Connection leak detection
   - Timeout configuration
   - Retry logic

2. **High Availability:**
   - MongoDB replica sets
   - Automatic failover
   - Read replicas
   - Multi-region deployment

3. **Monitoring:**
   - Connection pool metrics
   - Database health checks
   - Replication lag monitoring
   - Alert on connection issues

4. **Testing:**
   - Connection failure testing
   - Chaos engineering
   - Failover testing
   - Load testing

5. **Documentation:**
   - Connection string formats
   - Credential management
   - Failover procedures
   - Recovery runbooks

## Related Runbooks

- [High Error Rate](./high-error-rate.md)
- [High Latency](./high-latency.md)
- [SLA Violation](./sla-violation.md)
- [API Endpoint Down](./api-endpoint-down.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
