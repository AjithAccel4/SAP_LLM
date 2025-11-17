# Runbook: Low Disk Space

## Alert Details

**Alert Name:** DiskSpaceLow
**Severity:** Warning (< 20%), Critical (< 10%)
**Component:** Infrastructure
**Threshold:** Disk usage > 90% (< 10% free)

## Symptoms

- Disk space below threshold
- AlertManager firing DiskSpaceLow alert
- Write failures in logs
- Pod evictions due to disk pressure
- Database write errors
- Log rotation failures
- Application crashes due to inability to write

## Diagnosis Steps

### 1. Check Disk Usage

```bash
# Check disk usage on all nodes
kubectl get nodes -o custom-columns=NAME:.metadata.name,DISK:.status.allocatable.ephemeral-storage

# Check disk usage in pods
kubectl exec -it deployment/sap-llm -n sap-llm -- df -h

# Check specific mount points
kubectl exec -it deployment/sap-llm -n sap-llm -- df -h /var/log /tmp /data

# Query Prometheus for disk metrics
curl -s 'http://prometheus:9090/api/v1/query?query=node_filesystem_avail_bytes{mountpoint="/"}/node_filesystem_size_bytes{mountpoint="/"}*100' | jq .

# Check persistent volumes
kubectl get pv -o custom-columns=NAME:.metadata.name,CAPACITY:.spec.capacity.storage,STATUS:.status.phase
```

### 2. Identify Large Files and Directories

```bash
# Find largest directories
kubectl exec -it deployment/sap-llm -n sap-llm -- du -h / 2>/dev/null | sort -rh | head -20

# Find largest files
kubectl exec -it deployment/sap-llm -n sap-llm -- find / -type f -size +100M 2>/dev/null -exec ls -lh {} \; | sort -k5 -rh | head -20

# Check log directory size
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /var/log/*

# Check temporary files
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /tmp/*

# Check cache directories
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /tmp/cache /tmp/model_cache

# Check Docker images on node
kubectl debug node/<node-name> -it --image=busybox -- chroot /host du -sh /var/lib/docker
```

### 3. Check Container and Image Storage

```bash
# Check container image sizes
kubectl get pods -n sap-llm -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[0].image}{"\n"}{end}'

# Check for multiple image layers
kubectl debug node/<node-name> -it --image=busybox -- chroot /host docker images

# Check unused volumes
kubectl get pvc -A

# Check for zombie pods consuming space
kubectl get pods -A --field-selector=status.phase!=Running

# Check ephemeral storage usage
kubectl describe node <node-name> | grep -A 5 "Allocated resources"
```

### 4. Check Log Files

```bash
# Check log file sizes
kubectl exec -it deployment/sap-llm -n sap-llm -- ls -lh /var/log/*.log

# Check for log rotation
kubectl exec -it deployment/sap-llm -n sap-llm -- ls -lh /var/log/*.log.*

# Check application logs
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /var/log/sap-llm/

# Check system logs
kubectl debug node/<node-name> -it --image=busybox -- chroot /host du -sh /var/log/*

# Check container logs on node
kubectl debug node/<node-name> -it --image=busybox -- chroot /host du -sh /var/log/containers/
```

### 5. Check Database and Model Storage

```bash
# Check MongoDB data size
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.stats()"
kubectl exec -it deployment/mongodb -n sap-llm -- du -sh /data/db

# Check model storage
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /models/*

# Check for old model versions
kubectl exec -it deployment/sap-llm -n sap-llm -- ls -lht /models/ | head -20

# Check processed documents storage
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /data/documents/

# Check backup storage
kubectl exec -it deployment/mongodb -n sap-llm -- du -sh /data/backup/
```

## Common Root Causes

### 1. Log File Accumulation (40% of cases)

**Symptoms:** Large log files, log rotation not working, old logs not cleaned

**Resolution:**
```bash
# Check log sizes
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /var/log/*

# Truncate large log files immediately
kubectl exec -it deployment/sap-llm -n sap-llm -- sh -c "> /var/log/sap-llm/application.log"

# Remove old rotated logs
kubectl exec -it deployment/sap-llm -n sap-llm -- find /var/log -name "*.log.*" -mtime +7 -delete

# Configure log rotation
kubectl exec -it deployment/sap-llm -n sap-llm -- sh -c 'cat > /etc/logrotate.d/sap-llm << EOF
/var/log/sap-llm/*.log {
    daily
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 root root
    sharedscripts
    postrotate
        kill -USR1 \$(cat /var/run/sap-llm.pid)
    endscript
}
EOF'

# Force log rotation
kubectl exec -it deployment/sap-llm -n sap-llm -- logrotate -f /etc/logrotate.conf

# Reduce log verbosity
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "log_level":"INFO",
  "log_max_size_mb":"100",
  "log_retention_days":"7"
}}'

# Send logs to external system (e.g., CloudWatch, Elasticsearch)
kubectl patch deployment sap-llm -n sap-llm --type json -p='[
  {"op": "add", "path": "/spec/template/spec/containers/0/env/-", "value": {
    "name": "LOG_DESTINATION",
    "value": "cloudwatch"
  }}
]'

# Restart to apply changes
kubectl rollout restart deployment/sap-llm -n sap-llm
```

### 2. Temporary Files and Caches (30% of cases)

**Symptoms:** Large /tmp directory, old cache files, processing artifacts

**Resolution:**
```bash
# Check temp directory size
kubectl exec -it deployment/sap-llm -n sap-llm -- du -sh /tmp/*

# Clear temporary files older than 1 day
kubectl exec -it deployment/sap-llm -n sap-llm -- find /tmp -type f -mtime +1 -delete

# Clear application cache
kubectl exec -it deployment/sap-llm -n sap-llm -- rm -rf /tmp/cache/*
kubectl exec -it deployment/sap-llm -n sap-llm -- rm -rf /tmp/model_cache/*

# Clear failed processing artifacts
kubectl exec -it deployment/sap-llm -n sap-llm -- rm -rf /tmp/failed_documents/*

# Clear Redis cache to free memory (if applicable)
kubectl exec -it deployment/redis -n sap-llm -- redis-cli FLUSHDB

# Configure automatic cleanup
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "temp_file_max_age_hours":"24",
  "cache_max_size_gb":"5",
  "auto_cleanup_enabled":"true"
}}'

# Set up cron job for periodic cleanup
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup-temp-files
  namespace: sap-llm
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: cleanup
            image: busybox
            command:
            - /bin/sh
            - -c
            - find /tmp -type f -mtime +1 -delete
            volumeMounts:
            - name: tmp
              mountPath: /tmp
          restartPolicy: OnFailure
          volumes:
          - name: tmp
            emptyDir: {}
EOF

# Restart to apply changes
kubectl rollout restart deployment/sap-llm -n sap-llm
```

### 3. Database Growth (20% of cases)

**Symptoms:** Large database files, old data not purged, index bloat

**Resolution:**
```bash
# Check database size
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh --eval "db.stats(1024*1024*1024)"

# Check collection sizes
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "
  db.getCollectionNames().forEach(function(col) {
    var stats = db[col].stats(1024*1024);
    print(col + ': ' + stats.size + ' MB');
  })
"

# Compact database
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "db.runCommand({compact: 'documents'})"

# Remove old data
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "
  db.processing_history.deleteMany({
    created_at: {\$lt: new Date(Date.now() - 90*24*60*60*1000)}
  })
"

# Remove old logs from database
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "
  db.logs.deleteMany({
    timestamp: {\$lt: new Date(Date.now() - 30*24*60*60*1000)}
  })
"

# Set up TTL index for automatic deletion
kubectl exec -it deployment/mongodb -n sap-llm -- mongosh sap_llm --eval "
  db.processing_history.createIndex(
    {created_at: 1},
    {expireAfterSeconds: 7776000}  // 90 days
  )
"

# Configure data retention policy
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "data_retention_days":"90",
  "log_retention_days":"30"
}}'

# Consider archiving old data
kubectl exec -it deployment/mongodb -n sap-llm -- mongodump --archive=/data/backup/archive-$(date +%Y%m%d).gz --gzip --db=sap_llm --query='{"created_at": {"$lt": new Date("2024-01-01")}}'
```

### 4. Container Image and Volume Bloat (10% of cases)

**Symptoms:** Multiple image versions, unused volumes, dangling images

**Resolution:**
```bash
# Clean up unused images on nodes
kubectl debug node/<node-name> -it --image=busybox -- chroot /host docker image prune -a -f

# Remove dangling volumes
kubectl debug node/<node-name> -it --image=busybox -- chroot /host docker volume prune -f

# List and remove unused PVCs
kubectl get pvc -A --no-headers | awk '{if ($2 != "Bound") print $1, $2}' | while read ns pvc; do
  kubectl delete pvc $pvc -n $ns
done

# Remove old model versions
kubectl exec -it deployment/sap-llm -n sap-llm -- sh -c 'ls -t /models/*.bin | tail -n +4 | xargs rm -f'

# Clean up failed pod volumes
kubectl delete pod -n sap-llm --field-selector=status.phase=Failed

# Increase persistent volume size
kubectl patch pvc mongodb-data -n sap-llm -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'

# Add new volume for models if needed
kubectl apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: sap-llm
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
EOF

# Mount new volume to deployment
kubectl patch deployment sap-llm -n sap-llm --type json -p='[
  {"op": "add", "path": "/spec/template/spec/volumes/-", "value": {
    "name": "models",
    "persistentVolumeClaim": {"claimName": "models-pvc"}
  }},
  {"op": "add", "path": "/spec/template/spec/containers/0/volumeMounts/-", "value": {
    "name": "models",
    "mountPath": "/models"
  }}
]'
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Assess Severity:**
   ```bash
   # Check current disk usage
   kubectl exec -it deployment/sap-llm -n sap-llm -- df -h

   # Identify critical filesystems
   kubectl exec -it deployment/sap-llm -n sap-llm -- df -h | awk '$5+0 > 90'

   # Check for write errors
   kubectl logs -n sap-llm deployment/sap-llm --tail=50 | grep -i "disk\|space\|write error"
   ```

2. **Quick Space Recovery:**
   ```bash
   # Truncate large log files
   kubectl exec -it deployment/sap-llm -n sap-llm -- sh -c "> /var/log/sap-llm/application.log"

   # Clear temp files
   kubectl exec -it deployment/sap-llm -n sap-llm -- rm -rf /tmp/*

   # Clear cache
   kubectl exec -it deployment/sap-llm -n sap-llm -- rm -rf /tmp/cache/*

   # Verify space freed
   kubectl exec -it deployment/sap-llm -n sap-llm -- df -h
   ```

3. **Prevent Further Issues:**
   ```bash
   # Reduce log verbosity
   kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"log_level":"WARN"}}'

   # Disable non-critical logging
   kubectl set env deployment/sap-llm -n sap-llm DEBUG_MODE=false

   # Restart to apply changes
   kubectl rollout restart deployment/sap-llm -n sap-llm
   ```

### Short-term Fixes (5-30 minutes)

1. **If Log Accumulation:**
   - Configure log rotation
   - Truncate large logs
   - Set up external log shipping
   - Reduce log verbosity

2. **If Temp Files:**
   - Clear old temp files
   - Set up automatic cleanup
   - Configure cache limits
   - Remove processing artifacts

3. **If Database Growth:**
   - Archive old data
   - Purge unnecessary records
   - Compact database
   - Set up TTL indexes

4. **If Volume Full:**
   - Expand persistent volume
   - Add additional volumes
   - Clean up unused volumes
   - Implement data lifecycle

### Long-term Solutions (30+ minutes)

1. **Disk Management:**
   - Implement automated cleanup
   - Set up log aggregation
   - Configure data retention policies
   - Monitor disk usage trends

2. **Storage Architecture:**
   - Separate data and logs
   - Use appropriate storage classes
   - Implement tiered storage
   - Archive old data to cold storage

3. **Monitoring:**
   - Set up disk usage alerts
   - Track growth trends
   - Monitor I/O patterns
   - Capacity planning

## Escalation

### Level 1: On-Call Engineer (0-15 minutes)
- Follow this runbook
- Clear temporary files
- Truncate logs
- Free immediate space

### Level 2: Infrastructure Engineer (15-30 minutes)
- If disk expansion needed
- If persistent issues
- If storage reconfiguration needed
- If data migration required

### Level 3: Architecture Team (30+ minutes)
- If architectural changes needed
- If storage strategy redesign
- If data lifecycle changes
- If multi-tier storage needed

## Prevention

1. **Monitoring:**
   - Disk usage alerts (warning at 80%, critical at 90%)
   - Growth rate tracking
   - Capacity forecasting
   - Per-directory monitoring

2. **Automation:**
   - Automated log rotation
   - Scheduled cleanup jobs
   - Data archival automation
   - TTL indexes on time-series data

3. **Configuration:**
   - Appropriate log levels
   - Cache size limits
   - Data retention policies
   - Temp file cleanup

4. **Capacity Planning:**
   - Regular capacity reviews
   - Growth projections
   - Storage allocation strategy
   - Reserved capacity buffer

5. **Best Practices:**
   - External log aggregation
   - Separate volumes for data/logs
   - Regular cleanup schedules
   - Data lifecycle management

## Related Runbooks

- [High Memory](./high-memory.md)
- [Database Connection Failure](./database-connection-failure.md)
- [SLA Violation](./sla-violation.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
