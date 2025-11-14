# SAP_LLM Operations Playbooks

Complete operational procedures for running SAP_LLM in production.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Deployment Procedures](#deployment-procedures)
3. [Scaling Procedures](#scaling-procedures)
4. [Backup and Recovery](#backup-and-recovery)
5. [Monitoring and Alerts](#monitoring-and-alerts)
6. [Maintenance Windows](#maintenance-windows)
7. [Incident Response](#incident-response)

---

## Daily Operations

### Morning Health Check (Every Day, 8:00 AM)

**Duration:** 15 minutes

```bash
#!/bin/bash
# daily-health-check.sh

echo "=== SAP_LLM Daily Health Check ==="
echo "Date: $(date)"
echo ""

# 1. Check API Health
echo "1. API Health Check..."
API_STATUS=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$API_STATUS" != "healthy" ]; then
    echo "❌ API UNHEALTHY"
    exit 1
else
    echo "✅ API healthy"
fi

# 2. Check SLO Compliance
echo "2. SLO Compliance..."
SLO_DATA=$(curl -s http://localhost:8000/v1/slo)
AVAILABILITY=$(echo $SLO_DATA | jq -r '.availability.current')
LATENCY_P95=$(echo $SLO_DATA | jq -r '.latency.p95_current')
ERROR_RATE=$(echo $SLO_DATA | jq -r '.error_rate.current')

echo "   Availability: $AVAILABILITY (target: 99.99%)"
echo "   Latency P95: ${LATENCY_P95}ms (target: <100ms)"
echo "   Error Rate: ${ERROR_RATE}% (target: <1%)"

if (( $(echo "$AVAILABILITY < 99.99" | bc -l) )); then
    echo "⚠️  Availability below SLO"
fi

# 3. Check Resource Usage
echo "3. Resource Usage..."
kubectl top nodes
kubectl top pods -n sap-llm-production | head -10

# 4. Check Recent Errors
echo "4. Recent Errors (last 1 hour)..."
ERROR_COUNT=$(curl -s http://localhost:8000/metrics | grep "sap_llm_errors_total" | awk '{sum+=$2} END {print sum}')
echo "   Total errors: $ERROR_COUNT"

if [ "$ERROR_COUNT" -gt 100 ]; then
    echo "⚠️  High error count detected"
fi

# 5. Check Cost
echo "5. Cost Check..."
DAILY_COST=$(curl -s http://localhost:8000/v1/cost | jq -r '.daily_cost')
echo "   Cost (24h): \$$DAILY_COST"

# 6. Check Security
echo "6. Security Check..."
AUTH_FAILURES=$(curl -s http://localhost:8000/metrics | grep "authentication_failures" | awk '{print $2}')
echo "   Auth failures (24h): $AUTH_FAILURES"

if [ "$AUTH_FAILURES" -gt 50 ]; then
    echo "⚠️  High authentication failure rate"
fi

echo ""
echo "✅ Daily health check complete"
echo "Report sent to ops@company.com"
```

**Run:**
```bash
chmod +x daily-health-check.sh
./daily-health-check.sh | tee -a /var/log/sap_llm/daily-check.log
```

---

### Weekly Performance Review (Every Monday, 10:00 AM)

**Duration:** 30 minutes

**Checklist:**

1. **Review SLO Dashboard**
   - Open Grafana: http://grafana:3000/d/sap-llm-slo
   - Check error budget remaining
   - Document any SLO violations

2. **Analyze Trends**
   ```bash
   # Get 7-day metrics
   curl http://prometheus:9090/api/v1/query_range \
     --data-urlencode 'query=rate(sap_llm_requests_total[7d])' \
     --data-urlencode "start=$(date -d '7 days ago' +%s)" \
     --data-urlencode "end=$(date +%s)" \
     --data-urlencode 'step=3600'
   ```

3. **Review Cost Reports**
   ```bash
   # Generate weekly cost report
   curl http://localhost:8000/v1/cost/report?period=7d > weekly_cost_report.json
   ```

4. **Check Model Performance**
   ```bash
   # Get model accuracy trend
   curl http://localhost:8000/v1/learning/status | jq '.performance'
   ```

5. **Review Incidents**
   - Open incident log: `/var/log/sap_llm/incidents/`
   - Document lessons learned
   - Update runbooks if needed

**Output:** Weekly report email to stakeholders

---

## Deployment Procedures

### Standard Deployment (Non-Critical Update)

**When:** Every 2 weeks, Thursday 2:00 PM (low-traffic period)
**Duration:** 1 hour
**Impact:** None (rolling update)

**Pre-Deployment Checklist:**

- [ ] Code reviewed and approved
- [ ] All tests passing (unit, integration, load)
- [ ] Staging environment tested
- [ ] Rollback plan prepared
- [ ] Stakeholders notified
- [ ] Backup taken

**Procedure:**

```bash
#!/bin/bash
# standard-deployment.sh

set -e  # Exit on error

DEPLOYMENT_ID="DEPLOY-$(date +%Y%m%d-%H%M%S)"
echo "Starting deployment: $DEPLOYMENT_ID"

# 1. Pre-deployment backup
echo "Step 1: Creating backup..."
kubectl exec -it cosmos-backup -- /backup.sh

# 2. Build and push new image
echo "Step 2: Building new image..."
docker build -t sap-llm:$DEPLOYMENT_ID .
docker tag sap-llm:$DEPLOYMENT_ID registry.io/sap-llm:latest
docker push registry.io/sap-llm:latest

# 3. Update deployment (rolling update)
echo "Step 3: Updating deployment..."
kubectl set image deployment/sap-llm-api \
  api=registry.io/sap-llm:latest \
  --record

# 4. Monitor rollout
echo "Step 4: Monitoring rollout..."
kubectl rollout status deployment/sap-llm-api

# 5. Verify health
echo "Step 5: Verifying health..."
sleep 30  # Wait for pods to stabilize

HEALTH_STATUS=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$HEALTH_STATUS" != "healthy" ]; then
    echo "❌ Health check failed! Rolling back..."
    kubectl rollout undo deployment/sap-llm-api
    exit 1
fi

# 6. Run smoke tests
echo "Step 6: Running smoke tests..."
pytest tests/smoke/ -v

if [ $? -ne 0 ]; then
    echo "❌ Smoke tests failed! Rolling back..."
    kubectl rollout undo deployment/sap-llm-api
    exit 1
fi

# 7. Monitor for 15 minutes
echo "Step 7: Monitoring for 15 minutes..."
for i in {1..15}; do
    ERROR_RATE=$(curl -s http://localhost:8000/metrics | grep "errors_total" | awk '{sum+=$2} END {print sum}')
    echo "   Minute $i: Errors=$ERROR_RATE"

    if [ "$ERROR_RATE" -gt 10 ]; then
        echo "❌ High error rate! Rolling back..."
        kubectl rollout undo deployment/sap-llm-api
        exit 1
    fi

    sleep 60
done

echo "✅ Deployment successful: $DEPLOYMENT_ID"
echo "Deployment complete at $(date)"

# 8. Update documentation
git tag $DEPLOYMENT_ID
git push --tags

# 9. Notify stakeholders
curl -X POST https://slack.com/webhook \
  -d '{"text":"✅ SAP_LLM deployed successfully: '$DEPLOYMENT_ID'"}'
```

**Rollback Procedure:**

```bash
# Quick rollback (within 1 hour)
kubectl rollout undo deployment/sap-llm-api

# Rollback to specific revision
kubectl rollout history deployment/sap-llm-api
kubectl rollout undo deployment/sap-llm-api --to-revision=5
```

---

### Emergency Hotfix Deployment

**When:** Critical bug or security issue
**Duration:** 30 minutes
**Impact:** Minimal (rolling update with faster cadence)

```bash
#!/bin/bash
# emergency-hotfix.sh

echo "⚠️  EMERGENCY HOTFIX DEPLOYMENT"
echo "Reason: $1"
echo "Ticket: $2"

# Skip some checks for speed
# 1. Build
docker build -t sap-llm:hotfix-$(date +%s) .
docker push registry.io/sap-llm:hotfix

# 2. Deploy with aggressive rollout
kubectl set image deployment/sap-llm-api api=registry.io/sap-llm:hotfix
kubectl rollout status deployment/sap-llm-api --timeout=5m

# 3. Quick verification
curl http://localhost:8000/health

echo "Hotfix deployed. Monitor closely!"
```

---

## Scaling Procedures

### Horizontal Scaling (Add More Pods)

**When to Scale Up:**
- CPU usage > 70% for 5 minutes
- Request queue > 1000
- Latency P95 > 200ms

**Procedure:**

```bash
# Manual scaling
kubectl scale deployment sap-llm-api --replicas=10
kubectl scale deployment sap-llm-worker --replicas=20
kubectl scale deployment sap-llm-gpu --replicas=5

# Auto-scaling (recommended)
kubectl autoscale deployment sap-llm-api \
  --min=3 --max=20 --cpu-percent=70

kubectl autoscale deployment sap-llm-worker \
  --min=5 --max=50 --cpu-percent=70

kubectl autoscale deployment sap-llm-gpu \
  --min=2 --max=10 --cpu-percent=80
```

**When to Scale Down:**
- CPU usage < 30% for 15 minutes
- Request queue < 100
- Cost optimization needed

```bash
# Scale down gradually
kubectl scale deployment sap-llm-api --replicas=5
# Wait 5 minutes and monitor
kubectl scale deployment sap-llm-api --replicas=3
```

---

### Vertical Scaling (Increase Resources)

**When:** Persistent high memory/CPU usage

```bash
# Update resources
kubectl edit deployment sap-llm-api

# Change:
# resources:
#   requests:
#     memory: "8Gi"      # was 4Gi
#     cpu: "4000m"       # was 2000m
#   limits:
#     memory: "16Gi"     # was 8Gi
#     cpu: "8000m"       # was 4000m

# Apply changes (will trigger rolling restart)
kubectl rollout status deployment/sap-llm-api
```

---

## Backup and Recovery

### Daily Automated Backup

**Schedule:** Every day at 2:00 AM UTC
**Retention:** 30 days

```bash
#!/bin/bash
# automated-backup.sh

BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/sap_llm/$BACKUP_DATE"

mkdir -p $BACKUP_DIR

echo "Starting backup: $BACKUP_DATE"

# 1. Backup Cosmos DB
echo "Backing up Cosmos DB..."
az cosmosdb sql database show \
  -a sap-llm-cosmos \
  -g sap-llm-rg \
  -n sap_llm \
  --output json > $BACKUP_DIR/cosmos_metadata.json

# Point-in-time restore capability (automatic with Cosmos DB)

# 2. Backup Redis (snapshot)
echo "Backing up Redis..."
redis-cli --rdb $BACKUP_DIR/redis-snapshot.rdb

# 3. Backup Neo4j (PMG)
echo "Backing up Neo4j..."
kubectl exec neo4j-0 -- neo4j-admin backup \
  --backup-dir=/backup --name=pmg-$BACKUP_DATE

kubectl cp neo4j-0:/backup/pmg-$BACKUP_DATE $BACKUP_DIR/neo4j/

# 4. Backup Models
echo "Backing up models..."
kubectl cp sap-llm-gpu-0:/models $BACKUP_DIR/models/

# 5. Backup Configuration
echo "Backing up config..."
kubectl get configmap sap-llm-config -o yaml > $BACKUP_DIR/config.yaml
kubectl get secret sap-llm-secrets -o yaml > $BACKUP_DIR/secrets.yaml

# 6. Compress and upload to cloud storage
echo "Compressing..."
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR

echo "Uploading to Azure Blob..."
az storage blob upload \
  --account-name sapllmbackups \
  --container-name daily-backups \
  --name sap_llm-$BACKUP_DATE.tar.gz \
  --file $BACKUP_DIR.tar.gz

# 7. Cleanup old backups (keep 30 days)
find /backups/sap_llm/ -mtime +30 -delete

echo "✅ Backup complete: $BACKUP_DATE"
```

**Cron Schedule:**
```cron
0 2 * * * /scripts/automated-backup.sh >> /var/log/sap_llm/backup.log 2>&1
```

---

### Disaster Recovery Procedure

**RTO (Recovery Time Objective):** < 60 minutes
**RPO (Recovery Point Objective):** < 5 minutes

**Full System Recovery:**

```bash
#!/bin/bash
# disaster-recovery.sh

echo "⚠️  DISASTER RECOVERY INITIATED"
echo "Recovery Point: $1"  # e.g., 20250115

RECOVERY_POINT=$1

# 1. Download backup
echo "Step 1: Downloading backup..."
az storage blob download \
  --account-name sapllmbackups \
  --container-name daily-backups \
  --name sap_llm-$RECOVERY_POINT.tar.gz \
  --file /tmp/recovery.tar.gz

tar -xzf /tmp/recovery.tar.gz -C /tmp/

# 2. Restore Cosmos DB (Point-in-time restore)
echo "Step 2: Restoring Cosmos DB..."
az cosmosdb sql database restore \
  -a sap-llm-cosmos \
  -g sap-llm-rg \
  -n sap_llm \
  --restore-timestamp "$(date -d $RECOVERY_POINT +%Y-%m-%dT%H:%M:%S)"

# 3. Restore Redis
echo "Step 3: Restoring Redis..."
kubectl cp /tmp/backups/$RECOVERY_POINT/redis-snapshot.rdb redis-0:/data/dump.rdb
kubectl delete pod redis-0  # Restart to load snapshot

# 4. Restore Neo4j
echo "Step 4: Restoring Neo4j..."
kubectl exec neo4j-0 -- neo4j-admin restore \
  --from=/backup/pmg-$RECOVERY_POINT \
  --database=neo4j --force

kubectl delete pod neo4j-0  # Restart

# 5. Restore Models
echo "Step 5: Restoring models..."
kubectl cp /tmp/backups/$RECOVERY_POINT/models/ sap-llm-gpu-0:/models/

# 6. Restore Configuration
echo "Step 6: Restoring configuration..."
kubectl apply -f /tmp/backups/$RECOVERY_POINT/config.yaml
kubectl apply -f /tmp/backups/$RECOVERY_POINT/secrets.yaml

# 7. Restart all services
echo "Step 7: Restarting services..."
kubectl rollout restart deployment/sap-llm-api
kubectl rollout restart deployment/sap-llm-worker
kubectl rollout restart deployment/sap-llm-gpu

# 8. Wait for services to be ready
kubectl wait --for=condition=ready pod -l app=sap-llm-api --timeout=300s

# 9. Verify recovery
echo "Step 9: Verifying recovery..."
./scripts/smoke-test.sh

if [ $? -eq 0 ]; then
    echo "✅ RECOVERY SUCCESSFUL"
else
    echo "❌ RECOVERY FAILED - Manual intervention required"
    exit 1
fi

echo "Recovery completed at $(date)"
```

---

## Monitoring and Alerts

### Alert Configuration

**Critical Alerts (Immediate Response - PagerDuty)**

```yaml
# prometheus-alerts.yaml
groups:
  - name: sap_llm_critical
    interval: 1m
    rules:
      - alert: APIDown
        expr: up{job="sap-llm-api"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "SAP_LLM API is down"

      - alert: HighErrorRate
        expr: rate(sap_llm_errors_total[5m]) / rate(sap_llm_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate above 5%"

      - alert: SLOViolation
        expr: sap_llm_availability < 0.9999
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "SLO violated - availability below 99.99%"
```

**Warning Alerts (Email/Slack)**

```yaml
  - name: sap_llm_warnings
    interval: 5m
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(sap_llm_request_duration_seconds_bucket[5m])) > 0.2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P95 latency above 200ms"

      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Memory usage above 90%"
```

---

### On-Call Procedures

**On-Call Rotation:**
- Week 1: Engineer A
- Week 2: Engineer B
- Week 3: Engineer C
- Week 4: Engineer D

**On-Call Handoff (Every Monday, 9:00 AM):**

```markdown
## On-Call Handoff Template

**From:** [Current on-call]
**To:** [Next on-call]
**Date:** YYYY-MM-DD

### Status
- [ ] All systems green
- [ ] Known issues: [List any ongoing issues]
- [ ] Pending tasks: [List any pending items]

### Incidents This Week
- [INC-001] Brief description - Status: Resolved
- [INC-002] Brief description - Status: Monitoring

### Important Notes
- Recent deployments
- Configuration changes
- Upcoming maintenance

### Access Checklist
- [ ] PagerDuty access confirmed
- [ ] VPN credentials working
- [ ] kubectl access verified
- [ ] Azure portal access confirmed
- [ ] Runbooks reviewed

### Contact Numbers
- L2 Escalation: +1-xxx-xxx-xxxx
- L3 Escalation: +1-xxx-xxx-xxxx
- Security Team: +1-xxx-xxx-xxxx
```

---

## Maintenance Windows

### Planned Maintenance (Monthly)

**Schedule:** First Sunday of each month, 2:00 AM - 6:00 AM UTC
**Impact:** None (multi-region deployment ensures zero downtime)

**Procedure:**

```bash
#!/bin/bash
# planned-maintenance.sh

echo "Starting planned maintenance: $(date)"

# 1. Update notice
echo "Posting maintenance notice..."
curl -X POST http://status.company.com/maintenance \
  -d "Starting maintenance in region US-EAST"

# 2. Drain region US-EAST
echo "Draining US-EAST..."
# Traffic automatically routes to US-WEST and EU-CENTRAL

# 3. Update nodes
echo "Updating nodes..."
for node in $(kubectl get nodes -l region=us-east -o name); do
    kubectl drain $node --ignore-daemonsets --delete-emptydir-data
    # Perform OS updates
    ssh $node "sudo apt-get update && sudo apt-get upgrade -y"
    kubectl uncordon $node
done

# 4. Update Kubernetes
echo "Updating Kubernetes..."
kubectl version --short
# Follow cluster upgrade procedure

# 5. Verify region health
echo "Verifying health..."
./scripts/health-check.sh

# 6. Restore traffic to US-EAST
echo "Restoring traffic..."
curl -X POST http://traffic-manager.com/restore \
  -d "region=us-east"

# 7. Repeat for other regions
echo "Proceeding to US-WEST..."
# ... repeat steps for US-WEST
# ... repeat steps for EU-CENTRAL

echo "✅ Maintenance complete: $(date)"
```

---

## Incident Response

### Incident Response Plan

**P1 - Critical (Response Time: < 5 minutes)**

**Criteria:**
- Complete system outage
- Security breach
- Data loss
- SLA violation affecting > 50% of users

**Response:**

```bash
#!/bin/bash
# incident-p1-response.sh

# 1. Acknowledge incident
pagerduty-cli acknowledge $INCIDENT_ID

# 2. Assess impact
./scripts/impact-assessment.sh

# 3. Activate war room
zoom start incident-war-room

# 4. Notify stakeholders
./scripts/notify-stakeholders.sh --severity=P1

# 5. Start incident log
echo "=== INCIDENT LOG ===" > /tmp/incident-$INCIDENT_ID.log
echo "Time: $(date)" >> /tmp/incident-$INCIDENT_ID.log
echo "Severity: P1" >> /tmp/incident-$INCIDENT_ID.log

# 6. Follow runbook
case $INCIDENT_TYPE in
    "api_down")
        ./runbooks/api-down.sh
        ;;
    "database_failure")
        ./runbooks/database-failure.sh
        ;;
    "security_breach")
        ./runbooks/security-breach.sh
        ;;
esac

# 7. Monitor resolution
while [ "$STATUS" != "resolved" ]; do
    sleep 60
    ./scripts/check-status.sh
done

# 8. Post-incident review
./scripts/post-incident-review.sh $INCIDENT_ID
```

---

### Post-Incident Review (Within 48 hours)

**Template:**

```markdown
## Post-Incident Review

**Incident ID:** INC-YYYYMMDD-NNN
**Date:** YYYY-MM-DD
**Duration:** HH:MM
**Severity:** P1/P2/P3/P4

### What Happened
[Detailed description]

### Timeline
| Time | Event |
|------|-------|
| 00:00 | Incident started |
| 00:05 | Alert triggered |
| 00:10 | Engineer responded |
| 00:30 | Root cause identified |
| 01:00 | Fix applied |
| 01:15 | Incident resolved |

### Root Cause
[Deep dive into root cause]

### Impact
- Users affected: N
- Revenue impact: $X
- Services affected: [List]

### What Went Well
- Quick detection
- Fast response
- Clear communication

### What Could Be Improved
- [Improvement 1]
- [Improvement 2]

### Action Items
- [ ] Update runbook (Owner: X, Due: DATE)
- [ ] Add monitoring (Owner: Y, Due: DATE)
- [ ] Fix root cause (Owner: Z, Due: DATE)

### Lessons Learned
[Key takeaways]
```

---

## Operational Metrics

### Daily Metrics Report

```bash
#!/bin/bash
# daily-metrics-report.sh

echo "=== SAP_LLM Daily Metrics Report ==="
echo "Date: $(date +%Y-%m-%d)"
echo ""

# 1. Volume Metrics
DOCS_PROCESSED=$(curl -s http://localhost:8000/metrics | grep documents_processed_total | awk '{print $2}')
echo "Documents Processed: $DOCS_PROCESSED"

# 2. Performance Metrics
P95_LATENCY=$(curl -s http://localhost:8000/v1/slo | jq -r '.latency.p95_current')
echo "P95 Latency: ${P95_LATENCY}ms"

# 3. Availability
AVAILABILITY=$(curl -s http://localhost:8000/v1/slo | jq -r '.availability.current')
echo "Availability: ${AVAILABILITY}%"

# 4. Error Rate
ERROR_RATE=$(curl -s http://localhost:8000/v1/slo | jq -r '.error_rate.current')
echo "Error Rate: ${ERROR_RATE}%"

# 5. Cost
DAILY_COST=$(curl -s http://localhost:8000/v1/cost | jq -r '.daily_cost')
echo "Daily Cost: \$$DAILY_COST"

# 6. Cache Performance
CACHE_HIT_RATE=$(curl -s http://localhost:8000/metrics | grep cache_hit_rate | awk '{print $2}')
echo "Cache Hit Rate: ${CACHE_HIT_RATE}%"

# 7. Resource Utilization
echo ""
echo "Resource Utilization:"
kubectl top nodes | awk 'NR==1 || /gpu/'

# 8. Generate PDF report
./scripts/generate-pdf-report.py --date $(date +%Y-%m-%d)

# 9. Email report
mail -s "SAP_LLM Daily Report" stakeholders@company.com < daily-report.txt
```

---

## Summary

This operations playbook covers:

✅ **Daily Operations:** Health checks, monitoring, maintenance
✅ **Deployment:** Standard and emergency procedures
✅ **Scaling:** Horizontal and vertical scaling
✅ **Backup/Recovery:** Automated backups and DR procedures
✅ **Monitoring:** Alerts and on-call procedures
✅ **Maintenance:** Planned maintenance windows
✅ **Incidents:** Response plans and post-mortems

**Remember:** Operations is about consistency, documentation, and continuous improvement!
