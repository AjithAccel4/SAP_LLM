# Disaster Recovery Plan - SAP_LLM

## Table of Contents

- [Overview](#overview)
- [Recovery Objectives](#recovery-objectives)
- [Component Inventory](#component-inventory)
- [Backup Strategies](#backup-strategies)
- [Recovery Procedures](#recovery-procedures)
- [Failover Procedures](#failover-procedures)
- [Testing Schedule](#testing-schedule)
- [Roles and Responsibilities](#roles-and-responsibilities)
- [Contact Information](#contact-information)

## Overview

This document outlines the disaster recovery (DR) and business continuity plan for the SAP_LLM system. The plan ensures rapid recovery of critical services in the event of system failures, data loss, or catastrophic events.

### Document Version

- **Version**: 1.0
- **Last Updated**: 2025-11-14
- **Next Review Date**: 2026-02-14
- **Owner**: Infrastructure Team

### Scope

This DR plan covers:
- Model files and checkpoints
- Databases (MongoDB, Redis, Cosmos DB)
- Configuration files and secrets
- Application code and deployments
- Monitoring and logging infrastructure

## Recovery Objectives

### RTO and RPO Definitions

#### Recovery Time Objective (RTO)

The maximum acceptable downtime for each system component:

| Component | RTO | Criticality | Notes |
|-----------|-----|-------------|-------|
| **Core API Service** | 1 hour | Critical | Revenue-impacting |
| **Model Inference** | 1 hour | Critical | Core functionality |
| **MongoDB (Knowledge Base)** | 2 hours | High | Contains business rules |
| **Redis (Cache)** | 30 minutes | Medium | Performance impact only |
| **Cosmos DB (PMG)** | 4 hours | High | Historical data, learning |
| **SHWL Service** | 4 hours | Medium | Continuous improvement |
| **Monitoring (Prometheus/Grafana)** | 8 hours | Low | Observability |

#### Recovery Point Objective (RPO)

The maximum acceptable data loss for each system component:

| Component | RPO | Backup Frequency | Notes |
|-----------|-----|------------------|-------|
| **Model Checkpoints** | 24 hours | Daily | Large files, incremental |
| **MongoDB (Knowledge Base)** | 1 hour | Hourly snapshots | Critical business data |
| **Redis (Cache)** | N/A | None | Ephemeral cache data |
| **Cosmos DB (PMG)** | 15 minutes | Continuous replication | Transaction history |
| **Configuration Files** | 1 hour | On change + hourly | Version controlled |
| **Application Logs** | 5 minutes | Continuous streaming | ELK/Loki aggregation |

### Service Level Agreements (SLAs)

- **Availability Target**: 99.9% uptime (8.76 hours downtime/year)
- **Data Durability**: 99.999999999% (11 nines)
- **Mean Time To Recovery (MTTR)**: < 2 hours for critical services

## Component Inventory

### Critical Assets

#### 1. Model Files

**Location**: Azure Blob Storage (Primary), Local PVCs (Secondary)

```
/models/
├── vision_encoder/          # LayoutLMv3 (300M params, ~1.2GB)
├── language_decoder/        # LLaMA-2-7B (7B params, ~13GB)
├── reasoning_engine/        # Mixtral-8x7B (47B params, ~94GB)
├── sap_llm_unified/        # Unified model (~108GB)
└── checkpoints/            # Training checkpoints (variable size)
    ├── step_500/
    ├── step_1000/
    └── best_model/
```

**Total Storage**: ~220GB (compressed), ~300GB (uncompressed)

#### 2. Databases

##### MongoDB (Knowledge Base)

- **Database**: `sap_llm_kb`
- **Collections**:
  - `documents` - Processed document metadata
  - `results` - Extraction results
  - `exceptions` - Exception history
  - `field_mappings` - SAP field mappings
  - `business_rules` - Business validation rules
  - `transformations` - Data transformation functions
- **Estimated Size**: 50GB - 500GB (production)
- **Backup Location**: Azure Blob Storage `sap-llm-backups/mongodb/`

##### Redis (Cache)

- **Database**: 0
- **Type**: In-memory cache
- **Data**: Ephemeral (no backup required)
- **Recovery Strategy**: Warm-up from cold start

##### Cosmos DB (Process Memory Graph)

- **Database**: `sap_llm_pmg`
- **Container**: `transactions`
- **API**: Gremlin (Graph)
- **Backup**: Azure native point-in-time restore
- **Retention**: 30 days continuous backup

#### 3. Configuration Files

**Location**: Git repository (Primary), Azure Key Vault (Secrets)

```
/configs/
├── default_config.yaml      # Main configuration
├── production_config.yaml   # Production overrides
├── staging_config.yaml      # Staging environment
└── secrets/
    ├── cosmos_credentials   # Cosmos DB connection
    ├── api_keys            # Third-party API keys
    └── certificates/       # SSL/TLS certificates
```

#### 4. Application Code

- **Repository**: Git (Version controlled)
- **Container Images**: Azure Container Registry
- **Deployment Manifests**: Kubernetes YAML files

## Backup Strategies

### Automated Backup Schedule

```
┌─────────────┬──────────────┬───────────────┬──────────────┐
│  Component  │  Frequency   │   Retention   │   Location   │
├─────────────┼──────────────┼───────────────┼──────────────┤
│  Models     │  Daily       │  90 days      │  Azure Blob  │
│  MongoDB    │  Hourly      │  30 days      │  Azure Blob  │
│  Cosmos DB  │  Continuous  │  30 days      │  Azure Native│
│  Configs    │  On change   │  Indefinite   │  Git + KV    │
│  Logs       │  Streaming   │  90 days      │  Log Analytics│
└─────────────┴──────────────┴───────────────┴──────────────┘
```

### Backup Implementation

#### 1. Model File Backups

**Strategy**: Incremental backups with compression

```bash
# Automated via scripts/backup.sh
# Daily at 2:00 AM UTC
# Uploads to Azure Blob Storage with versioning
# Verifies checksums post-upload
```

**Storage Class**: Azure Blob Cool tier (90-day lifecycle to Archive)

**Compression**: gzip (70% reduction for model files)

**Verification**: SHA-256 checksums

#### 2. MongoDB Backups

**Strategy**: Hourly snapshots using mongodump

```bash
# Hourly snapshots
# Compressed BSON format
# Point-in-time recovery capability
# Incremental backups for large collections
```

**Backup Types**:
- **Full Backup**: Daily at 1:00 AM UTC
- **Incremental**: Hourly (changed documents only)
- **Oplog Backup**: Continuous replication

**Recovery Options**:
- Full restore from snapshot
- Point-in-time recovery using oplog
- Collection-level restore

#### 3. Cosmos DB Backups

**Strategy**: Azure native continuous backup

```yaml
# Automatic configuration
backup_policy:
  mode: Continuous
  tier: Continuous7Days  # or Continuous30Days
  retention_hours: 720   # 30 days
```

**Features**:
- Point-in-time restore to any second within retention
- No performance impact
- Geo-redundant storage
- Self-service restore via Azure Portal

#### 4. Configuration Backups

**Strategy**: Git version control + Azure Key Vault

**Git Repository**:
```bash
# All configuration files version controlled
# Encrypted secrets using git-crypt
# Protected branches (main, production)
# Automated commits for generated configs
```

**Azure Key Vault**:
```bash
# Secrets rotation policy: 90 days
# Soft-delete enabled: 90 days
# Purge protection: Enabled
# Access policies: Least privilege
```

#### 5. Redis Backup

**Strategy**: RDB snapshots (optional, for warm-up)

```redis
# Optional daily snapshot
save 900 1      # 15 min if 1 key changed
save 300 10     # 5 min if 10 keys changed
save 60 10000   # 1 min if 10000 keys changed
```

**Note**: Redis is ephemeral cache - full rebuild acceptable on restore

### Backup Storage Architecture

```
Azure Storage Account: sapllmprodbackups
│
├── Container: models
│   ├── daily/
│   │   ├── 2025-11-14/
│   │   │   ├── vision_encoder.tar.gz
│   │   │   ├── language_decoder.tar.gz
│   │   │   ├── reasoning_engine.tar.gz
│   │   │   └── checksums.sha256
│   │   └── ...
│   └── checkpoints/
│       ├── step_500.tar.gz
│       └── ...
│
├── Container: mongodb
│   ├── full/
│   │   ├── 2025-11-14_01-00.archive.gz
│   │   └── ...
│   ├── incremental/
│   │   ├── 2025-11-14_02-00.archive.gz
│   │   └── ...
│   └── oplog/
│       └── continuous/
│
├── Container: configs
│   ├── snapshots/
│   │   ├── 2025-11-14_config.tar.gz
│   │   └── ...
│   └── secrets/ (encrypted)
│
└── Container: logs (archived)
    ├── 2025-11/
    └── ...
```

### Backup Verification

**Automated Checks** (Post-backup):

1. **Integrity Checks**
   ```bash
   # Verify file checksums
   sha256sum -c checksums.sha256

   # Verify archive integrity
   gzip -t backup.tar.gz

   # Verify MongoDB backup
   mongorestore --dryRun
   ```

2. **Restore Testing**
   ```bash
   # Monthly automated restore to test environment
   # Verify data integrity
   # Performance benchmarks
   ```

3. **Monitoring**
   ```bash
   # Backup success/failure metrics
   # Storage utilization alerts
   # Backup age monitoring
   ```

## Recovery Procedures

### Failure Scenarios and Recovery Paths

```
Failure Type → Detection → Assessment → Recovery → Validation
     ↓             ↓            ↓           ↓           ↓
  [Event]      [Monitor]    [Impact]    [Restore]   [Test]
```

### Scenario 1: API Service Failure

**Symptoms**:
- Health check failures
- HTTP 5xx errors
- Pod crashes in Kubernetes

**Detection Time**: < 1 minute (health checks every 30s)

**Recovery Procedure**:

```bash
# 1. Assess the failure
kubectl get pods -n sap-llm
kubectl describe pod <pod-name> -n sap-llm
kubectl logs <pod-name> -n sap-llm --tail=100

# 2. Attempt automatic recovery (HPA/ReplicaSet)
# Kubernetes will automatically restart failed pods

# 3. If automatic recovery fails, manual intervention
kubectl rollout restart deployment/sap-llm-api -n sap-llm

# 4. If rollout fails, rollback to previous version
kubectl rollout undo deployment/sap-llm-api -n sap-llm

# 5. Verify recovery
kubectl rollout status deployment/sap-llm-api -n sap-llm
curl http://<service-ip>:8000/health

# 6. Check logs for root cause
kubectl logs deployment/sap-llm-api -n sap-llm --tail=500
```

**Expected RTO**: 5-15 minutes

### Scenario 2: Model File Corruption/Loss

**Symptoms**:
- Model loading errors
- Inference failures
- Checksum validation failures

**Detection Time**: Immediate (on model load)

**Recovery Procedure**:

```bash
# 1. Identify corrupted models
cd /home/user/SAP_LLM
./scripts/backup.sh --verify-local-models

# 2. Download backup from Azure Blob Storage
az login
export BACKUP_DATE="2025-11-14"  # Latest known good backup

# 3. Execute restore script
./scripts/restore.sh --component models --date $BACKUP_DATE

# 4. Verify model integrity
python -c "
from sap_llm.models import load_vision_encoder, load_language_decoder
encoder = load_vision_encoder()
decoder = load_language_decoder()
print('Models loaded successfully')
"

# 5. Restart affected services
kubectl rollout restart deployment/sap-llm-api -n sap-llm

# 6. Warm-up test
curl -X POST http://<service-ip>:8000/api/v1/process \
  -H "Content-Type: application/json" \
  -d '{"document_id": "test-001", "mode": "warm-up"}'
```

**Expected RTO**: 30-60 minutes (depending on download speed)

### Scenario 3: MongoDB Database Corruption

**Symptoms**:
- Database connection errors
- Data inconsistencies
- Replication lag

**Detection Time**: < 5 minutes (health checks)

**Recovery Procedure**:

```bash
# 1. Assess the damage
mongosh --eval "db.serverStatus()" mongodb://localhost:27017

# 2. Determine recovery point
# Option A: Latest hourly snapshot (RPO: 1 hour)
# Option B: Point-in-time using oplog (RPO: minutes)

# 3. Stop application writes
kubectl scale deployment/sap-llm-api --replicas=0 -n sap-llm

# 4. Backup current (potentially corrupted) data
mongodump --out=/backup/corrupted/$(date +%Y%m%d_%H%M%S)

# 5. Restore from backup
export RESTORE_DATE="2025-11-14_01-00"
./scripts/restore.sh --component mongodb --date $RESTORE_DATE

# Alternative: Point-in-time restore
./scripts/restore.sh --component mongodb --point-in-time "2025-11-14T14:30:00Z"

# 6. Verify data integrity
python scripts/verify_mongodb_integrity.py

# 7. Restart application
kubectl scale deployment/sap-llm-api --replicas=3 -n sap-llm

# 8. Monitor for issues
kubectl logs deployment/sap-llm-api -n sap-llm -f
```

**Expected RTO**: 1-2 hours (depending on database size)

### Scenario 4: Cosmos DB Failure

**Symptoms**:
- Graph query failures
- PMG lookup errors
- Connection timeouts

**Detection Time**: < 1 minute (health checks)

**Recovery Procedure**:

```bash
# Cosmos DB has built-in geo-redundancy and automatic failover

# 1. Check Cosmos DB status
az cosmosdb show --name <account-name> --resource-group <rg-name>

# 2. If automatic failover hasn't occurred, initiate manual failover
az cosmosdb failover-priority-change \
  --name <account-name> \
  --resource-group <rg-name> \
  --failover-policies <secondary-region>=0 <primary-region>=1

# 3. If data corruption, perform point-in-time restore
az cosmosdb restore \
  --account-name <account-name> \
  --resource-group <rg-name> \
  --restore-timestamp "2025-11-14T14:30:00Z" \
  --location <region>

# 4. Update application connection strings if needed
kubectl edit secret cosmos-credentials -n sap-llm

# 5. Restart affected services
kubectl rollout restart deployment/sap-llm-api -n sap-llm
kubectl rollout restart deployment/sap-llm-shwl -n sap-llm

# 6. Verify PMG functionality
python -c "
from sap_llm.pmg import ProcessMemoryGraph
pmg = ProcessMemoryGraph()
print(pmg.health_check())
"
```

**Expected RTO**: 15 minutes (automatic failover) to 4 hours (restore)

### Scenario 5: Redis Cache Failure

**Symptoms**:
- Cache miss rate spike
- Increased latency
- Connection refused errors

**Detection Time**: < 1 minute (health checks)

**Recovery Procedure**:

```bash
# 1. Check Redis status
kubectl get pods -n sap-llm | grep redis
kubectl logs redis-<pod-id> -n sap-llm --tail=100

# 2. Restart Redis pod
kubectl delete pod redis-<pod-id> -n sap-llm
# Kubernetes will automatically create a new pod

# 3. Verify Redis is running
kubectl exec -it redis-<new-pod-id> -n sap-llm -- redis-cli ping
# Expected output: PONG

# 4. Optional: Load warm-up data
./scripts/restore.sh --component redis --warm-up

# 5. Monitor cache hit rate recovery
# Cache will rebuild naturally from application requests
# Expected: Normal hit rate within 30-60 minutes
```

**Expected RTO**: 5-10 minutes (Redis restart)
**Cache Warm-up**: 30-60 minutes (automatic)

### Scenario 6: Complete Data Center Failure

**Symptoms**:
- All services unreachable
- Network connectivity loss
- Regional Azure outage

**Detection Time**: < 5 minutes (multi-region health checks)

**Recovery Procedure**:

```bash
# This is a catastrophic failure requiring DR site activation

# 1. Activate DR runbook
# Incident Commander declares disaster
# Activate DR team

# 2. Validate DR site readiness
az account set --subscription <dr-subscription>
kubectl config use-context <dr-cluster>

# 3. Deploy infrastructure to DR region
cd /home/user/SAP_LLM/deployments/terraform
terraform init
terraform workspace select dr-west-us
terraform apply -auto-approve

# 4. Restore data components

# 4a. Restore models from geo-redundant storage
export AZURE_STORAGE_ACCOUNT="sapllmdrbackups"
./scripts/restore.sh --component models --location westus

# 4b. Restore MongoDB from latest backup
./scripts/restore.sh --component mongodb --location westus

# 4c. Activate Cosmos DB in DR region
az cosmosdb failover-priority-change \
  --name <cosmos-account> \
  --resource-group <rg> \
  --failover-policies westus=0 eastus=1

# 5. Deploy application
helm upgrade --install sap-llm ./helm/sap-llm \
  --namespace sap-llm \
  --create-namespace \
  --values ./helm/sap-llm/values-dr.yaml

# 6. Update DNS/Traffic Manager
az network traffic-manager endpoint update \
  --name primary-endpoint \
  --profile-name sap-llm-tm \
  --resource-group sap-llm-rg \
  --type azureEndpoints \
  --target-resource-id <dr-lb-id> \
  --endpoint-status Enabled

# 7. Validate all services
./scripts/health_check.py --comprehensive --environment dr

# 8. Monitor and communicate
# Update status page
# Notify stakeholders
# Monitor for issues
```

**Expected RTO**: 4-8 hours (full DR activation)
**Expected RPO**: 1-24 hours (depending on component)

### Scenario 7: Configuration Corruption

**Symptoms**:
- Service misconfiguration
- Authentication failures
- Invalid settings

**Detection Time**: Immediate (on deployment)

**Recovery Procedure**:

```bash
# 1. Identify the issue
kubectl get configmap sap-llm-config -n sap-llm -o yaml
kubectl get secret sap-llm-secrets -n sap-llm -o yaml

# 2. Restore from Git
cd /home/user/SAP_LLM
git log --oneline configs/  # Find last known good commit
git checkout <commit-hash> -- configs/

# 3. Restore secrets from Azure Key Vault
az keyvault secret show --vault-name sap-llm-kv --name cosmos-key
az keyvault secret show --vault-name sap-llm-kv --name api-secret

# 4. Re-create Kubernetes configmap/secrets
kubectl delete configmap sap-llm-config -n sap-llm
kubectl create configmap sap-llm-config \
  --from-file=configs/ \
  -n sap-llm

kubectl delete secret sap-llm-secrets -n sap-llm
kubectl create secret generic sap-llm-secrets \
  --from-literal=cosmos-key=$(az keyvault secret show --vault-name sap-llm-kv --name cosmos-key --query value -o tsv) \
  --from-literal=api-secret=$(az keyvault secret show --vault-name sap-llm-kv --name api-secret --query value -o tsv) \
  -n sap-llm

# 5. Restart affected pods
kubectl rollout restart deployment/sap-llm-api -n sap-llm

# 6. Verify configuration
kubectl exec -it <pod-name> -n sap-llm -- env | grep -E 'COSMOS|API'
```

**Expected RTO**: 15-30 minutes

## Failover Procedures

### High Availability Architecture

```
Primary Region (East US)              DR Region (West US)
┌─────────────────────┐              ┌─────────────────────┐
│   AKS Cluster       │              │   AKS Cluster       │
│   ├── API (3 pods)  │◄────sync────►│   ├── API (warm)    │
│   ├── SHWL (2 pods) │              │   ├── SHWL (warm)   │
│   └── Workers       │              │   └── Workers       │
└─────────────────────┘              └─────────────────────┘
         ↓                                    ↓
┌─────────────────────┐              ┌─────────────────────┐
│  Storage Accounts   │              │  Storage Accounts   │
│  (GRS - Geo-Sync)   │◄────sync────►│  (Read-only)        │
└─────────────────────┘              └─────────────────────┘
         ↓                                    ↓
┌─────────────────────┐              ┌─────────────────────┐
│  Cosmos DB          │              │  Cosmos DB          │
│  (Multi-region)     │◄────sync────►│  (Replica)          │
└─────────────────────┘              └─────────────────────┘
         ↓                                    ↓
         └──────────────┐      ┌─────────────┘
                        ↓      ↓
                 ┌─────────────────┐
                 │ Traffic Manager │
                 │  (DNS failover) │
                 └─────────────────┘
```

### Automatic Failover Triggers

**Application-level** (Kubernetes):
```yaml
# Liveness probe failures → Pod restart
# Readiness probe failures → Remove from load balancer
# HPA metrics → Scale up/down
# Node failures → Pod rescheduling
```

**Database-level**:
```yaml
# Cosmos DB: Automatic regional failover
# MongoDB: Replica set automatic failover
# Redis: Sentinel-based failover (if configured)
```

**Infrastructure-level**:
```yaml
# Azure Traffic Manager: DNS-based failover
# Azure Load Balancer: Health probe-based routing
# Availability Zones: Cross-zone distribution
```

### Manual Failover Checklist

**Pre-Failover**:
- [ ] Assess severity and impact
- [ ] Notify stakeholders
- [ ] Verify DR site readiness
- [ ] Backup current state (if possible)
- [ ] Document decision and timestamp

**Failover Execution**:
- [ ] Update Traffic Manager to route to DR
- [ ] Activate DR region resources
- [ ] Restore data to DR (if needed)
- [ ] Validate all services operational
- [ ] Update monitoring dashboards

**Post-Failover**:
- [ ] Monitor DR site performance
- [ ] Document issues and resolutions
- [ ] Plan failback strategy
- [ ] Update runbooks based on lessons learned

### Failback Procedures

After primary region is restored:

```bash
# 1. Verify primary region health
./scripts/health_check.py --environment primary --comprehensive

# 2. Sync data from DR to primary
# MongoDB: Restore from DR backup
./scripts/restore.sh --source dr --target primary --component mongodb

# Cosmos DB: Data already synced (multi-region)

# Models: Already in geo-redundant storage

# 3. Deploy to primary region
kubectl config use-context primary-cluster
helm upgrade --install sap-llm ./helm/sap-llm \
  --namespace sap-llm \
  --values ./helm/sap-llm/values-primary.yaml

# 4. Validate primary region
./scripts/health_check.py --environment primary

# 5. Switch traffic back to primary (gradual)
# Start with 10% traffic to primary
az network traffic-manager endpoint update \
  --name primary-endpoint \
  --weight 10 \
  --profile-name sap-llm-tm

# Monitor for 30 minutes, increase gradually
# 10% → 25% → 50% → 75% → 100%

# 6. Full cutover to primary
az network traffic-manager endpoint update \
  --name primary-endpoint \
  --weight 100 \
  --profile-name sap-llm-tm

az network traffic-manager endpoint update \
  --name dr-endpoint \
  --weight 0 \
  --profile-name sap-llm-tm

# 7. Scale down DR region (keep warm for quick failover)
kubectl scale deployment/sap-llm-api --replicas=1 -n sap-llm \
  --context dr-cluster
```

## Testing Schedule

### DR Test Types

#### 1. Component-Level Testing

**Frequency**: Monthly

**Scope**: Individual component restore

```bash
# Test model restore
./scripts/restore.sh --component models --dry-run
./scripts/restore.sh --component models --environment test

# Test MongoDB restore
./scripts/restore.sh --component mongodb --environment test

# Verify restored data integrity
python scripts/verify_restore.py --component all
```

**Success Criteria**:
- ✅ Restore completes within RTO
- ✅ Data integrity verified (checksums match)
- ✅ Services start successfully with restored data
- ✅ All automated tests pass

#### 2. Service-Level Testing

**Frequency**: Quarterly

**Scope**: Full service recovery in test environment

```bash
# Simulate service failure
kubectl delete deployment sap-llm-api -n sap-llm-test

# Execute recovery procedures
# ... (follow runbook)

# Validate recovery
./scripts/health_check.py --environment test --comprehensive
pytest tests/integration/ --environment test
```

**Success Criteria**:
- ✅ Service recovers within RTO
- ✅ All health checks pass
- ✅ Integration tests pass
- ✅ Performance within acceptable range

#### 3. Full DR Drill

**Frequency**: Semi-annually (every 6 months)

**Scope**: Complete failover to DR region

**Participants**:
- Infrastructure team
- DevOps team
- Application team
- Management (observers)

**Procedure**:
1. **Planning** (Week 1)
   - Schedule drill date/time
   - Notify all stakeholders
   - Prepare test environment
   - Review runbooks

2. **Execution** (Day of drill)
   ```bash
   # T-0: Simulate primary region failure
   # Disable primary region routing

   # T+5min: Detect failure (automatic)
   # Monitoring alerts triggered

   # T+10min: Activate DR runbook
   # Team assembles, roles assigned

   # T+30min: Begin DR activation
   # Execute failover procedures

   # T+4hr: Services operational in DR
   # Validate all components

   # T+6hr: Full testing complete
   # Application tests, load tests
   ```

3. **Failback** (Within 24 hours)
   - Restore primary region
   - Gradual traffic shift
   - Monitor for issues

4. **Post-Mortem** (Within 1 week)
   - Document timeline
   - Identify gaps
   - Update runbooks
   - Share learnings

**Success Criteria**:
- ✅ DR activation within 4 hours
- ✅ All services operational
- ✅ RPO/RTO objectives met
- ✅ No data loss beyond RPO
- ✅ Failback successful

### Testing Documentation

After each test, document:
- **Date and time**
- **Test type and scope**
- **Participants**
- **Success/failure status**
- **RTO/RPO achieved**
- **Issues encountered**
- **Action items**
- **Runbook updates needed**

**Template**:
```markdown
# DR Test Report - YYYY-MM-DD

## Test Details
- Type: [Component/Service/Full DR]
- Date: YYYY-MM-DD HH:MM UTC
- Duration: X hours Y minutes
- Participants: [Names and roles]

## Objectives
- [ ] Objective 1
- [ ] Objective 2

## Timeline
| Time | Event | Owner | Status |
|------|-------|-------|--------|
| T+0  | ... | ... | ✅/❌ |

## Metrics
- **RTO Target**: X hours
- **RTO Achieved**: Y hours
- **RPO Target**: X minutes
- **RPO Achieved**: Y minutes

## Issues Encountered
1. Issue description
   - Impact: ...
   - Resolution: ...
   - Action item: ...

## Lessons Learned
- ...

## Action Items
- [ ] Update runbook section X
- [ ] Automate step Y
- [ ] Train team on Z
```

## Roles and Responsibilities

### Incident Response Team

#### Incident Commander
- **Primary**: DevOps Lead
- **Backup**: Infrastructure Lead
- **Responsibilities**:
  - Declare disaster
  - Coordinate recovery efforts
  - Communicate with stakeholders
  - Make go/no-go decisions

#### Technical Lead
- **Primary**: Senior SRE
- **Backup**: Platform Engineer
- **Responsibilities**:
  - Execute recovery procedures
  - Troubleshoot technical issues
  - Validate system integrity
  - Update technical status

#### Database Administrator
- **Primary**: Database Team Lead
- **Backup**: Senior DBA
- **Responsibilities**:
  - MongoDB restore and validation
  - Cosmos DB failover
  - Data integrity verification
  - Performance tuning post-restore

#### Security Lead
- **Primary**: Security Team Lead
- **Backup**: Security Engineer
- **Responsibilities**:
  - Validate security controls
  - Rotate compromised credentials
  - Audit access during recovery
  - Security incident response

#### Communications Lead
- **Primary**: Product Manager
- **Backup**: Engineering Manager
- **Responsibilities**:
  - Internal communications
  - Customer notifications
  - Status page updates
  - Post-incident report

### RACI Matrix

| Activity | Incident Commander | Technical Lead | DBA | Security | Comms |
|----------|-------------------|----------------|-----|----------|-------|
| Declare disaster | **R** | C | C | I | I |
| Execute runbook | A | **R** | **R** | C | I |
| Data restore | A | C | **R** | C | I |
| Security validation | A | C | C | **R** | I |
| Stakeholder comms | A | I | I | I | **R** |
| Post-mortem | **R** | C | C | C | C |

**Legend**: R=Responsible, A=Accountable, C=Consulted, I=Informed

## Contact Information

### Emergency Contacts

**24/7 On-Call**:
```
Primary On-Call: +1-XXX-XXX-XXXX (PagerDuty)
Backup On-Call:  +1-XXX-XXX-XXXX (PagerDuty)
```

**Escalation Chain**:
1. DevOps Team Lead
2. Infrastructure Manager
3. VP of Engineering
4. CTO

### External Vendors

| Vendor | Service | Contact | SLA |
|--------|---------|---------|-----|
| Microsoft Azure | Cloud Infrastructure | Azure Support Portal | 1 hour response (Severity A) |
| MongoDB | Database Support | support@mongodb.com | 1 hour response (P1) |
| Datadog/Prometheus | Monitoring | support@... | 4 hour response |

### Internal Teams

| Team | Purpose | Contact | Hours |
|------|---------|---------|-------|
| DevOps | Infrastructure | devops@company.com | 24/7 |
| Database | Data management | dba@company.com | 24/7 |
| Security | Security incidents | security@company.com | 24/7 |
| Support | Customer impact | support@company.com | Business hours |

## Appendices

### Appendix A: Backup Verification Commands

```bash
# Verify model backups
az storage blob list \
  --account-name sapllmprodbackups \
  --container-name models \
  --prefix "daily/$(date +%Y-%m-%d)/" \
  --output table

# Verify MongoDB backups
az storage blob list \
  --account-name sapllmprodbackups \
  --container-name mongodb \
  --prefix "full/$(date +%Y-%m-%d)_" \
  --output table

# Verify backup integrity
./scripts/backup.sh --verify-latest
```

### Appendix B: Quick Reference Commands

```bash
# Check overall system health
./scripts/health_check.py --comprehensive

# View recent backups
./scripts/backup.sh --list-recent

# Restore specific component
./scripts/restore.sh --component <name> --date <YYYY-MM-DD>

# Failover to DR
./scripts/failover.sh --region westus --mode manual

# Check Kubernetes cluster health
kubectl get nodes
kubectl get pods --all-namespaces
kubectl top nodes

# Check database status
mongosh --eval "rs.status()"
kubectl exec -it redis-0 -- redis-cli ping
```

### Appendix C: Monitoring Dashboard Links

- **Grafana**: https://grafana.sap-llm.company.com
- **Prometheus**: https://prometheus.sap-llm.company.com
- **Azure Portal**: https://portal.azure.com
- **Status Page**: https://status.sap-llm.company.com
- **Logs**: https://logs.sap-llm.company.com

### Appendix D: Compliance and Audit

**Regulatory Requirements**:
- SOC 2 Type II compliance
- GDPR data residency requirements
- Industry-specific regulations (finance, healthcare, etc.)

**Audit Trail**:
- All DR tests documented
- All production restores logged
- Quarterly DR readiness reviews
- Annual third-party DR audit

**Data Retention**:
- Backup retention: 90 days (models), 30 days (databases)
- Log retention: 90 days (active), 1 year (archived)
- Audit logs: 7 years

---

**Document Control**:
- **Classification**: Internal - Confidential
- **Review Frequency**: Quarterly
- **Next Review**: 2026-02-14
- **Approvers**: CTO, VP Engineering, Infrastructure Lead

**Version History**:

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-14 | DR Team | Initial version |

---
*End of Disaster Recovery Plan*
