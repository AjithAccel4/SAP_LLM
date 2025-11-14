# Self-Healing Workflow Loop - Deployment Guide

## Overview

This guide documents the implementation of the Self-Healing Workflow Loop (SHWL) deployment mechanism with progressive canary rollouts, health checks, and automatic rollback capabilities.

## Features Implemented

### 1. Configuration Loader (`sap_llm/shwl/config_loader.py`)

The `ConfigurationLoader` class provides:

- **Load healing rules from JSON configuration files**
  - Validates rule structure and metadata
  - Filters enabled rules
  - Supports multiple configuration files

- **Load deployment configuration**
  - Canary stage definitions
  - Rollback policies
  - Health check settings
  - Monitoring configuration

- **Save and manage rules**
  - Add new rules
  - Update existing rules
  - Delete rules
  - Validate rule format

- **Rule validation**
  - Required fields checking
  - Confidence score validation (0-1 range)
  - Risk level validation (low/medium/high)

### 2. Deployment Manager (`sap_llm/shwl/deployment_manager.py`)

The `DeploymentManager` class implements:

- **Progressive Canary Deployment**
  - Stage 1: 5% rollout (15 minutes monitoring)
  - Stage 2: 25% rollout (30 minutes monitoring)
  - Stage 3: 50% rollout (60 minutes monitoring)
  - Stage 4: 100% complete rollout

- **Health Checks**
  - Periodic health monitoring during each stage
  - Configurable check intervals
  - Success criteria validation (error rate, success rate, response time)
  - Automatic failure detection

- **Automatic Rollback**
  - Triggered on health check failures
  - Restores previous configuration from backup
  - Configurable retry attempts
  - Rollback delay for safety

- **Kubernetes Integration**
  - ConfigMap creation and updates
  - Namespace management
  - Label management for tracking
  - In-cluster and out-of-cluster support

- **Metrics Tracking**
  - Total deployments
  - Successful/failed deployments
  - Rollback count
  - Current canary stage
  - Last deployment timestamp

- **Dry Run Mode**
  - Test deployments without actual changes
  - Simulates all deployment steps
  - Safe for development and testing

### 3. Updated Healing Loop (`sap_llm/shwl/healing_loop.py`)

**Fixed TODO #1 (Line 193): Load Rules from Configuration**
- Integrated `ConfigurationLoader` to load existing healing rules
- Rules are loaded from `config/shwl/healing_rules.json`
- Loaded rules are passed to the rule generator for conflict detection

**Fixed TODO #2 (Line 280): Implement Deployment Mechanism**
- Integrated `DeploymentManager` for progressive deployments
- Converts proposals to healing rule format
- Saves rules to configuration files
- Deploys to Kubernetes ConfigMaps with canary rollout
- Monitors deployment status and logs metrics
- Handles deployment failures with rollback

## Configuration Files

### Healing Rules Configuration (`config/shwl/healing_rules.json`)

```json
{
  "rules": [
    {
      "rule_id": "RULE-001",
      "name": "Invoice date format validation",
      "description": "Validates invoice date format and handles common parsing errors",
      "rule_type": "validation",
      "category": "invoice_processing",
      "condition": {
        "exception_pattern": "ValueError: time data .* does not match format",
        "field": "invoice_date"
      },
      "action": {
        "type": "transform",
        "transformation": "normalize_date_format",
        "fallback_formats": ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d.%m.%Y"]
      },
      "metadata": {
        "created_at": "2024-01-15T10:00:00Z",
        "created_by": "system",
        "version": "1.0",
        "confidence": 0.95,
        "risk_level": "low"
      }
    }
  ],
  "configuration": {
    "enabled": true,
    "auto_apply": false,
    "require_human_approval": true,
    "max_rules": 100
  }
}
```

### Deployment Configuration (`config/shwl/deployment_config.json`)

```json
{
  "deployment": {
    "strategy": "canary",
    "namespace": "sap-llm",
    "configmap_name": "sap-llm-healing-rules",
    "canary_stages": [
      {
        "name": "initial",
        "percentage": 5,
        "duration_minutes": 15,
        "health_check_interval_seconds": 30,
        "success_criteria": {
          "error_rate_threshold": 0.01,
          "min_success_rate": 0.99,
          "max_response_time_p95_ms": 500
        }
      }
    ],
    "rollback": {
      "enabled": true,
      "automatic": true,
      "trigger_on_failure": true,
      "max_rollback_attempts": 3
    },
    "health_checks": {
      "enabled": true,
      "endpoint": "/health",
      "timeout_seconds": 10
    },
    "monitoring": {
      "enabled": true,
      "prometheus_enabled": true,
      "metrics": ["deployment_status", "canary_stage", "error_rate"]
    }
  }
}
```

## Usage

### Basic Usage

```python
from sap_llm.shwl import (
    ConfigurationLoader,
    DeploymentManager,
    SelfHealingWorkflowLoop,
)

# Initialize components
config_loader = ConfigurationLoader()
deployment_config = config_loader.load_deployment_config()
deployment_manager = DeploymentManager(
    deployment_config=deployment_config,
    dry_run=False,  # Set to True for testing
    in_cluster=True,  # True if running in Kubernetes
)

# Load healing rules
rules = config_loader.load_healing_rules()

# Deploy rules with canary rollout
result = deployment_manager.deploy_healing_rules(
    rules=rules,
    proposal_id="proposal-001",
)

if result["success"]:
    print(f"Deployment successful: {result['deployment_id']}")
else:
    print(f"Deployment failed: {result['error']}")
```

### With Healing Loop

```python
# Initialize SHWL with configuration
shwl = SelfHealingWorkflowLoop(
    pmg=pmg_client,
    reasoning_engine=reasoning_engine,
    config=config,
)

# Run healing cycle (includes deployment)
results = shwl.run_healing_cycle()

print(f"Deployed {results['fixes_deployed']} fixes")
```

## Deployment Flow

1. **Validation**
   - Validate rule structure
   - Check for conflicts
   - Perform dry run if enabled
   - Create backup of current configuration

2. **Initial Canary (5%)**
   - Deploy to 5% of instances
   - Monitor for 15 minutes
   - Check health metrics every 30 seconds
   - Proceed if healthy, rollback if not

3. **Expand Canary (25%)**
   - Deploy to 25% of instances
   - Monitor for 30 minutes
   - Check health metrics every 60 seconds
   - Proceed if healthy, rollback if not

4. **Majority Canary (50%)**
   - Deploy to 50% of instances
   - Monitor for 60 minutes
   - Check health metrics every 60 seconds
   - Proceed if healthy, rollback if not

5. **Complete Rollout (100%)**
   - Deploy to all instances
   - Mark deployment as completed
   - Update metrics

6. **Rollback (if failure)**
   - Restore previous configuration
   - Update deployment status
   - Log error details
   - Increment rollback counter

## Monitoring

### Deployment Metrics

```python
metrics = deployment_manager.get_metrics()
print(f"Total: {metrics['deployments_total']}")
print(f"Successful: {metrics['deployments_successful']}")
print(f"Failed: {metrics['deployments_failed']}")
print(f"Rollbacks: {metrics['rollbacks_total']}")
```

### Deployment Status

```python
status = deployment_manager.get_deployment_status(deployment_id)
print(f"Status: {status['status']}")
print(f"Canary Stage: {status['canary_stage']}")
print(f"Started: {status['started_at']}")
```

## Kubernetes Integration

### ConfigMap Structure

The deployment manager creates/updates a ConfigMap:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sap-llm-healing-rules
  namespace: sap-llm
  labels:
    app: sap-llm
    component: shwl
    managed-by: deployment-manager
data:
  healing_rules.json: |
    {
      "rules": [...],
      "deployment": {
        "canary_stage": "initial",
        "percentage": 5,
        "timestamp": "2024-11-14T..."
      }
    }
```

### Required Permissions

The deployment manager requires the following Kubernetes RBAC permissions:

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: shwl-deployment-manager
  namespace: sap-llm
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
```

## Safety Features

1. **Dry Run Mode**: Test deployments without making changes
2. **Backup Before Deploy**: Automatic backup of current configuration
3. **Progressive Rollout**: Gradual deployment with validation at each stage
4. **Health Checks**: Continuous monitoring during deployment
5. **Automatic Rollback**: Immediate rollback on failure detection
6. **Maximum Retry Limit**: Prevents infinite rollback loops
7. **Deployment History**: Track all deployment attempts
8. **Comprehensive Logging**: Detailed logs for debugging

## Best Practices

1. **Always test in dry run mode first**
   ```python
   deployment_manager = DeploymentManager(
       deployment_config=deployment_config,
       dry_run=True,
   )
   ```

2. **Configure appropriate health check intervals**
   - Balance between quick detection and system load
   - Recommended: 30-60 seconds

3. **Set realistic success criteria**
   - Don't set thresholds too strict
   - Account for normal variance

4. **Monitor deployment metrics**
   - Track success/failure rates
   - Analyze rollback patterns
   - Adjust configuration based on results

5. **Use meaningful proposal IDs**
   - Include date/time for tracking
   - Reference issue/ticket numbers

6. **Review deployment history**
   - Learn from failed deployments
   - Identify patterns in rollbacks

## Troubleshooting

### Deployment Failures

**Problem**: All deployments fail immediately
- Check Kubernetes connectivity
- Verify RBAC permissions
- Review ConfigMap name and namespace

**Problem**: Health checks always fail
- Adjust success criteria thresholds
- Increase health check intervals
- Verify monitoring endpoints

**Problem**: Rollbacks fail
- Check backup creation
- Verify ConfigMap update permissions
- Review rollback logs

### Performance Issues

**Problem**: Deployments take too long
- Reduce monitoring duration for stages
- Increase health check intervals
- Consider fewer canary stages

**Problem**: High resource usage
- Enable dry run for testing
- Batch rule updates
- Optimize health check queries

## Dependencies

- `kubernetes>=28.1.0` - Kubernetes client library

## Example Scripts

See `/home/user/SAP_LLM/examples/shwl_deployment_example.py` for a complete example.

## Summary of Changes

### Files Created

1. `/home/user/SAP_LLM/sap_llm/shwl/config_loader.py` (370 lines)
   - Configuration loader for healing rules and deployment settings

2. `/home/user/SAP_LLM/sap_llm/shwl/deployment_manager.py` (750 lines)
   - Progressive canary deployment with health checks and rollback

3. `/home/user/SAP_LLM/config/shwl/healing_rules.json` (130 lines)
   - Sample healing rules configuration with 5 example rules

4. `/home/user/SAP_LLM/config/shwl/deployment_config.json` (80 lines)
   - Deployment configuration with canary stages and policies

5. `/home/user/SAP_LLM/examples/shwl_deployment_example.py` (105 lines)
   - Example script demonstrating deployment functionality

6. `/home/user/SAP_LLM/docs/SHWL_DEPLOYMENT_GUIDE.md` (This file)
   - Comprehensive documentation

### Files Modified

1. `/home/user/SAP_LLM/sap_llm/shwl/healing_loop.py`
   - Added imports for ConfigurationLoader and DeploymentManager
   - Initialized config_loader and deployment_manager in __init__
   - Fixed TODO #1: Load rules from configuration (line 215-217)
   - Fixed TODO #2: Implement deployment mechanism (line 284-407)
   - Added _proposal_to_rule helper method

2. `/home/user/SAP_LLM/sap_llm/shwl/__init__.py`
   - Added ConfigurationLoader and DeploymentManager to exports

3. `/home/user/SAP_LLM/requirements.txt`
   - Added kubernetes>=28.1.0 dependency

## Next Steps

1. **Production Deployment**
   - Deploy to Kubernetes cluster
   - Configure RBAC permissions
   - Set up monitoring dashboards

2. **Integration**
   - Connect health checks to actual metrics
   - Integrate with Prometheus
   - Set up alerting

3. **Testing**
   - Add unit tests for ConfigurationLoader
   - Add unit tests for DeploymentManager
   - Add integration tests for full deployment flow

4. **Enhancements**
   - Add support for blue-green deployments
   - Implement A/B testing capabilities
   - Add canary analysis automation
