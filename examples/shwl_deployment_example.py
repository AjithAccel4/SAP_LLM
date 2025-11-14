"""
Example: Self-Healing Workflow Loop with Deployment

Demonstrates:
1. Loading healing rules from configuration
2. Deploying rules with progressive canary rollout
3. Monitoring deployment status
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sap_llm.shwl import (
    ConfigurationLoader,
    DeploymentManager,
)


def main():
    """Run SHWL deployment example."""
    print("=" * 80)
    print("Self-Healing Workflow Loop - Deployment Example")
    print("=" * 80)

    # Step 1: Initialize configuration loader
    print("\n[1] Initializing configuration loader...")
    config_loader = ConfigurationLoader()

    # Step 2: Load healing rules
    print("\n[2] Loading healing rules from configuration...")
    rules = config_loader.load_healing_rules()
    print(f"    Loaded {len(rules)} healing rules:")
    for rule in rules[:3]:  # Show first 3 rules
        print(f"    - {rule['rule_id']}: {rule['name']}")
    if len(rules) > 3:
        print(f"    ... and {len(rules) - 3} more rules")

    # Step 3: Load deployment configuration
    print("\n[3] Loading deployment configuration...")
    deployment_config = config_loader.load_deployment_config()
    print(f"    Strategy: {deployment_config['deployment']['strategy']}")
    print(f"    Namespace: {deployment_config['deployment']['namespace']}")
    print(f"    Canary stages: {len(deployment_config['deployment']['canary_stages'])}")

    # Step 4: Initialize deployment manager (dry run mode)
    print("\n[4] Initializing deployment manager (dry run mode)...")
    deployment_manager = DeploymentManager(
        deployment_config=deployment_config,
        dry_run=True,  # Dry run mode - no actual deployment
    )

    # Step 5: Simulate deployment
    print("\n[5] Simulating progressive canary deployment...")
    print("    Note: This is a dry run - no actual changes will be made")

    deployment_result = deployment_manager.deploy_healing_rules(
        rules=rules,
        proposal_id="example-001",
    )

    # Step 6: Display results
    print("\n[6] Deployment Results:")
    print(f"    Success: {deployment_result['success']}")
    print(f"    Deployment ID: {deployment_result['deployment_id']}")
    print(f"    Status: {deployment_result['status']}")
    print(f"    Message: {deployment_result['message']}")

    # Step 7: Display metrics
    print("\n[7] Deployment Metrics:")
    metrics = deployment_manager.get_metrics()
    print(f"    Total deployments: {metrics['deployments_total']}")
    print(f"    Successful: {metrics['deployments_successful']}")
    print(f"    Failed: {metrics['deployments_failed']}")
    print(f"    Rollbacks: {metrics['rollbacks_total']}")

    # Step 8: Show canary stages
    print("\n[8] Canary Deployment Stages:")
    for stage in deployment_config['deployment']['canary_stages']:
        print(f"    - {stage['name']}: {stage['percentage']}% "
              f"(monitor for {stage['duration_minutes']} minutes)")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
