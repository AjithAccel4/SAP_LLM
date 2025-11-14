"""
Configuration loader for Self-Healing Workflow Loop.

Loads healing rules and deployment configuration from JSON files.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigurationLoader:
    """Loads and validates healing rules and deployment configuration."""

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files.
                       Defaults to <project_root>/config/shwl
        """
        if config_dir is None:
            # Default to config/shwl in project root
            project_root = Path(__file__).parent.parent.parent
            self.config_dir = project_root / "config" / "shwl"
        else:
            self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            logger.warning(f"Configuration directory not found: {self.config_dir}")
            self.config_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Configuration loader initialized: {self.config_dir}")

    def load_healing_rules(
        self,
        filename: str = "healing_rules.json",
    ) -> List[Dict[str, Any]]:
        """
        Load healing rules from JSON configuration file.

        Args:
            filename: Name of the rules configuration file

        Returns:
            List of healing rule dictionaries
        """
        rules_file = self.config_dir / filename

        if not rules_file.exists():
            logger.warning(f"Rules file not found: {rules_file}")
            return []

        try:
            with open(rules_file, "r") as f:
                data = json.load(f)

            rules = data.get("rules", [])
            config = data.get("configuration", {})

            # Filter enabled rules
            if not config.get("enabled", True):
                logger.warning("Healing rules are disabled in configuration")
                return []

            # Validate rules
            validated_rules = []
            for rule in rules:
                if self._validate_rule(rule):
                    validated_rules.append(rule)
                else:
                    logger.warning(
                        f"Skipping invalid rule: {rule.get('rule_id', 'unknown')}"
                    )

            logger.info(
                f"Loaded {len(validated_rules)} healing rules from {rules_file}"
            )

            return validated_rules

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse rules file {rules_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load rules from {rules_file}: {e}")
            return []

    def load_deployment_config(
        self,
        filename: str = "deployment_config.json",
    ) -> Dict[str, Any]:
        """
        Load deployment configuration from JSON file.

        Args:
            filename: Name of the deployment configuration file

        Returns:
            Deployment configuration dictionary
        """
        config_file = self.config_dir / filename

        if not config_file.exists():
            logger.warning(f"Deployment config file not found: {config_file}")
            return self._get_default_deployment_config()

        try:
            with open(config_file, "r") as f:
                config = json.load(f)

            logger.info(f"Loaded deployment configuration from {config_file}")

            return config

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse deployment config {config_file}: {e}")
            return self._get_default_deployment_config()
        except Exception as e:
            logger.error(f"Failed to load deployment config from {config_file}: {e}")
            return self._get_default_deployment_config()

    def save_healing_rules(
        self,
        rules: List[Dict[str, Any]],
        filename: str = "healing_rules.json",
    ) -> bool:
        """
        Save healing rules to JSON configuration file.

        Args:
            rules: List of healing rule dictionaries
            filename: Name of the rules configuration file

        Returns:
            True if successful, False otherwise
        """
        rules_file = self.config_dir / filename

        try:
            # Load existing configuration to preserve settings
            existing_config = {}
            if rules_file.exists():
                with open(rules_file, "r") as f:
                    existing_data = json.load(f)
                    existing_config = existing_data.get("configuration", {})

            # Create new configuration
            data = {
                "rules": rules,
                "configuration": existing_config or {
                    "enabled": True,
                    "auto_apply": False,
                    "require_human_approval": True,
                    "max_rules": 100,
                    "rule_priority_order": [
                        "recovery",
                        "validation",
                        "transformation",
                        "enrichment",
                    ],
                },
            }

            # Write to file
            with open(rules_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(rules)} healing rules to {rules_file}")

            return True

        except Exception as e:
            logger.error(f"Failed to save rules to {rules_file}: {e}")
            return False

    def add_healing_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Add a new healing rule to the configuration.

        Args:
            rule: Healing rule dictionary

        Returns:
            True if successful, False otherwise
        """
        if not self._validate_rule(rule):
            logger.error("Invalid rule format")
            return False

        # Load existing rules
        rules = self.load_healing_rules()

        # Check for duplicate rule_id
        rule_id = rule.get("rule_id")
        if any(r.get("rule_id") == rule_id for r in rules):
            logger.error(f"Rule with ID {rule_id} already exists")
            return False

        # Add new rule
        rules.append(rule)

        # Save updated rules
        return self.save_healing_rules(rules)

    def update_healing_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update an existing healing rule.

        Args:
            rule_id: ID of the rule to update
            updates: Dictionary of updates to apply

        Returns:
            True if successful, False otherwise
        """
        # Load existing rules
        rules = self.load_healing_rules()

        # Find and update rule
        found = False
        for rule in rules:
            if rule.get("rule_id") == rule_id:
                rule.update(updates)
                found = True
                break

        if not found:
            logger.error(f"Rule not found: {rule_id}")
            return False

        # Save updated rules
        return self.save_healing_rules(rules)

    def delete_healing_rule(self, rule_id: str) -> bool:
        """
        Delete a healing rule.

        Args:
            rule_id: ID of the rule to delete

        Returns:
            True if successful, False otherwise
        """
        # Load existing rules
        rules = self.load_healing_rules()

        # Filter out the rule
        updated_rules = [r for r in rules if r.get("rule_id") != rule_id]

        if len(updated_rules) == len(rules):
            logger.error(f"Rule not found: {rule_id}")
            return False

        # Save updated rules
        return self.save_healing_rules(updated_rules)

    def _validate_rule(self, rule: Dict[str, Any]) -> bool:
        """
        Validate a healing rule.

        Args:
            rule: Healing rule dictionary

        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            "rule_id",
            "name",
            "rule_type",
            "condition",
            "action",
            "metadata",
        ]

        # Check required fields
        for field in required_fields:
            if field not in rule:
                logger.error(f"Missing required field in rule: {field}")
                return False

        # Validate metadata
        metadata = rule.get("metadata", {})
        if "confidence" not in metadata:
            logger.error("Missing confidence in rule metadata")
            return False

        if not 0 <= metadata.get("confidence", 0) <= 1:
            logger.error("Rule confidence must be between 0 and 1")
            return False

        # Validate risk level
        valid_risk_levels = ["low", "medium", "high"]
        if metadata.get("risk_level") not in valid_risk_levels:
            logger.error(
                f"Invalid risk level: {metadata.get('risk_level')}. "
                f"Must be one of {valid_risk_levels}"
            )
            return False

        return True

    def _get_default_deployment_config(self) -> Dict[str, Any]:
        """
        Get default deployment configuration.

        Returns:
            Default deployment configuration
        """
        return {
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
                            "max_response_time_p95_ms": 500,
                        },
                    },
                    {
                        "name": "expand",
                        "percentage": 25,
                        "duration_minutes": 30,
                        "health_check_interval_seconds": 60,
                        "success_criteria": {
                            "error_rate_threshold": 0.01,
                            "min_success_rate": 0.99,
                            "max_response_time_p95_ms": 500,
                        },
                    },
                    {
                        "name": "majority",
                        "percentage": 50,
                        "duration_minutes": 60,
                        "health_check_interval_seconds": 60,
                        "success_criteria": {
                            "error_rate_threshold": 0.01,
                            "min_success_rate": 0.99,
                            "max_response_time_p95_ms": 500,
                        },
                    },
                    {
                        "name": "complete",
                        "percentage": 100,
                        "duration_minutes": 0,
                        "health_check_interval_seconds": 0,
                        "success_criteria": None,
                    },
                ],
                "rollback": {
                    "enabled": True,
                    "automatic": True,
                    "trigger_on_failure": True,
                    "max_rollback_attempts": 3,
                    "rollback_delay_seconds": 5,
                },
                "health_checks": {
                    "enabled": True,
                    "endpoint": "/health",
                    "timeout_seconds": 10,
                    "healthy_threshold": 3,
                    "unhealthy_threshold": 2,
                },
                "monitoring": {
                    "enabled": True,
                    "prometheus_enabled": True,
                    "metrics": [
                        "deployment_status",
                        "canary_stage",
                        "error_rate",
                        "success_rate",
                        "response_time_p50",
                        "response_time_p95",
                        "response_time_p99",
                        "rollback_count",
                    ],
                    "alerts": {
                        "enabled": True,
                        "channels": ["slack", "email"],
                        "severity_levels": ["critical", "warning", "info"],
                    },
                },
            },
            "validation": {
                "enabled": True,
                "dry_run_first": True,
                "schema_validation": True,
                "conflict_detection": True,
                "backup_before_deploy": True,
            },
        }
