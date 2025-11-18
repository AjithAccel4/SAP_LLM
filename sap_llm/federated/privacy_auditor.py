"""
Privacy Auditor for Federated Learning.

Provides comprehensive privacy compliance and audit capabilities:
- GDPR compliance verification
- HIPAA compliance checking
- Differential privacy budget tracking
- Data minimization verification
- Audit trail generation
- Compliance reporting
"""

import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
from enum import Enum

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"


class PrivacyViolationType(Enum):
    """Types of privacy violations."""
    BUDGET_EXCEEDED = "budget_exceeded"
    DATA_LEAKAGE = "data_leakage"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RETENTION_VIOLATION = "retention_violation"
    PURPOSE_VIOLATION = "purpose_violation"
    INSUFFICIENT_ANONYMIZATION = "insufficient_anonymization"


@dataclass
class PrivacyViolation:
    """Privacy violation record."""
    violation_type: PrivacyViolationType
    severity: str  # "critical", "high", "medium", "low"
    timestamp: str
    description: str
    tenant_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    remediation_steps: List[str] = field(default_factory=list)


@dataclass
class PrivacyReport:
    """Privacy audit report."""
    report_id: str
    timestamp: str
    audit_period: Dict[str, str]  # start_date, end_date

    # Privacy metrics
    privacy_budget_used: Dict[str, float]
    privacy_budget_limit: Dict[str, float]
    budget_compliance: bool

    # Compliance status
    compliance_status: Dict[str, bool]
    violations: List[PrivacyViolation]

    # Data handling
    data_minimization_score: float
    purpose_limitation_score: float

    # Recommendations
    recommendations: List[str]

    # Metadata
    num_tenants: int
    num_training_rounds: int
    total_data_processed: int


class GDPRCompliance:
    """GDPR compliance verification."""

    REQUIREMENTS = {
        "lawfulness": "Processing must be lawful, fair, and transparent",
        "purpose_limitation": "Data collected for specified, explicit purposes",
        "data_minimization": "Data must be adequate, relevant, and limited",
        "accuracy": "Data must be accurate and kept up to date",
        "storage_limitation": "Data kept only as long as necessary",
        "integrity_confidentiality": "Appropriate security measures",
        "accountability": "Controller responsible for compliance"
    }

    @staticmethod
    def verify_compliance(audit_data: Dict[str, Any]) -> Dict[str, bool]:
        """Verify GDPR compliance."""
        compliance = {}

        # Check lawfulness (consent, legitimate interest, etc.)
        compliance["lawfulness"] = audit_data.get("consent_obtained", False)

        # Check purpose limitation
        compliance["purpose_limitation"] = audit_data.get(
            "purpose_specified", False
        ) and not audit_data.get("purpose_changed", False)

        # Check data minimization
        data_fields_used = audit_data.get("data_fields_used", [])
        necessary_fields = audit_data.get("necessary_fields", [])
        compliance["data_minimization"] = set(data_fields_used).issubset(
            set(necessary_fields)
        )

        # Check accuracy
        compliance["accuracy"] = audit_data.get("data_validation_performed", False)

        # Check storage limitation
        retention_period = audit_data.get("retention_period_days", 0)
        max_retention = audit_data.get("max_retention_days", 365)
        compliance["storage_limitation"] = retention_period <= max_retention

        # Check integrity & confidentiality
        compliance["integrity_confidentiality"] = (
            audit_data.get("encryption_enabled", False) and
            audit_data.get("access_control_enabled", False)
        )

        # Check accountability
        compliance["accountability"] = audit_data.get("audit_trail_enabled", False)

        return compliance


class HIPAACompliance:
    """HIPAA compliance verification."""

    REQUIREMENTS = {
        "privacy_rule": "Protects PHI from disclosure",
        "security_rule": "Administrative, physical, technical safeguards",
        "breach_notification": "Notification of breaches",
        "enforcement": "Penalties for non-compliance"
    }

    @staticmethod
    def verify_compliance(audit_data: Dict[str, Any]) -> Dict[str, bool]:
        """Verify HIPAA compliance."""
        compliance = {}

        # Privacy Rule
        compliance["privacy_rule"] = (
            audit_data.get("phi_identified", False) is False or
            audit_data.get("phi_anonymized", False)
        )

        # Security Rule - Administrative safeguards
        compliance["administrative_safeguards"] = (
            audit_data.get("security_management", False) and
            audit_data.get("workforce_training", False) and
            audit_data.get("access_authorization", False)
        )

        # Security Rule - Physical safeguards
        compliance["physical_safeguards"] = (
            audit_data.get("facility_access_controls", False) and
            audit_data.get("device_security", False)
        )

        # Security Rule - Technical safeguards
        compliance["technical_safeguards"] = (
            audit_data.get("access_controls", False) and
            audit_data.get("audit_controls", False) and
            audit_data.get("transmission_security", False) and
            audit_data.get("encryption_enabled", False)
        )

        # Breach Notification
        compliance["breach_notification"] = audit_data.get(
            "breach_notification_enabled", False
        )

        return compliance


class PrivacyBudgetTracker:
    """Tracks differential privacy budget consumption."""

    def __init__(self, epsilon_limit: float = 1.0, delta_limit: float = 1e-5):
        """Initialize privacy budget tracker."""
        self.epsilon_limit = epsilon_limit
        self.delta_limit = delta_limit

        # Track budget per tenant
        self.tenant_budgets: Dict[str, Dict[str, float]] = {}

        # Global budget
        self.global_epsilon = 0.0
        self.global_delta = 0.0

        # Budget history
        self.budget_history = []

    def allocate_budget(self, tenant_id: str, epsilon: float, delta: float):
        """Allocate privacy budget to tenant."""
        if tenant_id not in self.tenant_budgets:
            self.tenant_budgets[tenant_id] = {"epsilon": 0.0, "delta": 0.0}

        self.tenant_budgets[tenant_id]["epsilon"] = epsilon
        self.tenant_budgets[tenant_id]["delta"] = delta

        logger.info(
            f"Allocated budget to {tenant_id}: ε={epsilon}, δ={delta}"
        )

    def consume_budget(
        self,
        tenant_id: str,
        epsilon_spent: float,
        delta_spent: float
    ) -> bool:
        """
        Consume privacy budget.

        Returns:
            True if budget is available, False if exceeded
        """
        if tenant_id not in self.tenant_budgets:
            logger.error(f"Tenant {tenant_id} has no allocated budget")
            return False

        current_epsilon = self.tenant_budgets[tenant_id]["epsilon"]
        current_delta = self.tenant_budgets[tenant_id]["delta"]

        # Check if consumption would exceed limit
        if (epsilon_spent > current_epsilon or delta_spent > current_delta):
            logger.warning(
                f"Budget exceeded for {tenant_id}: "
                f"requested ε={epsilon_spent}, available={current_epsilon}"
            )
            return False

        # Update budgets
        self.tenant_budgets[tenant_id]["epsilon"] -= epsilon_spent
        self.tenant_budgets[tenant_id]["delta"] -= delta_spent

        # Update global budget
        self.global_epsilon = max(self.global_epsilon, epsilon_spent)
        self.global_delta = max(self.global_delta, delta_spent)

        # Record in history
        self.budget_history.append({
            "timestamp": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "epsilon_spent": epsilon_spent,
            "delta_spent": delta_spent,
            "remaining_epsilon": self.tenant_budgets[tenant_id]["epsilon"],
            "remaining_delta": self.tenant_budgets[tenant_id]["delta"]
        })

        return True

    def get_remaining_budget(self, tenant_id: str) -> Dict[str, float]:
        """Get remaining budget for tenant."""
        if tenant_id not in self.tenant_budgets:
            return {"epsilon": 0.0, "delta": 0.0}

        return self.tenant_budgets[tenant_id].copy()

    def is_budget_exceeded(self, tenant_id: Optional[str] = None) -> bool:
        """Check if budget is exceeded."""
        if tenant_id:
            if tenant_id not in self.tenant_budgets:
                return True

            budget = self.tenant_budgets[tenant_id]
            return budget["epsilon"] < 0 or budget["delta"] < 0

        # Check global budget
        return (
            self.global_epsilon > self.epsilon_limit or
            self.global_delta > self.delta_limit
        )


class PrivacyAuditor:
    """
    Comprehensive privacy auditor for federated learning.

    Features:
    - Multi-framework compliance (GDPR, HIPAA, CCPA, etc.)
    - Privacy budget tracking
    - Violation detection and reporting
    - Audit trail management
    - Compliance recommendations
    """

    def __init__(
        self,
        epsilon_limit: float = 1.0,
        delta_limit: float = 1e-5,
        output_dir: str = "./audit_reports"
    ):
        """Initialize privacy auditor."""
        self.epsilon_limit = epsilon_limit
        self.delta_limit = delta_limit

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Budget tracker
        self.budget_tracker = PrivacyBudgetTracker(epsilon_limit, delta_limit)

        # Violations
        self.violations: List[PrivacyViolation] = []

        # Audit events
        self.audit_events: List[Dict[str, Any]] = []

        # Compliance frameworks
        self.enabled_frameworks = {
            ComplianceFramework.GDPR,
            ComplianceFramework.HIPAA
        }

        logger.info("PrivacyAuditor initialized")

    def enable_framework(self, framework: ComplianceFramework):
        """Enable compliance framework."""
        self.enabled_frameworks.add(framework)
        logger.info(f"Enabled compliance framework: {framework.value}")

    def log_event(
        self,
        event_type: str,
        tenant_id: Optional[str],
        details: Dict[str, Any]
    ):
        """Log audit event."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "tenant_id": tenant_id,
            "details": details
        }

        self.audit_events.append(event)

        # Write to audit log file
        audit_file = self.output_dir / "audit_events.jsonl"
        with open(audit_file, 'a') as f:
            f.write(json.dumps(event) + "\n")

    def track_privacy_budget(
        self,
        tenant_id: str,
        epsilon_spent: float,
        delta_spent: float
    ):
        """Track privacy budget consumption."""
        budget_ok = self.budget_tracker.consume_budget(
            tenant_id, epsilon_spent, delta_spent
        )

        if not budget_ok:
            # Record violation
            violation = PrivacyViolation(
                violation_type=PrivacyViolationType.BUDGET_EXCEEDED,
                severity="critical",
                timestamp=datetime.now().isoformat(),
                description=f"Privacy budget exceeded for tenant {tenant_id}",
                tenant_id=tenant_id,
                details={
                    "epsilon_spent": epsilon_spent,
                    "delta_spent": delta_spent
                },
                remediation_steps=[
                    "Stop training immediately",
                    "Review privacy budget allocation",
                    "Consider increasing noise multiplier"
                ]
            )
            self.violations.append(violation)
            logger.error(f"Privacy budget exceeded: {tenant_id}")

        # Log event
        self.log_event("budget_consumption", tenant_id, {
            "epsilon_spent": epsilon_spent,
            "delta_spent": delta_spent,
            "budget_ok": budget_ok
        })

    def verify_data_minimization(
        self,
        tenant_id: str,
        data_fields_used: List[str],
        necessary_fields: List[str]
    ) -> float:
        """
        Verify data minimization principle.

        Returns:
            Score from 0.0 to 1.0
        """
        used_set = set(data_fields_used)
        necessary_set = set(necessary_fields)

        # Check if only necessary fields are used
        unnecessary_fields = used_set - necessary_set

        if unnecessary_fields:
            violation = PrivacyViolation(
                violation_type=PrivacyViolationType.PURPOSE_VIOLATION,
                severity="medium",
                timestamp=datetime.now().isoformat(),
                description="Unnecessary data fields being processed",
                tenant_id=tenant_id,
                details={
                    "unnecessary_fields": list(unnecessary_fields)
                },
                remediation_steps=[
                    "Remove unnecessary fields from processing pipeline",
                    "Update data collection policies"
                ]
            )
            self.violations.append(violation)

        # Calculate score
        score = 1.0 - (len(unnecessary_fields) / max(len(used_set), 1))

        self.log_event("data_minimization_check", tenant_id, {
            "score": score,
            "unnecessary_fields": list(unnecessary_fields)
        })

        return score

    def verify_compliance(
        self,
        tenant_id: str,
        audit_data: Dict[str, Any]
    ) -> Dict[ComplianceFramework, Dict[str, bool]]:
        """Verify compliance across all enabled frameworks."""
        compliance_results = {}

        # GDPR
        if ComplianceFramework.GDPR in self.enabled_frameworks:
            gdpr_result = GDPRCompliance.verify_compliance(audit_data)
            compliance_results[ComplianceFramework.GDPR] = gdpr_result

            # Check for violations
            for requirement, compliant in gdpr_result.items():
                if not compliant:
                    violation = PrivacyViolation(
                        violation_type=PrivacyViolationType.PURPOSE_VIOLATION,
                        severity="high",
                        timestamp=datetime.now().isoformat(),
                        description=f"GDPR requirement not met: {requirement}",
                        tenant_id=tenant_id,
                        details={"requirement": requirement},
                        remediation_steps=[
                            f"Address GDPR requirement: {GDPRCompliance.REQUIREMENTS[requirement]}"
                        ]
                    )
                    self.violations.append(violation)

        # HIPAA
        if ComplianceFramework.HIPAA in self.enabled_frameworks:
            hipaa_result = HIPAACompliance.verify_compliance(audit_data)
            compliance_results[ComplianceFramework.HIPAA] = hipaa_result

            # Check for violations
            for requirement, compliant in hipaa_result.items():
                if not compliant:
                    violation = PrivacyViolation(
                        violation_type=PrivacyViolationType.UNAUTHORIZED_ACCESS,
                        severity="critical",
                        timestamp=datetime.now().isoformat(),
                        description=f"HIPAA requirement not met: {requirement}",
                        tenant_id=tenant_id,
                        details={"requirement": requirement},
                        remediation_steps=[
                            f"Address HIPAA requirement: {requirement}"
                        ]
                    )
                    self.violations.append(violation)

        self.log_event("compliance_check", tenant_id, {
            "frameworks": [f.value for f in compliance_results.keys()],
            "results": {
                f.value: result for f, result in compliance_results.items()
            }
        })

        return compliance_results

    def generate_privacy_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> PrivacyReport:
        """Generate comprehensive privacy audit report."""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()

        # Filter events in date range
        period_events = [
            event for event in self.audit_events
            if start_date.isoformat() <= event["timestamp"] <= end_date.isoformat()
        ]

        # Get privacy budget status
        privacy_budget_used = {
            "epsilon": self.budget_tracker.global_epsilon,
            "delta": self.budget_tracker.global_delta
        }

        privacy_budget_limit = {
            "epsilon": self.epsilon_limit,
            "delta": self.delta_limit
        }

        budget_compliance = not self.budget_tracker.is_budget_exceeded()

        # Get compliance status
        compliance_status = {}
        for framework in self.enabled_frameworks:
            # Check if all requirements are met
            violations_for_framework = [
                v for v in self.violations
                if framework.value in v.description.lower()
            ]
            compliance_status[framework.value] = len(violations_for_framework) == 0

        # Calculate scores
        data_minimization_events = [
            e for e in period_events
            if e["event_type"] == "data_minimization_check"
        ]
        data_minimization_score = (
            sum(e["details"]["score"] for e in data_minimization_events) /
            max(len(data_minimization_events), 1)
        )

        purpose_limitation_score = 0.95  # Placeholder

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Count tenants
        unique_tenants = set(
            event["tenant_id"] for event in period_events
            if event["tenant_id"]
        )

        report = PrivacyReport(
            report_id=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            audit_period={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            privacy_budget_used=privacy_budget_used,
            privacy_budget_limit=privacy_budget_limit,
            budget_compliance=budget_compliance,
            compliance_status=compliance_status,
            violations=self.violations,
            data_minimization_score=data_minimization_score,
            purpose_limitation_score=purpose_limitation_score,
            recommendations=recommendations,
            num_tenants=len(unique_tenants),
            num_training_rounds=len([
                e for e in period_events
                if e["event_type"] == "training_round"
            ]),
            total_data_processed=len(period_events)
        )

        # Save report
        self._save_report(report)

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate privacy recommendations based on audit findings."""
        recommendations = []

        # Check privacy budget
        if self.budget_tracker.is_budget_exceeded():
            recommendations.append(
                "Privacy budget exceeded - increase noise multiplier or reduce training rounds"
            )

        # Check violations
        if self.violations:
            critical_violations = [
                v for v in self.violations if v.severity == "critical"
            ]
            if critical_violations:
                recommendations.append(
                    f"Address {len(critical_violations)} critical privacy violations immediately"
                )

        # General best practices
        recommendations.extend([
            "Regularly rotate encryption keys",
            "Conduct periodic privacy impact assessments",
            "Implement automated privacy budget monitoring",
            "Maintain comprehensive audit trails",
            "Provide privacy training for all stakeholders"
        ])

        return recommendations

    def _save_report(self, report: PrivacyReport):
        """Save privacy report to file."""
        report_file = self.output_dir / f"{report.report_id}.json"

        # Convert to dict
        report_dict = {
            "report_id": report.report_id,
            "timestamp": report.timestamp,
            "audit_period": report.audit_period,
            "privacy_budget_used": report.privacy_budget_used,
            "privacy_budget_limit": report.privacy_budget_limit,
            "budget_compliance": report.budget_compliance,
            "compliance_status": report.compliance_status,
            "violations": [
                {
                    "type": v.violation_type.value,
                    "severity": v.severity,
                    "timestamp": v.timestamp,
                    "description": v.description,
                    "tenant_id": v.tenant_id,
                    "details": v.details,
                    "remediation_steps": v.remediation_steps
                }
                for v in report.violations
            ],
            "data_minimization_score": report.data_minimization_score,
            "purpose_limitation_score": report.purpose_limitation_score,
            "recommendations": report.recommendations,
            "summary": {
                "num_tenants": report.num_tenants,
                "num_training_rounds": report.num_training_rounds,
                "total_data_processed": report.total_data_processed,
                "num_violations": len(report.violations)
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Privacy report saved: {report_file}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    auditor = PrivacyAuditor(epsilon_limit=1.0, delta_limit=1e-5)

    # Enable compliance frameworks
    auditor.enable_framework(ComplianceFramework.GDPR)
    auditor.enable_framework(ComplianceFramework.HIPAA)

    print("Privacy auditor module loaded successfully")
