"""
Comprehensive unit tests for SHWL (Self-Healing Workflow Learning) modules.

Tests cover:
- Healing loop orchestration
- Exception clustering
- Rule generation
- Proposal approval workflow
- Deployment management
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from sap_llm.shwl.healing_loop import SelfHealingWorkflowLoop


@pytest.fixture
def mock_pmg():
    """Mock Process Memory Graph."""
    pmg = Mock()
    pmg.query_exceptions.return_value = []
    return pmg


@pytest.fixture
def mock_reasoning_engine():
    """Mock reasoning engine."""
    engine = Mock()
    return engine


@pytest.fixture
def mock_config():
    """Mock SHWL configuration."""
    config = Mock()
    config.clustering = Mock()
    config.clustering.min_cluster_size = 15
    config.rule_generation = Mock()
    config.rule_generation.confidence_threshold = 0.90
    config.config_dir = None
    config.deployment = Mock()
    config.deployment.dry_run = True
    config.deployment.in_cluster = False
    config.schedule = Mock()
    config.schedule.lookback_days = 7
    config.approval = Mock()
    config.approval.auto_approve_threshold = 0.95
    config.approval.human_review_required = True
    return config


@pytest.fixture
def sample_exceptions():
    """Sample exceptions for testing."""
    return [
        {
            "id": "exc-1",
            "message": "Invalid vendor code format",
            "field": "vendor_code",
            "severity": "HIGH",
            "doc_type": "invoice"
        },
        {
            "id": "exc-2",
            "message": "Invalid vendor code format",
            "field": "vendor_code",
            "severity": "HIGH",
            "doc_type": "invoice"
        },
    ]


@pytest.fixture
def sample_clusters():
    """Sample exception clusters."""
    return [
        {
            "id": "cluster-1",
            "size": 25,
            "severity": "HIGH",
            "representative": {
                "message": "Invalid vendor code format",
                "field": "vendor_code"
            },
            "exceptions": []
        }
    ]


@pytest.fixture
def sample_proposal():
    """Sample fix proposal."""
    return {
        "rule_id": "RULE-001",
        "description": "Fix vendor code validation",
        "confidence": 0.92,
        "risk_level": "low",
        "fix_type": "validation",
        "fix_action": {"action": "transform", "pattern": r"^\d{6}$"},
        "exception_pattern": "Invalid vendor code",
        "field": "vendor_code"
    }


@pytest.mark.unit
class TestSelfHealingWorkflowLoop:
    """Tests for SelfHealingWorkflowLoop class."""

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_initialization_default(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test default initialization."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)

        assert shwl.pmg == mock_pmg
        assert shwl.lookback_days == 7
        assert shwl.auto_approve_threshold == 0.95
        assert shwl.human_review_required is True

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_initialization_with_config(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer,
        mock_pmg, mock_reasoning_engine, mock_config
    ):
        """Test initialization with configuration."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(
            pmg=mock_pmg,
            reasoning_engine=mock_reasoning_engine,
            config=mock_config
        )

        assert shwl.reasoning_engine == mock_reasoning_engine
        assert shwl.lookback_days == 7

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_run_healing_cycle_no_exceptions(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test healing cycle with no exceptions."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}
        mock_pmg.query_exceptions.return_value = []

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        result = shwl.run_healing_cycle()

        assert result["exceptions_fetched"] == 0
        assert result["clusters_found"] == 0
        assert result["proposals_generated"] == 0

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_run_healing_cycle_with_exceptions(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer,
        mock_pmg, sample_exceptions, sample_clusters
    ):
        """Test healing cycle with exceptions."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}
        mock_config_loader.return_value.load_healing_rules.return_value = []

        mock_pmg.query_exceptions.return_value = sample_exceptions

        # Mock clusterer
        mock_clusterer_instance = Mock()
        mock_clusterer_instance.cluster.return_value = sample_clusters
        mock_clusterer.return_value = mock_clusterer_instance

        # Mock rule generator
        mock_rule_gen_instance = Mock()
        mock_rule_gen_instance.generate_fix.return_value = {
            "rule_id": "RULE-001",
            "confidence": 0.85,
            "risk_level": "medium"
        }
        mock_rule_gen.return_value = mock_rule_gen_instance

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        result = shwl.run_healing_cycle()

        assert result["exceptions_fetched"] == 2
        assert result["clusters_found"] == 1
        assert result["proposals_generated"] == 1

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_fetch_exceptions_error(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test exception fetching with error."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}
        mock_pmg.query_exceptions.side_effect = Exception("Database error")

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        exceptions = shwl._fetch_exceptions()

        assert exceptions == []

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_cluster_exceptions_filters_by_size(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test that clustering filters by size and severity."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        # Create clusters with various sizes
        all_clusters = [
            {"id": "c1", "size": 25, "severity": "HIGH"},  # Keep
            {"id": "c2", "size": 10, "severity": "HIGH"},  # Filter out (too small)
            {"id": "c3", "size": 20, "severity": "MEDIUM"},  # Keep
            {"id": "c4", "size": 30, "severity": "LOW"},  # Filter out (low severity)
        ]

        mock_clusterer_instance = Mock()
        mock_clusterer_instance.cluster.return_value = all_clusters
        mock_clusterer.return_value = mock_clusterer_instance

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        clusters = shwl._cluster_exceptions([])

        # Should only include c1 and c3
        assert len(clusters) == 2

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_cluster_exceptions_error(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test clustering with error."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        mock_clusterer_instance = Mock()
        mock_clusterer_instance.cluster.side_effect = Exception("Clustering failed")
        mock_clusterer.return_value = mock_clusterer_instance

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        clusters = shwl._cluster_exceptions([])

        assert clusters == []

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_generate_proposals(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer,
        mock_pmg, sample_clusters, sample_proposal
    ):
        """Test proposal generation."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}
        mock_config_loader.return_value.load_healing_rules.return_value = []

        mock_rule_gen_instance = Mock()
        mock_rule_gen_instance.generate_fix.return_value = sample_proposal
        mock_rule_gen.return_value = mock_rule_gen_instance

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        proposals = shwl._generate_proposals(sample_clusters)

        assert len(proposals) == 1
        assert proposals[0]["rule_id"] == "RULE-001"
        assert "cluster_id" in proposals[0]
        assert "timestamp" in proposals[0]
        assert proposals[0]["status"] == "pending"

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_generate_proposals_with_error(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer,
        mock_pmg, sample_clusters
    ):
        """Test proposal generation with error."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}
        mock_config_loader.return_value.load_healing_rules.return_value = []

        mock_rule_gen_instance = Mock()
        mock_rule_gen_instance.generate_fix.side_effect = Exception("Generation failed")
        mock_rule_gen.return_value = mock_rule_gen_instance

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        proposals = shwl._generate_proposals(sample_clusters)

        assert proposals == []

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_process_approvals_auto_approve(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test automatic approval for high confidence proposals."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.human_review_required = False  # Disable human review

        proposals = [
            {
                "rule_id": "RULE-001",
                "confidence": 0.98,
                "risk_level": "low"
            }
        ]

        approved_count = shwl._process_approvals(proposals)

        assert approved_count == 1
        assert len(shwl.approved_proposals) == 1
        assert shwl.approved_proposals[0]["status"] == "approved"
        assert shwl.approved_proposals[0]["approval_method"] == "automatic"

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_process_approvals_human_review_required(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test proposals queue for human review."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.human_review_required = True

        proposals = [
            {
                "rule_id": "RULE-001",
                "confidence": 0.98,
                "risk_level": "low"
            }
        ]

        approved_count = shwl._process_approvals(proposals)

        assert approved_count == 0
        assert len(shwl.pending_proposals) == 1
        assert shwl.pending_proposals[0]["status"] == "pending_review"

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_process_approvals_low_confidence(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test low confidence proposals go to human review."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.human_review_required = False

        proposals = [
            {
                "rule_id": "RULE-001",
                "confidence": 0.80,  # Below threshold
                "risk_level": "low"
            }
        ]

        approved_count = shwl._process_approvals(proposals)

        assert approved_count == 0
        assert len(shwl.pending_proposals) == 1

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_deploy_approved_fixes(
        self, mock_deploy_class, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test deploying approved fixes."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}
        mock_config_loader.return_value.load_healing_rules.return_value = []
        mock_config_loader.return_value.save_healing_rules.return_value = True

        mock_deploy_instance = Mock()
        mock_deploy_instance.deploy_healing_rules.return_value = {
            "success": True,
            "deployment_id": "deploy-123"
        }
        mock_deploy_instance.get_metrics.return_value = {
            "deployments_total": 1,
            "deployments_successful": 1,
            "deployments_failed": 0,
            "rollbacks_total": 0
        }
        mock_deploy_class.return_value = mock_deploy_instance

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.approved_proposals = [
            {
                "rule_id": "RULE-001",
                "status": "approved",
                "description": "Fix rule"
            }
        ]

        deployed_count = shwl._deploy_approved_fixes()

        assert deployed_count == 1
        assert shwl.approved_proposals[0]["status"] == "deployed"
        assert "deployed_at" in shwl.approved_proposals[0]

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_deploy_approved_fixes_failure(
        self, mock_deploy_class, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test deployment failure handling."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}
        mock_config_loader.return_value.load_healing_rules.return_value = []
        mock_config_loader.return_value.save_healing_rules.return_value = True

        mock_deploy_instance = Mock()
        mock_deploy_instance.deploy_healing_rules.return_value = {
            "success": False,
            "error": "ConfigMap update failed"
        }
        mock_deploy_instance.get_metrics.return_value = {
            "deployments_total": 1,
            "deployments_successful": 0,
            "deployments_failed": 1,
            "rollbacks_total": 0
        }
        mock_deploy_class.return_value = mock_deploy_instance

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.approved_proposals = [
            {
                "rule_id": "RULE-001",
                "status": "approved"
            }
        ]

        deployed_count = shwl._deploy_approved_fixes()

        assert deployed_count == 0
        assert shwl.approved_proposals[0]["status"] == "deployment_failed"

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_approve_proposal(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test manual proposal approval."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.pending_proposals = [
            {
                "cluster_id": "cluster-1",
                "rule_id": "RULE-001",
                "status": "pending_review"
            }
        ]

        result = shwl.approve_proposal("cluster-1")

        assert result is True
        assert len(shwl.pending_proposals) == 0
        assert len(shwl.approved_proposals) == 1
        assert shwl.approved_proposals[0]["status"] == "approved"
        assert shwl.approved_proposals[0]["approval_method"] == "manual"

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_approve_proposal_not_found(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test approval of non-existent proposal."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)

        result = shwl.approve_proposal("nonexistent")

        assert result is False

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_reject_proposal(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test proposal rejection."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.pending_proposals = [
            {
                "cluster_id": "cluster-1",
                "rule_id": "RULE-001",
                "status": "pending_review"
            }
        ]

        result = shwl.reject_proposal("cluster-1", "Too risky")

        assert result is True
        assert len(shwl.pending_proposals) == 0
        assert len(shwl.rejected_proposals) == 1
        assert shwl.rejected_proposals[0]["status"] == "rejected"
        assert shwl.rejected_proposals[0]["rejection_reason"] == "Too risky"

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_reject_proposal_not_found(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test rejection of non-existent proposal."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)

        result = shwl.reject_proposal("nonexistent", "Not found")

        assert result is False

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_get_pending_proposals(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test getting pending proposals."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.pending_proposals = [
            {"rule_id": "RULE-001"},
            {"rule_id": "RULE-002"}
        ]

        pending = shwl.get_pending_proposals()

        assert len(pending) == 2
        # Should return a copy
        pending.append({"rule_id": "RULE-003"})
        assert len(shwl.pending_proposals) == 2

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_get_proposal_status_pending(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test getting status of pending proposal."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.pending_proposals = [
            {"rule_id": "RULE-001", "status": "pending_review"}
        ]

        status = shwl.get_proposal_status("RULE-001")

        assert status is not None
        assert status["status"] == "pending_review"

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_get_proposal_status_approved(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test getting status of approved proposal."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.approved_proposals = [
            {"rule_id": "RULE-001", "status": "approved"}
        ]

        status = shwl.get_proposal_status("RULE-001")

        assert status is not None
        assert status["status"] == "approved"

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_get_proposal_status_not_found(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test getting status of non-existent proposal."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)

        status = shwl.get_proposal_status("nonexistent")

        assert status is None

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_proposal_to_rule(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer,
        mock_pmg, sample_proposal
    ):
        """Test converting proposal to rule format."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        sample_proposal["cluster_id"] = "cluster-1"
        sample_proposal["cluster_size"] = 25

        rule = shwl._proposal_to_rule(sample_proposal)

        assert rule["rule_id"] == "RULE-001"
        assert rule["description"] == "Fix vendor code validation"
        assert rule["rule_type"] == "validation"
        assert "condition" in rule
        assert "action" in rule
        assert "metadata" in rule
        assert rule["metadata"]["confidence"] == 0.92


@pytest.mark.unit
class TestSelfHealingWorkflowLoopEdgeCases:
    """Edge case tests for SHWL."""

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_approve_by_rule_id(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test approval using rule_id instead of cluster_id."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.pending_proposals = [
            {
                "cluster_id": "cluster-1",
                "rule_id": "RULE-001",
                "status": "pending_review"
            }
        ]

        result = shwl.approve_proposal("RULE-001")

        assert result is True
        assert len(shwl.approved_proposals) == 1

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_healing_cycle_timing(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test that healing cycle includes timing information."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}
        mock_pmg.query_exceptions.return_value = []

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        result = shwl.run_healing_cycle()

        assert "start_time" in result
        # Validate ISO format
        datetime.fromisoformat(result["start_time"])

    @patch('sap_llm.shwl.healing_loop.ExceptionClusterer')
    @patch('sap_llm.shwl.healing_loop.RuleGenerator')
    @patch('sap_llm.shwl.healing_loop.ConfigurationLoader')
    @patch('sap_llm.shwl.healing_loop.DeploymentManager')
    def test_multiple_proposals_mixed_approval(
        self, mock_deploy, mock_config_loader, mock_rule_gen, mock_clusterer, mock_pmg
    ):
        """Test processing multiple proposals with mixed outcomes."""
        mock_config_loader.return_value.load_deployment_config.return_value = {}

        shwl = SelfHealingWorkflowLoop(pmg=mock_pmg)
        shwl.human_review_required = False

        proposals = [
            {"rule_id": "RULE-001", "confidence": 0.98, "risk_level": "low"},  # Auto-approve
            {"rule_id": "RULE-002", "confidence": 0.80, "risk_level": "low"},  # Human review
            {"rule_id": "RULE-003", "confidence": 0.99, "risk_level": "high"},  # Human review
        ]

        approved_count = shwl._process_approvals(proposals)

        assert approved_count == 1
        assert len(shwl.approved_proposals) == 1
        assert len(shwl.pending_proposals) == 2
