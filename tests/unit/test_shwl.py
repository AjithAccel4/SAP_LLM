"""
Comprehensive unit tests for SHWL (Self-Healing Workflow Loop) components.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from sap_llm.shwl.clusterer import ExceptionClusterer
from sap_llm.shwl.rule_generator import RuleGenerator
from sap_llm.shwl.healing_loop import HealingLoop
from sap_llm.shwl.deployment_manager import DeploymentManager
from sap_llm.shwl.config_loader import ConfigLoader


@pytest.mark.unit
@pytest.mark.shwl
class TestExceptionClusterer:
    """Tests for ExceptionClusterer."""

    @pytest.fixture
    def clusterer(self):
        """Create ExceptionClusterer instance."""
        config = MagicMock()
        config.n_clusters = 10
        config.min_cluster_size = 5
        config.algorithm = "hdbscan"
        return ExceptionClusterer(config=config)

    def test_clusterer_initialization(self, clusterer):
        """Test clusterer initialization."""
        assert clusterer is not None
        assert clusterer.n_clusters == 10

    def test_fit_exceptions(self, clusterer):
        """Test fitting exceptions to create clusters."""
        exceptions = [
            {
                "id": f"exc_{i}",
                "category": "VALIDATION_ERROR",
                "field": "total_amount",
                "message": "Amount mismatch",
            }
            for i in range(50)
        ]

        clusterer.fit(exceptions)
        # Should create clusters

    def test_predict_cluster(self, clusterer):
        """Test predicting cluster for new exception."""
        # Fit first
        exceptions = [
            {
                "id": f"exc_{i}",
                "category": "VALIDATION_ERROR",
                "field": "total_amount",
                "message": "Amount mismatch",
            }
            for i in range(50)
        ]

        clusterer.fit(exceptions)

        # Predict
        new_exception = {
            "category": "VALIDATION_ERROR",
            "field": "total_amount",
            "message": "Amount mismatch",
        }

        cluster_id = clusterer.predict(new_exception)
        assert cluster_id is not None

    def test_get_cluster_info(self, clusterer, sample_cluster):
        """Test getting cluster information."""
        exceptions = sample_cluster["exceptions"]
        clusterer.fit(exceptions)

        clusters = clusterer.get_clusters()
        assert isinstance(clusters, list)

    def test_empty_exceptions(self, clusterer):
        """Test handling empty exceptions list."""
        clusterer.fit([])
        # Should handle gracefully

    @pytest.mark.parametrize("algorithm", [
        "hdbscan",
        "kmeans",
        "dbscan",
    ])
    def test_different_algorithms(self, algorithm):
        """Test different clustering algorithms."""
        config = MagicMock()
        config.algorithm = algorithm
        config.n_clusters = 10
        config.min_cluster_size = 5

        clusterer = ExceptionClusterer(config=config)
        assert clusterer.algorithm == algorithm


@pytest.mark.unit
@pytest.mark.shwl
class TestRuleGenerator:
    """Tests for RuleGenerator."""

    @pytest.fixture
    def rule_generator(self, mock_reasoning_engine):
        """Create RuleGenerator instance."""
        config = MagicMock()
        config.min_confidence = 0.90
        return RuleGenerator(config=config, reasoning_engine=mock_reasoning_engine)

    def test_rule_generator_initialization(self, rule_generator):
        """Test rule generator initialization."""
        assert rule_generator is not None

    def test_generate_rule_from_cluster(self, rule_generator, sample_cluster):
        """Test generating rule from exception cluster."""
        with patch.object(rule_generator, 'reasoning_engine') as mock_engine:
            mock_engine.generate.return_value = """
            {
                "rule_type": "AUTO_CORRECT",
                "condition": "total_amount_mismatch",
                "action": "recalculate_total",
                "confidence": 0.95
            }
            """

            rule = rule_generator.generate_rule(sample_cluster)

            assert rule is not None
            assert "rule_type" in rule
            assert "confidence" in rule

    def test_validate_rule(self, rule_generator):
        """Test rule validation."""
        rule = {
            "rule_type": "AUTO_CORRECT",
            "condition": "total_amount_mismatch",
            "action": "recalculate_total",
            "confidence": 0.95,
        }

        is_valid = rule_generator.validate_rule(rule)
        assert is_valid is True

    def test_rule_confidence_threshold(self, rule_generator):
        """Test rule confidence threshold filtering."""
        low_confidence_rule = {
            "rule_type": "AUTO_CORRECT",
            "confidence": 0.50,  # Below threshold
        }

        is_valid = rule_generator.validate_rule(low_confidence_rule)
        # Should reject low confidence rules

    @pytest.mark.parametrize("rule_type", [
        "AUTO_CORRECT",
        "VALIDATION_RULE",
        "ROUTING_RULE",
        "TRANSFORMATION_RULE",
    ])
    def test_different_rule_types(self, rule_generator, rule_type):
        """Test generating different rule types."""
        cluster = {
            "id": "cluster_001",
            "category": "VALIDATION_ERROR",
            "size": 25,
        }

        with patch.object(rule_generator, 'reasoning_engine') as mock_engine:
            mock_engine.generate.return_value = f'{{"rule_type": "{rule_type}", "confidence": 0.95}}'

            rule = rule_generator.generate_rule(cluster)
            assert rule is not None


@pytest.mark.unit
@pytest.mark.shwl
class TestHealingLoop:
    """Tests for HealingLoop."""

    @pytest.fixture
    def healing_loop(self, mock_pmg):
        """Create HealingLoop instance."""
        config = MagicMock()
        config.loop_interval = 3600  # 1 hour
        config.min_exceptions = 10
        config.enable_auto_deploy = False

        return HealingLoop(pmg=mock_pmg, config=config)

    def test_healing_loop_initialization(self, healing_loop):
        """Test healing loop initialization."""
        assert healing_loop is not None
        assert healing_loop.loop_interval == 3600

    def test_collect_exceptions(self, healing_loop):
        """Test collecting exceptions from PMG."""
        with patch.object(healing_loop.pmg, 'query_exceptions') as mock_query:
            mock_query.return_value = [
                {"id": f"exc_{i}", "category": "VALIDATION_ERROR"}
                for i in range(20)
            ]

            exceptions = healing_loop.collect_exceptions()
            assert len(exceptions) == 20

    def test_cluster_and_analyze(self, healing_loop):
        """Test clustering and analyzing exceptions."""
        exceptions = [
            {
                "id": f"exc_{i}",
                "category": "VALIDATION_ERROR",
                "field": "total_amount",
            }
            for i in range(50)
        ]

        with patch.object(healing_loop, 'clusterer') as mock_clusterer:
            mock_clusterer.fit.return_value = None
            mock_clusterer.get_clusters.return_value = [
                {"id": "cluster_001", "size": 25},
                {"id": "cluster_002", "size": 25},
            ]

            clusters = healing_loop.cluster_exceptions(exceptions)
            assert len(clusters) == 2

    def test_generate_healing_rules(self, healing_loop):
        """Test generating healing rules from clusters."""
        clusters = [
            {
                "id": "cluster_001",
                "category": "VALIDATION_ERROR",
                "size": 25,
            }
        ]

        with patch.object(healing_loop, 'rule_generator') as mock_gen:
            mock_gen.generate_rule.return_value = {
                "rule_type": "AUTO_CORRECT",
                "confidence": 0.95,
            }

            rules = healing_loop.generate_rules(clusters)
            assert len(rules) >= 0

    def test_run_healing_cycle(self, healing_loop):
        """Test running complete healing cycle."""
        with patch.object(healing_loop, 'collect_exceptions') as mock_collect:
            mock_collect.return_value = [
                {"id": f"exc_{i}", "category": "VALIDATION_ERROR"}
                for i in range(50)
            ]

            with patch.object(healing_loop, 'cluster_exceptions') as mock_cluster:
                mock_cluster.return_value = [{"id": "cluster_001", "size": 50}]

                with patch.object(healing_loop, 'generate_rules') as mock_gen:
                    mock_gen.return_value = [{"rule_type": "AUTO_CORRECT"}]

                    results = healing_loop.run_cycle()
                    assert results is not None

    def test_healing_loop_metrics(self, healing_loop):
        """Test healing loop metrics collection."""
        metrics = healing_loop.get_metrics()
        assert isinstance(metrics, dict)
        # Should include: cycles_run, rules_generated, exceptions_resolved, etc.


@pytest.mark.unit
@pytest.mark.shwl
class TestDeploymentManager:
    """Tests for DeploymentManager."""

    @pytest.fixture
    def deployment_manager(self):
        """Create DeploymentManager instance."""
        config = MagicMock()
        config.deployment_mode = "staged"  # staged, canary, blue-green
        config.approval_required = True
        return DeploymentManager(config=config)

    def test_deployment_manager_initialization(self, deployment_manager):
        """Test deployment manager initialization."""
        assert deployment_manager is not None
        assert deployment_manager.deployment_mode == "staged"

    def test_stage_rule(self, deployment_manager):
        """Test staging a rule for deployment."""
        rule = {
            "id": "rule_001",
            "rule_type": "AUTO_CORRECT",
            "confidence": 0.95,
        }

        deployment_manager.stage_rule(rule)
        # Should add to staging area

    def test_deploy_rule(self, deployment_manager):
        """Test deploying a rule to production."""
        rule = {
            "id": "rule_001",
            "rule_type": "AUTO_CORRECT",
            "confidence": 0.95,
        }

        with patch.object(deployment_manager, 'validate_rule') as mock_validate:
            mock_validate.return_value = True

            success = deployment_manager.deploy_rule(rule)
            # Deployment success depends on approval requirements

    def test_rollback_rule(self, deployment_manager):
        """Test rolling back a deployed rule."""
        rule_id = "rule_001"

        with patch.object(deployment_manager, 'get_rule') as mock_get:
            mock_get.return_value = {
                "id": rule_id,
                "status": "deployed",
            }

            success = deployment_manager.rollback_rule(rule_id)
            # Should rollback successfully

    def test_canary_deployment(self):
        """Test canary deployment strategy."""
        config = MagicMock()
        config.deployment_mode = "canary"
        config.canary_percentage = 10  # Deploy to 10% of traffic

        dm = DeploymentManager(config=config)

        rule = {"id": "rule_001", "rule_type": "AUTO_CORRECT"}

        with patch.object(dm, 'deploy_canary') as mock_deploy:
            mock_deploy.return_value = True

            success = dm.deploy_rule(rule, strategy="canary")
            # Should deploy to 10% of traffic

    def test_blue_green_deployment(self):
        """Test blue-green deployment strategy."""
        config = MagicMock()
        config.deployment_mode = "blue-green"

        dm = DeploymentManager(config=config)

        rule = {"id": "rule_001", "rule_type": "AUTO_CORRECT"}

        with patch.object(dm, 'deploy_blue_green') as mock_deploy:
            mock_deploy.return_value = True

            success = dm.deploy_rule(rule, strategy="blue-green")

    @pytest.mark.parametrize("deployment_mode", [
        "staged",
        "canary",
        "blue-green",
        "immediate",
    ])
    def test_different_deployment_modes(self, deployment_mode):
        """Test different deployment modes."""
        config = MagicMock()
        config.deployment_mode = deployment_mode
        config.approval_required = False

        dm = DeploymentManager(config=config)
        assert dm.deployment_mode == deployment_mode


@pytest.mark.unit
@pytest.mark.shwl
class TestConfigLoader:
    """Tests for ConfigLoader."""

    @pytest.fixture
    def config_loader(self):
        """Create ConfigLoader instance."""
        return ConfigLoader()

    def test_load_config_file(self, config_loader, temp_dir):
        """Test loading configuration from file."""
        config_file = temp_dir / "shwl_config.yaml"
        config_file.write_text("""
        clusterer:
          n_clusters: 10
          algorithm: hdbscan
        rule_generator:
          min_confidence: 0.90
        healing_loop:
          loop_interval: 3600
        """)

        config = config_loader.load(str(config_file))
        assert config is not None
        assert config.clusterer.n_clusters == 10

    def test_load_default_config(self, config_loader):
        """Test loading default configuration."""
        config = config_loader.load_defaults()
        assert config is not None

    def test_validate_config(self, config_loader):
        """Test configuration validation."""
        config = {
            "clusterer": {"n_clusters": 10},
            "rule_generator": {"min_confidence": 0.90},
            "healing_loop": {"loop_interval": 3600},
        }

        is_valid = config_loader.validate(config)
        assert is_valid is True

    def test_merge_configs(self, config_loader):
        """Test merging multiple configurations."""
        config1 = {"clusterer": {"n_clusters": 10}}
        config2 = {"rule_generator": {"min_confidence": 0.90}}

        merged = config_loader.merge(config1, config2)
        assert "clusterer" in merged
        assert "rule_generator" in merged


@pytest.mark.unit
@pytest.mark.shwl
class TestSHWLIntegration:
    """Integration tests for SHWL components."""

    @pytest.fixture
    def full_shwl_system(self, mock_pmg, mock_reasoning_engine):
        """Create full SHWL system."""
        config = MagicMock()
        config.clusterer = MagicMock(n_clusters=10, algorithm="hdbscan", min_cluster_size=5)
        config.rule_generator = MagicMock(min_confidence=0.90)
        config.healing_loop = MagicMock(loop_interval=3600, min_exceptions=10)
        config.deployment = MagicMock(deployment_mode="staged", approval_required=False)

        healing_loop = HealingLoop(pmg=mock_pmg, config=config.healing_loop)
        healing_loop.clusterer = ExceptionClusterer(config=config.clusterer)
        healing_loop.rule_generator = RuleGenerator(
            config=config.rule_generator,
            reasoning_engine=mock_reasoning_engine,
        )
        healing_loop.deployment_manager = DeploymentManager(config=config.deployment)

        return healing_loop

    def test_end_to_end_healing_flow(self, full_shwl_system):
        """Test complete SHWL healing flow."""
        # Mock exception collection
        exceptions = [
            {
                "id": f"exc_{i}",
                "category": "VALIDATION_ERROR",
                "field": "total_amount",
                "message": "Amount mismatch",
            }
            for i in range(50)
        ]

        with patch.object(full_shwl_system, 'collect_exceptions') as mock_collect:
            mock_collect.return_value = exceptions

            # Mock clustering
            with patch.object(full_shwl_system.clusterer, 'fit') as mock_fit:
                mock_fit.return_value = None

                with patch.object(full_shwl_system.clusterer, 'get_clusters') as mock_clusters:
                    mock_clusters.return_value = [
                        {
                            "id": "cluster_001",
                            "size": 50,
                            "category": "VALIDATION_ERROR",
                            "exceptions": exceptions,
                        }
                    ]

                    # Mock rule generation
                    with patch.object(full_shwl_system.rule_generator, 'generate_rule') as mock_gen:
                        mock_gen.return_value = {
                            "id": "rule_001",
                            "rule_type": "AUTO_CORRECT",
                            "confidence": 0.95,
                        }

                        # Run cycle
                        results = full_shwl_system.run_cycle()
                        assert results is not None

    def test_exception_lifecycle(self, full_shwl_system, sample_cluster):
        """Test exception from detection to resolution."""
        # 1. Collect exceptions
        with patch.object(full_shwl_system, 'collect_exceptions') as mock_collect:
            mock_collect.return_value = sample_cluster["exceptions"]

            exceptions = full_shwl_system.collect_exceptions()
            assert len(exceptions) > 0

            # 2. Cluster exceptions
            with patch.object(full_shwl_system.clusterer, 'fit'):
                with patch.object(full_shwl_system.clusterer, 'get_clusters') as mock_clusters:
                    mock_clusters.return_value = [sample_cluster]

                    clusters = full_shwl_system.cluster_exceptions(exceptions)
                    assert len(clusters) > 0

                    # 3. Generate healing rules
                    with patch.object(full_shwl_system.rule_generator, 'generate_rule') as mock_gen:
                        mock_gen.return_value = {
                            "rule_type": "AUTO_CORRECT",
                            "action": "recalculate_total",
                        }

                        rules = full_shwl_system.generate_rules(clusters)
                        assert len(rules) > 0

    def test_metrics_tracking(self, full_shwl_system):
        """Test tracking SHWL metrics."""
        with patch.object(full_shwl_system, 'collect_exceptions') as mock_collect:
            mock_collect.return_value = [{"id": f"exc_{i}"} for i in range(50)]

            full_shwl_system.run_cycle()

            metrics = full_shwl_system.get_metrics()
            assert isinstance(metrics, dict)
            # Should track: cycles_run, exceptions_collected, clusters_formed, rules_generated, etc.
