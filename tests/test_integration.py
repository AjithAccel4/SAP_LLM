"""
Integration tests for full pipeline.
"""

import pytest
from pathlib import Path

from sap_llm.stages.inbox import InboxStage
from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.stages.validation import ValidationStage


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end integration tests."""

    @pytest.mark.slow
    @pytest.mark.requires_models
    def test_full_pipeline_purchase_order(self, test_config, sample_document_image, temp_dir):
        """Test full pipeline with purchase order."""
        # This would require all models loaded
        # For now, test individual stages in sequence

        # Save sample image
        image_path = temp_dir / "test_po.png"
        sample_document_image.save(image_path)

        with open(image_path, "rb") as f:
            file_content = f.read()

        # Stage 1: Inbox
        inbox = InboxStage(config=test_config.stages.inbox)
        inbox_result = inbox.process({
            "file_path": str(image_path),
            "file_content": file_content,
        })

        assert inbox_result is not None
        assert "document_hash" in inbox_result

        # Stage 2: Preprocessing
        preprocessing = PreprocessingStage(config=test_config.stages.preprocessing)
        prep_result = preprocessing.process(inbox_result)

        assert prep_result is not None
        assert "ocr_text" in prep_result

        # Stage 3-8 would require models
        # Skipping for unit tests

    @pytest.mark.slow
    def test_pipeline_error_handling(self, test_config):
        """Test pipeline error handling."""
        # Test invalid input
        inbox = InboxStage(config=test_config.stages.inbox)

        with pytest.raises(Exception):
            inbox.process({})  # Missing required fields

    def test_pipeline_stage_order(self):
        """Test that stages are processed in correct order."""
        # This would test the orchestration
        pass


@pytest.mark.integration
@pytest.mark.pmg
class TestPMGIntegration:
    """Integration tests for PMG."""

    @pytest.mark.requires_cosmos
    def test_pmg_store_and_retrieve(self, test_config, sample_adc):
        """Test storing and retrieving from PMG."""
        # This would require Cosmos DB
        pass

    def test_pmg_similarity_search(self, test_config, sample_adc):
        """Test PMG similarity search."""
        # This would require PMG with vector store
        pass


@pytest.mark.integration
@pytest.mark.shwl
class TestSHWLIntegration:
    """Integration tests for SHWL."""

    def test_shwl_healing_cycle(self, test_config, sample_cluster, mock_pmg, mock_reasoning_engine):
        """Test SHWL healing cycle."""
        from sap_llm.shwl.healing_loop import SelfHealingWorkflowLoop

        # Create SHWL with mocks
        shwl = SelfHealingWorkflowLoop(
            pmg=mock_pmg,
            reasoning_engine=mock_reasoning_engine,
            config=test_config.shwl,
        )

        # Mock PMG to return exceptions
        mock_pmg.query_exceptions.return_value = sample_cluster["exceptions"]

        # Run healing cycle
        result = shwl.run_healing_cycle()

        assert result is not None
        assert "exceptions_fetched" in result
        assert "clusters_found" in result
        assert "proposals_generated" in result

    def test_shwl_exception_clustering(self, test_config, sample_cluster):
        """Test exception clustering."""
        from sap_llm.shwl.clusterer import ExceptionClusterer

        clusterer = ExceptionClusterer(min_cluster_size=15)

        # Cluster exceptions
        clusters = clusterer.cluster(sample_cluster["exceptions"])

        assert isinstance(clusters, list)

    def test_shwl_rule_generation(self, test_config, sample_cluster, mock_reasoning_engine):
        """Test rule generation."""
        from sap_llm.shwl.rule_generator import RuleGenerator

        generator = RuleGenerator(reasoning_engine=mock_reasoning_engine)

        # Generate fix
        fix = generator.generate_fix(sample_cluster, existing_rules=[])

        assert fix is not None
        assert "rule_id" in fix or "modification_type" in fix


@pytest.mark.integration
@pytest.mark.knowledge_base
class TestKnowledgeBaseIntegration:
    """Integration tests for Knowledge Base."""

    def test_knowledge_base_search(self, test_config):
        """Test knowledge base search."""
        from sap_llm.knowledge_base.storage import KnowledgeBaseStorage

        storage = KnowledgeBaseStorage(mongo_uri=None)  # Mock mode

        # Search should not crash
        results = storage.search_apis("purchase order", k=5)
        assert isinstance(results, list)

    def test_knowledge_base_crawler(self):
        """Test knowledge base crawler."""
        import asyncio
        from sap_llm.knowledge_base.crawler import SAPAPICrawler

        crawler = SAPAPICrawler()

        # Test mock data generation
        data = asyncio.run(crawler.crawl_mock_data())

        assert data is not None
        assert "api_schemas" in data
        assert "field_mappings" in data
        assert "business_rules" in data


@pytest.mark.integration
class TestPerformance:
    """Performance and benchmark tests."""

    @pytest.mark.performance
    def test_inbox_throughput(self, test_config, sample_image, temp_dir, benchmark):
        """Benchmark inbox throughput."""
        inbox = InboxStage(config=test_config.stages.inbox)

        # Save sample image
        image_path = temp_dir / "test_perf.png"
        sample_image.save(image_path)

        with open(image_path, "rb") as f:
            file_content = f.read()

        input_data = {
            "file_path": str(image_path),
            "file_content": file_content,
        }

        # Benchmark
        result = benchmark(inbox.process, input_data)

        assert result is not None

    @pytest.mark.performance
    @pytest.mark.slow
    def test_preprocessing_latency(self, test_config, sample_document_image, temp_dir, benchmark):
        """Benchmark preprocessing latency."""
        preprocessing = PreprocessingStage(config=test_config.stages.preprocessing)

        # Save sample image
        image_path = temp_dir / "test_perf.png"
        sample_document_image.save(image_path)

        input_data = {
            "file_path": str(image_path),
            "image": sample_document_image,
        }

        # Benchmark
        result = benchmark(preprocessing.process, input_data)

        assert result is not None
        assert "ocr_text" in result

    @pytest.mark.performance
    def test_validation_latency(self, test_config, sample_adc, benchmark):
        """Benchmark validation latency."""
        validation = ValidationStage(config=test_config.stages.validation)

        input_data = {
            "adc": sample_adc,
            "quality_score": 0.95,
            "document_type": "purchase_order",
        }

        # Benchmark
        result = benchmark(validation.process, input_data)

        assert result is not None
