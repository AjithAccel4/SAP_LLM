"""
Unit tests for Intelligent Web Search Triggering.

Tests the enhanced context_aware_processor.py with web search integration.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from sap_llm.inference.context_aware_processor import ContextAwareProcessor


class TestIntelligentWebSearchTriggering:
    """Test suite for web search triggering logic."""

    @pytest.fixture
    def mock_web_search_agent(self):
        """Mock WebSearchAgent for testing."""
        agent = Mock()
        agent.search.return_value = [
            {
                "url": "https://api.sap.com/invoice",
                "title": "SAP Invoice API",
                "snippet": "Official SAP Invoice processing API",
                "trust_score": 0.95
            },
            {
                "url": "https://help.sap.com/invoice",
                "title": "Invoice Processing Guide",
                "snippet": "Best practices for invoice processing",
                "trust_score": 0.90
            }
        ]
        return agent

    @pytest.fixture
    def processor_with_web_search(self, mock_web_search_agent):
        """Create processor with mocked web search."""
        with patch('sap_llm.inference.context_aware_processor.WebSearchAgent',
                   return_value=mock_web_search_agent):
            processor = ContextAwareProcessor(
                enable_web_search=True,
                web_search_config={}
            )
            processor.web_search_agent = mock_web_search_agent
            return processor

    def test_processor_initialization_with_web_search(self):
        """Test that processor initializes with web search enabled."""
        with patch('sap_llm.inference.context_aware_processor.WebSearchAgent'):
            processor = ContextAwareProcessor(enable_web_search=True)

            assert processor.enable_web_search is True
            assert hasattr(processor, 'web_search_agent')
            assert hasattr(processor, 'web_search_threshold')
            assert processor.web_search_threshold == 0.65

    def test_web_search_not_triggered_high_confidence(self, processor_with_web_search):
        """Test that web search is NOT triggered when confidence is high."""
        document = {"doc_type": "invoice"}

        # Mock high confidence result
        with patch.object(processor_with_web_search, '_initial_prediction',
                         return_value={"confidence": 0.85, "extracted_fields": {}}):
            result = processor_with_web_search.process_document(document)

            # Web search should not be triggered
            processor_with_web_search.web_search_agent.search.assert_not_called()
            assert "web_search_used" not in result

    def test_web_search_triggered_low_confidence(self, processor_with_web_search):
        """Test that web search IS triggered when confidence is low."""
        document = {"doc_type": "invoice"}

        # Mock low confidence result
        with patch.object(processor_with_web_search, '_initial_prediction',
                         return_value={"confidence": 0.60, "extracted_fields": {}}):
            with patch.object(processor_with_web_search.retriever, 'retrieve_context',
                            return_value=[]):
                result = processor_with_web_search.process_document(document)

                # Web search should be triggered
                processor_with_web_search.web_search_agent.search.assert_called_once()
                assert result.get("web_search_used") is True

    def test_web_search_confidence_boost(self, processor_with_web_search):
        """Test that web search results boost confidence."""
        document = {"doc_type": "invoice"}

        initial_confidence = 0.60
        with patch.object(processor_with_web_search, '_initial_prediction',
                         return_value={"confidence": initial_confidence, "extracted_fields": {}}):
            with patch.object(processor_with_web_search.retriever, 'retrieve_context',
                            return_value=[]):
                result = processor_with_web_search.process_document(document)

                # Confidence should be boosted
                assert result["confidence"] > initial_confidence
                assert result.get("web_knowledge") is not None

    def test_web_search_query_construction(self, processor_with_web_search):
        """Test that web search queries are constructed correctly."""
        document = {
            "doc_type": "invoice",
            "module": "MM"
        }

        with patch.object(processor_with_web_search, '_initial_prediction',
                         return_value={"confidence": 0.60, "extracted_fields": {"vendor": None}}):
            with patch.object(processor_with_web_search.retriever, 'retrieve_context',
                            return_value=[]):
                processor_with_web_search.process_document(document)

                # Check search was called with correct context
                call_args = processor_with_web_search.web_search_agent.search.call_args
                assert call_args is not None
                assert "invoice" in call_args[1]["query"].lower()

    def test_web_search_statistics_tracking(self, processor_with_web_search):
        """Test that web search triggers are tracked in statistics."""
        document = {"doc_type": "invoice"}

        with patch.object(processor_with_web_search, '_initial_prediction',
                         return_value={"confidence": 0.60, "extracted_fields": {}}):
            with patch.object(processor_with_web_search.retriever, 'retrieve_context',
                            return_value=[]):
                processor_with_web_search.process_document(document)

                # Check statistics
                stats = processor_with_web_search.get_statistics()
                assert stats["web_search_triggered"] == 1

    def test_rag_before_web_search_cascade(self, processor_with_web_search):
        """Test that RAG is tried before web search."""
        document = {"doc_type": "invoice"}

        # Mock low initial confidence
        with patch.object(processor_with_web_search, '_initial_prediction',
                         return_value={"confidence": 0.65, "extracted_fields": {}}):

            # Mock RAG improvement to just below web search threshold
            mock_context = [Mock(success=True, similarity=0.7)]
            with patch.object(processor_with_web_search.retriever, 'retrieve_context',
                            return_value=mock_context):
                with patch.object(processor_with_web_search, '_context_aware_prediction',
                                return_value={"confidence": 0.64, "extracted_fields": {}}):

                    result = processor_with_web_search.process_document(document)

                    # Both RAG and web search should be used
                    assert processor_with_web_search.stats["context_used"] == 1
                    assert processor_with_web_search.stats["web_search_triggered"] == 1

    def test_web_search_disabled_flag(self):
        """Test that web search can be disabled."""
        processor = ContextAwareProcessor(enable_web_search=False)

        document = {"doc_type": "invoice"}

        with patch.object(processor, '_initial_prediction',
                         return_value={"confidence": 0.50, "extracted_fields": {}}):
            result = processor.process_document(document)

            # Web search should not be used
            assert "web_search_used" not in result

    def test_web_search_error_handling(self, processor_with_web_search):
        """Test that errors in web search are handled gracefully."""
        document = {"doc_type": "invoice"}

        # Make web search raise an exception
        processor_with_web_search.web_search_agent.search.side_effect = Exception("API Error")

        with patch.object(processor_with_web_search, '_initial_prediction',
                         return_value={"confidence": 0.60, "extracted_fields": {}}):
            with patch.object(processor_with_web_search.retriever, 'retrieve_context',
                            return_value=[]):

                # Should not raise exception
                result = processor_with_web_search.process_document(document)

                # Should return result without web search enhancement
                assert result["confidence"] == 0.60


class TestWebSearchEnhancementMethod:
    """Test the _web_search_enhancement method specifically."""

    @pytest.fixture
    def processor(self):
        """Create processor with mocked web search."""
        mock_agent = Mock()
        mock_agent.search.return_value = [
            {"url": "test.com", "title": "Test", "snippet": "Test", "trust_score": 0.8}
        ]

        processor = ContextAwareProcessor(enable_web_search=False)
        processor.web_search_agent = mock_agent
        return processor

    def test_web_search_enhancement_with_missing_fields(self, processor):
        """Test web search enhancement when fields are missing."""
        document = {"doc_type": "invoice"}
        current_result = {
            "confidence": 0.60,
            "extracted_fields": {"vendor": None, "amount": ""}
        }

        enhanced = processor._web_search_enhancement(document, current_result)

        # Should have web search metadata
        assert enhanced.get("web_search_used") is True
        assert enhanced.get("web_knowledge") is not None
        assert len(enhanced["web_knowledge"]) > 0

    def test_confidence_boost_calculation(self, processor):
        """Test that confidence boost is calculated correctly."""
        document = {"doc_type": "invoice"}
        current_result = {"confidence": 0.60, "extracted_fields": {}}

        # Mock high trust score results
        processor.web_search_agent.search.return_value = [
            {"url": "test1.com", "title": "T1", "snippet": "S1", "trust_score": 0.9},
            {"url": "test2.com", "title": "T2", "snippet": "S2", "trust_score": 0.8},
        ]

        enhanced = processor._web_search_enhancement(document, current_result)

        # Average trust = 0.85, boost = 0.85 * 0.15 = 0.1275
        # New confidence should be 0.60 + 0.1275 = 0.7275
        assert enhanced["confidence"] > 0.60
        assert enhanced["confidence"] <= 0.95  # Max cap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
