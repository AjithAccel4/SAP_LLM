"""
Web Search Agent for SAP_LLM.

Provides high-level interface for intelligent web searching with
SAP domain expertise, semantic ranking, and knowledge extraction.
"""

from typing import Any, Dict, List, Optional

from sap_llm.utils.logger import get_logger
from sap_llm.web_search.search_engine import SearchMode, WebSearchEngine

logger = get_logger(__name__)


class WebSearchAgent:
    """
    Intelligent web search agent with SAP domain expertise.

    Features:
    - Multi-provider search with automatic fallback
    - Context-aware query refinement
    - Semantic result ranking
    - SAP source validation and trust scoring
    - Knowledge extraction and learning
    - Caching and rate limiting

    Example:
        >>> agent = WebSearchAgent(config)
        >>> results = agent.search("How to get invoice price from SAP BAPI?")
        >>> knowledge = agent.search_and_learn("SAP OData API authentication")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize web search agent.

        Args:
            config: Configuration dictionary for search engine
        """
        self.config = config or {}
        self.engine = WebSearchEngine(config)

        logger.info("WebSearchAgent initialized")

    def search(
        self,
        query: str,
        num_results: int = 10,
        context: Optional[Dict[str, Any]] = None,
        search_mode: str = "web"
    ) -> List[Dict[str, Any]]:
        """
        Perform intelligent web search.

        Args:
            query: Search query
            num_results: Number of results to return
            context: Optional context (document_type, module, etc.)
            search_mode: Search mode (web, news, api, etc.)

        Returns:
            List of ranked search results with trust scores

        Example:
            >>> results = agent.search(
            ...     "SAP vendor invoice BAPI",
            ...     context={"document_type": "invoice", "module": "MM"}
            ... )
            >>> print(results[0]["title"], results[0]["trust_score"])
        """
        mode_mapping = {
            "web": SearchMode.WEB,
            "news": SearchMode.NEWS,
            "images": SearchMode.IMAGES,
            "api": SearchMode.WEB  # Use web mode with API filters
        }

        mode = mode_mapping.get(search_mode, SearchMode.WEB)

        # Add API-specific context if needed
        if search_mode == "api":
            context = context or {}
            context["require_official_docs"] = True

        results = self.engine.search(
            query=query,
            num_results=num_results,
            mode=mode,
            context=context,
            use_semantic_ranking=True,
            use_sap_validation=True
        )

        return results

    def search_and_learn(
        self,
        query: str,
        num_results: int = 10,
        context: Optional[Dict[str, Any]] = None,
        extract_knowledge: bool = True,
        min_trust_score: float = 0.7
    ) -> Dict[str, Any]:
        """
        Search and extract knowledge for learning.

        Args:
            query: Search query
            num_results: Number of results to return
            context: Optional context
            extract_knowledge: Whether to extract knowledge entries
            min_trust_score: Minimum trust score for knowledge extraction

        Returns:
            Dictionary with results and extracted knowledge

        Example:
            >>> response = agent.search_and_learn(
            ...     "SAP S/4HANA OData API best practices"
            ... )
            >>> print(f"Found {len(response['results'])} results")
            >>> print(f"Extracted {len(response['knowledge'])} knowledge entries")
        """
        # Perform search
        results = self.search(
            query=query,
            num_results=num_results,
            context=context
        )

        response = {
            "query": query,
            "results": results,
            "result_count": len(results),
            "knowledge": []
        }

        # Extract knowledge if requested
        if extract_knowledge and results:
            knowledge_entries = self.engine.extract_knowledge(
                results,
                min_trust_score=min_trust_score
            )
            response["knowledge"] = [entry.to_dict() for entry in knowledge_entries]
            response["knowledge_count"] = len(knowledge_entries)

            logger.info(
                f"Extracted {len(knowledge_entries)} knowledge entries "
                f"from {len(results)} search results"
            )

        return response

    def search_sap_documentation(
        self,
        topic: str,
        doc_type: str = "api"
    ) -> List[Dict[str, Any]]:
        """
        Search SAP official documentation.

        Args:
            topic: Topic to search for
            doc_type: Documentation type (api, help, guide, etc.)

        Returns:
            List of SAP documentation results

        Example:
            >>> docs = agent.search_sap_documentation(
            ...     "Business Partner",
            ...     doc_type="api"
            ... )
        """
        return self.engine.search_sap_documentation(topic, doc_type)

    def verify_sap_information(
        self,
        claim: str,
        min_sources: int = 3
    ) -> Dict[str, Any]:
        """
        Verify SAP-related information using multiple sources.

        Args:
            claim: Claim to verify
            min_sources: Minimum number of confirming sources

        Returns:
            Verification result with confidence score

        Example:
            >>> result = agent.verify_sap_information(
            ...     "BAPI_VENDOR_GETDETAIL retrieves vendor master data"
            ... )
            >>> print(result["verified"], result["confidence"])
        """
        return self.engine.verify_fact(claim, min_sources)

    def lookup_sap_entity(
        self,
        entity_name: str,
        entity_type: str,
        attributes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Look up information about SAP entity (vendor, customer, etc.).

        Args:
            entity_name: Entity name
            entity_type: Entity type (vendor, customer, product, etc.)
            attributes: Specific attributes to find

        Returns:
            Entity information

        Example:
            >>> info = agent.lookup_sap_entity(
            ...     "ACME Corporation",
            ...     "vendor",
            ...     attributes=["tax_id", "address"]
            ... )
        """
        return self.engine.lookup_entity(entity_name, entity_type, attributes)

    def get_api_documentation(
        self,
        api_name: str,
        include_examples: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive API documentation.

        Args:
            api_name: API or BAPI name
            include_examples: Whether to include code examples

        Returns:
            API documentation with endpoints, parameters, and examples

        Example:
            >>> api_doc = agent.get_api_documentation("BAPI_VENDOR_GETDETAIL")
            >>> print(api_doc["endpoints"])
        """
        # Search for API documentation
        query = f"{api_name} API documentation parameters examples"
        context = {
            "require_official_docs": True,
            "require_sap_domain": True
        }

        results = self.search(
            query=query,
            num_results=10,
            context=context
        )

        # Extract knowledge
        knowledge = self.engine.extract_knowledge(results, min_trust_score=0.8)

        # Compile API documentation
        api_doc = {
            "api_name": api_name,
            "description": "",
            "endpoints": [],
            "parameters": [],
            "examples": [],
            "sources": []
        }

        for entry in knowledge:
            if entry.source_type == "api_documentation":
                # Extract API-specific metadata
                metadata = entry.metadata
                api_doc["endpoints"].extend(metadata.get("endpoints", []))
                api_doc["parameters"].extend(metadata.get("parameters", []))

                if include_examples:
                    api_doc["examples"].extend(metadata.get("code_snippets", []))

                api_doc["sources"].append({
                    "url": entry.source_url,
                    "title": entry.title,
                    "trust_score": entry.trust_score
                })

        return api_doc

    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive search statistics.

        Returns:
            Statistics including cache hit rate, provider status, etc.
        """
        return self.engine.get_statistics()

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of search system.

        Returns:
            Health status of all components
        """
        return self.engine.health_check()

    def clear_cache(self) -> None:
        """Clear all search caches."""
        self.engine.clear_cache()
        logger.info("Search cache cleared")

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Update agent configuration.

        Args:
            config: New configuration dictionary
        """
        self.config.update(config)
        logger.info("Agent configuration updated")


def create_web_search_agent(config: Optional[Dict[str, Any]] = None) -> WebSearchAgent:
    """
    Factory function to create web search agent.

    Args:
        config: Optional configuration dictionary

    Returns:
        Configured WebSearchAgent instance

    Example:
        >>> config = {
        ...     "providers": {
        ...         "serpapi": {"enabled": True, "api_key": "..."},
        ...         "brave": {"enabled": True, "api_key": "..."}
        ...     },
        ...     "cache": {"enabled": True},
        ...     "semantic_ranking": {"use_gpu": False}
        ... }
        >>> agent = create_web_search_agent(config)
    """
    return WebSearchAgent(config)
