"""
Web Search Module for SAP_LLM.

Provides multi-provider web search capabilities with 3-tier caching, rate limiting,
entity enrichment, and pipeline integration.
"""

from sap_llm.web_search.cache_manager import SearchCacheManager
from sap_llm.web_search.entity_enrichment import EntityEnricher
from sap_llm.web_search.integrations import (
    ExtractionEnhancer,
    KnowledgeBaseUpdater,
    QualityCheckEnhancer,
    RoutingEnhancer,
    ValidationEnhancer,
)
from sap_llm.web_search.knowledge_extractor import KnowledgeEntry, KnowledgeExtractor
from sap_llm.web_search.query_analyzer import QueryAnalyzer
from sap_llm.web_search.rate_limiter import RateLimiter
from sap_llm.web_search.result_processor import ResultProcessor
from sap_llm.web_search.sap_validator import SAPSourceValidator
from sap_llm.web_search.search_engine import SearchMode, WebSearchEngine
from sap_llm.web_search.search_providers import (
    BingSearchProvider,
    BraveSearchProvider,
    DuckDuckGoProvider,
    GoogleSearchProvider,
    SearchProvider,
    SerpAPIProvider,
    TavilySearchProvider,
)
from sap_llm.web_search.semantic_ranker import SemanticRanker
from sap_llm.web_search.deduplication import (
    deduplicate_results,
    Deduplicator,
    find_duplicates,
)

__all__ = [
    # Core
    "WebSearchEngine",
    "SearchMode",
    # Providers
    "SearchProvider",
    "SerpAPIProvider",
    "BraveSearchProvider",
    "GoogleSearchProvider",
    "BingSearchProvider",
    "TavilySearchProvider",
    "DuckDuckGoProvider",
    # Components
    "SearchCacheManager",
    "RateLimiter",
    "ResultProcessor",
    "EntityEnricher",
    # Advanced Components
    "SemanticRanker",
    "QueryAnalyzer",
    "SAPSourceValidator",
    "KnowledgeExtractor",
    "KnowledgeEntry",
    # Integrations
    "ExtractionEnhancer",
    "ValidationEnhancer",
    "RoutingEnhancer",
    "QualityCheckEnhancer",
    "KnowledgeBaseUpdater",
    # Deduplication
    "deduplicate_results",
    "Deduplicator",
    "find_duplicates",
]
