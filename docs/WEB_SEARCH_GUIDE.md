# Multi-Provider Web Search System Guide

## Overview

The SAP_LLM Multi-Provider Web Search System provides enterprise-grade search capabilities with intelligent query refinement, semantic ranking, SAP-specific validation, and automated knowledge extraction.

## Features

### Core Capabilities
- **6+ Search Providers**: SerpAPI, Brave, Google, Bing, Tavily, DuckDuckGo
- **Automatic Failover**: Seamless switching between providers on failure
- **3-Tier Caching**: L1 (memory), L2 (Redis), L3 (disk) for optimal performance
- **Semantic Ranking**: Embedding-based result ranking using SentenceTransformers
- **SAP Domain Expertise**: Query refinement with SAP terminology and validation
- **Trust Scoring**: Multi-factor validation of source credibility
- **Knowledge Extraction**: Automated structured knowledge extraction from results

### Advanced Features
- Context-aware query refinement
- Semantic duplicate detection
- Rate limiting per provider
- Result diversity scoring
- Official SAP documentation prioritization
- Automated knowledge base updates

## Quick Start

### Basic Usage

```python
from sap_llm.agents.web_search_agent import WebSearchAgent

# Initialize agent
config = {
    "providers": {
        "serpapi": {
            "enabled": True,
            "api_key": "your_serpapi_key"
        },
        "brave": {
            "enabled": True,
            "api_key": "your_brave_key"
        }
    }
}

agent = WebSearchAgent(config)

# Simple search
results = agent.search("SAP BAPI vendor master data")

for result in results:
    print(f"{result['title']} (trust: {result['trust_score']:.2f})")
    print(f"  {result['url']}")
```

### Search with Context

```python
# Context-aware search
results = agent.search(
    "How to get invoice price?",
    context={
        "document_type": "invoice",
        "module": "MM",
        "require_sap_domain": True
    }
)
```

### Search and Learn

```python
# Extract knowledge from search results
response = agent.search_and_learn(
    "SAP OData API authentication methods",
    extract_knowledge=True,
    min_trust_score=0.8
)

print(f"Found {response['result_count']} results")
print(f"Extracted {response['knowledge_count']} knowledge entries")

for entry in response['knowledge']:
    print(f"  - {entry['title']} ({entry['source_type']})")
```

## Configuration

### Complete Configuration Example

```python
config = {
    # Enable/disable search
    "enabled": True,
    "offline_mode": False,

    # Provider configuration
    "providers": {
        "serpapi": {
            "enabled": True,
            "api_key": "your_serpapi_key"
        },
        "brave": {
            "enabled": True,
            "api_key": "your_brave_key"
        },
        "google": {
            "enabled": False,
            "api_key": "your_google_key",
            "cx": "your_custom_search_engine_id"
        },
        "bing": {
            "enabled": False,
            "api_key": "your_bing_key"
        },
        "tavily": {
            "enabled": False,
            "api_key": "your_tavily_key"
        },
        "duckduckgo": {
            "enabled": True  # No API key required
        }
    },

    # Provider priority (order matters)
    "provider_priority": ["serpapi", "brave", "tavily", "google", "bing", "duckduckgo"],

    # Rate limits per provider (requests per minute / day)
    "rate_limits": {
        "serpapi": 100,
        "serpapi_daily": 10000,
        "brave": 60,
        "brave_daily": 2000,
        "google": 100,
        "google_daily": 10000,
        "bing": 100,
        "bing_daily": 10000,
        "tavily": 60,
        "tavily_daily": 1000,
        "duckduckgo": 30,
        "duckduckgo_daily": 1000
    },

    # Cache configuration
    "cache": {
        "enabled": True,
        "ttl": 86400,  # 24 hours
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "disk_cache_dir": "/tmp/sap_llm_cache",
        "max_disk_cache_size_mb": 1000
    },

    # Semantic ranking configuration
    "semantic_ranking": {
        "model_name": "all-MiniLM-L6-v2",
        "use_gpu": False,
        "batch_size": 32
    },

    # SAP validator configuration
    "sap_validator": {
        "min_trust_score": 0.5,
        "freshness_weight": 0.1,
        "require_https": True
    },

    # Knowledge extractor configuration
    "knowledge_extractor": {
        "fetch_full_content": False,
        "max_content_length": 10000
    },

    # Result filtering
    "trusted_domains": [
        "sap.com",
        "help.sap.com",
        "api.sap.com"
    ],
    "blocked_domains": [],
    "min_relevance_score": 0.5
}
```

## Components

### 1. Search Providers

#### SerpAPI Provider
```python
from sap_llm.web_search.search_providers import SerpAPIProvider

provider = SerpAPIProvider(api_key="your_key")
results = provider.search("SAP BAPI documentation", num_results=10)
```

**Features:**
- Google search results via API
- No Google API credentials needed
- Up to 100 results per request
- News and image search support

#### Brave Search Provider
```python
from sap_llm.web_search.search_providers import BraveSearchProvider

provider = BraveSearchProvider(api_key="your_key")
results = provider.search("SAP vendor management", num_results=20)
```

**Features:**
- Independent search index
- Privacy-focused
- Up to 20 results per request
- Freshness filters

### 2. Semantic Ranker

```python
from sap_llm.web_search.semantic_ranker import SemanticRanker

ranker = SemanticRanker(
    model_name="all-MiniLM-L6-v2",
    use_gpu=False
)

# Rank results by semantic similarity
ranked = ranker.rank_results(
    query="SAP vendor BAPI",
    results=search_results,
    context="Looking for API documentation"
)

# Remove semantic duplicates
unique = ranker.remove_semantic_duplicates(
    results,
    threshold=0.90
)

# Compute diversity score
diversity = ranker.compute_result_diversity(results)
print(f"Result diversity: {diversity:.2f}")
```

**Performance:**
- <100ms for 10 results (with cache)
- Embedding cache (LRU, 1000 entries)
- Batch processing support

### 3. Query Analyzer

```python
from sap_llm.web_search.query_analyzer import QueryAnalyzer

analyzer = QueryAnalyzer()

# Refine query with SAP terminology
refined = analyzer.refine_query(
    "How to get invoice price?",
    context={"document_type": "invoice", "module": "FI"}
)

print(f"Generated {len(refined)} query variations:")
for i, q in enumerate(refined):
    print(f"  {i+1}. {q}")

# Analyze query intent
intent = analyzer._analyze_intent("SAP BAPI error message")
print(f"Intent: {intent}")  # "troubleshoot"

# Extract SAP entities
entities = analyzer.extract_entities("Use ME21N transaction in MM module")
print(f"Transactions: {entities['transactions']}")  # ['ME21N']
print(f"Modules: {entities['modules']}")  # ['MM']

# Suggest search domains
domains = analyzer.suggest_search_domains("SAP API documentation")
print(f"Suggested domains: {domains}")  # ['api.sap.com', 'developers.sap.com', ...]
```

**SAP Knowledge Base:**
- 50+ SAP term mappings
- 10+ SAP module mappings
- Transaction code patterns
- Intent detection patterns

### 4. SAP Validator

```python
from sap_llm.web_search.sap_validator import SAPSourceValidator

validator = SAPSourceValidator(
    min_trust_score=0.6,
    freshness_weight=0.1,
    require_https=True
)

# Validate search results
validated = validator.validate_results(
    search_results,
    require_sap_domain=True
)

for result in validated:
    print(f"{result['title']}")
    print(f"  Trust: {result['trust_score']:.2f} ({result['trust_level']})")
    print(f"  Official: {result['trust_metadata']['is_official']}")
    print(f"  Domain: {result['trust_metadata']['domain_authority']:.2f}")

# Get trust summary
summary = validator.get_trust_summary(validated)
print(f"Average trust: {summary['avg_trust_score']:.2f}")
print(f"High trust: {summary['high_trust_percentage']:.1f}%")
```

**Trust Factors:**
- Domain authority (40% weight)
- Content type (30% weight)
- Official verification (20% weight)
- Freshness (10% weight)
- HTTPS bonus

**Trusted Domains (with authority levels):**
- help.sap.com: 1.0
- api.sap.com: 1.0
- developers.sap.com: 0.9
- community.sap.com: 0.85
- blogs.sap.com: 0.8

### 5. Knowledge Extractor

```python
from sap_llm.web_search.knowledge_extractor import KnowledgeExtractor

extractor = KnowledgeExtractor(
    fetch_full_content=True,
    max_content_length=10000
)

# Extract knowledge from search results
entries = extractor.extract_from_results(
    search_results,
    min_trust_score=0.7
)

for entry in entries:
    print(f"\n{entry.title}")
    print(f"  Type: {entry.source_type}")
    print(f"  Trust: {entry.trust_score:.2f}")
    print(f"  URL: {entry.source_url}")

    if entry.source_type == "api_documentation":
        print(f"  Endpoints: {len(entry.metadata.get('endpoints', []))}")
        print(f"  Parameters: {len(entry.metadata.get('parameters', []))}")

# Export knowledge to JSON
extractor.export_to_json(entries, "knowledge.json")
```

**Extraction Capabilities:**
- API endpoints and parameters
- Code snippets (Python, JSON, SQL, etc.)
- Tutorial steps
- Response codes
- Documentation structure

## Advanced Usage

### Multi-Query Search

```python
# Search multiple related queries
queries = [
    "SAP BAPI vendor master data",
    "SAP vendor information retrieval",
    "SAP supplier data API"
]

results = engine.search_multiple(queries, parallel=True)

for query, query_results in results.items():
    print(f"\n{query}: {len(query_results)} results")
```

### Fact Verification

```python
# Verify SAP-related claims
result = agent.verify_sap_information(
    claim="BAPI_VENDOR_GETDETAIL retrieves vendor master data",
    min_sources=3
)

print(f"Verified: {result['verified']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Sources: {result['confirming_sources']}")
```

### API Documentation Retrieval

```python
# Get comprehensive API documentation
api_doc = agent.get_api_documentation(
    api_name="BAPI_VENDOR_GETDETAIL",
    include_examples=True
)

print(f"API: {api_doc['api_name']}")
print(f"\nEndpoints:")
for endpoint in api_doc['endpoints']:
    print(f"  {endpoint['method']} {endpoint['path']}")

print(f"\nParameters:")
for param in api_doc['parameters']:
    print(f"  {param['name']}: {param['type']}")

print(f"\nExamples:")
for example in api_doc['examples']:
    print(f"  [{example['language']}]")
    print(f"  {example['code'][:100]}...")
```

### Search SAP Documentation

```python
# Search official SAP documentation only
docs = agent.search_sap_documentation(
    topic="Business Partner",
    doc_type="api"
)

for doc in docs:
    print(f"{doc['title']}")
    print(f"  {doc['url']}")
    print(f"  Trust: {doc['trust_score']:.2f}")
```

## Performance Optimization

### Caching Strategy

The system uses a 3-tier caching strategy:

1. **L1 (Memory)**: Fastest, volatile, 1000 entries
2. **L2 (Redis)**: Fast, shared, persistent
3. **L3 (Disk)**: Slowest, most persistent, 1GB default

**Cache Hit Rates:**
- Target: >80% overall
- L1: ~40-50%
- L2: ~30-35%
- L3: ~5-10%

### Rate Limiting

Configure per-provider rate limits:

```python
config = {
    "rate_limits": {
        "serpapi": 100,  # requests per minute
        "serpapi_daily": 10000,  # requests per day
        # ... other providers
    }
}
```

**Rate Limit Strategies:**
- Token bucket for burst handling
- Sliding window for accurate limits
- Automatic midnight reset

### GPU Acceleration

Enable GPU for semantic ranking:

```python
config = {
    "semantic_ranking": {
        "use_gpu": True,  # Requires CUDA
        "model_name": "all-MiniLM-L6-v2"
    }
}
```

**Performance Impact:**
- CPU: ~200ms for 10 results
- GPU: ~50ms for 10 results

## Monitoring and Statistics

### Get Search Statistics

```python
stats = agent.get_search_statistics()

print(f"Total searches: {stats['total_searches']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
print(f"Average response time: {stats['avg_response_time_ms']:.0f}ms")

# Provider status
for provider, status in stats['provider_status']['providers'].items():
    print(f"\n{provider}:")
    print(f"  Available: {status['available']}")
    print(f"  Failures: {status['failures']}")
    print(f"  Rate limited: {status.get('rate_limited', False)}")
```

### Health Check

```python
health = agent.get_health_status()

print(f"Overall healthy: {health['overall_healthy']}")

# Component health
for component, status in health.items():
    if isinstance(status, dict) and 'healthy' in status:
        print(f"\n{component}:")
        print(f"  Healthy: {status['healthy']}")
        if 'latency_ms' in status:
            print(f"  Latency: {status['latency_ms']:.0f}ms")
```

## Best Practices

### 1. API Key Management

```python
import os

config = {
    "providers": {
        "serpapi": {
            "enabled": True,
            "api_key": os.getenv("SERPAPI_KEY")
        }
    }
}
```

### 2. Context Usage

Always provide context for better results:

```python
results = agent.search(
    "vendor invoice processing",
    context={
        "document_type": "invoice",
        "module": "MM",
        "require_official_docs": True,
        "context_description": "Looking for SAP standard invoice posting procedures"
    }
)
```

### 3. Trust Score Thresholds

Use appropriate trust thresholds:

- **Critical operations**: 0.8+
- **General information**: 0.6+
- **Exploratory search**: 0.4+

```python
# High-trust results only
high_trust = [r for r in results if r['trust_score'] >= 0.8]
```

### 4. Error Handling

```python
try:
    results = agent.search("SAP vendor API")
    if not results:
        print("No results found")
except Exception as e:
    logger.error(f"Search failed: {e}")
    # Fallback to cached results or alternative source
```

### 5. Resource Cleanup

```python
# Clear cache periodically
agent.clear_cache()

# Or configure automatic cleanup
config = {
    "cache": {
        "max_disk_cache_size_mb": 1000,  # Auto-cleanup when exceeded
        "ttl": 86400  # 24-hour TTL
    }
}
```

## Troubleshooting

### Common Issues

#### 1. No Search Results

**Symptoms:** `search()` returns empty list

**Causes:**
- All providers rate-limited
- All providers failed
- Too restrictive filters

**Solutions:**
```python
# Check provider status
status = agent.get_health_status()
print(status['provider_status'])

# Reduce filtering
results = agent.search(
    query,
    use_sap_validation=False,  # Disable strict validation
    context={"require_sap_domain": False}
)
```

#### 2. Slow Search Performance

**Symptoms:** Search takes >5 seconds

**Causes:**
- Cache disabled or not working
- Semantic ranking on CPU
- Full content fetching enabled

**Solutions:**
```python
# Enable caching
config["cache"]["enabled"] = True

# Use GPU for semantic ranking
config["semantic_ranking"]["use_gpu"] = True

# Disable full content fetching
config["knowledge_extractor"]["fetch_full_content"] = False
```

#### 3. Low Trust Scores

**Symptoms:** All results have trust score <0.5

**Causes:**
- Non-SAP domains in results
- HTTP-only sources
- Community-only results

**Solutions:**
```python
# Adjust trust threshold
config["sap_validator"]["min_trust_score"] = 0.3

# Focus on official domains
context = {
    "require_sap_domain": True,
    "require_official_docs": True
}
```

#### 4. Semantic Ranking Not Working

**Symptoms:** No `semantic_score` in results

**Causes:**
- sentence-transformers not installed
- Model loading failed

**Solutions:**
```bash
pip install sentence-transformers

# Or use lighter model
config["semantic_ranking"]["model_name"] = "paraphrase-MiniLM-L3-v2"
```

## Requirements

### Dependencies

```bash
pip install requests beautifulsoup4 sentence-transformers redis numpy
```

### Optional Dependencies

```bash
# For DuckDuckGo support
pip install duckduckgo-search

# For GPU acceleration
pip install torch torchvision

# For enhanced HTML parsing
pip install lxml html5lib
```

## Performance Benchmarks

### Search Latency (10 results)

| Configuration | P50 | P95 | P99 |
|--------------|-----|-----|-----|
| With cache | 15ms | 50ms | 100ms |
| Without cache | 1.2s | 2.5s | 3.5s |
| With semantic ranking (CPU) | 300ms | 600ms | 900ms |
| With semantic ranking (GPU) | 100ms | 200ms | 350ms |

### Cache Hit Rates

| Cache Tier | Hit Rate | Latency |
|-----------|----------|---------|
| L1 (Memory) | 40-50% | <10ms |
| L2 (Redis) | 30-35% | 10-30ms |
| L3 (Disk) | 5-10% | 30-100ms |
| **Total** | **80-85%** | **<50ms** |

### Resource Usage

- Memory: ~200MB (baseline) + ~500MB (with models)
- Disk cache: Up to 1GB (configurable)
- Network: Depends on cache hit rate

## API Reference

See individual module documentation:
- `sap_llm.agents.web_search_agent` - Agent interface
- `sap_llm.web_search.search_engine` - Core search engine
- `sap_llm.web_search.semantic_ranker` - Semantic ranking
- `sap_llm.web_search.query_analyzer` - Query analysis
- `sap_llm.web_search.sap_validator` - Source validation
- `sap_llm.web_search.knowledge_extractor` - Knowledge extraction

## Support

For issues, questions, or contributions:
- GitHub Issues: [SAP_LLM Issues](https://github.com/yourusername/SAP_LLM/issues)
- Documentation: See `/docs` directory
- Examples: See `/examples/web_search` directory
