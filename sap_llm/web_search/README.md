# Web Search Module

Enterprise-grade web search and internet connectivity for SAP_LLM system, enabling real-time information retrieval, entity enrichment, and validation.

## Features

### 🔍 Multi-Provider Search
- **Google Custom Search API** - High-quality results with Google's index
- **Bing Search API** - Microsoft's search engine integration
- **Tavily AI Search** - AI-optimized search for LLM applications
- **DuckDuckGo** - No API key required, always available as fallback

### ⚡ Performance & Reliability
- **Redis-based caching** - Avoid redundant API calls
- **Automatic failover** - Switches providers on errors
- **Rate limiting** - Per-minute and per-day limits
- **Compression** - Automatic compression of large results
- **Offline mode** - Graceful degradation without internet

### 🛡️ Safety & Security
- **Domain whitelisting** - Boost trusted sources
- **Domain blacklisting** - Filter out untrusted sources
- **Result sanitization** - Clean and validate all results
- **URL validation** - Ensure valid URLs
- **Rate limiting** - Prevent API quota exhaustion

### 📊 Entity Enrichment
- **Vendor verification** - Validate company information
- **Address validation** - Verify and normalize addresses
- **Tax ID validation** - Check VAT/Tax ID formats
- **IBAN validation** - Validate bank account numbers
- **Price validation** - Compare against market prices
- **Exchange rates** - Real-time currency conversion

### 🔗 Pipeline Integration
- **Validation stage** - Verify extracted entities
- **Routing stage** - Discover SAP API endpoints
- **Quality check** - Cross-reference with external sources
- **Knowledge base** - Auto-update with latest documentation

## Architecture

```
sap_llm/web_search/
├── __init__.py                 # Package exports
├── search_engine.py            # Main search engine
├── search_providers.py         # Provider implementations
├── cache_manager.py            # Redis-based caching
├── rate_limiter.py             # Rate limiting
├── result_processor.py         # Result ranking & validation
├── entity_enrichment.py        # Entity enrichment & validation
└── integrations.py             # Pipeline stage integrations
```

## Quick Start

### 1. Install Dependencies

```bash
pip install redis requests pydantic pyyaml
# Optional for DuckDuckGo
pip install duckduckgo-search
```

### 2. Configure API Keys

Set environment variables for your chosen providers:

```bash
# Google Custom Search
export GOOGLE_SEARCH_API_KEY="your-api-key"
export GOOGLE_SEARCH_CX="your-cx-id"

# Bing Search
export BING_SEARCH_API_KEY="your-api-key"

# Tavily AI Search
export TAVILY_API_KEY="your-api-key"

# Redis (optional, uses in-memory fallback)
export REDIS_HOST="localhost"
export REDIS_PASSWORD="your-password"
```

### 3. Update Configuration

Edit `configs/default_config.yaml`:

```yaml
web_search:
  enabled: true
  offline_mode: false  # Set true to disable internet

  providers:
    google:
      enabled: true
    tavily:
      enabled: true
    duckduckgo:
      enabled: true  # Always available
```

### 4. Basic Usage

```python
from sap_llm.config import load_config
from sap_llm.web_search import WebSearchEngine

# Load configuration
config = load_config()
web_config = config.web_search.model_dump()

# Initialize search engine
engine = WebSearchEngine(web_config)

# Perform search
results = engine.search("SAP S/4HANA API documentation", num_results=10)

for result in results:
    print(f"{result['title']}: {result['url']}")
```

## Usage Examples

### Entity Enrichment

```python
from sap_llm.web_search import EntityEnricher

enricher = EntityEnricher(engine)

# Enrich vendor information
vendor_info = enricher.enrich_vendor(
    vendor_name="ACME Corporation",
    country="Germany"
)

print(f"Verified: {vendor_info['verified']}")
print(f"Confidence: {vendor_info['confidence']}")
print(f"Address: {vendor_info['extracted_data']['addresses']}")
print(f"Tax IDs: {vendor_info['extracted_data']['tax_ids']}")
```

### Address Validation

```python
# Validate address
validation = enricher.validate_address(
    address="1234 Main Street, New York, NY 10001",
    country="USA"
)

print(f"Valid: {validation['valid']}")
print(f"Normalized: {validation['normalized_address']}")
```

### Tax ID Validation

```python
# Validate tax ID
tax_validation = enricher.validate_tax_id(
    tax_id="DE123456789",
    country="DE",
    company_name="ACME GmbH"
)

print(f"Format valid: {tax_validation['format_valid']}")
print(f"Web verified: {tax_validation['web_verified']}")
```

### IBAN Validation

```python
# Validate IBAN
iban_validation = enricher.validate_iban("DE89 3704 0044 0532 0130 00")

print(f"Valid: {iban_validation['valid']}")
print(f"Country: {iban_validation['country_code']}")
print(f"Checksum: {iban_validation['checksum_valid']}")
```

### Invoice Validation

```python
from sap_llm.web_search.integrations import ValidationEnhancer

validator = ValidationEnhancer(engine)

invoice_data = {
    "vendor_name": "ACME Corp",
    "vendor_address": "123 Main St, Berlin",
    "vendor_country": "Germany",
    "vendor_tax_id": "DE123456789",
    "line_items": [
        {"description": "Laptop", "unit_price": 1200, "quantity": 1}
    ]
}

validation = validator.validate_invoice_data(invoice_data)

print(f"Valid: {validation['overall_valid']}")
print(f"Confidence: {validation['confidence']}")
```

### SAP API Discovery

```python
from sap_llm.web_search.integrations import RoutingEnhancer

router = RoutingEnhancer(engine)

# Find SAP API for creating invoices
api_info = router.find_sap_api(
    document_type="SUPPLIER_INVOICE",
    operation="CREATE",
    sap_system="S/4HANA"
)

print(f"Found: {api_info['found']}")
for endpoint in api_info['endpoints']:
    print(f"  {endpoint['type']}: {endpoint['name']}")
```

### Market Price Lookup

```python
# Get market price
price_info = enricher.get_market_price(
    product_name="iPhone 15 Pro",
    quantity=1,
    unit="piece"
)

print(f"Found: {price_info['found']}")
print(f"Avg Price: ${price_info.get('avg_price', 0):.2f}")
print(f"Range: ${price_info.get('min_price', 0):.2f} - ${price_info.get('max_price', 0):.2f}")
```

### Exchange Rate Lookup

```python
# Get exchange rate
rate = engine.get_exchange_rate("USD", "EUR")

print(f"1 USD = {rate} EUR")
```

### SAP Documentation Search

```python
# Search SAP official documentation
sap_docs = engine.search_sap_documentation(
    topic="purchase order BAPI",
    doc_type="api"
)

for doc in sap_docs:
    print(f"{doc['title']}: {doc['url']}")
```

## Pipeline Integration

### Validation Stage

```python
from sap_llm.stages.validation import ValidationStage
from sap_llm.web_search.integrations import ValidationEnhancer

# In your ValidationStage class
class EnhancedValidationStage(ValidationStage):
    def __init__(self, config):
        super().__init__(config)

        # Initialize web validation
        if config.web_search.integrations.validation.enabled:
            self.web_validator = ValidationEnhancer(
                WebSearchEngine(config.web_search.model_dump())
            )

    def validate(self, data):
        # Standard validation
        result = super().validate(data)

        # Web-enhanced validation
        if hasattr(self, 'web_validator'):
            web_result = self.web_validator.validate_vendor_data(
                vendor_name=data['vendor_name'],
                address=data.get('vendor_address'),
                tax_id=data.get('vendor_tax_id')
            )

            # Merge results
            result['web_validation'] = web_result

        return result
```

### Routing Stage

```python
from sap_llm.web_search.integrations import RoutingEnhancer

# Find correct SAP API endpoint
router = RoutingEnhancer(engine)

api_info = router.find_sap_api(
    document_type=extracted_data['document_type'],
    operation="CREATE"
)

# Use discovered endpoint
if api_info['found']:
    endpoint = api_info['endpoints'][0]
    # Route to endpoint...
```

### Quality Check Stage

```python
from sap_llm.web_search.integrations import QualityCheckEnhancer

checker = QualityCheckEnhancer(engine)

# Verify extracted data
verification = checker.verify_extracted_data(
    extracted_data=data,
    document_type=doc_type
)

if not verification['verified']:
    # Flag for human review
    handle_verification_failure(verification['issues'])
```

## Configuration

### Full Configuration Example

```yaml
web_search:
  enabled: true
  offline_mode: false
  cache_enabled: true

  providers:
    google:
      enabled: true
      api_key: "${GOOGLE_SEARCH_API_KEY}"
      cx: "${GOOGLE_SEARCH_CX}"

    bing:
      enabled: true
      api_key: "${BING_SEARCH_API_KEY}"

    tavily:
      enabled: true
      api_key: "${TAVILY_API_KEY}"

    duckduckgo:
      enabled: true

  provider_priority:
    - tavily
    - google
    - bing
    - duckduckgo

  rate_limits:
    google: 100           # Requests per minute
    google_daily: 10000   # Requests per day
    bing: 100
    bing_daily: 10000
    tavily: 60
    tavily_daily: 1000
    duckduckgo: 30
    duckduckgo_daily: 1000

  cache:
    host: "localhost"
    port: 6379
    db: 1
    ttl: 86400  # 24 hours
    compress_threshold: 1024

  trusted_domains:
    - sap.com
    - help.sap.com
    - api.sap.com
    - wikipedia.org

  blocked_domains:
    - spam.com
    - example.com

  min_relevance_score: 0.3

  features:
    vendor_verification: true
    address_validation: true
    tax_id_validation: true
    price_validation: true
    exchange_rates: true
    sap_doc_search: true

  integrations:
    validation:
      enabled: true
      verify_vendors: true
      verify_addresses: true
    routing:
      enabled: true
      api_discovery: true
    quality_check:
      enabled: true
      fact_verification: true
```

## Monitoring & Statistics

```python
# Get search engine statistics
stats = engine.get_statistics()

print(f"Total searches: {stats['total_searches']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Avg response time: {stats['avg_response_time_ms']:.2f}ms")
print(f"Provider failures: {stats['provider_failures']}")

# Get cache statistics
cache_stats = engine.cache_manager.get_stats()
print(f"Cache hits: {cache_stats['hits']}")
print(f"Cache misses: {cache_stats['misses']}")
print(f"Hit rate: {cache_stats['hit_rate']:.2%}")

# Get rate limiter statistics
rate_stats = engine.rate_limiters['google'].get_stats()
print(f"Requests today: {rate_stats['current_daily_count']}")
print(f"Requests this minute: {rate_stats['current_minute_count']}")
print(f"Block rate: {rate_stats['block_rate']:.2%}")
```

## Performance Optimization

### 1. Enable Caching

```python
# Search with caching (default)
results = engine.search("query", use_cache=True)

# Search bypassing cache
results = engine.search("query", use_cache=False)

# Clear cache
engine.clear_cache()
```

### 2. Batch Searches

```python
# Search multiple queries in parallel
queries = ["query1", "query2", "query3"]
results = engine.search_multiple(queries, parallel=True)

for query, query_results in results.items():
    print(f"{query}: {len(query_results)} results")
```

### 3. Adjust Cache TTL

```yaml
cache:
  ttl: 3600  # Cache for 1 hour (shorter for frequently changing data)
  ttl: 86400  # Cache for 24 hours (longer for stable data)
  ttl: 604800  # Cache for 1 week (very stable data)
```

## Error Handling

```python
try:
    results = engine.search("query")
except Exception as e:
    # All providers failed
    logger.error(f"Search failed: {e}")

    # Check if offline mode is needed
    if should_enable_offline_mode():
        engine.offline_mode = True
```

## Testing

Run the example file to test all functionality:

```bash
python examples/web_search_example.py
```

## API Reference

See individual module docstrings for detailed API documentation:
- `search_engine.py` - Main search engine API
- `entity_enrichment.py` - Entity enrichment methods
- `integrations.py` - Pipeline integration helpers

## Troubleshooting

### No results returned
- Check API keys are correctly configured
- Verify internet connectivity
- Check rate limits haven't been exceeded
- Try different providers

### Slow searches
- Enable caching
- Check Redis connection
- Reduce number of results
- Use Tavily (optimized for LLMs)

### Cache not working
- Verify Redis is running
- Check Redis connection settings
- Ensure cache is enabled in config

### Provider failures
- Check API keys
- Verify API quotas
- Check provider status pages
- Enable fallback providers

## Production Checklist

- [ ] Configure API keys for at least 2 providers
- [ ] Set up Redis for production caching
- [ ] Configure appropriate rate limits
- [ ] Set up trusted/blocked domain lists
- [ ] Enable monitoring and logging
- [ ] Test failover scenarios
- [ ] Configure offline mode fallback
- [ ] Set appropriate cache TTLs
- [ ] Review and adjust relevance thresholds
- [ ] Test all integration points

## License

Part of SAP_LLM system. See main repository for license information.
