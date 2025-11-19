# Web Search API Setup Guide

Complete guide for configuring and testing web search integration in SAP_LLM.

**Part of:** TODO #9 - Web Search API Configuration & Integration Testing
**Status:** Implementation Complete - Configuration Required
**Estimated Setup Time:** 30-60 minutes

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [API Key Setup](#api-key-setup)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Cost Optimization](#cost-optimization)

---

## Overview

SAP_LLM uses a multi-provider web search system with automatic failover:

1. **Tavily AI** (Primary) - AI-powered search, best results
2. **Google Custom Search** (Secondary) - Reliable fallback
3. **Bing Search API** (Tertiary) - Additional fallback
4. **DuckDuckGo** (Fallback) - Free, no API key required

### Features

✓ **Automatic Failover** - Switches providers on failure
✓ **3-Tier Caching** - Memory → Redis → Disk (>80% hit rate)
✓ **Entity Enrichment** - Vendor, product, tax rate validation
✓ **Cost Optimization** - <$0.001/document target
✓ **Rate Limiting** - Prevents API quota exhaustion

---

## Prerequisites

- SAP_LLM installed and configured
- Internet connection
- Email accounts for API signups
- Credit card (optional, for paid tiers)

---

## API Key Setup

### 1. Tavily AI (Recommended - Primary Provider)

**Why Tavily?**
- AI-powered results with high relevance
- Best for SAP documentation and vendor lookup
- Affordable: $0.001 per request
- Free tier: 1,000 requests/month

**Setup Steps:**

1. Visit [https://tavily.com](https://tavily.com)
2. Click "Sign Up" and create account
3. Navigate to Dashboard → API Keys
4. Copy your API key (format: `tvly-xxxxxxxxxx`)
5. Add to `.env`:
   ```bash
   TAVILY_API_KEY=tvly-your-actual-key-here
   ```

**Free Tier Limits:**
- 1,000 requests/month
- Rate limit: 10 requests/second
- Adequate for: Development, testing, <50K docs/month

**Paid Tier:**
- $0.001/request ($1 per 1,000 requests)
- 1M requests = $1,000/month
- Best for: Production with >50K docs/month

---

### 2. Google Custom Search (Recommended - Secondary Provider)

**Why Google?**
- Most reliable fallback option
- Excellent for general web search
- 100 free queries/day

**Setup Steps:**

1. **Get API Key:**
   - Visit [Google Cloud Console](https://console.cloud.google.com/)
   - Create new project or select existing
   - Enable "Custom Search API"
   - Navigate to Credentials → Create API Key
   - Copy API key (format: `AIzaSy...`)

2. **Create Custom Search Engine:**
   - Visit [Programmable Search Engine](https://programmablesearchengine.google.com/)
   - Click "Add" to create new search engine
   - Settings:
     - Name: "SAP LLM Web Search"
     - Search the entire web: ON
     - SafeSearch: OFF (optional)
   - Copy Search Engine ID (format: `xxxxxxxxxx`)

3. **Add to `.env`:**
   ```bash
   GOOGLE_SEARCH_API_KEY=AIzaSy-your-actual-key-here
   GOOGLE_SEARCH_CX=your-search-engine-id-here
   ```

**Free Tier Limits:**
- 100 queries/day
- Good for: Development, testing

**Paid Tier:**
- $5 per 1,000 queries ($0.005/query)
- 10,000 queries/day limit
- Good for: Production fallback

---

### 3. Bing Search API (Optional - Tertiary Provider)

**Why Bing?**
- Additional fallback option
- Good for Microsoft/enterprise content
- Free tier: 1,000 queries/month

**Setup Steps:**

1. Visit [Azure Portal](https://portal.azure.com/)
2. Create "Bing Search v7" resource
3. Select pricing tier (F1 for free)
4. Copy API key from Keys section
5. Add to `.env`:
   ```bash
   BING_SEARCH_API_KEY=your-bing-key-here
   ```

**Free Tier Limits:**
- 1,000 transactions/month
- 3 transactions/second
- Good for: Fallback only

**Paid Tier:**
- S1: $5 per 1,000 queries
- Good for: Additional redundancy

---

### 4. DuckDuckGo (Built-in - No Setup Required)

**Why DuckDuckGo?**
- No API key required
- Free forever
- Privacy-focused

**Limitations:**
- Rate limited (~30 requests/minute)
- Lower result quality
- Use only as emergency fallback

**Configuration:**
```bash
# No API key needed - works out of the box
# Automatically used when other providers fail
```

---

## Configuration

### Basic Configuration

Edit your `.env` file:

```bash
# Enable web search
WEB_SEARCH_ENABLED=true

# Provider API keys
TAVILY_API_KEY=tvly-your-key
GOOGLE_SEARCH_API_KEY=AIzaSy-your-key
GOOGLE_SEARCH_CX=your-cx-id
BING_SEARCH_API_KEY=your-bing-key  # Optional

# Web search settings
WEB_SEARCH_OFFLINE_MODE=false
WEB_SEARCH_CACHE_TTL=86400  # 24 hours
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TIMEOUT=10  # seconds
WEB_SEARCH_RETRY_ATTEMPTS=3

# Entity enrichment
ENABLE_ENTITY_ENRICHMENT=true
ENTITY_ENRICHMENT_CONFIDENCE_THRESHOLD=0.7
```

### Advanced Configuration

For production deployments, configure in `configs/default_config.yaml`:

```yaml
web_search:
  enabled: true
  offline_mode: false

  # Provider priority (failover order)
  providers:
    - tavily      # Primary
    - google      # Secondary
    - bing        # Tertiary
    - duckduckgo  # Fallback

  # Cache configuration
  cache:
    enabled: true
    ttl: 86400  # 24 hours
    max_size: 10000  # entries
    backend: redis  # or "memory", "disk"

  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst: 10

  # Cost optimization
  cost_optimization:
    enable_caching: true
    cache_hit_target: 0.8  # 80%
    max_cost_per_document: 0.001  # $0.001
```

---

## Testing

### Quick Test

Run the integration test script:

```bash
# Test all providers
python scripts/test_web_search_integration.py

# Test specific provider
python scripts/test_web_search_integration.py --provider tavily

# Test failover behavior
python scripts/test_web_search_integration.py --test-failover

# Test caching
python scripts/test_web_search_integration.py --test-cache

# Test entity enrichment
python scripts/test_web_search_integration.py --test-enrichment
```

### Expected Output

```
Testing All Search Providers

✓ Tavily: 5 results in 245.32ms
✓ Google: 5 results in 389.21ms
✓ Bing: 5 results in 412.18ms
✓ Duckduckgo: 5 results in 523.45ms

Testing Cache Performance

Making first request (should be cache miss)...
Making second request (should be cache hit)...

Cache miss latency: 245.32ms
Cache hit latency: 12.45ms
Speedup: 19.7x
✓ Cache working correctly!

=================================================================
Web Search Integration Test Summary
=================================================================

Provider Test Results
┏━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Provider    ┃ Status ┃ Results ┃ Latency (ms) ┃ Error ┃
┡━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━┩
│ Tavily      │ ✓ PASS │ 5       │ 245.32       │       │
│ Google      │ ✓ PASS │ 5       │ 389.21       │       │
│ Bing        │ ✓ PASS │ 5       │ 412.18       │       │
│ Duckduckgo  │ ✓ PASS │ 5       │ 523.45       │       │
└─────────────┴────────┴─────────┴──────────────┴───────┘

Overall: 4/4 providers working
✓ Web search is operational (multiple providers working)
```

### Manual Testing

Test in Python:

```python
import asyncio
from sap_llm.web_search import WebSearchEngine
from sap_llm.config import Config

async def test_search():
    config = Config()
    engine = WebSearchEngine(config)

    # Test basic search
    results = await engine.search(
        query="SAP S/4HANA purchase order API",
        mode="web",
        max_results=5
    )

    print(f"Found {len(results)} results")
    for r in results:
        print(f"- {r['title']}: {r['url']}")

asyncio.run(test_search())
```

---

## Troubleshooting

### Common Issues

#### ❌ "API key not configured"

**Solution:**
1. Check `.env` file has correct API key
2. Verify environment variable is loaded: `echo $TAVILY_API_KEY`
3. Restart application to reload environment

#### ❌ "Authentication failed"

**Solution:**
1. Verify API key is correct (no extra spaces)
2. Check API key is active in provider dashboard
3. Ensure billing is set up (for paid tiers)

#### ❌ "Rate limit exceeded"

**Solution:**
1. Enable caching: `WEB_SEARCH_CACHE_TTL=86400`
2. Reduce `WEB_SEARCH_MAX_RESULTS`
3. Upgrade to paid tier
4. Add more providers for failover

#### ❌ "No results returned"

**Solution:**
1. Check internet connectivity
2. Verify query is not empty
3. Try different provider: `--provider google`
4. Check provider status page

#### ❌ "Slow performance"

**Solution:**
1. Enable Redis caching
2. Reduce timeout: `WEB_SEARCH_TIMEOUT=5`
3. Lower `WEB_SEARCH_MAX_RESULTS`
4. Check network latency

---

## Cost Optimization

### Estimated Costs

**Development (10K docs/month):**
- Tavily free tier: $0
- Google free tier: $0
- **Total: $0/month**

**Small Production (100K docs/month):**
- Tavily: ~10K searches × $0.001 = $10/month
- Cache hit rate: 80% → 2K actual searches = $2/month
- **Total: ~$2-5/month**

**Large Production (1M docs/month):**
- Tavily: ~100K searches × $0.001 = $100/month
- Cache hit rate: 85% → 15K actual searches = $15/month
- **Total: ~$15-30/month**

### Optimization Strategies

1. **Enable Aggressive Caching**
   ```bash
   WEB_SEARCH_CACHE_TTL=604800  # 7 days
   ```

2. **Reduce Search Frequency**
   - Only search for unknown vendors
   - Cache vendor/product databases locally
   - Batch similar queries

3. **Use Free Tiers Effectively**
   - DuckDuckGo for development
   - Google free tier (100/day) for testing
   - Tavily free tier (1K/month) for small deployments

4. **Monitor Usage**
   ```bash
   # Check cache hit rate
   python scripts/web_search_performance_report.py
   ```

---

## Integration with Pipeline

### Automatic Entity Enrichment

Web search automatically enriches entities during extraction:

```python
# In extraction stage
extracted_data = {
    "vendor_name": "Acme Corporation",
    "vendor_address": None,  # Missing
    "vendor_vat": None,      # Missing
}

# Web search enrichment (automatic if enabled)
if ENABLE_ENTITY_ENRICHMENT:
    enriched = await entity_enrichment.enrich_vendor(
        vendor_name="Acme Corporation"
    )

    # Merged result
    {
        "vendor_name": "Acme Corporation",
        "vendor_address": "123 Main St, New York, NY",  # Enriched
        "vendor_vat": "US123456789",                    # Enriched
        "enrichment_confidence": 0.87
    }
```

### Knowledge Base Updates

Weekly SAP API documentation updates:

```bash
# Scheduled cron job
0 2 * * 0 python -m sap_llm.knowledge_base.updater
```

---

## Security Best Practices

1. **Never commit API keys** to git
2. **Use environment variables** or secrets manager
3. **Rotate keys** every 90 days
4. **Monitor usage** for anomalies
5. **Set budget alerts** in provider dashboards
6. **Use read-only keys** when available

---

## Next Steps

After configuration:

1. ✅ Run integration tests
2. ✅ Verify all providers working
3. ✅ Test failover behavior
4. ✅ Validate entity enrichment
5. ✅ Monitor cache hit rate (target >80%)
6. ✅ Set up cost alerts
7. ✅ Document any custom configuration

---

## Support

**Issues?**
- Check logs: `tail -f logs/web_search.log`
- Review test output: `python scripts/test_web_search_integration.py`
- GitHub Issues: [SAP_LLM/issues](https://github.com/qorsync/sap-llm/issues)

**Cost Concerns?**
- Email: ai@qorsync.com
- Slack: #sap-llm-support

---

**Last Updated:** 2025-11-18
**Version:** 1.0.0
**Status:** ✅ Ready for Configuration
