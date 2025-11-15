"""
Web Search Example for SAP_LLM.

Demonstrates how to use web search capabilities for entity enrichment,
validation, and real-time information retrieval.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sap_llm.config import load_config
from sap_llm.web_search import WebSearchEngine, EntityEnricher
from sap_llm.web_search.integrations import (
    ValidationEnhancer,
    RoutingEnhancer,
    QualityCheckEnhancer,
    KnowledgeBaseUpdater
)


def example_basic_search():
    """Example: Basic web search."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Web Search")
    print("="*80)

    # Load configuration
    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    # Initialize search engine
    engine = WebSearchEngine(web_config)

    # Perform search
    query = "SAP S/4HANA invoice API BAPI"
    print(f"\nSearching for: {query}")

    results = engine.search(query, num_results=5)

    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
        print(f"   Relevance: {result.get('relevance_score', 0):.2f}")


def example_vendor_enrichment():
    """Example: Vendor information enrichment."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Vendor Enrichment")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)
    enricher = EntityEnricher(engine)

    # Enrich vendor information
    vendor_name = "SAP SE"
    country = "Germany"

    print(f"\nEnriching vendor: {vendor_name} ({country})")

    vendor_info = enricher.enrich_vendor(
        vendor_name=vendor_name,
        country=country
    )

    print(f"\nVendor verified: {vendor_info['verified']}")
    print(f"Confidence: {vendor_info['confidence']:.2f}")
    print(f"\nExtracted data:")

    for key, values in vendor_info['extracted_data'].items():
        if values:
            print(f"  {key}: {values[:2]}")  # Show first 2 items


def example_address_validation():
    """Example: Address validation."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Address Validation")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)
    enricher = EntityEnricher(engine)

    # Validate address
    address = "Dietmar-Hopp-Allee 16, 69190 Walldorf, Germany"

    print(f"\nValidating address: {address}")

    validation = enricher.validate_address(
        address=address,
        country="Germany"
    )

    print(f"\nAddress valid: {validation['valid']}")
    print(f"Confidence: {validation['confidence']:.2f}")
    print(f"Components: {validation['components']}")


def example_tax_id_validation():
    """Example: Tax ID validation."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Tax ID Validation")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)
    enricher = EntityEnricher(engine)

    # Validate tax ID
    tax_id = "DE811234567"  # Example German VAT
    country = "DE"
    company = "SAP SE"

    print(f"\nValidating tax ID: {tax_id}")

    validation = enricher.validate_tax_id(
        tax_id=tax_id,
        country=country,
        company_name=company
    )

    print(f"\nFormat valid: {validation['format_valid']}")
    print(f"Web verified: {validation['web_verified']}")
    print(f"Confidence: {validation['confidence']:.2f}")


def example_iban_validation():
    """Example: IBAN validation."""
    print("\n" + "="*80)
    print("EXAMPLE 5: IBAN Validation")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)
    enricher = EntityEnricher(engine)

    # Validate IBAN
    iban = "DE89 3704 0044 0532 0130 00"

    print(f"\nValidating IBAN: {iban}")

    validation = enricher.validate_iban(iban)

    print(f"\nIBAN valid: {validation['valid']}")
    print(f"Country: {validation['country_code']}")
    print(f"Format valid: {validation['format_valid']}")
    print(f"Length valid: {validation['length_valid']}")
    print(f"Checksum valid: {validation['checksum_valid']}")


def example_invoice_validation():
    """Example: Complete invoice validation."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Invoice Data Validation")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)
    validator = ValidationEnhancer(engine, enabled=True)

    # Sample invoice data
    invoice_data = {
        "vendor_name": "SAP SE",
        "vendor_address": "Dietmar-Hopp-Allee 16, 69190 Walldorf",
        "vendor_country": "Germany",
        "vendor_tax_id": "DE811234567",
        "invoice_number": "INV-2024-001",
        "invoice_date": "2024-01-15",
        "total_amount": 10000.00,
        "currency": "EUR",
        "iban": "DE89370400440532013000"
    }

    print("\nValidating invoice data...")

    validation = validator.validate_invoice_data(invoice_data)

    print(f"\nOverall valid: {validation['overall_valid']}")
    print(f"Confidence: {validation['confidence']:.2f}")

    print("\nValidation details:")
    for key, result in validation['validations'].items():
        print(f"\n  {key}:")
        if isinstance(result, dict):
            if 'overall_valid' in result:
                print(f"    Valid: {result['overall_valid']}")
            if 'valid' in result:
                print(f"    Valid: {result['valid']}")
            if 'confidence' in result:
                print(f"    Confidence: {result.get('confidence', 0):.2f}")


def example_sap_api_discovery():
    """Example: SAP API endpoint discovery."""
    print("\n" + "="*80)
    print("EXAMPLE 7: SAP API Discovery")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)
    router = RoutingEnhancer(engine, enabled=True)

    # Find API for invoice posting
    doc_type = "SUPPLIER_INVOICE"
    operation = "CREATE"

    print(f"\nFinding SAP API for: {doc_type} - {operation}")

    api_info = router.find_sap_api(
        document_type=doc_type,
        operation=operation,
        sap_system="S/4HANA"
    )

    print(f"\nAPI found: {api_info['found']}")

    if api_info['endpoints']:
        print(f"\nDiscovered endpoints:")
        for endpoint in api_info['endpoints'][:5]:
            print(f"  - {endpoint['type']}: {endpoint['name']}")
            print(f"    Source: {endpoint['source'][:60]}...")


def example_price_validation():
    """Example: Market price validation."""
    print("\n" + "="*80)
    print("EXAMPLE 8: Market Price Validation")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)
    enricher = EntityEnricher(engine)

    # Get market price
    product = "Microsoft Surface Laptop"

    print(f"\nGetting market price for: {product}")

    price_info = enricher.get_market_price(
        product_name=product,
        quantity=1,
        unit="piece"
    )

    print(f"\nPrices found: {price_info['found']}")

    if price_info.get('prices'):
        print(f"\nPrice range:")
        print(f"  Min: ${price_info.get('min_price', 0):.2f}")
        print(f"  Avg: ${price_info.get('avg_price', 0):.2f}")
        print(f"  Max: ${price_info.get('max_price', 0):.2f}")


def example_exchange_rate():
    """Example: Currency exchange rate lookup."""
    print("\n" + "="*80)
    print("EXAMPLE 9: Exchange Rate Lookup")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)

    # Get exchange rate
    from_curr = "USD"
    to_curr = "EUR"

    print(f"\nGetting exchange rate: {from_curr} -> {to_curr}")

    rate = engine.get_exchange_rate(from_curr, to_curr)

    if rate:
        print(f"\nExchange rate: 1 {from_curr} = {rate:.4f} {to_curr}")
        print(f"Example: $100 USD = €{100 * rate:.2f} EUR")
    else:
        print("\nExchange rate not found")


def example_sap_documentation_search():
    """Example: Search SAP documentation."""
    print("\n" + "="*80)
    print("EXAMPLE 10: SAP Documentation Search")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)

    # Search SAP docs
    topic = "purchase order BAPI"

    print(f"\nSearching SAP documentation for: {topic}")

    results = engine.search_sap_documentation(
        topic=topic,
        doc_type="api"
    )

    print(f"\nFound {len(results)} SAP documentation results:")

    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['url']}")
        print(f"   {result['snippet'][:100]}...")


def example_statistics():
    """Example: Get search engine statistics."""
    print("\n" + "="*80)
    print("EXAMPLE 11: Search Engine Statistics")
    print("="*80)

    config = load_config()
    web_config = config.web_search.model_dump() if config.web_search else {}

    engine = WebSearchEngine(web_config)

    # Perform a few searches first
    engine.search("SAP BAPI", num_results=5)
    engine.search("SAP BAPI", num_results=5)  # Should hit cache
    engine.search("Invoice processing", num_results=5)

    # Get statistics
    stats = engine.get_statistics()

    print("\nSearch Engine Statistics:")
    print(f"  Total searches: {stats['total_searches']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.2%}")
    print(f"  Average response time: {stats['avg_response_time_ms']:.2f}ms")
    print(f"  Providers available: {', '.join(stats['providers_available'])}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("SAP_LLM Web Search Examples")
    print("="*80)

    examples = [
        ("Basic Search", example_basic_search),
        ("Vendor Enrichment", example_vendor_enrichment),
        ("Address Validation", example_address_validation),
        ("Tax ID Validation", example_tax_id_validation),
        ("IBAN Validation", example_iban_validation),
        ("Invoice Validation", example_invoice_validation),
        ("SAP API Discovery", example_sap_api_discovery),
        ("Price Validation", example_price_validation),
        ("Exchange Rate", example_exchange_rate),
        ("SAP Documentation", example_sap_documentation_search),
        ("Statistics", example_statistics),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...")

    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
