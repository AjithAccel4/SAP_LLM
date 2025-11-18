#!/usr/bin/env python3
"""
Web Search Integration Test Script

Tests all web search providers, failover behavior, caching, and entity enrichment.
Part of TODO #9: Web Search API Configuration & Integration Testing

Usage:
    python scripts/test_web_search_integration.py
    python scripts/test_web_search_integration.py --provider tavily
    python scripts/test_web_search_integration.py --test-failover
    python scripts/test_web_search_integration.py --test-enrichment
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Load environment variables
load_dotenv()

console = Console()


class WebSearchTester:
    """Comprehensive web search integration tester"""

    def __init__(self):
        self.results = {
            "tavily": {"success": False, "latency_ms": None, "result_count": 0, "error": None},
            "google": {"success": False, "latency_ms": None, "result_count": 0, "error": None},
            "bing": {"success": False, "latency_ms": None, "result_count": 0, "error": None},
            "duckduckgo": {"success": False, "latency_ms": None, "result_count": 0, "error": None},
        }
        self.cache_stats = {"hits": 0, "misses": 0, "hit_rate": 0.0}

    async def test_provider(self, provider: str, query: str = "OpenAI GPT-4") -> Dict:
        """Test a single search provider"""
        try:
            from sap_llm.web_search import WebSearchEngine
            from sap_llm.config import Config

            config = Config()
            search_engine = WebSearchEngine(config)

            start = time.time()
            results = await search_engine.search(
                query=query,
                mode="web",
                provider=provider,
                max_results=5
            )
            latency_ms = (time.time() - start) * 1000

            return {
                "success": True,
                "latency_ms": latency_ms,
                "result_count": len(results),
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "latency_ms": None,
                "result_count": 0,
                "error": str(e)
            }

    async def test_all_providers(self):
        """Test all search providers"""
        console.print("\n[bold cyan]Testing All Search Providers[/bold cyan]\n")

        test_query = "SAP S/4HANA invoice processing API"

        for provider in ["tavily", "google", "bing", "duckduckgo"]:
            with console.status(f"[bold green]Testing {provider.capitalize()}..."):
                result = await self.test_provider(provider, test_query)
                self.results[provider] = result

                if result["success"]:
                    console.print(
                        f"✓ {provider.capitalize()}: "
                        f"{result['result_count']} results in {result['latency_ms']:.2f}ms",
                        style="green"
                    )
                else:
                    console.print(
                        f"✗ {provider.capitalize()}: {result['error']}",
                        style="red"
                    )

    async def test_failover(self):
        """Test automatic failover behavior"""
        console.print("\n[bold cyan]Testing Failover Behavior[/bold cyan]\n")

        try:
            from sap_llm.web_search import WebSearchEngine
            from sap_llm.config import Config

            config = Config()
            search_engine = WebSearchEngine(config)

            # Save original Tavily key
            original_key = os.environ.get("TAVILY_API_KEY", "")

            # Set invalid key to force failover
            os.environ["TAVILY_API_KEY"] = "invalid_key_to_force_failover"

            console.print("Testing with invalid Tavily key (should failover to Google)...")

            results = await search_engine.search(
                query="Test failover query",
                mode="web",
                max_results=3
            )

            if len(results) > 0:
                console.print(
                    f"✓ Failover successful! Got {len(results)} results from fallback provider",
                    style="green"
                )
            else:
                console.print("✗ Failover failed - no results returned", style="red")

            # Restore original key
            if original_key:
                os.environ["TAVILY_API_KEY"] = original_key

        except Exception as e:
            console.print(f"✗ Failover test error: {e}", style="red")

    async def test_caching(self):
        """Test cache performance"""
        console.print("\n[bold cyan]Testing Cache Performance[/bold cyan]\n")

        try:
            from sap_llm.web_search import WebSearchEngine
            from sap_llm.config import Config

            config = Config()
            search_engine = WebSearchEngine(config)

            query = "SAP BAPI invoice create"

            # First request (cache miss)
            console.print("Making first request (should be cache miss)...")
            start = time.time()
            results1 = await search_engine.search(query=query, mode="web", max_results=5)
            latency1 = (time.time() - start) * 1000

            # Second request (cache hit)
            console.print("Making second request (should be cache hit)...")
            start = time.time()
            results2 = await search_engine.search(query=query, mode="web", max_results=5)
            latency2 = (time.time() - start) * 1000

            speedup = latency1 / latency2 if latency2 > 0 else 0

            console.print(f"\nCache miss latency: {latency1:.2f}ms", style="yellow")
            console.print(f"Cache hit latency: {latency2:.2f}ms", style="green")
            console.print(f"Speedup: {speedup:.1f}x", style="bold green")

            if latency2 < latency1 * 0.1:  # Cache should be at least 10x faster
                console.print("✓ Cache working correctly!", style="green")
            else:
                console.print("⚠ Cache may not be working optimally", style="yellow")

        except Exception as e:
            console.print(f"✗ Cache test error: {e}", style="red")

    async def test_entity_enrichment(self):
        """Test entity enrichment functionality"""
        console.print("\n[bold cyan]Testing Entity Enrichment[/bold cyan]\n")

        try:
            from sap_llm.web_search.entity_enrichment import EntityEnrichment
            from sap_llm.web_search import WebSearchEngine
            from sap_llm.config import Config

            config = Config()
            search_engine = WebSearchEngine(config)
            enrichment = EntityEnrichment(search_engine)

            # Test vendor enrichment
            console.print("Testing vendor enrichment...")
            vendor_info = await enrichment.enrich_entity(
                entity_name="Microsoft Corporation",
                entity_type="vendor"
            )

            if vendor_info:
                console.print(f"✓ Vendor: {vendor_info.get('name', 'N/A')}", style="green")
                console.print(f"  Address: {vendor_info.get('address', 'N/A')}")
                console.print(f"  Country: {vendor_info.get('country', 'N/A')}")
                console.print(f"  VAT: {vendor_info.get('vat_number', 'N/A')}")
            else:
                console.print("⚠ No vendor information found", style="yellow")

            # Test product enrichment
            console.print("\nTesting product enrichment...")
            product_info = await enrichment.enrich_entity(
                entity_name="Apple MacBook Pro M3",
                entity_type="product"
            )

            if product_info:
                console.print(f"✓ Product: {product_info.get('name', 'N/A')}", style="green")
                console.print(f"  Category: {product_info.get('category', 'N/A')}")
                console.print(f"  Price: {product_info.get('price', 'N/A')}")
            else:
                console.print("⚠ No product information found", style="yellow")

        except ImportError:
            console.print("⚠ Entity enrichment module not available", style="yellow")
        except Exception as e:
            console.print(f"✗ Entity enrichment test error: {e}", style="red")

    def print_summary(self):
        """Print test summary"""
        console.print("\n" + "="*70)
        console.print("[bold cyan]Web Search Integration Test Summary[/bold cyan]")
        console.print("="*70 + "\n")

        # Create results table
        table = Table(title="Provider Test Results")
        table.add_column("Provider", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Results", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Error", style="red")

        for provider, result in self.results.items():
            status = "✓ PASS" if result["success"] else "✗ FAIL"
            status_style = "green" if result["success"] else "red"

            table.add_row(
                provider.capitalize(),
                f"[{status_style}]{status}[/{status_style}]",
                str(result["result_count"]) if result["success"] else "0",
                f"{result['latency_ms']:.2f}" if result["latency_ms"] else "N/A",
                result["error"] or ""
            )

        console.print(table)

        # Overall status
        successful_providers = sum(1 for r in self.results.values() if r["success"])
        console.print(f"\n[bold]Overall: {successful_providers}/4 providers working[/bold]")

        if successful_providers >= 2:
            console.print("[green]✓ Web search is operational (multiple providers working)[/green]")
        elif successful_providers == 1:
            console.print("[yellow]⚠ Web search partially operational (only 1 provider working)[/yellow]")
        else:
            console.print("[red]✗ Web search not operational (no providers working)[/red]")

        # Recommendations
        console.print("\n[bold]Recommendations:[/bold]")
        for provider, result in self.results.items():
            if not result["success"]:
                if "API key" in str(result["error"]) or "authentication" in str(result["error"]).lower():
                    console.print(f"- Configure {provider.upper()}_API_KEY in .env file", style="yellow")
                elif result["error"]:
                    console.print(f"- Check {provider} configuration: {result['error']}", style="yellow")


async def main():
    """Main test runner"""
    import argparse

    parser = argparse.ArgumentParser(description="Test SAP_LLM Web Search Integration")
    parser.add_argument("--provider", choices=["tavily", "google", "bing", "duckduckgo"],
                       help="Test specific provider only")
    parser.add_argument("--test-failover", action="store_true",
                       help="Test failover behavior")
    parser.add_argument("--test-cache", action="store_true",
                       help="Test cache performance")
    parser.add_argument("--test-enrichment", action="store_true",
                       help="Test entity enrichment")
    parser.add_argument("--all", action="store_true",
                       help="Run all tests (default)")

    args = parser.parse_args()

    # Check if web search is enabled
    if os.getenv("WEB_SEARCH_ENABLED", "false").lower() != "true":
        console.print("[yellow]⚠ WEB_SEARCH_ENABLED is not set to true in .env[/yellow]")
        console.print("[yellow]  Set WEB_SEARCH_ENABLED=true to enable web search[/yellow]\n")

    tester = WebSearchTester()

    # Run selected tests
    if args.provider:
        console.print(f"\n[bold cyan]Testing {args.provider.capitalize()} Only[/bold cyan]\n")
        result = await tester.test_provider(args.provider)
        tester.results[args.provider] = result
    elif args.test_failover:
        await tester.test_failover()
    elif args.test_cache:
        await tester.test_caching()
    elif args.test_enrichment:
        await tester.test_entity_enrichment()
    else:
        # Run all tests
        await tester.test_all_providers()

        if any(r["success"] for r in tester.results.values()):
            await tester.test_caching()
            await tester.test_failover()
            await tester.test_enrichment()

    # Print summary if we tested providers
    if not (args.test_failover or args.test_cache or args.test_enrichment):
        tester.print_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)
