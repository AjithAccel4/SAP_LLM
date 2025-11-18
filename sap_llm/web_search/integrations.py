"""
Pipeline Integration for Web Search.

Provides integration points for web search capabilities into SAP_LLM
pipeline stages (validation, routing, quality check).
"""

from typing import Any, Dict, List, Optional

from sap_llm.utils.logger import get_logger
from sap_llm.web_search.entity_enrichment import EntityEnricher
from sap_llm.web_search.search_engine import WebSearchEngine

logger = get_logger(__name__)


class ExtractionEnhancer:
    """
    Enhances extraction stage with real-time web search for missing fields.

    Provides:
    - Automatic vendor name lookup for missing vendor_name fields
    - Product code enrichment from manufacturer websites
    - Currency exchange rate lookup for multi-currency invoices
    - Tax rate validation from government sources
    """

    def __init__(
        self,
        search_engine: WebSearchEngine,
        enabled: bool = True
    ):
        """
        Initialize extraction enhancer.

        Args:
            search_engine: WebSearchEngine instance
            enabled: Whether web enrichment is enabled
        """
        self.search_engine = search_engine
        self.enricher = EntityEnricher(search_engine)
        self.enabled = enabled
        self.enrichment_stats = {
            "vendor_enrichments": 0,
            "product_enrichments": 0,
            "currency_lookups": 0,
            "tax_rate_lookups": 0
        }

    def enrich_extracted_data(
        self,
        extracted_data: Dict[str, Any],
        document_type: str = "INVOICE"
    ) -> Dict[str, Any]:
        """
        Enrich extracted data with web search in real-time.

        Args:
            extracted_data: Extracted data from document
            document_type: Type of document (INVOICE, PURCHASE_ORDER, etc.)

        Returns:
            Enriched extracted data with additional fields
        """
        if not self.enabled:
            return extracted_data

        enriched = extracted_data.copy()
        enriched["web_enrichments"] = {}

        logger.info(f"Enriching extracted {document_type} data with web search")

        # 1. Enrich vendor information if missing or incomplete
        if "vendor_name" in enriched:
            vendor_enrichment = self._enrich_vendor_field(
                enriched.get("vendor_name"),
                enriched.get("vendor_id"),
                enriched.get("country")
            )
            if vendor_enrichment:
                enriched["web_enrichments"]["vendor"] = vendor_enrichment
                self.enrichment_stats["vendor_enrichments"] += 1

                # Fill missing fields
                if not enriched.get("vendor_address") and vendor_enrichment.get("address"):
                    enriched["vendor_address"] = vendor_enrichment["address"]
                if not enriched.get("vendor_tax_id") and vendor_enrichment.get("tax_id"):
                    enriched["vendor_tax_id"] = vendor_enrichment["tax_id"]

        # 2. Enrich product codes and descriptions
        if "line_items" in enriched:
            for i, item in enumerate(enriched["line_items"]):
                product_enrichment = self._enrich_product_field(
                    item.get("description"),
                    item.get("material_number"),
                    item.get("manufacturer")
                )
                if product_enrichment:
                    if "line_item_enrichments" not in enriched["web_enrichments"]:
                        enriched["web_enrichments"]["line_item_enrichments"] = {}
                    enriched["web_enrichments"]["line_item_enrichments"][i] = product_enrichment
                    self.enrichment_stats["product_enrichments"] += 1

                    # Fill missing product code if found
                    if not item.get("material_number") and product_enrichment.get("product_code"):
                        item["material_number"] = product_enrichment["product_code"]

        # 3. Lookup currency exchange rates for multi-currency invoices
        if enriched.get("currency") and enriched["currency"] != "USD":
            exchange_rate = self._get_exchange_rate(
                enriched["currency"],
                "USD",
                enriched.get("invoice_date")
            )
            if exchange_rate:
                enriched["web_enrichments"]["exchange_rate"] = exchange_rate
                enriched["amount_usd"] = enriched.get("total_amount", 0) * exchange_rate["rate"]
                self.enrichment_stats["currency_lookups"] += 1

        # 4. Lookup tax rates for validation
        if enriched.get("country") and enriched.get("tax_amount"):
            tax_rate_info = self._get_tax_rate(
                enriched["country"],
                enriched.get("tax_type", "VAT")
            )
            if tax_rate_info:
                enriched["web_enrichments"]["tax_rate"] = tax_rate_info
                self.enrichment_stats["tax_rate_lookups"] += 1

        return enriched

    def _enrich_vendor_field(
        self,
        vendor_name: Optional[str],
        vendor_id: Optional[str],
        country: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Enrich vendor information using web search."""
        if not vendor_name:
            return None

        try:
            vendor_info = self.enricher.enrich_vendor(
                vendor_name,
                country=country,
                additional_context={"vendor_id": vendor_id}
            )

            if vendor_info.get("verified"):
                extracted_data = vendor_info.get("extracted_data", {})
                return {
                    "verified": True,
                    "confidence": vendor_info.get("confidence", 0.0),
                    "address": extracted_data.get("address"),
                    "tax_id": extracted_data.get("tax_id"),
                    "website": extracted_data.get("website"),
                    "sources": vendor_info.get("sources", [])[:2]
                }
        except Exception as e:
            logger.debug(f"Vendor enrichment error: {e}")

        return None

    def _enrich_product_field(
        self,
        description: Optional[str],
        material_number: Optional[str],
        manufacturer: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Enrich product information using web search."""
        if not description:
            return None

        try:
            product_info = self.enricher.enrich_product(
                description,
                manufacturer=manufacturer,
                product_id=material_number
            )

            if product_info.get("verified"):
                extracted_data = product_info.get("extracted_data", {})
                return {
                    "verified": True,
                    "confidence": product_info.get("confidence", 0.0),
                    "product_code": extracted_data.get("product_code"),
                    "manufacturer": extracted_data.get("manufacturer"),
                    "market_price": extracted_data.get("price"),
                    "sources": product_info.get("sources", [])[:2]
                }
        except Exception as e:
            logger.debug(f"Product enrichment error: {e}")

        return None

    def _get_exchange_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get currency exchange rate."""
        try:
            rate_info = self.search_engine.get_exchange_rate(
                from_currency=from_currency,
                to_currency=to_currency,
                date=date
            )

            if rate_info and rate_info.get("rate"):
                return {
                    "from_currency": from_currency,
                    "to_currency": to_currency,
                    "rate": rate_info["rate"],
                    "date": rate_info.get("date"),
                    "source": rate_info.get("source")
                }
        except Exception as e:
            logger.debug(f"Exchange rate lookup error: {e}")

        return None

    def _get_tax_rate(
        self,
        country: str,
        tax_type: str = "VAT"
    ) -> Optional[Dict[str, Any]]:
        """Get tax rate for country."""
        try:
            query = f"{country} {tax_type} rate official 2024 2025"
            results = self.search_engine.search(
                query,
                num_results=5,
                filters={"domains": ["gov", "europa.eu", "oecd.org"]}
            )

            if results:
                # Simple extraction - could be enhanced with NLP
                import re
                for result in results:
                    snippet = result.get("snippet", "")
                    # Look for percentage patterns
                    rate_match = re.search(r'(\d+(?:\.\d+)?)\s*%', snippet)
                    if rate_match:
                        return {
                            "country": country,
                            "tax_type": tax_type,
                            "rate": float(rate_match.group(1)),
                            "source": result.get("url")
                        }
        except Exception as e:
            logger.debug(f"Tax rate lookup error: {e}")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get enrichment statistics."""
        return self.enrichment_stats.copy()


class ValidationEnhancer:
    """
    Enhances validation stage with web search capabilities.

    Provides:
    - Vendor address verification
    - Tax ID validation
    - Company existence verification
    - IBAN validation
    - Price reasonableness checks
    """

    def __init__(
        self,
        search_engine: WebSearchEngine,
        enabled: bool = True
    ):
        """
        Initialize validation enhancer.

        Args:
            search_engine: WebSearchEngine instance
            enabled: Whether web validation is enabled
        """
        self.search_engine = search_engine
        self.enricher = EntityEnricher(search_engine)
        self.enabled = enabled

    def validate_vendor_data(
        self,
        vendor_name: str,
        address: Optional[str] = None,
        tax_id: Optional[str] = None,
        country: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate vendor data using web search.

        Args:
            vendor_name: Vendor company name
            address: Vendor address
            tax_id: Vendor tax ID
            country: Vendor country

        Returns:
            Validation results with confidence scores
        """
        if not self.enabled:
            return {"validated": False, "reason": "Web validation disabled"}

        logger.info(f"Validating vendor: {vendor_name}")

        results = {
            "vendor_name": vendor_name,
            "validations": {},
            "overall_valid": True,
            "confidence": 1.0
        }

        # Verify company exists
        company_check = self.enricher.verify_company_exists(
            vendor_name,
            country=country
        )
        results["validations"]["company_exists"] = company_check

        if not company_check["exists"]:
            results["overall_valid"] = False
            results["confidence"] *= 0.5

        # Validate address if provided
        if address:
            address_check = self.enricher.validate_address(
                address,
                country=country
            )
            results["validations"]["address"] = address_check

            if not address_check["valid"]:
                results["overall_valid"] = False
                results["confidence"] *= 0.7

        # Validate tax ID if provided
        if tax_id and country:
            tax_check = self.enricher.validate_tax_id(
                tax_id,
                country=country,
                company_name=vendor_name
            )
            results["validations"]["tax_id"] = tax_check

            if not tax_check["format_valid"]:
                results["overall_valid"] = False
                results["confidence"] *= 0.6

        logger.info(
            f"Vendor validation complete: {vendor_name} - "
            f"Valid: {results['overall_valid']}, "
            f"Confidence: {results['confidence']:.2f}"
        )

        return results

    def validate_invoice_data(
        self,
        invoice_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate invoice data.

        Args:
            invoice_data: Extracted invoice data

        Returns:
            Validation results
        """
        if not self.enabled:
            return {"validated": False, "reason": "Web validation disabled"}

        results = {
            "validations": {},
            "overall_valid": True,
            "confidence": 1.0
        }

        # Validate vendor
        if "vendor_name" in invoice_data:
            vendor_validation = self.validate_vendor_data(
                vendor_name=invoice_data["vendor_name"],
                address=invoice_data.get("vendor_address"),
                tax_id=invoice_data.get("vendor_tax_id"),
                country=invoice_data.get("vendor_country")
            )
            results["validations"]["vendor"] = vendor_validation

            if not vendor_validation["overall_valid"]:
                results["overall_valid"] = False
                results["confidence"] *= vendor_validation["confidence"]

        # Validate line items (check product prices)
        if "line_items" in invoice_data:
            item_validations = []

            for item in invoice_data["line_items"]:
                item_validation = self.validate_line_item(item)
                item_validations.append(item_validation)

            results["validations"]["line_items"] = item_validations

        # Validate bank details if present
        if "iban" in invoice_data:
            iban_validation = self.enricher.validate_iban(invoice_data["iban"])
            results["validations"]["iban"] = iban_validation

            if not iban_validation["valid"]:
                results["overall_valid"] = False
                results["confidence"] *= 0.8

        return results

    def validate_line_item(
        self,
        line_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate invoice line item (product and price).

        Args:
            line_item: Line item data

        Returns:
            Validation results
        """
        validation = {
            "product": line_item.get("description"),
            "valid": True,
            "warnings": []
        }

        # Get market price if possible
        if "description" in line_item and "unit_price" in line_item:
            price_info = self.enricher.get_market_price(
                product_name=line_item["description"],
                quantity=line_item.get("quantity"),
                unit=line_item.get("unit")
            )

            if price_info["found"]:
                # Check if invoice price is within reasonable range
                invoice_price = line_item["unit_price"]
                avg_price = price_info.get("avg_price")

                if avg_price:
                    deviation = abs(invoice_price - avg_price) / avg_price

                    if deviation > 0.5:  # More than 50% deviation
                        validation["warnings"].append({
                            "type": "price_deviation",
                            "message": f"Price deviates {deviation*100:.0f}% from market average",
                            "invoice_price": invoice_price,
                            "market_avg": avg_price,
                            "market_range": f"{price_info.get('min_price')}-{price_info.get('max_price')}"
                        })

                validation["market_price_info"] = price_info

        return validation


class RoutingEnhancer:
    """
    Enhances routing stage with web search for API endpoint discovery.

    Provides:
    - SAP API endpoint lookup
    - Documentation search
    - API version detection
    - Parameter mapping discovery
    """

    def __init__(
        self,
        search_engine: WebSearchEngine,
        enabled: bool = True
    ):
        """
        Initialize routing enhancer.

        Args:
            search_engine: WebSearchEngine instance
            enabled: Whether web routing is enabled
        """
        self.search_engine = search_engine
        self.enabled = enabled

    def find_sap_api(
        self,
        document_type: str,
        operation: str,
        sap_system: Optional[str] = "S/4HANA"
    ) -> Dict[str, Any]:
        """
        Find appropriate SAP API for document type and operation.

        Args:
            document_type: Type of document (INVOICE, PURCHASE_ORDER, etc.)
            operation: Operation (CREATE, UPDATE, READ, etc.)
            sap_system: SAP system type

        Returns:
            API endpoint information
        """
        if not self.enabled:
            return {"found": False, "reason": "Web search disabled"}

        logger.info(f"Finding SAP API for: {document_type} - {operation}")

        # Search SAP documentation
        query = f"SAP {sap_system} {document_type} {operation} API endpoint BAPI OData"

        results = self.search_engine.search_sap_documentation(
            topic=f"{document_type} {operation}",
            doc_type="api"
        )

        api_info = {
            "document_type": document_type,
            "operation": operation,
            "sap_system": sap_system,
            "found": len(results) > 0,
            "endpoints": [],
            "sources": results[:3]
        }

        if results:
            # Extract API endpoints from results
            endpoints = self._extract_api_endpoints(results)
            api_info["endpoints"] = endpoints

        return api_info

    def _extract_api_endpoints(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Extract API endpoint information from search results."""
        import re

        endpoints = []

        for result in results:
            snippet = result.get("snippet", "")
            title = result.get("title", "")

            # Look for BAPI names
            bapi_pattern = r'BAPI_[A-Z0-9_]+'
            bapis = re.findall(bapi_pattern, snippet + " " + title)

            for bapi in bapis:
                endpoints.append({
                    "type": "BAPI",
                    "name": bapi,
                    "source": result["url"]
                })

            # Look for OData service names
            odata_pattern = r'/[A-Z][A-Z0-9_]+_SRV'
            odata_services = re.findall(odata_pattern, snippet)

            for service in odata_services:
                endpoints.append({
                    "type": "OData",
                    "name": service,
                    "source": result["url"]
                })

        # Deduplicate
        seen = set()
        unique_endpoints = []
        for ep in endpoints:
            key = f"{ep['type']}:{ep['name']}"
            if key not in seen:
                seen.add(key)
                unique_endpoints.append(ep)

        return unique_endpoints


class QualityCheckEnhancer:
    """
    Enhances quality check stage with external verification.

    Provides:
    - Cross-reference verification
    - Fact checking against external sources
    - Data consistency validation
    """

    def __init__(
        self,
        search_engine: WebSearchEngine,
        enabled: bool = True
    ):
        """
        Initialize quality check enhancer.

        Args:
            search_engine: WebSearchEngine instance
            enabled: Whether web quality checks are enabled
        """
        self.search_engine = search_engine
        self.enabled = enabled

    def verify_extracted_data(
        self,
        extracted_data: Dict[str, Any],
        document_type: str
    ) -> Dict[str, Any]:
        """
        Verify extracted data against external sources.

        Args:
            extracted_data: Data extracted from document
            document_type: Type of document

        Returns:
            Verification results
        """
        if not self.enabled:
            return {"verified": False, "reason": "Web verification disabled"}

        logger.info(f"Verifying extracted data for {document_type}")

        verification = {
            "document_type": document_type,
            "fields_verified": {},
            "overall_confidence": 1.0,
            "issues": []
        }

        # Verify vendor name if present
        if "vendor_name" in extracted_data:
            vendor_verified = self.search_engine.verify_fact(
                claim=f"{extracted_data['vendor_name']} company",
                min_sources=2
            )

            verification["fields_verified"]["vendor_name"] = vendor_verified

            if not vendor_verified["verified"]:
                verification["issues"].append({
                    "field": "vendor_name",
                    "issue": "Could not verify company exists",
                    "value": extracted_data["vendor_name"]
                })
                verification["overall_confidence"] *= 0.7

        # Verify dates are reasonable
        if "invoice_date" in extracted_data:
            date_check = self._verify_date_reasonable(
                extracted_data["invoice_date"]
            )

            verification["fields_verified"]["invoice_date"] = date_check

            if not date_check["reasonable"]:
                verification["issues"].append({
                    "field": "invoice_date",
                    "issue": date_check["reason"],
                    "value": extracted_data["invoice_date"]
                })
                verification["overall_confidence"] *= 0.8

        return verification

    def _verify_date_reasonable(
        self,
        date_str: str
    ) -> Dict[str, Any]:
        """Verify date is reasonable (not too old or in future)."""
        from datetime import datetime, timedelta

        try:
            # Parse date (simplified - support multiple formats)
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d.%m.%Y"]:
                try:
                    date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            else:
                return {
                    "reasonable": False,
                    "reason": "Could not parse date format"
                }

            now = datetime.now()

            # Check if date is in future
            if date > now:
                return {
                    "reasonable": False,
                    "reason": "Date is in the future"
                }

            # Check if date is too old (e.g., more than 5 years)
            if date < now - timedelta(days=365 * 5):
                return {
                    "reasonable": False,
                    "reason": "Date is more than 5 years old"
                }

            return {
                "reasonable": True,
                "reason": "Date is within acceptable range"
            }

        except Exception as e:
            return {
                "reasonable": False,
                "reason": f"Error validating date: {e}"
            }


class KnowledgeBaseUpdater:
    """
    Updates SAP knowledge base with latest documentation from web.

    Provides:
    - Automatic documentation updates
    - New API discovery
    - Field mapping updates
    - Business rule updates
    """

    def __init__(
        self,
        search_engine: WebSearchEngine,
        enabled: bool = True
    ):
        """
        Initialize knowledge base updater.

        Args:
            search_engine: WebSearchEngine instance
            enabled: Whether auto-updates are enabled
        """
        self.search_engine = search_engine
        self.enabled = enabled

    def fetch_latest_sap_docs(
        self,
        topic: str,
        max_results: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetch latest SAP documentation for topic.

        Args:
            topic: Documentation topic
            max_results: Maximum results to fetch

        Returns:
            List of documentation results
        """
        if not self.enabled:
            return []

        logger.info(f"Fetching latest SAP docs for: {topic}")

        results = self.search_engine.search_sap_documentation(
            topic=topic,
            doc_type="api"
        )

        return results[:max_results]

    def discover_new_apis(
        self,
        document_types: List[str]
    ) -> Dict[str, List[str]]:
        """
        Discover new SAP APIs for document types.

        Args:
            document_types: List of document types to search

        Returns:
            Dictionary mapping document types to discovered APIs
        """
        if not self.enabled:
            return {}

        logger.info(f"Discovering new APIs for {len(document_types)} document types")

        discovered = {}

        for doc_type in document_types:
            # Search for APIs
            query = f"SAP S/4HANA {doc_type} API BAPI OData 2024 2025"

            results = self.search_engine.search(
                query,
                num_results=10,
                filters={"domains": ["help.sap.com", "api.sap.com"]}
            )

            # Extract API names
            import re
            apis = set()

            for result in results:
                content = result.get("snippet", "") + " " + result.get("title", "")

                # Find BAPIs
                bapis = re.findall(r'BAPI_[A-Z0-9_]+', content)
                apis.update(bapis)

                # Find OData services
                odata = re.findall(r'/[A-Z][A-Z0-9_]+_SRV', content)
                apis.update(odata)

            discovered[doc_type] = list(apis)

        return discovered
