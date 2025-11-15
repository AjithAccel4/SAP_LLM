"""
Entity Enrichment and Verification.

Enriches and validates entities using web search: vendors, products,
addresses, tax IDs, etc.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class EntityEnricher:
    """
    Enriches and validates entities using web search.

    Supported entity types:
    - Vendor/Supplier (company information, addresses, tax IDs)
    - Customer (company information, contact details)
    - Product (descriptions, prices, specifications)
    - Address (validation, normalization)
    - Tax ID / VAT number (validation)
    - Bank details (IBAN, SWIFT validation)

    Example:
        >>> enricher = EntityEnricher(search_engine)
        >>> vendor_info = enricher.enrich_vendor("ACME Corp", "Germany")
        >>> print(vendor_info["address"], vendor_info["tax_id"])
    """

    def __init__(self, search_engine):
        """
        Initialize entity enricher.

        Args:
            search_engine: WebSearchEngine instance
        """
        self.search_engine = search_engine

    def enrich_vendor(
        self,
        vendor_name: str,
        country: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enrich vendor/supplier information.

        Args:
            vendor_name: Vendor company name
            country: Vendor country (helps with search accuracy)
            additional_context: Additional context (city, industry, etc.)

        Returns:
            Enriched vendor information dictionary
        """
        logger.info(f"Enriching vendor: {vendor_name}")

        # Build search query
        query = f'"{vendor_name}" company'
        if country:
            query += f" {country}"
        if additional_context and "city" in additional_context:
            query += f' {additional_context["city"]}'

        query += " address headquarters contact VAT tax"

        # Perform search
        results = self.search_engine.search(query, num_results=10)

        # Extract information
        enriched = {
            "vendor_name": vendor_name,
            "country": country,
            "verified": len(results) > 0,
            "confidence": self._calculate_confidence(results, vendor_name),
            "sources": results[:3],
            "extracted_data": {}
        }

        if results:
            # Extract structured data from results
            enriched["extracted_data"] = self._extract_vendor_data(
                results,
                vendor_name
            )

        return enriched

    def enrich_customer(
        self,
        customer_name: str,
        country: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrich customer information.

        Args:
            customer_name: Customer company name
            country: Customer country

        Returns:
            Enriched customer information
        """
        # Similar to vendor enrichment
        return self.enrich_vendor(customer_name, country)

    def enrich_product(
        self,
        product_name: str,
        manufacturer: Optional[str] = None,
        product_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrich product information.

        Args:
            product_name: Product name or description
            manufacturer: Product manufacturer
            product_id: Product ID/SKU/EAN

        Returns:
            Enriched product information
        """
        logger.info(f"Enriching product: {product_name}")

        # Build query
        query = f'"{product_name}"'
        if manufacturer:
            query += f' "{manufacturer}"'
        if product_id:
            query += f" {product_id}"

        query += " price specifications features"

        # Search
        results = self.search_engine.search(query, num_results=10)

        enriched = {
            "product_name": product_name,
            "manufacturer": manufacturer,
            "product_id": product_id,
            "verified": len(results) > 0,
            "confidence": self._calculate_confidence(results, product_name),
            "sources": results[:3],
            "extracted_data": {}
        }

        if results:
            enriched["extracted_data"] = self._extract_product_data(
                results,
                product_name
            )

        return enriched

    def validate_address(
        self,
        address: str,
        country: Optional[str] = None,
        postal_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate and normalize address.

        Args:
            address: Address string
            country: Country code or name
            postal_code: Postal/ZIP code

        Returns:
            Validation result with normalized address
        """
        logger.info(f"Validating address: {address[:50]}...")

        # Build query
        query = f'"{address}"'
        if country:
            query += f" {country}"
        if postal_code:
            query += f" {postal_code}"

        # Use local search mode
        from sap_llm.web_search.search_engine import SearchMode
        results = self.search_engine.search(
            query,
            num_results=5,
            mode=SearchMode.LOCAL
        )

        # Parse address components
        components = self._parse_address(address)

        validation = {
            "original_address": address,
            "normalized_address": address,  # Placeholder
            "components": components,
            "valid": len(results) > 0,
            "confidence": min(0.9, len(results) * 0.2),
            "sources": results[:2]
        }

        return validation

    def validate_tax_id(
        self,
        tax_id: str,
        country: str,
        company_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate tax ID / VAT number.

        Args:
            tax_id: Tax ID or VAT number
            country: Country code
            company_name: Company name for additional verification

        Returns:
            Validation result
        """
        logger.info(f"Validating tax ID: {tax_id} ({country})")

        # Format-based validation first
        format_valid = self._validate_tax_id_format(tax_id, country)

        # Web search verification
        query = f"{tax_id}"
        if company_name:
            query += f' "{company_name}"'
        query += f" {country} tax VAT"

        results = self.search_engine.search(query, num_results=5)

        validation = {
            "tax_id": tax_id,
            "country": country,
            "format_valid": format_valid,
            "web_verified": len(results) > 0,
            "confidence": 0.5 if format_valid else 0.0,
            "sources": results[:2]
        }

        # Increase confidence if both format and web verification pass
        if format_valid and len(results) > 0:
            validation["confidence"] = 0.8

        return validation

    def validate_iban(self, iban: str) -> Dict[str, Any]:
        """
        Validate IBAN (International Bank Account Number).

        Args:
            iban: IBAN string

        Returns:
            Validation result
        """
        # Remove spaces and convert to uppercase
        iban_clean = iban.replace(" ", "").upper()

        # Basic format validation
        iban_pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z0-9]+$')
        format_valid = bool(iban_pattern.match(iban_clean))

        # Length validation (country-specific)
        country_code = iban_clean[:2]
        length_valid = self._validate_iban_length(iban_clean, country_code)

        # Checksum validation (mod-97)
        checksum_valid = False
        if format_valid and length_valid:
            checksum_valid = self._validate_iban_checksum(iban_clean)

        validation = {
            "iban": iban,
            "iban_normalized": iban_clean,
            "country_code": country_code,
            "format_valid": format_valid,
            "length_valid": length_valid,
            "checksum_valid": checksum_valid,
            "valid": format_valid and length_valid and checksum_valid,
            "confidence": 1.0 if checksum_valid else 0.0
        }

        return validation

    def verify_company_exists(
        self,
        company_name: str,
        country: Optional[str] = None,
        min_sources: int = 2
    ) -> Dict[str, Any]:
        """
        Verify that a company exists.

        Args:
            company_name: Company name
            country: Country
            min_sources: Minimum number of sources for verification

        Returns:
            Verification result
        """
        # Use fact verification
        claim = f'"{company_name}" company'
        if country:
            claim += f" {country}"

        result = self.search_engine.verify_fact(
            claim=claim,
            min_sources=min_sources,
            confidence_threshold=0.6
        )

        return {
            "company_name": company_name,
            "country": country,
            "exists": result["verified"],
            "confidence": result["confidence"],
            "sources": result.get("sources", [])
        }

    def get_market_price(
        self,
        product_name: str,
        quantity: Optional[int] = None,
        unit: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get market price for a product.

        Args:
            product_name: Product name
            quantity: Quantity
            unit: Unit of measure

        Returns:
            Price information
        """
        query = f'"{product_name}" price'
        if quantity and unit:
            query += f" per {unit}"

        results = self.search_engine.search(query, num_results=10)

        # Extract prices from results
        prices = self._extract_prices(results)

        price_info = {
            "product_name": product_name,
            "quantity": quantity,
            "unit": unit,
            "found": len(prices) > 0,
            "prices": prices,
            "sources": results[:3]
        }

        if prices:
            # Calculate statistics
            price_values = [p["amount"] for p in prices if p.get("amount")]
            if price_values:
                price_info["min_price"] = min(price_values)
                price_info["max_price"] = max(price_values)
                price_info["avg_price"] = sum(price_values) / len(price_values)

        return price_info

    def _extract_vendor_data(
        self,
        results: List[Dict[str, Any]],
        vendor_name: str
    ) -> Dict[str, Any]:
        """Extract structured vendor data from search results."""
        data = {
            "addresses": [],
            "emails": [],
            "phones": [],
            "tax_ids": [],
            "websites": []
        }

        for result in results[:5]:
            snippet = result.get("snippet", "")
            url = result.get("url", "")

            # Extract addresses (simplified - in production use NER)
            address_patterns = [
                r'\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)[,\s]+[\w\s]+',
                r'[\w\s]+,\s*\d{5}(?:-\d{4})?',  # US ZIP
                r'[\w\s]+,\s*[A-Z]{2}\s+\d{5}'  # State + ZIP
            ]

            for pattern in address_patterns:
                matches = re.findall(pattern, snippet)
                data["addresses"].extend(matches)

            # Extract emails
            emails = re.findall(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                snippet
            )
            data["emails"].extend(emails)

            # Extract phones
            phones = re.findall(
                r'\+?\d{1,3}?[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}',
                snippet
            )
            data["phones"].extend(phones)

            # Extract potential tax IDs (varies by country)
            # Example: DE123456789 (German VAT)
            tax_patterns = [
                r'\b[A-Z]{2}\d{9,12}\b',  # EU VAT format
                r'\b\d{2}-\d{7}\b'  # US EIN format
            ]

            for pattern in tax_patterns:
                matches = re.findall(pattern, snippet)
                data["tax_ids"].extend(matches)

            # Website from URL
            data["websites"].append(url)

        # Deduplicate
        for key in data:
            if isinstance(data[key], list):
                data[key] = list(set(data[key]))[:5]  # Keep top 5 unique

        return data

    def _extract_product_data(
        self,
        results: List[Dict[str, Any]],
        product_name: str
    ) -> Dict[str, Any]:
        """Extract structured product data from search results."""
        data = {
            "descriptions": [],
            "prices": [],
            "specifications": [],
            "manufacturers": []
        }

        for result in results[:5]:
            snippet = result.get("snippet", "")

            # Description
            data["descriptions"].append(snippet)

            # Extract prices
            prices = self._extract_prices([result])
            data["prices"].extend(prices)

        # Deduplicate descriptions
        data["descriptions"] = list(set(data["descriptions"]))[:3]

        return data

    def _extract_prices(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract price information from search results."""
        prices = []

        for result in results:
            snippet = result.get("snippet", "") + " " + result.get("title", "")

            # Price patterns
            # $99.99, €99.99, £99.99, 99.99 USD, etc.
            price_patterns = [
                r'[\$€£¥]\s*(\d+(?:[.,]\d{2})?)',  # Symbol prefix
                r'(\d+(?:[.,]\d{2})?)\s*(?:USD|EUR|GBP|JPY)',  # Currency suffix
            ]

            for pattern in price_patterns:
                matches = re.findall(pattern, snippet)
                for match in matches:
                    try:
                        # Convert to float
                        amount = float(match.replace(',', ''))

                        prices.append({
                            "amount": amount,
                            "currency": self._detect_currency(snippet, match),
                            "source": result.get("url", "")
                        })
                    except ValueError:
                        continue

        return prices

    def _detect_currency(self, text: str, price_str: str) -> str:
        """Detect currency from text context."""
        # Look for currency symbols near price
        context = text[max(0, text.find(price_str) - 10):text.find(price_str) + 20]

        if '$' in context:
            return "USD"
        elif '€' in context:
            return "EUR"
        elif '£' in context:
            return "GBP"
        elif '¥' in context:
            return "JPY"
        elif 'USD' in context:
            return "USD"
        elif 'EUR' in context:
            return "EUR"

        return "USD"  # Default

    def _parse_address(self, address: str) -> Dict[str, Optional[str]]:
        """Parse address into components."""
        components = {
            "street": None,
            "city": None,
            "state": None,
            "postal_code": None,
            "country": None
        }

        # Very simplified parsing - in production use address parsing library
        parts = address.split(',')

        if len(parts) >= 1:
            components["street"] = parts[0].strip()
        if len(parts) >= 2:
            components["city"] = parts[1].strip()
        if len(parts) >= 3:
            # Try to extract state and ZIP
            last_part = parts[-1].strip()
            state_zip = re.search(r'([A-Z]{2})\s+(\d{5})', last_part)
            if state_zip:
                components["state"] = state_zip.group(1)
                components["postal_code"] = state_zip.group(2)

        return components

    def _validate_tax_id_format(self, tax_id: str, country: str) -> bool:
        """Validate tax ID format for specific country."""
        tax_id_clean = tax_id.replace(" ", "").replace("-", "").upper()

        # Country-specific patterns
        patterns = {
            "DE": r'^DE\d{9}$',  # Germany
            "FR": r'^FR[A-Z0-9]{2}\d{9}$',  # France
            "GB": r'^GB\d{9}$',  # UK
            "US": r'^\d{2}-?\d{7}$',  # US EIN
            "NL": r'^NL\d{9}B\d{2}$',  # Netherlands
        }

        pattern = patterns.get(country.upper())
        if pattern:
            return bool(re.match(pattern, tax_id_clean))

        return False  # Unknown country format

    def _validate_iban_length(self, iban: str, country_code: str) -> bool:
        """Validate IBAN length for country."""
        lengths = {
            "DE": 22, "FR": 27, "GB": 22, "IT": 27, "ES": 24,
            "NL": 18, "BE": 16, "AT": 20, "CH": 21, "PL": 28
        }

        expected_length = lengths.get(country_code)
        if expected_length:
            return len(iban) == expected_length

        return False

    def _validate_iban_checksum(self, iban: str) -> bool:
        """Validate IBAN checksum using mod-97."""
        try:
            # Move first 4 characters to end
            rearranged = iban[4:] + iban[:4]

            # Replace letters with numbers (A=10, B=11, etc.)
            numeric = ""
            for char in rearranged:
                if char.isdigit():
                    numeric += char
                else:
                    numeric += str(ord(char) - ord('A') + 10)

            # Check if mod 97 equals 1
            return int(numeric) % 97 == 1

        except Exception:
            return False

    def _calculate_confidence(
        self,
        results: List[Dict[str, Any]],
        entity_name: str
    ) -> float:
        """Calculate confidence score for enrichment."""
        if not results:
            return 0.0

        # Base confidence on number of results
        confidence = min(0.5, len(results) * 0.1)

        # Increase if entity name appears in top results
        entity_lower = entity_name.lower()
        for result in results[:3]:
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()

            if entity_lower in title:
                confidence += 0.2
            elif entity_lower in snippet:
                confidence += 0.1

        return min(1.0, confidence)
