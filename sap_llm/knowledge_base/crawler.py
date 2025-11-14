"""
SAP API Crawler

Crawls SAP API documentation and extracts:
- API endpoint schemas
- Field definitions and types
- Business rules and validations
- Example requests/responses
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SAPAPICrawler:
    """
    Crawls SAP API documentation and extracts structured information.

    Supports multiple SAP API types:
    - OData V2/V4
    - REST APIs
    - SOAP services (legacy)
    - BAPI function modules
    """

    def __init__(
        self,
        base_urls: Optional[List[str]] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_concurrent_requests: int = 10,
        request_timeout: int = 30,
    ):
        """
        Initialize SAP API crawler.

        Args:
            base_urls: List of base URLs to crawl
            embedding_model: Model for generating embeddings
            max_concurrent_requests: Max concurrent HTTP requests
            request_timeout: Request timeout in seconds
        """
        # Default SAP API documentation URLs
        self.base_urls = base_urls or [
            "https://api.sap.com/",
            "https://help.sap.com/docs/",
        ]

        self.max_concurrent_requests = max_concurrent_requests
        self.request_timeout = request_timeout

        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Track visited URLs
        self.visited_urls: set = set()

        # Store crawled data
        self.api_schemas: List[Dict[str, Any]] = []
        self.field_mappings: List[Dict[str, Any]] = []
        self.business_rules: List[Dict[str, Any]] = []
        self.examples: List[Dict[str, Any]] = []

        logger.info("SAP API Crawler initialized")

    async def crawl_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Crawl all configured SAP API documentation.

        Returns:
            Dictionary containing all crawled data
        """
        logger.info("Starting SAP API documentation crawl...")

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.request_timeout)
        ) as session:
            # Crawl all base URLs
            tasks = [self.crawl_url(session, url) for url in self.base_urls]
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info(
            f"Crawl complete: {len(self.api_schemas)} APIs, "
            f"{len(self.field_mappings)} field mappings, "
            f"{len(self.business_rules)} business rules"
        )

        return {
            "api_schemas": self.api_schemas,
            "field_mappings": self.field_mappings,
            "business_rules": self.business_rules,
            "examples": self.examples,
        }

    async def crawl_url(
        self, session: aiohttp.ClientSession, url: str, depth: int = 0
    ) -> None:
        """
        Crawl a single URL and extract API information.

        Args:
            session: aiohttp session
            url: URL to crawl
            depth: Current crawl depth
        """
        # Depth limit
        if depth > 3:
            return

        # Skip if already visited
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)

        try:
            logger.debug(f"Crawling: {url}")

            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return

                content_type = response.headers.get("Content-Type", "")

                # Handle different content types
                if "application/json" in content_type:
                    await self._process_json(url, await response.text())
                elif "text/html" in content_type:
                    await self._process_html(session, url, await response.text(), depth)
                elif "application/xml" in content_type:
                    await self._process_xml(url, await response.text())

        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching {url}")
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")

    async def _process_json(self, url: str, content: str) -> None:
        """
        Process JSON content (OData metadata, OpenAPI specs).

        Args:
            url: Source URL
            content: JSON content
        """
        try:
            data = json.loads(content)

            # Check if it's OpenAPI/Swagger spec
            if "openapi" in data or "swagger" in data:
                await self._extract_openapi_schema(url, data)

            # Check if it's OData metadata
            elif "$metadata" in url or "edmx" in content.lower():
                await self._extract_odata_schema(url, data)

            # Generic JSON schema
            else:
                await self._extract_generic_schema(url, data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {url}: {e}")

    async def _process_html(
        self, session: aiohttp.ClientSession, url: str, content: str, depth: int
    ) -> None:
        """
        Process HTML content (documentation pages).

        Args:
            session: aiohttp session
            url: Source URL
            content: HTML content
            depth: Current depth
        """
        soup = BeautifulSoup(content, "html.parser")

        # Extract API documentation links
        for link in soup.find_all("a", href=True):
            href = link["href"]

            # Skip non-API links
            if not self._is_api_link(href):
                continue

            # Make absolute URL
            absolute_url = urljoin(url, href)

            # Recursively crawl
            if absolute_url not in self.visited_urls:
                await self.crawl_url(session, absolute_url, depth + 1)

        # Extract field mappings from tables
        await self._extract_field_mappings_from_html(url, soup)

        # Extract business rules from documentation
        await self._extract_business_rules_from_html(url, soup)

    async def _process_xml(self, url: str, content: str) -> None:
        """
        Process XML content (EDMX, WSDL).

        Args:
            url: Source URL
            content: XML content
        """
        # TODO: Implement XML parsing for EDMX/WSDL
        logger.debug(f"Processing XML from {url}")

    def _is_api_link(self, href: str) -> bool:
        """
        Check if link is API-related.

        Args:
            href: Link href

        Returns:
            True if API-related
        """
        api_keywords = [
            "api",
            "odata",
            "rest",
            "endpoint",
            "schema",
            "metadata",
            "swagger",
            "openapi",
        ]

        href_lower = href.lower()
        return any(keyword in href_lower for keyword in api_keywords)

    async def _extract_openapi_schema(self, url: str, spec: Dict[str, Any]) -> None:
        """
        Extract API schema from OpenAPI specification.

        Args:
            url: Source URL
            spec: OpenAPI specification
        """
        api_info = {
            "source": url,
            "type": "openapi",
            "version": spec.get("openapi", spec.get("swagger", "unknown")),
            "title": spec.get("info", {}).get("title", "Unknown"),
            "description": spec.get("info", {}).get("description", ""),
            "endpoints": [],
        }

        # Extract endpoints
        paths = spec.get("paths", {})
        for path, methods in paths.items():
            for method, details in methods.items():
                if method.upper() not in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
                    continue

                endpoint = {
                    "path": path,
                    "method": method.upper(),
                    "summary": details.get("summary", ""),
                    "description": details.get("description", ""),
                    "parameters": details.get("parameters", []),
                    "request_body": details.get("requestBody", {}),
                    "responses": details.get("responses", {}),
                }

                # Extract field mappings from request body
                if "requestBody" in details:
                    await self._extract_fields_from_schema(
                        url, path, details["requestBody"]
                    )

                api_info["endpoints"].append(endpoint)

        # Generate embedding
        api_text = f"{api_info['title']} {api_info['description']}"
        api_info["embedding"] = self.embedding_model.encode(api_text).tolist()

        self.api_schemas.append(api_info)
        logger.info(f"Extracted OpenAPI schema: {api_info['title']}")

    async def _extract_odata_schema(self, url: str, metadata: Dict[str, Any]) -> None:
        """
        Extract API schema from OData metadata.

        Args:
            url: Source URL
            metadata: OData metadata
        """
        # TODO: Implement OData metadata parsing
        logger.debug(f"Extracting OData schema from {url}")

    async def _extract_generic_schema(self, url: str, data: Dict[str, Any]) -> None:
        """
        Extract generic API schema.

        Args:
            url: Source URL
            data: JSON data
        """
        # Create basic schema
        api_info = {
            "source": url,
            "type": "generic",
            "fields": self._extract_fields_recursive(data),
        }

        # Generate embedding
        fields_text = " ".join([f["name"] for f in api_info["fields"]])
        api_info["embedding"] = self.embedding_model.encode(fields_text).tolist()

        self.api_schemas.append(api_info)

    def _extract_fields_recursive(
        self, data: Dict[str, Any], prefix: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Recursively extract fields from nested structure.

        Args:
            data: Data structure
            prefix: Field name prefix

        Returns:
            List of field definitions
        """
        fields = []

        for key, value in data.items():
            field_name = f"{prefix}.{key}" if prefix else key

            field_info = {
                "name": field_name,
                "type": type(value).__name__,
            }

            if isinstance(value, dict):
                # Recursively extract nested fields
                fields.extend(self._extract_fields_recursive(value, field_name))
            else:
                fields.append(field_info)

        return fields

    async def _extract_fields_from_schema(
        self, url: str, endpoint: str, schema: Dict[str, Any]
    ) -> None:
        """
        Extract field mappings from schema.

        Args:
            url: Source URL
            endpoint: API endpoint
            schema: Field schema
        """
        # TODO: Implement detailed field extraction
        logger.debug(f"Extracting fields from {endpoint}")

    async def _extract_field_mappings_from_html(
        self, url: str, soup: BeautifulSoup
    ) -> None:
        """
        Extract field mappings from HTML tables.

        Args:
            url: Source URL
            soup: BeautifulSoup object
        """
        # Find tables with field mappings
        for table in soup.find_all("table"):
            # Check if table contains field information
            headers = [th.get_text(strip=True).lower() for th in table.find_all("th")]

            if "field" in headers or "parameter" in headers or "property" in headers:
                rows = table.find_all("tr")[1:]  # Skip header row

                for row in rows:
                    cells = [td.get_text(strip=True) for td in row.find_all("td")]

                    if len(cells) >= 2:
                        mapping = {
                            "source": url,
                            "field_name": cells[0],
                            "description": cells[1] if len(cells) > 1 else "",
                            "type": cells[2] if len(cells) > 2 else "string",
                            "required": "required" in " ".join(cells).lower(),
                        }

                        # Generate embedding
                        mapping_text = f"{mapping['field_name']} {mapping['description']}"
                        mapping["embedding"] = self.embedding_model.encode(
                            mapping_text
                        ).tolist()

                        self.field_mappings.append(mapping)

    async def _extract_business_rules_from_html(
        self, url: str, soup: BeautifulSoup
    ) -> None:
        """
        Extract business rules from documentation.

        Args:
            url: Source URL
            soup: BeautifulSoup object
        """
        # Look for validation rules, constraints, business logic
        rule_keywords = ["validation", "constraint", "rule", "must", "should", "required"]

        for paragraph in soup.find_all(["p", "li"]):
            text = paragraph.get_text(strip=True)

            # Check if paragraph contains business rule
            if any(keyword in text.lower() for keyword in rule_keywords):
                rule = {
                    "source": url,
                    "description": text,
                    "type": self._classify_rule(text),
                }

                # Generate embedding
                rule["embedding"] = self.embedding_model.encode(text).tolist()

                self.business_rules.append(rule)

    def _classify_rule(self, text: str) -> str:
        """
        Classify business rule type.

        Args:
            text: Rule text

        Returns:
            Rule type
        """
        text_lower = text.lower()

        if "validation" in text_lower or "valid" in text_lower:
            return "validation"
        elif "format" in text_lower:
            return "format"
        elif "range" in text_lower or "between" in text_lower:
            return "range"
        elif "required" in text_lower or "must" in text_lower:
            return "required"
        elif "calculation" in text_lower or "formula" in text_lower:
            return "calculation"
        else:
            return "general"

    async def crawl_mock_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate mock SAP API data for testing.

        Returns:
            Dictionary containing mock API data
        """
        logger.info("Generating mock SAP API data...")

        # Mock Purchase Order API
        po_api = {
            "source": "mock://sap/po",
            "type": "odata",
            "title": "Purchase Order API",
            "description": "SAP Purchase Order OData service",
            "endpoints": [
                {
                    "path": "/sap/opu/odata/sap/MM_PUR_PO_MAINT_V2/PurchaseOrder",
                    "method": "POST",
                    "summary": "Create Purchase Order",
                    "fields": [
                        "PurchaseOrderType",
                        "CompanyCode",
                        "VendorID",
                        "PurchaseOrderDate",
                        "TotalAmount",
                        "Currency",
                    ],
                }
            ],
            "embedding": self.embedding_model.encode(
                "Purchase Order API SAP PO creation"
            ).tolist(),
        }

        # Mock Invoice API
        invoice_api = {
            "source": "mock://sap/invoice",
            "type": "odata",
            "title": "Invoice API",
            "description": "SAP Invoice OData service",
            "endpoints": [
                {
                    "path": "/sap/opu/odata/sap/FI_INVOICE_MAINT_V2/Invoice",
                    "method": "POST",
                    "summary": "Create Invoice",
                    "fields": [
                        "InvoiceNumber",
                        "InvoiceDate",
                        "VendorID",
                        "TotalAmount",
                        "TaxAmount",
                        "Currency",
                    ],
                }
            ],
            "embedding": self.embedding_model.encode(
                "Invoice API SAP FI invoice creation"
            ).tolist(),
        }

        # Mock field mappings
        field_mappings = [
            {
                "source": "mock://sap/fields",
                "field_name": "PO_NUMBER",
                "sap_field": "PurchaseOrderID",
                "description": "Purchase order number",
                "type": "string",
                "pattern": "^[0-9]{10}$",
                "embedding": self.embedding_model.encode(
                    "Purchase order number PO number"
                ).tolist(),
            },
            {
                "source": "mock://sap/fields",
                "field_name": "VENDOR_ID",
                "sap_field": "VendorID",
                "description": "Vendor identifier",
                "type": "string",
                "pattern": "^[0-9]{6}$",
                "embedding": self.embedding_model.encode(
                    "Vendor identifier supplier ID"
                ).tolist(),
            },
            {
                "source": "mock://sap/fields",
                "field_name": "TOTAL_AMOUNT",
                "sap_field": "TotalAmount",
                "description": "Total amount including tax",
                "type": "number",
                "embedding": self.embedding_model.encode(
                    "Total amount sum gross amount"
                ).tolist(),
            },
        ]

        # Mock business rules
        business_rules = [
            {
                "source": "mock://sap/rules",
                "rule_id": "RULE_001",
                "description": "Total amount must equal subtotal plus tax",
                "type": "calculation",
                "formula": "total_amount = subtotal + tax_amount",
                "embedding": self.embedding_model.encode(
                    "Total amount calculation validation subtotal tax"
                ).tolist(),
            },
            {
                "source": "mock://sap/rules",
                "rule_id": "RULE_002",
                "description": "PO number must be 10 digits",
                "type": "validation",
                "pattern": "^[0-9]{10}$",
                "embedding": self.embedding_model.encode(
                    "PO number format validation 10 digits"
                ).tolist(),
            },
        ]

        self.api_schemas = [po_api, invoice_api]
        self.field_mappings = field_mappings
        self.business_rules = business_rules

        logger.info(
            f"Generated mock data: {len(self.api_schemas)} APIs, "
            f"{len(self.field_mappings)} field mappings, "
            f"{len(self.business_rules)} business rules"
        )

        return {
            "api_schemas": self.api_schemas,
            "field_mappings": self.field_mappings,
            "business_rules": self.business_rules,
            "examples": [],
        }
