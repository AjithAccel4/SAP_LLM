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
import xml.etree.ElementTree as ET
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
        logger.debug(f"Processing XML from {url}")

        try:
            root = ET.fromstring(content)

            # Determine XML type based on root element and namespaces
            if "edmx" in root.tag.lower() or "metadata" in url.lower():
                # EDMX (Entity Data Model XML) - OData metadata
                await self._parse_edmx(url, root)

            elif "wsdl" in root.tag.lower() or "definitions" in root.tag.lower():
                # WSDL (Web Services Description Language)
                await self._parse_wsdl(url, root)

            else:
                # Generic XML schema
                logger.debug(f"Processing generic XML schema from {url}")
                await self._parse_generic_xml(url, root)

        except ET.ParseError as e:
            logger.error(f"Failed to parse XML from {url}: {e}")
        except Exception as e:
            logger.error(f"Error processing XML from {url}: {e}")

    async def _parse_edmx(self, url: str, root: ET.Element) -> None:
        """
        Parse EDMX (OData metadata) XML.

        Args:
            url: Source URL
            root: XML root element
        """
        try:
            # EDMX namespaces
            namespaces = {
                "edmx": "http://docs.oasis-open.org/odata/ns/edmx",
                "edm": "http://docs.oasis-open.org/odata/ns/edm",
                "": "http://schemas.microsoft.com/ado/2009/11/edm",
            }

            # Extract service info
            api_info = {
                "source": url,
                "type": "odata",
                "title": "OData Service",
                "description": f"OData metadata from {url}",
                "entities": [],
                "endpoints": [],
            }

            # Find all EntityType definitions
            for entity in root.findall(".//edm:EntityType", namespaces):
                entity_name = entity.get("Name", "Unknown")

                # Extract properties/fields
                fields = []
                for prop in entity.findall(".//edm:Property", namespaces):
                    field_info = {
                        "name": prop.get("Name", ""),
                        "type": prop.get("Type", ""),
                        "nullable": prop.get("Nullable", "true"),
                        "max_length": prop.get("MaxLength"),
                    }
                    fields.append(field_info)

                    # Store field mapping
                    await self._store_field_from_edmx(url, entity_name, field_info)

                # Extract navigation properties (relationships)
                nav_props = []
                for nav in entity.findall(".//edm:NavigationProperty", namespaces):
                    nav_info = {
                        "name": nav.get("Name", ""),
                        "type": nav.get("Type", ""),
                        "partner": nav.get("Partner"),
                    }
                    nav_props.append(nav_info)

                entity_info = {
                    "name": entity_name,
                    "fields": fields,
                    "navigation": nav_props,
                }

                api_info["entities"].append(entity_info)

                # Create endpoint for entity
                endpoint = {
                    "path": f"/{entity_name}",
                    "method": "GET",
                    "summary": f"Query {entity_name}",
                    "description": f"OData endpoint for {entity_name} entity",
                }
                api_info["endpoints"].append(endpoint)

            # Generate embedding
            entities_text = " ".join([e["name"] for e in api_info["entities"]])
            api_text = f"{api_info['title']} {entities_text}"
            api_info["embedding"] = self.embedding_model.encode(api_text).tolist()

            self.api_schemas.append(api_info)
            logger.info(f"Extracted EDMX schema with {len(api_info['entities'])} entities")

        except Exception as e:
            logger.error(f"Error parsing EDMX: {e}")

    async def _parse_wsdl(self, url: str, root: ET.Element) -> None:
        """
        Parse WSDL (SOAP service) XML.

        Args:
            url: Source URL
            root: XML root element
        """
        try:
            # WSDL namespaces
            namespaces = {
                "wsdl": "http://schemas.xmlsoap.org/wsdl/",
                "soap": "http://schemas.xmlsoap.org/wsdl/soap/",
                "xsd": "http://www.w3.org/2001/XMLSchema",
            }

            # Extract service info
            service_elem = root.find(".//wsdl:service", namespaces)
            service_name = service_elem.get("name") if service_elem is not None else "SOAP Service"

            api_info = {
                "source": url,
                "type": "soap",
                "title": service_name,
                "description": f"SOAP service from {url}",
                "operations": [],
                "endpoints": [],
            }

            # Find all operations
            for operation in root.findall(".//wsdl:operation", namespaces):
                op_name = operation.get("name", "Unknown")

                # Extract input/output messages
                input_elem = operation.find(".//wsdl:input", namespaces)
                output_elem = operation.find(".//wsdl:output", namespaces)

                operation_info = {
                    "name": op_name,
                    "input": input_elem.get("message") if input_elem is not None else None,
                    "output": output_elem.get("message") if output_elem is not None else None,
                }

                api_info["operations"].append(operation_info)

                # Create endpoint
                endpoint = {
                    "path": f"/{op_name}",
                    "method": "POST",
                    "summary": f"SOAP operation: {op_name}",
                    "description": f"SOAP operation {op_name}",
                }
                api_info["endpoints"].append(endpoint)

            # Generate embedding
            ops_text = " ".join([op["name"] for op in api_info["operations"]])
            api_text = f"{api_info['title']} {ops_text}"
            api_info["embedding"] = self.embedding_model.encode(api_text).tolist()

            self.api_schemas.append(api_info)
            logger.info(f"Extracted WSDL schema with {len(api_info['operations'])} operations")

        except Exception as e:
            logger.error(f"Error parsing WSDL: {e}")

    async def _parse_generic_xml(self, url: str, root: ET.Element) -> None:
        """
        Parse generic XML schema.

        Args:
            url: Source URL
            root: XML root element
        """
        try:
            # Extract fields from XML structure
            fields = self._extract_xml_fields(root)

            api_info = {
                "source": url,
                "type": "xml",
                "title": root.tag,
                "description": f"XML schema from {url}",
                "fields": fields,
            }

            # Generate embedding
            fields_text = " ".join([f["name"] for f in fields])
            api_info["embedding"] = self.embedding_model.encode(fields_text).tolist()

            self.api_schemas.append(api_info)
            logger.info(f"Extracted generic XML schema with {len(fields)} fields")

        except Exception as e:
            logger.error(f"Error parsing generic XML: {e}")

    def _extract_xml_fields(self, element: ET.Element, prefix: str = "") -> List[Dict[str, Any]]:
        """
        Recursively extract fields from XML element.

        Args:
            element: XML element
            prefix: Field name prefix

        Returns:
            List of field definitions
        """
        fields = []

        for child in element:
            # Get tag name without namespace
            tag_name = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            field_name = f"{prefix}.{tag_name}" if prefix else tag_name

            field_info = {
                "name": field_name,
                "type": "string",
                "value": child.text.strip() if child.text else None,
            }

            # Add attributes
            if child.attrib:
                field_info["attributes"] = child.attrib

            fields.append(field_info)

            # Recursively process children
            if len(child) > 0:
                fields.extend(self._extract_xml_fields(child, field_name))

        return fields

    async def _store_field_from_edmx(
        self, url: str, entity_name: str, field_info: Dict[str, Any]
    ) -> None:
        """
        Store field mapping extracted from EDMX.

        Args:
            url: Source URL
            entity_name: Entity name
            field_info: Field information
        """
        mapping = {
            "source": url,
            "entity": entity_name,
            "field_name": field_info["name"],
            "sap_field": field_info["name"],
            "type": field_info["type"],
            "description": f"{entity_name}.{field_info['name']}",
            "nullable": field_info.get("nullable", "true") == "true",
            "max_length": field_info.get("max_length"),
        }

        # Generate embedding
        mapping_text = f"{entity_name} {field_info['name']} {field_info['type']}"
        mapping["embedding"] = self.embedding_model.encode(mapping_text).tolist()

        self.field_mappings.append(mapping)

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
        logger.debug(f"Extracting OData schema from {url}")

        try:
            api_info = {
                "source": url,
                "type": "odata",
                "title": "OData Service",
                "description": f"OData service from {url}",
                "entities": [],
                "endpoints": [],
            }

            # Check for different OData metadata formats
            # OData v4 JSON format
            if "@odata.context" in metadata or "value" in metadata:
                # Parse OData v4 JSON metadata
                await self._parse_odata_v4_json(url, metadata, api_info)

            # OData v2/v3 JSON format
            elif "d" in metadata and "EntitySets" in metadata.get("d", {}):
                # Parse OData v2/v3 JSON metadata
                await self._parse_odata_v2_json(url, metadata, api_info)

            # Check for entity collections in response
            elif "d" in metadata and "results" in metadata.get("d", {}):
                # This is a data response, extract entity structure
                results = metadata["d"]["results"]
                if results:
                    await self._extract_entity_from_data(url, results[0], api_info)

            # Generic OData response
            else:
                # Try to infer structure from response
                await self._extract_generic_odata_structure(url, metadata, api_info)

            # Only add if we found entities
            if api_info["entities"] or api_info["endpoints"]:
                # Generate embedding
                entities_text = " ".join([e.get("name", "") for e in api_info["entities"]])
                api_text = f"{api_info['title']} {entities_text}"
                api_info["embedding"] = self.embedding_model.encode(api_text).tolist()

                self.api_schemas.append(api_info)
                logger.info(f"Extracted OData schema with {len(api_info['entities'])} entities")

        except Exception as e:
            logger.error(f"Error extracting OData schema: {e}")

    async def _parse_odata_v4_json(
        self, url: str, metadata: Dict[str, Any], api_info: Dict[str, Any]
    ) -> None:
        """
        Parse OData v4 JSON metadata.

        Args:
            url: Source URL
            metadata: Metadata dictionary
            api_info: API info to populate
        """
        # Extract context
        context = metadata.get("@odata.context", "")
        api_info["title"] = f"OData v4 Service: {context.split('/')[-1]}"

        # Parse entity sets from value array
        if "value" in metadata:
            for entity_set in metadata["value"]:
                entity_name = entity_set.get("name", "")
                entity_kind = entity_set.get("kind", "")

                if entity_kind == "EntitySet":
                    entity_info = {
                        "name": entity_name,
                        "kind": entity_kind,
                        "url": entity_set.get("url", entity_name),
                    }

                    api_info["entities"].append(entity_info)

                    # Create CRUD endpoints
                    for method, action in [
                        ("GET", "Query"),
                        ("POST", "Create"),
                        ("PUT", "Update"),
                        ("DELETE", "Delete"),
                    ]:
                        endpoint = {
                            "path": f"/{entity_name}",
                            "method": method,
                            "summary": f"{action} {entity_name}",
                            "description": f"OData {action} operation for {entity_name}",
                        }
                        api_info["endpoints"].append(endpoint)

    async def _parse_odata_v2_json(
        self, url: str, metadata: Dict[str, Any], api_info: Dict[str, Any]
    ) -> None:
        """
        Parse OData v2/v3 JSON metadata.

        Args:
            url: Source URL
            metadata: Metadata dictionary
            api_info: API info to populate
        """
        api_info["title"] = "OData v2/v3 Service"

        # Extract entity sets
        d_obj = metadata.get("d", {})
        entity_sets = d_obj.get("EntitySets", [])

        for entity_name in entity_sets:
            entity_info = {
                "name": entity_name,
                "kind": "EntitySet",
            }

            api_info["entities"].append(entity_info)

            # Create endpoints
            endpoint = {
                "path": f"/{entity_name}",
                "method": "GET",
                "summary": f"Query {entity_name}",
                "description": f"OData query for {entity_name}",
            }
            api_info["endpoints"].append(endpoint)

    async def _extract_entity_from_data(
        self, url: str, sample_data: Dict[str, Any], api_info: Dict[str, Any]
    ) -> None:
        """
        Extract entity structure from OData data response.

        Args:
            url: Source URL
            sample_data: Sample data record
            api_info: API info to populate
        """
        # Infer entity name from metadata
        metadata = sample_data.get("__metadata", {})
        entity_type = metadata.get("type", "Entity")

        # Extract fields from sample data
        fields = []
        for key, value in sample_data.items():
            if not key.startswith("__"):
                field_info = {
                    "name": key,
                    "type": type(value).__name__,
                    "sample_value": str(value)[:50] if value else None,
                }
                fields.append(field_info)

                # Store field mapping
                mapping = {
                    "source": url,
                    "entity": entity_type,
                    "field_name": key,
                    "sap_field": key,
                    "type": field_info["type"],
                    "description": f"{entity_type}.{key}",
                }

                mapping_text = f"{entity_type} {key}"
                mapping["embedding"] = self.embedding_model.encode(mapping_text).tolist()
                self.field_mappings.append(mapping)

        entity_info = {
            "name": entity_type,
            "fields": fields,
        }

        api_info["entities"].append(entity_info)

    async def _extract_generic_odata_structure(
        self, url: str, metadata: Dict[str, Any], api_info: Dict[str, Any]
    ) -> None:
        """
        Extract generic OData structure from response.

        Args:
            url: Source URL
            metadata: Metadata dictionary
            api_info: API info to populate
        """
        # Try to find entity collections
        for key, value in metadata.items():
            if isinstance(value, dict) and "results" in value:
                # Found entity collection
                results = value["results"]
                if results and isinstance(results, list):
                    await self._extract_entity_from_data(url, results[0], api_info)

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
        logger.debug(f"Extracting fields from {endpoint}")

        try:
            # Navigate to the actual schema content
            content = schema.get("content", {})

            for content_type, content_schema in content.items():
                schema_def = content_schema.get("schema", {})

                # Extract fields from schema definition
                await self._extract_fields_recursive(
                    url, endpoint, schema_def, content_type
                )

        except Exception as e:
            logger.error(f"Error extracting fields from schema: {e}")

    async def _extract_fields_recursive(
        self,
        url: str,
        endpoint: str,
        schema: Dict[str, Any],
        content_type: str,
        prefix: str = "",
        parent_required: Optional[List[str]] = None,
    ) -> None:
        """
        Recursively extract fields from nested schema.

        Args:
            url: Source URL
            endpoint: API endpoint
            schema: Schema definition
            content_type: Content type (e.g., application/json)
            prefix: Field name prefix for nested fields
            parent_required: Required fields from parent level
        """
        try:
            # Handle schema references ($ref)
            if "$ref" in schema:
                # Reference to another schema - we'll skip for now
                # In production, you'd resolve these references
                logger.debug(f"Skipping $ref schema: {schema['$ref']}")
                return

            # Get schema type
            schema_type = schema.get("type", "object")

            # Extract properties for object type
            if schema_type == "object":
                properties = schema.get("properties", {})
                required_fields = schema.get("required", parent_required or [])

                for prop_name, prop_def in properties.items():
                    field_name = f"{prefix}.{prop_name}" if prefix else prop_name

                    # Extract field details
                    field_mapping = await self._create_field_mapping(
                        url=url,
                        endpoint=endpoint,
                        field_name=field_name,
                        field_def=prop_def,
                        content_type=content_type,
                        required=prop_name in required_fields,
                    )

                    if field_mapping:
                        self.field_mappings.append(field_mapping)

                    # Recursively process nested objects
                    if prop_def.get("type") == "object":
                        await self._extract_fields_recursive(
                            url, endpoint, prop_def, content_type, field_name, required_fields
                        )

                    # Process array items
                    elif prop_def.get("type") == "array":
                        items = prop_def.get("items", {})
                        if items.get("type") == "object":
                            await self._extract_fields_recursive(
                                url, endpoint, items, content_type, f"{field_name}[]", required_fields
                            )

            # Handle allOf, anyOf, oneOf schemas
            for combine_key in ["allOf", "anyOf", "oneOf"]:
                if combine_key in schema:
                    for sub_schema in schema[combine_key]:
                        await self._extract_fields_recursive(
                            url, endpoint, sub_schema, content_type, prefix, parent_required
                        )

        except Exception as e:
            logger.error(f"Error in recursive field extraction: {e}")

    async def _create_field_mapping(
        self,
        url: str,
        endpoint: str,
        field_name: str,
        field_def: Dict[str, Any],
        content_type: str,
        required: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Create detailed field mapping from OpenAPI field definition.

        Args:
            url: Source URL
            endpoint: API endpoint
            field_name: Field name
            field_def: Field definition from OpenAPI schema
            content_type: Content type
            required: Whether field is required

        Returns:
            Field mapping dictionary or None
        """
        try:
            # Extract field properties
            field_type = field_def.get("type", "string")
            description = field_def.get("description", "")
            format_type = field_def.get("format")
            default_value = field_def.get("default")
            example = field_def.get("example")

            # Constraints
            min_length = field_def.get("minLength")
            max_length = field_def.get("maxLength")
            minimum = field_def.get("minimum")
            maximum = field_def.get("maximum")
            pattern = field_def.get("pattern")
            enum = field_def.get("enum")

            # Create comprehensive field mapping
            mapping = {
                "source": url,
                "endpoint": endpoint,
                "field_name": field_name,
                "sap_field": field_name,  # Default to same name
                "type": field_type,
                "format": format_type,
                "description": description,
                "required": required,
                "content_type": content_type,
            }

            # Add optional properties
            if default_value is not None:
                mapping["default"] = default_value

            if example is not None:
                mapping["example"] = example

            if min_length is not None:
                mapping["min_length"] = min_length

            if max_length is not None:
                mapping["max_length"] = max_length

            if minimum is not None:
                mapping["minimum"] = minimum

            if maximum is not None:
                mapping["maximum"] = maximum

            if pattern:
                mapping["pattern"] = pattern

            if enum:
                mapping["enum"] = enum

            # Extract business rules from field definition
            await self._extract_rules_from_field(url, field_name, field_def, mapping)

            # Generate embedding for semantic search
            embedding_text = f"{field_name} {description} {field_type}"
            if format_type:
                embedding_text += f" {format_type}"

            mapping["embedding"] = self.embedding_model.encode(embedding_text).tolist()

            return mapping

        except Exception as e:
            logger.error(f"Error creating field mapping for {field_name}: {e}")
            return None

    async def _extract_rules_from_field(
        self,
        url: str,
        field_name: str,
        field_def: Dict[str, Any],
        mapping: Dict[str, Any],
    ) -> None:
        """
        Extract business rules from field definition.

        Args:
            url: Source URL
            field_name: Field name
            field_def: Field definition
            mapping: Field mapping (for context)
        """
        try:
            # Create validation rules from constraints
            if "pattern" in field_def:
                rule = {
                    "source": url,
                    "rule_id": f"PATTERN_{field_name}",
                    "description": f"Field '{field_name}' must match pattern: {field_def['pattern']}",
                    "type": "validation",
                    "pattern": field_def["pattern"],
                    "field": field_name,
                }

                rule_text = f"{field_name} pattern validation {field_def['pattern']}"
                rule["embedding"] = self.embedding_model.encode(rule_text).tolist()
                self.business_rules.append(rule)

            # Required field rule
            if mapping.get("required"):
                rule = {
                    "source": url,
                    "rule_id": f"REQUIRED_{field_name}",
                    "description": f"Field '{field_name}' is required",
                    "type": "required",
                    "field": field_name,
                }

                rule_text = f"{field_name} required validation"
                rule["embedding"] = self.embedding_model.encode(rule_text).tolist()
                self.business_rules.append(rule)

            # Range validation
            if "minimum" in field_def or "maximum" in field_def:
                min_val = field_def.get("minimum", "")
                max_val = field_def.get("maximum", "")

                rule = {
                    "source": url,
                    "rule_id": f"RANGE_{field_name}",
                    "description": f"Field '{field_name}' must be between {min_val} and {max_val}",
                    "type": "range",
                    "field": field_name,
                    "minimum": min_val,
                    "maximum": max_val,
                }

                rule_text = f"{field_name} range validation {min_val} {max_val}"
                rule["embedding"] = self.embedding_model.encode(rule_text).tolist()
                self.business_rules.append(rule)

            # Enum validation
            if "enum" in field_def:
                enum_values = field_def["enum"]

                rule = {
                    "source": url,
                    "rule_id": f"ENUM_{field_name}",
                    "description": f"Field '{field_name}' must be one of: {', '.join(map(str, enum_values))}",
                    "type": "validation",
                    "field": field_name,
                    "enum": enum_values,
                }

                rule_text = f"{field_name} enum validation {' '.join(map(str, enum_values))}"
                rule["embedding"] = self.embedding_model.encode(rule_text).tolist()
                self.business_rules.append(rule)

        except Exception as e:
            logger.error(f"Error extracting rules from field {field_name}: {e}")

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
