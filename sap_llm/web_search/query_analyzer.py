"""
Context-Aware Query Analyzer for SAP Domain.

Analyzes search queries and refines them with SAP-specific terminology,
context expansion, and intelligent query variations for better search results.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class QueryAnalyzer:
    """
    Intelligent query analyzer with SAP domain knowledge.

    Features:
    - SAP terminology expansion
    - Query intent detection
    - Context-aware refinement
    - Document type specific queries
    - Synonym expansion
    - Query variation generation

    Example:
        >>> analyzer = QueryAnalyzer()
        >>> refined = analyzer.refine_query(
        ...     "How to get invoice price?",
        ...     context={"document_type": "invoice"}
        ... )
        >>> print(refined)
        ['How to get invoice price?',
         'How to get supplier invoice net price?',
         'SAP invoice pricing condition',
         ...]
    """

    def __init__(self):
        """Initialize query analyzer with SAP domain knowledge."""
        # SAP terminology mappings
        self.sap_term_mappings = {
            # Document types
            "invoice": [
                "supplier invoice",
                "vendor invoice",
                "A/P invoice",
                "incoming invoice",
                "MIRO transaction"
            ],
            "purchase order": [
                "PO",
                "procurement order",
                "requisition",
                "ME21N transaction",
                "purchase document"
            ],
            "sales order": [
                "SO",
                "customer order",
                "VA01 transaction",
                "sales document"
            ],
            "delivery": [
                "outbound delivery",
                "VL01N transaction",
                "goods issue",
                "delivery document"
            ],

            # Pricing
            "price": [
                "net price",
                "gross price",
                "unit price",
                "pricing condition",
                "condition record",
                "pricing procedure"
            ],
            "discount": [
                "pricing condition",
                "condition type",
                "rebate",
                "allowance"
            ],

            # Master data
            "vendor": [
                "supplier",
                "business partner",
                "BP",
                "XK01 transaction",
                "creditor"
            ],
            "customer": [
                "sold-to party",
                "business partner",
                "BP",
                "XD01 transaction",
                "debtor"
            ],
            "material": [
                "product",
                "article",
                "SKU",
                "MM01 transaction",
                "material master"
            ],

            # Financial
            "payment": [
                "payment terms",
                "payment method",
                "F110 transaction",
                "payment run",
                "clearing"
            ],
            "account": [
                "GL account",
                "general ledger",
                "FS00 transaction",
                "chart of accounts"
            ],

            # Logistics
            "warehouse": [
                "storage location",
                "plant",
                "warehouse management",
                "WM",
                "extended warehouse management"
            ],
            "inventory": [
                "stock",
                "material stock",
                "MB51 transaction",
                "inventory management"
            ],

            # Technical
            "API": [
                "OData service",
                "BAPI",
                "function module",
                "web service",
                "REST API"
            ],
            "field": [
                "data element",
                "table field",
                "structure field",
                "DDIC field"
            ]
        }

        # SAP modules
        self.sap_modules = {
            "MM": "Materials Management",
            "SD": "Sales and Distribution",
            "FI": "Financial Accounting",
            "CO": "Controlling",
            "PP": "Production Planning",
            "QM": "Quality Management",
            "PM": "Plant Maintenance",
            "HR": "Human Resources",
            "WM": "Warehouse Management",
            "PS": "Project System"
        }

        # Common SAP patterns
        self.transaction_pattern = re.compile(r'\b[A-Z]{2,4}\d{1,3}[A-Z]?\b')
        self.table_pattern = re.compile(r'\b[A-Z]{3,}_[A-Z0-9]+\b')

        # Query intent patterns
        self.intent_patterns = {
            "how_to": [
                r'\bhow\s+to\b',
                r'\bhow\s+do\s+i\b',
                r'\bhow\s+can\s+i\b'
            ],
            "what_is": [
                r'\bwhat\s+is\b',
                r'\bdefine\b',
                r'\bexplain\b'
            ],
            "troubleshoot": [
                r'\berror\b',
                r'\bissue\b',
                r'\bproblem\b',
                r'\bfailing\b',
                r'\bnot\s+working\b'
            ],
            "api_lookup": [
                r'\bAPI\b',
                r'\bendpoint\b',
                r'\bservice\b',
                r'\bBAPI\b'
            ],
            "configuration": [
                r'\bconfigure\b',
                r'\bsetup\b',
                r'\bsettings\b',
                r'\bcustomizing\b'
            ]
        }

        logger.info("QueryAnalyzer initialized with SAP domain knowledge")

    def refine_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_variations: int = 5
    ) -> List[str]:
        """
        Refine query with SAP-specific terms and context.

        Args:
            query: Original search query
            context: Optional context (document_type, module, etc.)
            max_variations: Maximum number of query variations to generate

        Returns:
            List of refined queries (original + variations)
        """
        context = context or {}
        refined_queries = [query]  # Always include original

        # Detect query intent
        intent = self._analyze_intent(query)
        logger.debug(f"Query intent detected: {intent}")

        # Expand with SAP-specific terms
        sap_variations = self._expand_with_sap_terms(query, intent)
        refined_queries.extend(sap_variations)

        # Add context-specific refinements
        if context.get("document_type"):
            doc_variations = self._add_document_type_context(
                query,
                context["document_type"]
            )
            refined_queries.extend(doc_variations)

        if context.get("module"):
            module_variations = self._add_module_context(
                query,
                context["module"]
            )
            refined_queries.extend(module_variations)

        # Add API-specific queries if applicable
        if intent == "api_lookup" or "API" in query:
            api_variations = self._add_api_context(query)
            refined_queries.extend(api_variations)

        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in refined_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)

        # Limit to max_variations
        result = unique_queries[:max_variations]

        logger.info(
            f"Query refined: '{query}' -> {len(result)} variations "
            f"(intent: {intent})"
        )

        return result

    def _analyze_intent(self, query: str) -> str:
        """
        Analyze query intent.

        Args:
            query: Search query

        Returns:
            Intent classification
        """
        query_lower = query.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent

        return "general"

    def _expand_with_sap_terms(
        self,
        query: str,
        intent: str
    ) -> List[str]:
        """
        Expand query with SAP-specific terminology.

        Args:
            query: Original query
            intent: Query intent

        Returns:
            List of expanded queries
        """
        variations = []
        query_lower = query.lower()

        # Check each SAP term mapping
        for generic_term, sap_terms in self.sap_term_mappings.items():
            if generic_term in query_lower:
                # Generate variations with SAP terms
                for sap_term in sap_terms[:2]:  # Limit to top 2 per term
                    variation = re.sub(
                        rf'\b{re.escape(generic_term)}\b',
                        sap_term,
                        query,
                        flags=re.IGNORECASE
                    )
                    if variation != query:
                        variations.append(variation)

        # Add "SAP" prefix if not present and appropriate
        if "sap" not in query_lower and intent in ["what_is", "how_to", "configuration"]:
            variations.append(f"SAP {query}")

        return variations

    def _add_document_type_context(
        self,
        query: str,
        document_type: str
    ) -> List[str]:
        """
        Add document type specific context to query.

        Args:
            query: Original query
            document_type: Document type (invoice, purchase_order, etc.)

        Returns:
            List of queries with document context
        """
        variations = []
        doc_type_lower = document_type.lower().replace("_", " ")

        # Add document type if not in query
        if doc_type_lower not in query.lower():
            variations.append(f"{query} {doc_type_lower}")
            variations.append(f"SAP {doc_type_lower} {query}")

        # Add SAP-specific document variations
        if doc_type_lower in self.sap_term_mappings:
            for sap_doc_type in self.sap_term_mappings[doc_type_lower][:2]:
                variations.append(f"{query} {sap_doc_type}")

        return variations

    def _add_module_context(
        self,
        query: str,
        module: str
    ) -> List[str]:
        """
        Add SAP module context to query.

        Args:
            query: Original query
            module: SAP module code (MM, SD, FI, etc.)

        Returns:
            List of queries with module context
        """
        variations = []
        module_upper = module.upper()

        if module_upper in self.sap_modules:
            module_name = self.sap_modules[module_upper]

            # Add module code
            if module_upper not in query.upper():
                variations.append(f"SAP {module_upper} {query}")

            # Add module full name
            if module_name.lower() not in query.lower():
                variations.append(f"SAP {module_name} {query}")

        return variations

    def _add_api_context(self, query: str) -> List[str]:
        """
        Add API-specific context to query.

        Args:
            query: Original query

        Returns:
            List of API-specific queries
        """
        variations = []

        # Add API documentation sites
        variations.append(f"{query} site:api.sap.com")
        variations.append(f"{query} OData service")
        variations.append(f"{query} BAPI function module")

        # Add API-specific terms
        if "get" in query.lower():
            variations.append(query.replace("get", "retrieve", 1))
            variations.append(query.replace("get", "fetch", 1))

        return variations

    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract SAP-specific entities from query.

        Args:
            query: Search query

        Returns:
            Dictionary of entity types and values
        """
        entities = {
            "transactions": [],
            "tables": [],
            "modules": [],
            "terms": []
        }

        # Extract transaction codes
        transactions = self.transaction_pattern.findall(query)
        entities["transactions"] = list(set(transactions))

        # Extract table names
        tables = self.table_pattern.findall(query)
        entities["tables"] = list(set(tables))

        # Extract modules
        for module in self.sap_modules.keys():
            if re.search(rf'\b{module}\b', query, re.IGNORECASE):
                entities["modules"].append(module)

        # Extract SAP terms
        query_lower = query.lower()
        for term in self.sap_term_mappings.keys():
            if term in query_lower:
                entities["terms"].append(term)

        return entities

    def suggest_search_domains(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Suggest SAP domains to search based on query.

        Args:
            query: Search query
            context: Optional context

        Returns:
            List of recommended domains
        """
        intent = self._analyze_intent(query)
        entities = self.extract_entities(query)

        domains = []

        # API-related domains
        if intent == "api_lookup" or entities["transactions"]:
            domains.extend([
                "api.sap.com",
                "developers.sap.com"
            ])

        # Documentation domains
        if intent in ["what_is", "how_to"]:
            domains.extend([
                "help.sap.com",
                "support.sap.com"
            ])

        # Community for troubleshooting
        if intent == "troubleshoot":
            domains.extend([
                "community.sap.com",
                "answers.sap.com"
            ])

        # Always include main SAP domains
        if not domains:
            domains = [
                "help.sap.com",
                "api.sap.com",
                "community.sap.com"
            ]

        return list(set(domains))

    def get_search_filters(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate search filters based on query analysis.

        Args:
            query: Search query
            context: Optional context

        Returns:
            Dictionary of search filters
        """
        filters = {}

        # Suggest domains
        domains = self.suggest_search_domains(query, context)
        if domains:
            filters["domains"] = domains

        # Date range for time-sensitive queries
        if any(term in query.lower() for term in ["latest", "recent", "new", "2024", "2025"]):
            filters["date_range"] = "y"  # Past year

        # Add context-based filters
        if context:
            if context.get("require_official_docs"):
                filters["domains"] = ["help.sap.com", "api.sap.com", "support.sap.com"]

        return filters

    def get_stats(self) -> Dict[str, Any]:
        """
        Get query analyzer statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "sap_term_count": len(self.sap_term_mappings),
            "module_count": len(self.sap_modules),
            "intent_types": len(self.intent_patterns)
        }
