"""
PMG Query Engine - High-level queries for common use cases
"""

from typing import Any, Dict, List, Optional

from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.pmg.vector_store import PMGVectorStore
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class PMGQueryEngine:
    """
    High-level query interface for PMG.

    Combines graph queries and vector search for common use cases.
    """

    def __init__(
        self,
        graph: ProcessMemoryGraph,
        vector_store: Optional[PMGVectorStore] = None,
    ):
        """
        Initialize query engine.

        Args:
            graph: PMG graph client
            vector_store: Optional vector store for semantic search
        """
        self.graph = graph
        self.vector_store = vector_store

        logger.info("PMG Query Engine initialized")

    def get_routing_context(
        self,
        doc_type: str,
        supplier_id: Optional[str] = None,
        company_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get routing context for a document.

        Returns similar successful routing decisions and statistics.

        Args:
            doc_type: Document type
            supplier_id: Optional supplier filter
            company_code: Optional company code filter

        Returns:
            Routing context dictionary
        """
        # Get similar routing decisions
        similar_routings = self.graph.get_similar_routing(
            doc_type=doc_type,
            supplier=supplier_id,
            company_code=company_code,
            limit=20,
        )

        # Calculate statistics
        if similar_routings:
            endpoints = [r.get("endpoint") for r in similar_routings]
            most_common_endpoint = max(set(endpoints), key=endpoints.count)
            success_rate = len(similar_routings) / max(len(similar_routings), 1)
        else:
            most_common_endpoint = None
            success_rate = 0.0

        return {
            "similar_routings": similar_routings[:5],  # Top 5
            "num_similar": len(similar_routings),
            "recommended_endpoint": most_common_endpoint,
            "success_rate": success_rate,
        }

    def get_exception_patterns(
        self,
        category: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get exception patterns and statistics.

        Args:
            category: Optional exception category filter
            days: Lookback period

        Returns:
            Exception pattern analysis
        """
        exceptions = self.graph.query_exceptions(
            days=days,
            category=category,
        )

        if not exceptions:
            return {
                "total_exceptions": 0,
                "by_category": {},
                "by_severity": {},
                "trending": [],
            }

        # Analyze by category
        by_category = {}
        for exc in exceptions:
            cat = exc.get("category", "UNKNOWN")
            by_category[cat] = by_category.get(cat, 0) + 1

        # Analyze by severity
        by_severity = {}
        for exc in exceptions:
            sev = exc.get("severity", "MEDIUM")
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_exceptions": len(exceptions),
            "by_category": by_category,
            "by_severity": by_severity,
            "trending": list(by_category.items())[:10],
        }

    def find_similar_exceptions(
        self,
        exception: Dict[str, Any],
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find similar exceptions using vector search.

        Args:
            exception: Exception to find similar cases for
            limit: Maximum results

        Returns:
            List of similar exceptions
        """
        if self.vector_store is None:
            logger.warning("Vector store not available, using graph query")
            return self._find_similar_exceptions_graph(exception, limit)

        # Create search query
        query = self._exception_to_text(exception)

        # Search vector store
        results = self.vector_store.search(query, k=limit, min_similarity=0.7)

        return results

    def _find_similar_exceptions_graph(
        self,
        exception: Dict[str, Any],
        limit: int,
    ) -> List[Dict[str, Any]]:
        """Fallback to graph-based exception search."""
        return self.graph.query_exceptions(
            category=exception.get("category"),
            severity=exception.get("severity"),
        )[:limit]

    def _exception_to_text(self, exception: Dict[str, Any]) -> str:
        """Convert exception to searchable text."""
        parts = [
            f"Category: {exception.get('category', '')}",
            f"Severity: {exception.get('severity', '')}",
            f"Field: {exception.get('field', '')}",
            f"Message: {exception.get('message', '')}",
        ]
        return " | ".join(parts)

    def get_document_statistics(
        self,
        doc_type: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get document processing statistics.

        Args:
            doc_type: Optional document type filter
            days: Lookback period

        Returns:
            Processing statistics
        """
        # This would query PMG for various metrics
        # For now, return mock data

        return {
            "total_documents": 0,
            "by_type": {},
            "avg_confidence": 0.0,
            "success_rate": 0.0,
            "avg_processing_time_ms": 0,
        }

    def get_supplier_history(
        self,
        supplier_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get processing history for a supplier.

        Args:
            supplier_id: Supplier ID
            limit: Maximum results

        Returns:
            List of documents for this supplier
        """
        # Find all documents for this supplier
        documents = self.graph.find_similar_documents(
            doc_type="SUPPLIER_INVOICE",
            supplier_id=supplier_id,
            limit=limit,
        )

        return documents

    def verify_three_way_match(
        self,
        po_number: str,
        invoice_number: str,
        gr_number: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify three-way match between PO, Invoice, and GR.

        Args:
            po_number: Purchase order number
            invoice_number: Invoice number
            gr_number: Optional goods receipt number

        Returns:
            Match verification results
        """
        # Get documents from PMG
        po_doc = None  # self.graph.get_document_by_number("PO", po_number)
        invoice_doc = None  # self.graph.get_document_by_number("INV", invoice_number)
        gr_doc = None  # self.graph.get_document_by_number("GR", gr_number) if gr_number else None

        # Verify match
        # In production, would check amounts, quantities, etc.

        return {
            "po_found": po_doc is not None,
            "invoice_found": invoice_doc is not None,
            "gr_found": gr_doc is not None if gr_number else None,
            "price_match": False,
            "quantity_match": False,
            "variance_percentage": 0.0,
        }
