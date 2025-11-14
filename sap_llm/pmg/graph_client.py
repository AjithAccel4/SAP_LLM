"""
Process Memory Graph Client - Cosmos DB Gremlin API Interface

Manages graph database operations for storing and retrieving processing history,
routing decisions, exceptions, and business rules.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from gremlin_python.driver import client, serializer
from gremlin_python.driver.protocol import GremlinServerError

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ProcessMemoryGraph:
    """
    Interface to Cosmos DB Gremlin API for Process Memory Graph.

    Graph Schema:
    - Nodes: Document, Rule, Exception, RoutingDecision, SAPResponse
    - Edges: CLASSIFIED_AS, VALIDATED_BY, RAISED_EXCEPTION, ROUTED_TO, GOT_RESPONSE, SIMILAR_TO
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        database: str = "qorsync",
        container: str = "pmg",
    ):
        """
        Initialize PMG client.

        Args:
            endpoint: Cosmos DB endpoint (or from COSMOS_ENDPOINT env var)
            key: Cosmos DB key (or from COSMOS_KEY env var)
            database: Database name
            container: Container/collection name
        """
        self.endpoint = endpoint or os.getenv("COSMOS_ENDPOINT")
        self.key = key or os.getenv("COSMOS_KEY")
        self.database = database
        self.container = container

        if not self.endpoint or not self.key:
            logger.warning(
                "Cosmos DB credentials not provided. PMG will run in mock mode."
            )
            self.client = None
            self.mock_mode = True
        else:
            self.mock_mode = False
            self._init_client()

        logger.info(f"PMG initialized (mock_mode={self.mock_mode})")

    def _init_client(self):
        """Initialize Gremlin client."""
        try:
            # Extract host from endpoint
            host = self.endpoint.replace("https://", "").replace(":443/", "")

            self.client = client.Client(
                f"wss://{host}:443/",
                "g",
                username=f"/dbs/{self.database}/colls/{self.container}",
                password=self.key,
                message_serializer=serializer.GraphSONSerializersV2d0(),
            )

            logger.info("Gremlin client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Gremlin client: {e}")
            self.client = None
            self.mock_mode = True

    def store_transaction(
        self,
        document: Dict[str, Any],
        routing_decision: Optional[Dict[str, Any]] = None,
        sap_response: Optional[Dict[str, Any]] = None,
        exceptions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Store complete transaction in PMG.

        Args:
            document: Extracted document data (ADC format)
            routing_decision: Routing decision details
            sap_response: SAP API response
            exceptions: List of exceptions/violations

        Returns:
            Document ID (UUID)
        """
        doc_id = str(uuid.uuid4())

        if self.mock_mode:
            logger.debug(f"[MOCK] Storing transaction: {doc_id}")
            return doc_id

        try:
            # Create document vertex
            self._create_document_vertex(doc_id, document)

            # Create routing decision vertex
            if routing_decision:
                decision_id = self._create_routing_decision_vertex(
                    routing_decision
                )
                self._create_edge(doc_id, decision_id, "ROUTED_TO")

                # Create SAP response vertex
                if sap_response:
                    response_id = self._create_sap_response_vertex(sap_response)
                    self._create_edge(decision_id, response_id, "GOT_RESPONSE")

            # Create exception vertices
            if exceptions:
                for exc in exceptions:
                    exc_id = self._create_exception_vertex(exc)
                    self._create_edge(doc_id, exc_id, "RAISED_EXCEPTION")

            logger.info(f"Transaction stored: {doc_id}")
            return doc_id

        except Exception as e:
            logger.error(f"Failed to store transaction: {e}")
            return doc_id

    def _create_document_vertex(self, doc_id: str, document: Dict[str, Any]):
        """Create document vertex."""
        query = f"""
        g.addV('Document')
          .property('id', '{doc_id}')
          .property('doc_type', '{document.get('doc_type', '')}')
          .property('doc_subtype', '{document.get('doc_subtype', '')}')
          .property('supplier_id', '{document.get('supplier_id', '')}')
          .property('company_code', '{document.get('company_code', '')}')
          .property('total_amount', {document.get('total_amount', 0)})
          .property('currency', '{document.get('currency', 'USD')}')
          .property('ingestion_timestamp', '{datetime.now().isoformat()}')
          .property('adc_json', '{self._escape_json(document)}')
        """
        self._execute_query(query)

    def _create_routing_decision_vertex(
        self, decision: Dict[str, Any]
    ) -> str:
        """Create routing decision vertex."""
        decision_id = str(uuid.uuid4())

        query = f"""
        g.addV('RoutingDecision')
          .property('id', '{decision_id}')
          .property('endpoint', '{decision.get('endpoint', '')}')
          .property('method', '{decision.get('method', 'POST')}')
          .property('confidence', {decision.get('confidence', 0.0)})
          .property('reasoning', '{self._escape_string(decision.get('reasoning', ''))}')
          .property('timestamp', '{datetime.now().isoformat()}')
        """
        self._execute_query(query)
        return decision_id

    def _create_sap_response_vertex(self, response: Dict[str, Any]) -> str:
        """Create SAP response vertex."""
        response_id = str(uuid.uuid4())

        query = f"""
        g.addV('SAPResponse')
          .property('id', '{response_id}')
          .property('status_code', {response.get('status_code', 0)})
          .property('success', {str(response.get('success', False)).lower()})
          .property('timestamp', '{datetime.now().isoformat()}')
        """
        self._execute_query(query)
        return response_id

    def _create_exception_vertex(self, exception: Dict[str, Any]) -> str:
        """Create exception vertex."""
        exc_id = str(uuid.uuid4())

        query = f"""
        g.addV('Exception')
          .property('id', '{exc_id}')
          .property('category', '{exception.get('category', '')}')
          .property('severity', '{exception.get('severity', 'MEDIUM')}')
          .property('field', '{exception.get('field', '')}')
          .property('message', '{self._escape_string(exception.get('message', ''))}')
          .property('timestamp', '{datetime.now().isoformat()}')
        """
        self._execute_query(query)
        return exc_id

    def _create_edge(self, from_id: str, to_id: str, label: str):
        """Create edge between vertices."""
        query = f"""
        g.V('{from_id}').addE('{label}').to(g.V('{to_id}'))
        """
        self._execute_query(query)

    def find_similar_documents(
        self,
        doc_type: str,
        supplier_id: Optional[str] = None,
        company_code: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Find similar documents from PMG.

        Args:
            doc_type: Document type
            supplier_id: Optional supplier filter
            company_code: Optional company code filter
            limit: Maximum results

        Returns:
            List of similar documents
        """
        if self.mock_mode:
            logger.debug("[MOCK] Finding similar documents")
            return []

        try:
            query = f"g.V().has('Document', 'doc_type', '{doc_type}')"

            if supplier_id:
                query += f".has('supplier_id', '{supplier_id}')"

            if company_code:
                query += f".has('company_code', '{company_code}')"

            query += f".limit({limit}).valueMap()"

            results = self._execute_query(query)
            return [self._parse_vertex(r) for r in results]

        except Exception as e:
            logger.error(f"Failed to find similar documents: {e}")
            return []

    def get_similar_routing(
        self,
        doc_type: str,
        supplier: Optional[str] = None,
        company_code: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Query PMG for similar successful routing decisions.

        Args:
            doc_type: Document type
            supplier: Optional supplier filter
            company_code: Optional company code filter
            limit: Maximum results

        Returns:
            List of routing decisions
        """
        if self.mock_mode:
            logger.debug("[MOCK] Getting similar routing")
            return []

        try:
            query = f"""
            g.V().has('Document', 'doc_type', '{doc_type}')
            """

            if supplier:
                query += f".has('supplier_id', '{supplier}')"

            if company_code:
                query += f".has('company_code', '{company_code}')"

            query += f"""
              .out('ROUTED_TO')
              .as('routing')
              .out('GOT_RESPONSE')
              .has('success', true)
              .select('routing')
              .limit({limit})
              .valueMap()
            """

            results = self._execute_query(query)
            return [self._parse_vertex(r) for r in results]

        except Exception as e:
            logger.error(f"Failed to get similar routing: {e}")
            return []

    def query_exceptions(
        self,
        days: int = 7,
        category: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query exceptions from PMG.

        Args:
            days: Lookback period in days
            category: Optional category filter
            severity: Optional severity filter

        Returns:
            List of exceptions
        """
        if self.mock_mode:
            logger.debug("[MOCK] Querying exceptions")
            return []

        try:
            # Calculate timestamp threshold
            from datetime import timedelta

            threshold = (datetime.now() - timedelta(days=days)).isoformat()

            query = f"""
            g.V().hasLabel('Exception')
              .has('timestamp', gt('{threshold}'))
            """

            if category:
                query += f".has('category', '{category}')"

            if severity:
                query += f".has('severity', '{severity}')"

            query += ".valueMap()"

            results = self._execute_query(query)
            return [self._parse_vertex(r) for r in results]

        except Exception as e:
            logger.error(f"Failed to query exceptions: {e}")
            return []

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        if self.mock_mode:
            return None

        try:
            query = f"g.V('{doc_id}').valueMap()"
            results = self._execute_query(query)

            if results:
                return self._parse_vertex(results[0])

            return None

        except Exception as e:
            logger.error(f"Failed to get document: {e}")
            return None

    def get_workflow_by_correlation_id(
        self, correlation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve workflow history by correlation ID.

        Args:
            correlation_id: Correlation ID to search for

        Returns:
            Workflow history with steps and status
        """
        if self.mock_mode:
            logger.debug(f"[MOCK] Getting workflow for correlation_id: {correlation_id}")
            return None

        try:
            # Query for documents with correlation_id in their metadata
            # Note: This assumes correlation_id is stored in adc_json or as a property
            query = f"""
            g.V().hasLabel('Document')
              .has('id', '{correlation_id}')
              .project('document', 'routing_decisions', 'responses', 'exceptions')
              .by(valueMap())
              .by(out('ROUTED_TO').valueMap().fold())
              .by(out('ROUTED_TO').out('GOT_RESPONSE').valueMap().fold())
              .by(out('RAISED_EXCEPTION').valueMap().fold())
            """

            results = self._execute_query(query)

            if results and len(results) > 0:
                result = results[0]

                # Parse the workflow data
                document = self._parse_vertex(result.get("document", {}))
                routing_decisions = [
                    self._parse_vertex(r) for r in result.get("routing_decisions", [])
                ]
                responses = [
                    self._parse_vertex(r) for r in result.get("responses", [])
                ]
                exceptions = [
                    self._parse_vertex(e) for e in result.get("exceptions", [])
                ]

                return {
                    "correlation_id": correlation_id,
                    "document": document,
                    "routing_decisions": routing_decisions,
                    "responses": responses,
                    "exceptions": exceptions,
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get workflow by correlation_id: {e}")
            return None

    def get_workflow_steps(
        self, correlation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get ordered workflow steps by correlation ID.

        Args:
            correlation_id: Correlation ID

        Returns:
            List of workflow steps in chronological order
        """
        if self.mock_mode:
            logger.debug(f"[MOCK] Getting workflow steps for: {correlation_id}")
            return []

        try:
            workflow = self.get_workflow_by_correlation_id(correlation_id)

            if not workflow:
                return []

            # Build step history from routing decisions
            steps = []
            for idx, decision in enumerate(workflow.get("routing_decisions", [])):
                step = {
                    "step_number": idx + 1,
                    "endpoint": decision.get("endpoint", "unknown"),
                    "timestamp": decision.get("timestamp"),
                    "confidence": decision.get("confidence", 0.0),
                    "status": "completed",
                }
                steps.append(step)

            return steps

        except Exception as e:
            logger.error(f"Failed to get workflow steps: {e}")
            return []

    def _execute_query(self, query: str) -> List[Any]:
        """Execute Gremlin query."""
        if self.client is None:
            return []

        try:
            callback = self.client.submitAsync(query)
            results = callback.result()
            return list(results)

        except GremlinServerError as e:
            logger.error(f"Gremlin query error: {e}")
            return []

    def _parse_vertex(self, vertex: Dict) -> Dict[str, Any]:
        """Parse vertex from Gremlin response."""
        parsed = {}

        for key, value in vertex.items():
            if isinstance(value, list) and len(value) > 0:
                parsed[key] = value[0]
            else:
                parsed[key] = value

        return parsed

    def _escape_json(self, data: Dict) -> str:
        """Escape JSON for Gremlin query."""
        json_str = json.dumps(data)
        return json_str.replace("'", "\\'").replace('"', '\\"')

    def _escape_string(self, s: str) -> str:
        """Escape string for Gremlin query."""
        return s.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")

    def close(self):
        """Close client connection."""
        if self.client is not None:
            self.client.close()
            logger.info("PMG client closed")
