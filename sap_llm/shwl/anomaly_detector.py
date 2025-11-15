"""
SHWL Phase 1: Anomaly Detection

Detects processing anomalies from Process Memory Graph:
- Low-confidence predictions (<0.7)
- SAP API failures (HTTP 4xx, 5xx)
- Validation failures
- Routing errors
- Business rule violations

Queries PMG and classifies anomalies by severity and type.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict

# Import PMG components
from sap_llm.pmg.graph_client import ProcessMemoryGraph

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    """Represents a detected anomaly."""
    anomaly_id: str
    anomaly_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    doc_id: str
    document: Dict[str, Any]
    confidence_score: Optional[float]
    error_message: Optional[str]
    field_name: Optional[str]
    sap_response: Optional[Dict[str, Any]]
    timestamp: str
    metadata: Dict[str, Any]


@dataclass
class DetectionConfig:
    """Configuration for anomaly detection."""
    lookback_days: int = 7
    min_confidence_threshold: float = 0.7
    include_sap_failures: bool = True
    include_low_confidence: bool = True
    include_validation_failures: bool = True
    include_routing_errors: bool = True
    severity_levels: List[str] = None

    def __post_init__(self):
        if self.severity_levels is None:
            self.severity_levels = ["CRITICAL", "HIGH", "MEDIUM"]


class AnomalyDetector:
    """
    SHWL Phase 1: Detect processing anomalies from PMG.

    Detection Sources:
    - Low-confidence predictions
    - SAP API failures
    - Validation exceptions
    - Routing failures
    - Business rule violations
    """

    def __init__(
        self,
        pmg: ProcessMemoryGraph,
        config: Optional[DetectionConfig] = None
    ):
        """
        Initialize anomaly detector.

        Args:
            pmg: Process Memory Graph client
            config: Detection configuration
        """
        self.pmg = pmg
        self.config = config or DetectionConfig()

        # Statistics
        self.stats = {
            "total_scanned": 0,
            "total_anomalies": 0,
            "by_type": defaultdict(int),
            "by_severity": defaultdict(int)
        }

        logger.info(
            f"AnomalyDetector initialized "
            f"(lookback={self.config.lookback_days}d, "
            f"confidence_threshold={self.config.min_confidence_threshold})"
        )

    def detect_anomalies(self) -> List[Anomaly]:
        """
        Run comprehensive anomaly detection.

        Returns:
            List of detected anomalies
        """
        logger.info("Starting anomaly detection...")

        all_anomalies = []

        # Detection 1: Low-confidence predictions
        if self.config.include_low_confidence:
            low_conf_anomalies = self._detect_low_confidence()
            all_anomalies.extend(low_conf_anomalies)
            logger.info(f"Detected {len(low_conf_anomalies)} low-confidence anomalies")

        # Detection 2: SAP API failures
        if self.config.include_sap_failures:
            sap_failures = self._detect_sap_failures()
            all_anomalies.extend(sap_failures)
            logger.info(f"Detected {len(sap_failures)} SAP failure anomalies")

        # Detection 3: Validation failures
        if self.config.include_validation_failures:
            validation_failures = self._detect_validation_failures()
            all_anomalies.extend(validation_failures)
            logger.info(f"Detected {len(validation_failures)} validation failure anomalies")

        # Detection 4: Routing errors
        if self.config.include_routing_errors:
            routing_errors = self._detect_routing_errors()
            all_anomalies.extend(routing_errors)
            logger.info(f"Detected {len(routing_errors)} routing error anomalies")

        # Filter by severity
        filtered_anomalies = self._filter_by_severity(all_anomalies)

        # Update statistics
        self._update_stats(filtered_anomalies)

        logger.info(
            f"Anomaly detection complete: {len(filtered_anomalies)} anomalies "
            f"(scanned {self.stats['total_scanned']} documents)"
        )

        return filtered_anomalies

    def _detect_low_confidence(self) -> List[Anomaly]:
        """
        Detect low-confidence predictions from PMG.

        Queries for documents with confidence scores below threshold.
        """
        anomalies = []

        # Calculate time window
        cutoff_date = datetime.now() - timedelta(days=self.config.lookback_days)

        logger.debug(f"Querying PMG for low-confidence predictions since {cutoff_date}")

        # Mock query (in production, would use Gremlin/Cypher)
        # g.V().has('Document', 'confidence', lt(0.7))
        #   .has('processing_timestamp', gte(cutoff_date))

        if self.pmg.mock_mode:
            # Generate mock anomalies for demonstration
            mock_docs = self._generate_mock_low_confidence_docs(10)

            for doc in mock_docs:
                anomaly = Anomaly(
                    anomaly_id=f"LC-{doc['id'][:8]}",
                    anomaly_type="low_confidence",
                    severity=self._determine_confidence_severity(doc.get('confidence', 0)),
                    doc_id=doc['id'],
                    document=doc,
                    confidence_score=doc.get('confidence'),
                    error_message=f"Low confidence: {doc.get('confidence', 0):.3f}",
                    field_name=doc.get('low_confidence_field'),
                    sap_response=None,
                    timestamp=doc.get('timestamp', datetime.now().isoformat()),
                    metadata={'detection_method': 'confidence_threshold'}
                )
                anomalies.append(anomaly)

            self.stats["total_scanned"] += len(mock_docs)

        return anomalies

    def _detect_sap_failures(self) -> List[Anomaly]:
        """
        Detect SAP API failures from PMG.

        Queries for documents with failed SAP responses.
        """
        anomalies = []

        # Mock query for SAP failures
        # g.V().has('Document').out('GOT_RESPONSE')
        #   .has('SAPResponse', 'status_code', gte(400))

        if self.pmg.mock_mode:
            mock_failures = self._generate_mock_sap_failures(8)

            for doc in mock_failures:
                sap_response = doc.get('sap_response', {})
                status_code = sap_response.get('status_code', 500)

                anomaly = Anomaly(
                    anomaly_id=f"SF-{doc['id'][:8]}",
                    anomaly_type="sap_failure",
                    severity=self._determine_sap_failure_severity(status_code),
                    doc_id=doc['id'],
                    document=doc,
                    confidence_score=None,
                    error_message=sap_response.get('error_message', 'SAP API failure'),
                    field_name=None,
                    sap_response=sap_response,
                    timestamp=doc.get('timestamp', datetime.now().isoformat()),
                    metadata={
                        'detection_method': 'sap_response_check',
                        'status_code': status_code
                    }
                )
                anomalies.append(anomaly)

            self.stats["total_scanned"] += len(mock_failures)

        return anomalies

    def _detect_validation_failures(self) -> List[Anomaly]:
        """
        Detect validation failures from PMG.

        Queries for documents with raised exceptions.
        """
        anomalies = []

        # Mock query for validation failures
        # g.V().has('Document').out('RAISED_EXCEPTION')
        #   .has('Exception', 'category', 'validation')

        if self.pmg.mock_mode:
            mock_validation_failures = self._generate_mock_validation_failures(12)

            for doc in mock_validation_failures:
                exceptions = doc.get('exceptions', [])

                for exc in exceptions:
                    anomaly = Anomaly(
                        anomaly_id=f"VF-{doc['id'][:8]}-{exc.get('field', 'unknown')}",
                        anomaly_type="validation_failure",
                        severity=exc.get('severity', 'MEDIUM'),
                        doc_id=doc['id'],
                        document=doc,
                        confidence_score=None,
                        error_message=exc.get('message', 'Validation failed'),
                        field_name=exc.get('field'),
                        sap_response=None,
                        timestamp=doc.get('timestamp', datetime.now().isoformat()),
                        metadata={
                            'detection_method': 'validation_exception',
                            'exception_category': exc.get('category', 'validation')
                        }
                    )
                    anomalies.append(anomaly)

            self.stats["total_scanned"] += len(mock_validation_failures)

        return anomalies

    def _detect_routing_errors(self) -> List[Anomaly]:
        """
        Detect routing errors from PMG.

        Queries for documents with failed routing decisions.
        """
        anomalies = []

        # Mock query for routing errors
        # g.V().has('Document').out('ROUTED_TO')
        #   .has('RoutingDecision', 'confidence', lt(0.5))

        if self.pmg.mock_mode:
            mock_routing_errors = self._generate_mock_routing_errors(5)

            for doc in mock_routing_errors:
                routing_decision = doc.get('routing_decision', {})

                anomaly = Anomaly(
                    anomaly_id=f"RE-{doc['id'][:8]}",
                    anomaly_type="routing_error",
                    severity="MEDIUM",
                    doc_id=doc['id'],
                    document=doc,
                    confidence_score=routing_decision.get('confidence'),
                    error_message=f"Uncertain routing: {routing_decision.get('confidence', 0):.3f}",
                    field_name=None,
                    sap_response=None,
                    timestamp=doc.get('timestamp', datetime.now().isoformat()),
                    metadata={
                        'detection_method': 'routing_confidence',
                        'suggested_endpoint': routing_decision.get('endpoint')
                    }
                )
                anomalies.append(anomaly)

            self.stats["total_scanned"] += len(mock_routing_errors)

        return anomalies

    def _filter_by_severity(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Filter anomalies by configured severity levels."""
        return [
            a for a in anomalies
            if a.severity in self.config.severity_levels
        ]

    def _determine_confidence_severity(self, confidence: float) -> str:
        """Determine severity based on confidence score."""
        if confidence < 0.3:
            return "CRITICAL"
        elif confidence < 0.5:
            return "HIGH"
        elif confidence < 0.7:
            return "MEDIUM"
        else:
            return "LOW"

    def _determine_sap_failure_severity(self, status_code: int) -> str:
        """Determine severity based on SAP status code."""
        if status_code >= 500:
            return "CRITICAL"  # Server errors
        elif status_code >= 400:
            return "HIGH"  # Client errors
        else:
            return "MEDIUM"

    def _update_stats(self, anomalies: List[Anomaly]):
        """Update detection statistics."""
        self.stats["total_anomalies"] = len(anomalies)

        for anomaly in anomalies:
            self.stats["by_type"][anomaly.anomaly_type] += 1
            self.stats["by_severity"][anomaly.severity] += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_scanned": self.stats["total_scanned"],
            "total_anomalies": self.stats["total_anomalies"],
            "by_type": dict(self.stats["by_type"]),
            "by_severity": dict(self.stats["by_severity"]),
            "detection_rate": (
                self.stats["total_anomalies"] / self.stats["total_scanned"]
                if self.stats["total_scanned"] > 0 else 0
            )
        }

    # Mock data generators for demonstration
    def _generate_mock_low_confidence_docs(self, count: int) -> List[Dict]:
        """Generate mock low-confidence documents."""
        import random
        docs = []
        for i in range(count):
            docs.append({
                'id': f'doc_lc_{i:04d}',
                'doc_type': random.choice(['invoice', 'purchase_order']),
                'confidence': random.uniform(0.3, 0.69),
                'low_confidence_field': random.choice(['total_amount', 'vendor_name', 'po_number']),
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat()
            })
        return docs

    def _generate_mock_sap_failures(self, count: int) -> List[Dict]:
        """Generate mock SAP failure documents."""
        import random
        docs = []
        for i in range(count):
            status_code = random.choice([400, 401, 404, 422, 500, 502, 503])
            docs.append({
                'id': f'doc_sf_{i:04d}',
                'doc_type': 'invoice',
                'sap_response': {
                    'status_code': status_code,
                    'error_message': f'SAP API error: {status_code}',
                    'success': False
                },
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat()
            })
        return docs

    def _generate_mock_validation_failures(self, count: int) -> List[Dict]:
        """Generate mock validation failure documents."""
        import random
        docs = []
        for i in range(count):
            docs.append({
                'id': f'doc_vf_{i:04d}',
                'doc_type': 'purchase_order',
                'exceptions': [{
                    'category': 'validation',
                    'severity': random.choice(['HIGH', 'MEDIUM']),
                    'field': random.choice(['total_amount', 'line_items', 'vendor_id']),
                    'message': 'Validation rule violated'
                }],
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat()
            })
        return docs

    def _generate_mock_routing_errors(self, count: int) -> List[Dict]:
        """Generate mock routing error documents."""
        import random
        docs = []
        for i in range(count):
            docs.append({
                'id': f'doc_re_{i:04d}',
                'doc_type': 'delivery_note',
                'routing_decision': {
                    'confidence': random.uniform(0.3, 0.49),
                    'endpoint': '/api/sap/deliveries',
                    'reasoning': 'Uncertain document type classification'
                },
                'timestamp': (datetime.now() - timedelta(days=random.randint(0, 7))).isoformat()
            })
        return docs


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize detector
    pmg = ProcessMemoryGraph()
    detector = AnomalyDetector(pmg)

    # Detect anomalies
    anomalies = detector.detect_anomalies()

    print(f"Detected {len(anomalies)} anomalies:")
    for anomaly in anomalies[:10]:
        print(f"  - {anomaly.anomaly_type}: {anomaly.severity} - {anomaly.error_message}")

    # Get statistics
    stats = detector.get_statistics()
    print(f"\nStatistics: {stats}")
