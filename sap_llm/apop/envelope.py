"""
APOP Envelope - CloudEvents 1.0 Compliant Message Format

Implements the CloudEvents specification with APOP extensions for
autonomous agent orchestration.
"""

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class APOPEnvelope:
    """
    APOP Envelope structure (CloudEvents compliant).

    CloudEvents Base Fields:
    - id: Unique event ID
    - source: Event source (service name)
    - type: Event type (e.g., "classify.done")
    - specversion: CloudEvents spec version (1.0)
    - time: Event timestamp (ISO-8601)
    - datacontenttype: Content type of data

    APOP Extensions:
    - next_action_hint: Suggested next action for routing
    - correlation_id: Correlation ID for tracking
    - traceparent: W3C Trace Context
    - tenant_id: Multi-tenant support
    - signature: ECDSA signature for verification
    - data: Event payload
    """

    # CloudEvents required fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "sap_llm"
    type: str = "document.processed"
    specversion: str = "1.0"

    # CloudEvents optional fields
    time: Optional[str] = None
    datacontenttype: str = "application/json"

    # APOP extensions
    next_action_hint: Optional[str] = None
    correlation_id: Optional[str] = None
    traceparent: Optional[str] = None
    tenant_id: Optional[str] = None
    signature: Optional[str] = None

    # Payload
    data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Set default time if not provided."""
        if self.time is None:
            self.time = datetime.now().isoformat()

        if self.correlation_id is None:
            self.correlation_id = self.id

    def to_dict(self) -> Dict[str, Any]:
        """Convert envelope to dictionary."""
        envelope_dict = asdict(self)

        # Remove None values
        envelope_dict = {k: v for k, v in envelope_dict.items() if v is not None}

        return envelope_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APOPEnvelope":
        """Create envelope from dictionary."""
        # Extract known fields
        envelope_data = {}

        for key in [
            "id",
            "source",
            "type",
            "specversion",
            "time",
            "datacontenttype",
            "next_action_hint",
            "correlation_id",
            "traceparent",
            "tenant_id",
            "signature",
            "data",
        ]:
            if key in data:
                envelope_data[key] = data[key]

        return cls(**envelope_data)

    def validate(self) -> bool:
        """
        Validate envelope against CloudEvents spec.

        Returns:
            True if valid, False otherwise
        """
        # Required fields
        if not self.id:
            logger.error("Envelope missing required field: id")
            return False

        if not self.source:
            logger.error("Envelope missing required field: source")
            return False

        if not self.type:
            logger.error("Envelope missing required field: type")
            return False

        if self.specversion != "1.0":
            logger.error(f"Unsupported specversion: {self.specversion}")
            return False

        return True

    def clone(self, **kwargs) -> "APOPEnvelope":
        """
        Create a copy of the envelope with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New envelope instance
        """
        envelope_dict = self.to_dict()
        envelope_dict.update(kwargs)
        return APOPEnvelope.from_dict(envelope_dict)


def create_envelope(
    source: str,
    event_type: str,
    data: Dict[str, Any],
    next_action_hint: Optional[str] = None,
    correlation_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
) -> APOPEnvelope:
    """
    Create an APOP envelope.

    Args:
        source: Source service name
        event_type: Event type (e.g., "classify.done")
        data: Event payload
        next_action_hint: Optional next action hint
        correlation_id: Optional correlation ID
        tenant_id: Optional tenant ID

    Returns:
        APOP envelope

    Example:
        >>> envelope = create_envelope(
        ...     source="classifier",
        ...     event_type="classify.done",
        ...     data={"doc_type": "INVOICE", "confidence": 0.95},
        ...     next_action_hint="extract.fields"
        ... )
    """
    envelope = APOPEnvelope(
        source=source,
        type=event_type,
        data=data,
        next_action_hint=next_action_hint,
        correlation_id=correlation_id,
        tenant_id=tenant_id,
    )

    # Generate trace context
    envelope.traceparent = _generate_traceparent()

    return envelope


def _generate_traceparent() -> str:
    """
    Generate W3C Trace Context traceparent.

    Format: {version}-{trace-id}-{parent-id}-{trace-flags}
    Example: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
    """
    version = "00"
    trace_id = uuid.uuid4().hex + uuid.uuid4().hex[:16]  # 32 hex chars
    parent_id = uuid.uuid4().hex[:16]  # 16 hex chars
    trace_flags = "01"  # Sampled

    return f"{version}-{trace_id}-{parent_id}-{trace_flags}"
