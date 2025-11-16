"""
APOP (Agentic Process Orchestration Protocol) Implementation.

CloudEvents-based protocol for agent-to-agent communication:
1. ECDSA signature verification for security
2. <5ms routing decision latency
3. Standardized message format (CloudEvents)
4. Agent capability discovery
5. Priority-based routing
6. Message tracing and correlation

Target Metrics:
- Routing latency: <5ms P95
- Signature verification: <1ms
- Message throughput: 100k/sec
- Protocol overhead: <100 bytes per message
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class MessageType(Enum):
    """APOP message types."""
    ENVELOPE = "com.sap.apop.envelope.v1"
    ROUTING_DECISION = "com.sap.apop.routing.v1"
    AGENT_CAPABILITY = "com.sap.apop.capability.v1"
    AGENT_HEALTH = "com.sap.apop.health.v1"
    SYSTEM_EVENT = "com.sap.apop.system.v1"


@dataclass
class CloudEventsMessage:
    """
    CloudEvents v1.0 compliant message.

    Standard fields:
    - id: Unique message ID
    - source: Message source (agent ID)
    - specversion: CloudEvents version (1.0)
    - type: Message type
    - datacontenttype: Content type (application/json)
    - time: Timestamp
    - data: Message payload
    """

    id: str
    source: str
    specversion: str
    type: str
    datacontenttype: str
    time: str
    data: Dict[str, Any]

    # APOP extensions
    priority: int = MessagePriority.NORMAL.value
    correlation_id: Optional[str] = None
    signature: Optional[str] = None
    trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CloudEventsMessage":
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "CloudEventsMessage":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


class ECDSASigner:
    """
    ECDSA signature generation and verification.

    Uses NIST P-256 curve for fast signing/verification.
    """

    def __init__(self):
        """Initialize ECDSA signer."""
        # Generate private/public key pair
        self.private_key = ec.generate_private_key(
            ec.SECP256R1(),  # NIST P-256 curve
            default_backend(),
        )

        self.public_key = self.private_key.public_key()

    def sign(self, message: bytes) -> bytes:
        """
        Sign message with ECDSA.

        Args:
            message: Message bytes to sign

        Returns:
            Signature bytes
        """
        signature = self.private_key.sign(
            message,
            ec.ECDSA(hashes.SHA256()),
        )

        return signature

    def verify(
        self,
        message: bytes,
        signature: bytes,
        public_key: Optional[ec.EllipticCurvePublicKey] = None,
    ) -> bool:
        """
        Verify ECDSA signature.

        Args:
            message: Message bytes
            signature: Signature bytes
            public_key: Public key (uses own if None)

        Returns:
            True if signature is valid
        """
        key = public_key or self.public_key

        try:
            key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256()),
            )
            return True
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return False

    def get_public_key_pem(self) -> str:
        """Get public key in PEM format."""
        pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return pem.decode('utf-8')


class APOPProtocol:
    """
    APOP Protocol implementation.

    Handles:
    - Message creation and parsing
    - Signature generation/verification
    - Routing decisions
    - Agent capability management
    """

    def __init__(
        self,
        agent_id: str,
        enable_signatures: bool = True,
    ):
        """
        Initialize APOP protocol.

        Args:
            agent_id: Unique agent ID
            enable_signatures: Enable ECDSA signatures
        """
        self.agent_id = agent_id
        self.enable_signatures = enable_signatures

        # ECDSA signer
        if enable_signatures:
            self.signer = ECDSASigner()
        else:
            self.signer = None

        # Message statistics
        self.messages_sent = 0
        self.messages_received = 0
        self.routing_latencies: List[float] = []

        logger.info(f"APOP Protocol initialized for agent: {agent_id}")
        logger.info(f"  Signatures: {enable_signatures}")

    def create_envelope_message(
        self,
        envelope_data: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
    ) -> CloudEventsMessage:
        """
        Create APOP envelope message.

        Args:
            envelope_data: Envelope payload
            priority: Message priority
            correlation_id: Correlation ID for tracking

        Returns:
            CloudEvents message
        """
        message = CloudEventsMessage(
            id=str(uuid.uuid4()),
            source=f"agent://{self.agent_id}",
            specversion="1.0",
            type=MessageType.ENVELOPE.value,
            datacontenttype="application/json",
            time=datetime.utcnow().isoformat() + "Z",
            data=envelope_data,
            priority=priority.value,
            correlation_id=correlation_id or str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
        )

        # Sign message
        if self.enable_signatures and self.signer:
            message.signature = self._sign_message(message)

        self.messages_sent += 1

        return message

    def create_routing_decision(
        self,
        envelope_id: str,
        next_agent: str,
        routing_metadata: Optional[Dict[str, Any]] = None,
    ) -> CloudEventsMessage:
        """
        Create routing decision message.

        Args:
            envelope_id: Envelope ID being routed
            next_agent: Next agent to route to
            routing_metadata: Additional routing metadata

        Returns:
            CloudEvents message
        """
        routing_data = {
            "envelope_id": envelope_id,
            "next_agent": next_agent,
            "routing_timestamp": time.time(),
            "metadata": routing_metadata or {},
        }

        message = CloudEventsMessage(
            id=str(uuid.uuid4()),
            source=f"agent://{self.agent_id}",
            specversion="1.0",
            type=MessageType.ROUTING_DECISION.value,
            datacontenttype="application/json",
            time=datetime.utcnow().isoformat() + "Z",
            data=routing_data,
            priority=MessagePriority.HIGH.value,
            correlation_id=envelope_id,
        )

        # Sign message
        if self.enable_signatures and self.signer:
            message.signature = self._sign_message(message)

        self.messages_sent += 1

        return message

    def route_envelope(
        self,
        envelope: CloudEventsMessage,
        agent_capabilities: Dict[str, List[str]],
    ) -> str:
        """
        Route envelope to appropriate agent.

        Ultra-fast routing using capability matching.

        Args:
            envelope: Envelope message
            agent_capabilities: Map of agent_id -> capabilities

        Returns:
            Selected agent ID
        """
        start_time = time.perf_counter()

        # Extract document type from envelope
        doc_type = envelope.data.get("doc_type", "UNKNOWN")

        # Find agents with matching capability
        matching_agents = []

        for agent_id, capabilities in agent_capabilities.items():
            if doc_type in capabilities or "ALL" in capabilities:
                matching_agents.append(agent_id)

        # Select agent (simple round-robin for now)
        if not matching_agents:
            selected_agent = "default_agent"
        else:
            # Use hash-based selection for consistency
            hash_val = hash(envelope.id)
            idx = hash_val % len(matching_agents)
            selected_agent = matching_agents[idx]

        # Record latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.routing_latencies.append(latency_ms)

        logger.debug(f"Routed {envelope.id} to {selected_agent} in {latency_ms:.3f}ms")

        return selected_agent

    def verify_message(self, message: CloudEventsMessage) -> bool:
        """
        Verify message signature.

        Args:
            message: Message to verify

        Returns:
            True if signature is valid
        """
        if not self.enable_signatures:
            return True  # Signatures disabled

        if not message.signature:
            logger.warning("Message has no signature")
            return False

        # For simplicity, verify with our own public key
        # In production, would look up sender's public key
        message_bytes = self._get_message_bytes_for_signing(message)
        signature_bytes = bytes.fromhex(message.signature)

        return self.signer.verify(message_bytes, signature_bytes)

    def _sign_message(self, message: CloudEventsMessage) -> str:
        """Sign message and return signature as hex string."""
        if not self.signer:
            return ""

        message_bytes = self._get_message_bytes_for_signing(message)
        signature_bytes = self.signer.sign(message_bytes)

        return signature_bytes.hex()

    def _get_message_bytes_for_signing(self, message: CloudEventsMessage) -> bytes:
        """Get canonical message bytes for signing."""
        # Create canonical representation for signing
        canonical = {
            "id": message.id,
            "source": message.source,
            "type": message.type,
            "time": message.time,
            "data": message.data,
        }

        canonical_json = json.dumps(canonical, sort_keys=True)
        return canonical_json.encode('utf-8')

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get routing performance statistics.

        Returns:
            Routing statistics
        """
        if not self.routing_latencies:
            return {
                "total_routings": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            }

        import numpy as np

        latencies = np.array(self.routing_latencies)

        return {
            "total_routings": len(latencies),
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
        }

    def get_protocol_stats(self) -> Dict[str, Any]:
        """Get overall protocol statistics."""
        return {
            "agent_id": self.agent_id,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "signatures_enabled": self.enable_signatures,
            "routing_stats": self.get_routing_stats(),
        }
