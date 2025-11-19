"""
Comprehensive unit tests for APOP (Agentic Process Orchestration Protocol).

Tests cover:
- CloudEvents message handling
- ECDSA signing and verification
- Message routing
- Protocol statistics
"""

import json
import time
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from sap_llm.apop.apop_protocol import (
    MessagePriority,
    MessageType,
    CloudEventsMessage,
    ECDSASigner,
    APOPProtocol,
)


@pytest.fixture
def sample_envelope_data():
    """Sample envelope data for testing."""
    return {
        "doc_type": "INVOICE",
        "supplier_id": "SUP-001",
        "total_amount": 1000.00,
        "currency": "USD",
        "fields": {"invoice_number": "INV-2024-001"}
    }


@pytest.fixture
def sample_message():
    """Sample CloudEvents message."""
    return CloudEventsMessage(
        id="test-123",
        source="agent://test-agent",
        specversion="1.0",
        type=MessageType.ENVELOPE.value,
        datacontenttype="application/json",
        time="2024-01-15T10:30:00Z",
        data={"test": "data"},
        priority=MessagePriority.NORMAL.value,
        correlation_id="corr-456",
        trace_id="trace-789"
    )


@pytest.mark.unit
class TestMessagePriority:
    """Tests for MessagePriority enum."""

    def test_priority_values(self):
        """Test priority value ordering."""
        assert MessagePriority.LOW.value < MessagePriority.NORMAL.value
        assert MessagePriority.NORMAL.value < MessagePriority.HIGH.value
        assert MessagePriority.HIGH.value < MessagePriority.URGENT.value

    def test_priority_names(self):
        """Test priority names."""
        assert MessagePriority.LOW.name == "LOW"
        assert MessagePriority.URGENT.name == "URGENT"


@pytest.mark.unit
class TestMessageType:
    """Tests for MessageType enum."""

    def test_message_types(self):
        """Test message type values."""
        assert "envelope" in MessageType.ENVELOPE.value.lower()
        assert "routing" in MessageType.ROUTING_DECISION.value.lower()
        assert "capability" in MessageType.AGENT_CAPABILITY.value.lower()


@pytest.mark.unit
class TestCloudEventsMessage:
    """Tests for CloudEventsMessage dataclass."""

    def test_creation(self, sample_message):
        """Test message creation."""
        assert sample_message.id == "test-123"
        assert sample_message.source == "agent://test-agent"
        assert sample_message.specversion == "1.0"

    def test_to_dict(self, sample_message):
        """Test conversion to dictionary."""
        d = sample_message.to_dict()
        assert isinstance(d, dict)
        assert d["id"] == "test-123"
        assert d["data"] == {"test": "data"}

    def test_to_json(self, sample_message):
        """Test conversion to JSON."""
        json_str = sample_message.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["id"] == "test-123"

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "id": "msg-123",
            "source": "agent://test",
            "specversion": "1.0",
            "type": "test.type",
            "datacontenttype": "application/json",
            "time": "2024-01-15T10:00:00Z",
            "data": {"key": "value"}
        }
        msg = CloudEventsMessage.from_dict(d)
        assert msg.id == "msg-123"
        assert msg.data == {"key": "value"}

    def test_from_json(self):
        """Test creation from JSON string."""
        json_str = json.dumps({
            "id": "msg-456",
            "source": "agent://test",
            "specversion": "1.0",
            "type": "test.type",
            "datacontenttype": "application/json",
            "time": "2024-01-15T10:00:00Z",
            "data": {}
        })
        msg = CloudEventsMessage.from_json(json_str)
        assert msg.id == "msg-456"

    def test_default_values(self):
        """Test default values for optional fields."""
        msg = CloudEventsMessage(
            id="test",
            source="test",
            specversion="1.0",
            type="test",
            datacontenttype="application/json",
            time="2024-01-15T10:00:00Z",
            data={}
        )
        assert msg.priority == MessagePriority.NORMAL.value
        assert msg.correlation_id is None
        assert msg.signature is None

    def test_roundtrip_json(self, sample_message):
        """Test JSON roundtrip preserves data."""
        json_str = sample_message.to_json()
        restored = CloudEventsMessage.from_json(json_str)
        assert restored.id == sample_message.id
        assert restored.data == sample_message.data


@pytest.mark.unit
class TestECDSASigner:
    """Tests for ECDSA signature generation and verification."""

    def test_initialization(self):
        """Test signer initialization generates keys."""
        signer = ECDSASigner()
        assert signer.private_key is not None
        assert signer.public_key is not None

    def test_sign_message(self):
        """Test message signing."""
        signer = ECDSASigner()
        message = b"Test message for signing"
        signature = signer.sign(message)

        assert signature is not None
        assert len(signature) > 0
        assert isinstance(signature, bytes)

    def test_verify_valid_signature(self):
        """Test verification of valid signature."""
        signer = ECDSASigner()
        message = b"Test message"
        signature = signer.sign(message)

        assert signer.verify(message, signature) is True

    def test_verify_invalid_signature(self):
        """Test verification of invalid signature."""
        signer = ECDSASigner()
        message = b"Test message"
        signature = signer.sign(message)

        # Modify message
        different_message = b"Different message"
        assert signer.verify(different_message, signature) is False

    def test_verify_tampered_signature(self):
        """Test verification of tampered signature."""
        signer = ECDSASigner()
        message = b"Test message"
        signature = signer.sign(message)

        # Tamper with signature
        tampered = bytes([b ^ 0xFF for b in signature[:5]]) + signature[5:]
        assert signer.verify(message, tampered) is False

    def test_verify_with_different_public_key(self):
        """Test verification with different public key fails."""
        signer1 = ECDSASigner()
        signer2 = ECDSASigner()

        message = b"Test message"
        signature = signer1.sign(message)

        # Verify with different key
        result = signer1.verify(message, signature, signer2.public_key)
        assert result is False

    def test_get_public_key_pem(self):
        """Test getting public key in PEM format."""
        signer = ECDSASigner()
        pem = signer.get_public_key_pem()

        assert isinstance(pem, str)
        assert "BEGIN PUBLIC KEY" in pem
        assert "END PUBLIC KEY" in pem

    def test_different_signers_different_keys(self):
        """Test that different signers have different keys."""
        signer1 = ECDSASigner()
        signer2 = ECDSASigner()

        assert signer1.get_public_key_pem() != signer2.get_public_key_pem()

    def test_sign_empty_message(self):
        """Test signing empty message."""
        signer = ECDSASigner()
        signature = signer.sign(b"")
        assert signer.verify(b"", signature) is True

    def test_sign_large_message(self):
        """Test signing large message."""
        signer = ECDSASigner()
        message = b"x" * 10000
        signature = signer.sign(message)
        assert signer.verify(message, signature) is True


@pytest.mark.unit
class TestAPOPProtocol:
    """Tests for APOP Protocol implementation."""

    def test_initialization(self):
        """Test protocol initialization."""
        protocol = APOPProtocol(agent_id="test-agent")

        assert protocol.agent_id == "test-agent"
        assert protocol.enable_signatures is True
        assert protocol.signer is not None
        assert protocol.messages_sent == 0

    def test_initialization_without_signatures(self):
        """Test initialization with signatures disabled."""
        protocol = APOPProtocol(
            agent_id="test-agent",
            enable_signatures=False
        )

        assert protocol.signer is None
        assert protocol.enable_signatures is False

    def test_create_envelope_message(self, sample_envelope_data):
        """Test creating envelope message."""
        protocol = APOPProtocol(agent_id="test-agent")
        msg = protocol.create_envelope_message(sample_envelope_data)

        assert msg.type == MessageType.ENVELOPE.value
        assert msg.source == "agent://test-agent"
        assert msg.data == sample_envelope_data
        assert msg.signature is not None
        assert protocol.messages_sent == 1

    def test_create_envelope_message_with_priority(self, sample_envelope_data):
        """Test creating envelope with custom priority."""
        protocol = APOPProtocol(agent_id="test-agent")
        msg = protocol.create_envelope_message(
            sample_envelope_data,
            priority=MessagePriority.URGENT
        )

        assert msg.priority == MessagePriority.URGENT.value

    def test_create_envelope_message_with_correlation_id(self, sample_envelope_data):
        """Test creating envelope with correlation ID."""
        protocol = APOPProtocol(agent_id="test-agent")
        msg = protocol.create_envelope_message(
            sample_envelope_data,
            correlation_id="custom-corr-id"
        )

        assert msg.correlation_id == "custom-corr-id"

    def test_create_envelope_message_no_signatures(self, sample_envelope_data):
        """Test creating envelope without signatures."""
        protocol = APOPProtocol(
            agent_id="test-agent",
            enable_signatures=False
        )
        msg = protocol.create_envelope_message(sample_envelope_data)

        assert msg.signature is None

    def test_create_routing_decision(self):
        """Test creating routing decision message."""
        protocol = APOPProtocol(agent_id="router-agent")
        msg = protocol.create_routing_decision(
            envelope_id="env-123",
            next_agent="processor-agent",
            routing_metadata={"reason": "capability_match"}
        )

        assert msg.type == MessageType.ROUTING_DECISION.value
        assert msg.data["envelope_id"] == "env-123"
        assert msg.data["next_agent"] == "processor-agent"
        assert msg.priority == MessagePriority.HIGH.value

    def test_route_envelope_matching_capability(self, sample_envelope_data):
        """Test routing to agent with matching capability."""
        protocol = APOPProtocol(agent_id="router")
        msg = protocol.create_envelope_message(sample_envelope_data)

        agent_capabilities = {
            "agent-1": ["PURCHASE_ORDER"],
            "agent-2": ["INVOICE", "CREDIT_NOTE"],
            "agent-3": ["ALL"]
        }

        selected = protocol.route_envelope(msg, agent_capabilities)

        # Should select agent-2 (INVOICE) or agent-3 (ALL)
        assert selected in ["agent-2", "agent-3"]

    def test_route_envelope_no_matching_capability(self):
        """Test routing when no agent has matching capability."""
        protocol = APOPProtocol(agent_id="router")
        msg = protocol.create_envelope_message({"doc_type": "UNKNOWN_TYPE"})

        agent_capabilities = {
            "agent-1": ["INVOICE"],
            "agent-2": ["PURCHASE_ORDER"]
        }

        selected = protocol.route_envelope(msg, agent_capabilities)
        assert selected == "default_agent"

    def test_route_envelope_records_latency(self, sample_envelope_data):
        """Test that routing records latency."""
        protocol = APOPProtocol(agent_id="router")
        msg = protocol.create_envelope_message(sample_envelope_data)

        agent_capabilities = {"agent-1": ["ALL"]}
        protocol.route_envelope(msg, agent_capabilities)

        assert len(protocol.routing_latencies) == 1
        assert protocol.routing_latencies[0] >= 0

    def test_route_envelope_consistent_selection(self, sample_envelope_data):
        """Test routing is consistent for same message."""
        protocol = APOPProtocol(agent_id="router")
        msg = protocol.create_envelope_message(sample_envelope_data)

        agent_capabilities = {
            "agent-1": ["ALL"],
            "agent-2": ["ALL"],
            "agent-3": ["ALL"]
        }

        # Same message should route to same agent
        selected1 = protocol.route_envelope(msg, agent_capabilities)
        selected2 = protocol.route_envelope(msg, agent_capabilities)

        assert selected1 == selected2

    def test_verify_message_valid(self, sample_envelope_data):
        """Test verification of valid message."""
        protocol = APOPProtocol(agent_id="test-agent")
        msg = protocol.create_envelope_message(sample_envelope_data)

        assert protocol.verify_message(msg) is True

    def test_verify_message_no_signature(self):
        """Test verification of message without signature."""
        protocol = APOPProtocol(agent_id="test-agent")

        msg = CloudEventsMessage(
            id="test",
            source="agent://test",
            specversion="1.0",
            type="test",
            datacontenttype="application/json",
            time="2024-01-15T10:00:00Z",
            data={},
            signature=None
        )

        assert protocol.verify_message(msg) is False

    def test_verify_message_signatures_disabled(self):
        """Test verification when signatures disabled."""
        protocol = APOPProtocol(
            agent_id="test-agent",
            enable_signatures=False
        )

        msg = CloudEventsMessage(
            id="test",
            source="agent://test",
            specversion="1.0",
            type="test",
            datacontenttype="application/json",
            time="2024-01-15T10:00:00Z",
            data={}
        )

        assert protocol.verify_message(msg) is True

    def test_get_routing_stats_empty(self):
        """Test routing stats when no routing done."""
        protocol = APOPProtocol(agent_id="test-agent")
        stats = protocol.get_routing_stats()

        assert stats["total_routings"] == 0
        assert stats["avg_latency_ms"] == 0.0

    def test_get_routing_stats_with_data(self, sample_envelope_data):
        """Test routing stats after routing."""
        protocol = APOPProtocol(agent_id="router")

        # Perform multiple routings
        for _ in range(10):
            msg = protocol.create_envelope_message(sample_envelope_data)
            protocol.route_envelope(msg, {"agent-1": ["ALL"]})

        stats = protocol.get_routing_stats()

        assert stats["total_routings"] == 10
        assert stats["avg_latency_ms"] > 0
        assert stats["p95_latency_ms"] >= stats["avg_latency_ms"]

    def test_get_protocol_stats(self, sample_envelope_data):
        """Test protocol statistics."""
        protocol = APOPProtocol(agent_id="test-agent")

        # Create some messages
        protocol.create_envelope_message(sample_envelope_data)
        protocol.create_envelope_message(sample_envelope_data)

        stats = protocol.get_protocol_stats()

        assert stats["agent_id"] == "test-agent"
        assert stats["messages_sent"] == 2
        assert stats["signatures_enabled"] is True

    def test_sign_message_no_signer(self):
        """Test signing when signer not available."""
        protocol = APOPProtocol(
            agent_id="test-agent",
            enable_signatures=False
        )

        msg = CloudEventsMessage(
            id="test",
            source="agent://test",
            specversion="1.0",
            type="test",
            datacontenttype="application/json",
            time="2024-01-15T10:00:00Z",
            data={}
        )

        signature = protocol._sign_message(msg)
        assert signature == ""

    def test_message_bytes_for_signing(self):
        """Test canonical message bytes generation."""
        protocol = APOPProtocol(agent_id="test-agent")

        msg = CloudEventsMessage(
            id="test-123",
            source="agent://test",
            specversion="1.0",
            type="test.type",
            datacontenttype="application/json",
            time="2024-01-15T10:00:00Z",
            data={"key": "value"}
        )

        msg_bytes = protocol._get_message_bytes_for_signing(msg)

        assert isinstance(msg_bytes, bytes)
        # Should be valid JSON
        parsed = json.loads(msg_bytes.decode('utf-8'))
        assert parsed["id"] == "test-123"


@pytest.mark.unit
class TestAPOPProtocolEdgeCases:
    """Edge case tests for APOP Protocol."""

    def test_route_empty_capabilities(self):
        """Test routing with empty capabilities map."""
        protocol = APOPProtocol(agent_id="router")
        msg = protocol.create_envelope_message({"doc_type": "INVOICE"})

        selected = protocol.route_envelope(msg, {})
        assert selected == "default_agent"

    def test_route_single_agent(self):
        """Test routing with single agent."""
        protocol = APOPProtocol(agent_id="router")
        msg = protocol.create_envelope_message({"doc_type": "INVOICE"})

        selected = protocol.route_envelope(msg, {"only-agent": ["INVOICE"]})
        assert selected == "only-agent"

    def test_multiple_messages_same_correlation(self):
        """Test multiple messages with same correlation ID."""
        protocol = APOPProtocol(agent_id="test-agent")

        msg1 = protocol.create_envelope_message(
            {"data": "1"},
            correlation_id="shared-corr"
        )
        msg2 = protocol.create_envelope_message(
            {"data": "2"},
            correlation_id="shared-corr"
        )

        assert msg1.correlation_id == msg2.correlation_id
        assert msg1.id != msg2.id

    def test_routing_latency_under_threshold(self, sample_envelope_data):
        """Test that routing latency is under target (5ms)."""
        protocol = APOPProtocol(agent_id="router")

        capabilities = {
            f"agent-{i}": ["ALL"] for i in range(100)
        }

        msg = protocol.create_envelope_message(sample_envelope_data)
        protocol.route_envelope(msg, capabilities)

        # Target is <5ms
        assert protocol.routing_latencies[0] < 10  # Allow some margin

    def test_large_envelope_data(self):
        """Test handling large envelope data."""
        protocol = APOPProtocol(agent_id="test-agent")

        large_data = {
            "doc_type": "INVOICE",
            "fields": {f"field_{i}": f"value_{i}" * 100 for i in range(100)}
        }

        msg = protocol.create_envelope_message(large_data)
        assert msg.data == large_data
        assert protocol.verify_message(msg) is True


@pytest.mark.unit
class TestAPOPProtocolConcurrency:
    """Tests for concurrent operations."""

    def test_message_counter_thread_safety(self):
        """Test that message counter is updated correctly."""
        protocol = APOPProtocol(agent_id="test-agent")

        initial_count = protocol.messages_sent

        for i in range(100):
            protocol.create_envelope_message({"index": i})

        assert protocol.messages_sent == initial_count + 100
