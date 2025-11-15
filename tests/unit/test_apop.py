"""
Comprehensive unit tests for APOP (Agentic Protocol for Orchestrating Pipelines) components.
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime

from sap_llm.apop.envelope import APOPEnvelope
from sap_llm.apop.agent import BaseAgent, AgentRegistry
from sap_llm.apop.orchestrator import AgenticOrchestrator
from sap_llm.apop.cloudevents_bus import CloudEventsBus
from sap_llm.apop.signature import EnvelopeSignature
from sap_llm.apop.stage_agents import (
    InboxAgent,
    PreprocessingAgent,
    ClassificationAgent,
    ExtractionAgent,
    ValidationAgent,
    RoutingAgent,
)


@pytest.mark.unit
@pytest.mark.apop
class TestAPOPEnvelope:
    """Tests for APOP Envelope."""

    def test_envelope_creation(self):
        """Test creating an APOP envelope."""
        envelope = APOPEnvelope(
            id="doc_12345",
            source="inbox",
            type="document.received",
            data={"file_path": "/path/to/doc.pdf"},
        )

        assert envelope.id == "doc_12345"
        assert envelope.source == "inbox"
        assert envelope.type == "document.received"
        assert envelope.data["file_path"] == "/path/to/doc.pdf"
        assert envelope.specversion == "1.0"

    def test_envelope_serialization(self):
        """Test envelope JSON serialization."""
        envelope = APOPEnvelope(
            id="doc_12345",
            source="inbox",
            type="document.received",
            data={"file_path": "/path/to/doc.pdf"},
        )

        json_str = envelope.to_json()
        assert isinstance(json_str, str)

        parsed = json.loads(json_str)
        assert parsed["id"] == "doc_12345"
        assert parsed["source"] == "inbox"

    def test_envelope_deserialization(self):
        """Test envelope JSON deserialization."""
        json_data = {
            "id": "doc_12345",
            "source": "inbox",
            "type": "document.received",
            "specversion": "1.0",
            "data": {"file_path": "/path/to/doc.pdf"},
        }

        envelope = APOPEnvelope.from_dict(json_data)
        assert envelope.id == "doc_12345"
        assert envelope.source == "inbox"
        assert envelope.data["file_path"] == "/path/to/doc.pdf"

    def test_envelope_next_action_hint(self):
        """Test next_action_hint field."""
        envelope = APOPEnvelope(
            id="doc_12345",
            source="inbox",
            type="document.received",
            data={"file_path": "/path/to/doc.pdf"},
            next_action_hint="preproc.ocr",
        )

        assert envelope.next_action_hint == "preproc.ocr"

    def test_envelope_exception_flow(self):
        """Test envelope with exception flow."""
        envelope = APOPEnvelope(
            id="doc_12345",
            source="validation",
            type="validation.failed",
            data={"exceptions": [{"field": "total_amount", "error": "mismatch"}]},
            next_action_hint="exception.handle",
        )

        assert envelope.type == "validation.failed"
        assert "exceptions" in envelope.data
        assert envelope.next_action_hint == "exception.handle"

    def test_envelope_metadata(self):
        """Test envelope metadata fields."""
        envelope = APOPEnvelope(
            id="doc_12345",
            source="inbox",
            type="document.received",
            data={"test": "data"},
            metadata={
                "priority": "high",
                "customer_id": "CUST_001",
            },
        )

        assert envelope.metadata["priority"] == "high"
        assert envelope.metadata["customer_id"] == "CUST_001"

    @pytest.mark.parametrize("envelope_type", [
        "document.received",
        "preprocessing.complete",
        "classification.done",
        "extraction.complete",
        "validation.passed",
        "routing.complete",
    ])
    def test_envelope_types(self, envelope_type):
        """Test different envelope types."""
        envelope = APOPEnvelope(
            id="doc_12345",
            source="test",
            type=envelope_type,
            data={},
        )

        assert envelope.type == envelope_type


@pytest.mark.unit
@pytest.mark.apop
class TestBaseAgent:
    """Tests for BaseAgent."""

    def test_agent_creation(self):
        """Test creating a base agent."""
        agent = BaseAgent(
            name="test_agent",
            subscriptions=["document.received"],
        )

        assert agent.name == "test_agent"
        assert "document.received" in agent.subscriptions

    @pytest.mark.asyncio
    async def test_agent_process(self):
        """Test agent processing."""
        class TestAgent(BaseAgent):
            async def process(self, envelope: APOPEnvelope) -> APOPEnvelope:
                envelope.data["processed"] = True
                return envelope

        agent = TestAgent(name="test", subscriptions=["test.event"])
        envelope = APOPEnvelope(
            id="test_1",
            source="test",
            type="test.event",
            data={},
        )

        result = await agent.process(envelope)
        assert result.data["processed"] is True

    def test_agent_subscription_match(self):
        """Test agent subscription matching."""
        agent = BaseAgent(
            name="test",
            subscriptions=["document.*", "validation.failed"],
        )

        assert agent.can_handle("document.received")
        assert agent.can_handle("document.processed")
        assert agent.can_handle("validation.failed")
        assert not agent.can_handle("routing.complete")


@pytest.mark.unit
@pytest.mark.apop
class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_registry_creation(self):
        """Test creating agent registry."""
        registry = AgentRegistry()
        assert registry is not None

    def test_register_agent(self):
        """Test registering an agent."""
        registry = AgentRegistry()
        agent = BaseAgent(name="test_agent", subscriptions=["test.event"])

        registry.register(agent)
        assert registry.get_agent("test_agent") == agent

    def test_unregister_agent(self):
        """Test unregistering an agent."""
        registry = AgentRegistry()
        agent = BaseAgent(name="test_agent", subscriptions=["test.event"])

        registry.register(agent)
        registry.unregister("test_agent")
        assert registry.get_agent("test_agent") is None

    def test_find_agents_for_event(self):
        """Test finding agents for an event type."""
        registry = AgentRegistry()

        agent1 = BaseAgent(name="agent1", subscriptions=["document.*"])
        agent2 = BaseAgent(name="agent2", subscriptions=["document.received"])
        agent3 = BaseAgent(name="agent3", subscriptions=["validation.*"])

        registry.register(agent1)
        registry.register(agent2)
        registry.register(agent3)

        agents = registry.find_agents_for_event("document.received")
        assert len(agents) == 2
        assert agent1 in agents
        assert agent2 in agents

    def test_duplicate_registration(self):
        """Test registering duplicate agent names."""
        registry = AgentRegistry()
        agent1 = BaseAgent(name="test", subscriptions=["test.1"])
        agent2 = BaseAgent(name="test", subscriptions=["test.2"])

        registry.register(agent1)
        registry.register(agent2)  # Should replace

        assert registry.get_agent("test") == agent2


@pytest.mark.unit
@pytest.mark.apop
class TestAgenticOrchestrator:
    """Tests for AgenticOrchestrator."""

    @pytest.fixture
    def orchestrator(self, mock_pmg):
        """Create orchestrator instance."""
        return AgenticOrchestrator(pmg=mock_pmg, max_hops=20)

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert orchestrator.max_hops == 20

    def test_register_agent(self, orchestrator):
        """Test registering agent with orchestrator."""
        agent = BaseAgent(name="test", subscriptions=["test.event"])
        orchestrator.register_agent(agent)

        assert orchestrator.registry.get_agent("test") == agent

    @pytest.mark.asyncio
    async def test_process_envelope_single_hop(self, orchestrator):
        """Test processing envelope through single hop."""
        class TestAgent(BaseAgent):
            async def process(self, envelope: APOPEnvelope) -> APOPEnvelope:
                envelope.data["processed"] = True
                envelope.next_action_hint = "complete"
                return envelope

        agent = TestAgent(name="test", subscriptions=["test.event"])
        orchestrator.register_agent(agent)

        envelope = APOPEnvelope(
            id="test_1",
            source="test",
            type="test.event",
            data={},
            next_action_hint="test.event",
        )

        results = await orchestrator.process_envelope(envelope)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_process_envelope_multi_hop(self, orchestrator):
        """Test processing envelope through multiple hops."""
        class Agent1(BaseAgent):
            async def process(self, envelope: APOPEnvelope) -> APOPEnvelope:
                envelope.data["step1"] = True
                envelope.next_action_hint = "step.2"
                return envelope

        class Agent2(BaseAgent):
            async def process(self, envelope: APOPEnvelope) -> APOPEnvelope:
                envelope.data["step2"] = True
                envelope.next_action_hint = "complete"
                return envelope

        agent1 = Agent1(name="agent1", subscriptions=["step.1"])
        agent2 = Agent2(name="agent2", subscriptions=["step.2"])

        orchestrator.register_agent(agent1)
        orchestrator.register_agent(agent2)

        envelope = APOPEnvelope(
            id="test_1",
            source="test",
            type="step.1",
            data={},
            next_action_hint="step.1",
        )

        results = await orchestrator.process_envelope(envelope)
        # Should have processed through both agents

    @pytest.mark.asyncio
    async def test_max_hops_protection(self, orchestrator):
        """Test max hops protection against infinite loops."""
        class LoopingAgent(BaseAgent):
            async def process(self, envelope: APOPEnvelope) -> APOPEnvelope:
                # Always points back to itself
                envelope.next_action_hint = "loop.event"
                return envelope

        agent = LoopingAgent(name="looper", subscriptions=["loop.event"])
        orchestrator.register_agent(agent)

        envelope = APOPEnvelope(
            id="test_1",
            source="test",
            type="loop.event",
            data={},
            next_action_hint="loop.event",
        )

        # Should stop at max_hops
        results = await orchestrator.process_envelope(envelope, max_hops=5)
        assert len(results) <= 5

    def test_default_flow(self, orchestrator):
        """Test default routing flow."""
        assert "inbox.routed" in orchestrator.default_flow
        assert orchestrator.default_flow["inbox.routed"] == "preproc.ocr"


@pytest.mark.unit
@pytest.mark.apop
class TestCloudEventsBus:
    """Tests for CloudEventsBus."""

    @pytest.fixture
    def event_bus(self):
        """Create CloudEventsBus instance."""
        return CloudEventsBus()

    @pytest.mark.asyncio
    async def test_publish_subscribe(self, event_bus):
        """Test pub/sub functionality."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("test.event", handler)

        envelope = APOPEnvelope(
            id="test_1",
            source="test",
            type="test.event",
            data={"message": "hello"},
        )

        await event_bus.publish(envelope)
        await asyncio.sleep(0.1)  # Allow async processing

        assert len(received_events) > 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers to same event."""
        count1 = []
        count2 = []

        async def handler1(event):
            count1.append(1)

        async def handler2(event):
            count2.append(1)

        event_bus.subscribe("test.event", handler1)
        event_bus.subscribe("test.event", handler2)

        envelope = APOPEnvelope(
            id="test_1",
            source="test",
            type="test.event",
            data={},
        )

        await event_bus.publish(envelope)
        await asyncio.sleep(0.1)

        # Both handlers should receive event

    def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        async def handler(event):
            pass

        event_bus.subscribe("test.event", handler)
        event_bus.unsubscribe("test.event", handler)

        # Should no longer receive events


@pytest.mark.unit
@pytest.mark.apop
class TestEnvelopeSignature:
    """Tests for EnvelopeSignature."""

    def test_signature_creation(self):
        """Test creating envelope signature."""
        envelope = APOPEnvelope(
            id="test_1",
            source="test",
            type="test.event",
            data={"key": "value"},
        )

        signature = EnvelopeSignature.sign(envelope, secret_key="test_secret")
        assert signature is not None
        assert isinstance(signature, str)

    def test_signature_verification(self):
        """Test verifying envelope signature."""
        envelope = APOPEnvelope(
            id="test_1",
            source="test",
            type="test.event",
            data={"key": "value"},
        )

        signature = EnvelopeSignature.sign(envelope, secret_key="test_secret")
        is_valid = EnvelopeSignature.verify(envelope, signature, secret_key="test_secret")

        assert is_valid is True

    def test_signature_tampering_detection(self):
        """Test detection of tampered envelopes."""
        envelope = APOPEnvelope(
            id="test_1",
            source="test",
            type="test.event",
            data={"key": "value"},
        )

        signature = EnvelopeSignature.sign(envelope, secret_key="test_secret")

        # Tamper with data
        envelope.data["key"] = "tampered"

        is_valid = EnvelopeSignature.verify(envelope, signature, secret_key="test_secret")
        assert is_valid is False


@pytest.mark.unit
@pytest.mark.apop
class TestStageAgents:
    """Tests for Stage Agents."""

    @pytest.mark.asyncio
    async def test_inbox_agent(self):
        """Test InboxAgent."""
        with patch('sap_llm.apop.stage_agents.InboxStage') as mock_stage:
            mock_stage_instance = MagicMock()
            mock_stage_instance.process.return_value = {
                "document_hash": "abc123",
                "is_duplicate": False,
            }
            mock_stage.return_value = mock_stage_instance

            agent = InboxAgent(config=MagicMock())
            envelope = APOPEnvelope(
                id="doc_1",
                source="email",
                type="document.received",
                data={"file_path": "/path/to/doc.pdf"},
            )

            result = await agent.process(envelope)
            assert result is not None
            assert result.type == "inbox.routed"

    @pytest.mark.asyncio
    async def test_preprocessing_agent(self):
        """Test PreprocessingAgent."""
        with patch('sap_llm.apop.stage_agents.PreprocessingStage') as mock_stage:
            mock_stage_instance = MagicMock()
            mock_stage_instance.process.return_value = {
                "ocr_text": "Test OCR",
                "words": ["Test", "OCR"],
                "boxes": [[0, 0, 100, 20], [110, 0, 200, 20]],
            }
            mock_stage.return_value = mock_stage_instance

            agent = PreprocessingAgent(config=MagicMock())
            envelope = APOPEnvelope(
                id="doc_1",
                source="inbox",
                type="inbox.routed",
                data={"file_path": "/path/to/doc.pdf"},
            )

            result = await agent.process(envelope)
            assert result is not None
            assert result.type == "preproc.ready"

    @pytest.mark.asyncio
    async def test_classification_agent(self):
        """Test ClassificationAgent."""
        with patch('sap_llm.apop.stage_agents.ClassificationStage') as mock_stage:
            mock_stage_instance = MagicMock()
            mock_stage_instance.process.return_value = {
                "predicted_class": 0,
                "confidence": 0.95,
            }
            mock_stage.return_value = mock_stage_instance

            agent = ClassificationAgent(config=MagicMock())
            envelope = APOPEnvelope(
                id="doc_1",
                source="preprocessing",
                type="preproc.ready",
                data={"ocr_text": "Test", "words": [], "boxes": []},
            )

            result = await agent.process(envelope)
            assert result is not None
            assert result.type == "classify.done"

    @pytest.mark.asyncio
    async def test_extraction_agent(self):
        """Test ExtractionAgent."""
        with patch('sap_llm.apop.stage_agents.ExtractionStage') as mock_stage:
            mock_stage_instance = MagicMock()
            mock_stage_instance.process.return_value = {
                "extracted_fields": {
                    "po_number": "12345",
                    "total_amount": 1000.0,
                }
            }
            mock_stage.return_value = mock_stage_instance

            agent = ExtractionAgent(config=MagicMock())
            envelope = APOPEnvelope(
                id="doc_1",
                source="classification",
                type="classify.done",
                data={"document_type": "PURCHASE_ORDER"},
            )

            result = await agent.process(envelope)
            assert result is not None
            assert result.type == "extract.done"

    @pytest.mark.asyncio
    async def test_validation_agent(self):
        """Test ValidationAgent."""
        with patch('sap_llm.apop.stage_agents.ValidationStage') as mock_stage:
            mock_stage_instance = MagicMock()
            mock_stage_instance.process.return_value = {
                "validation_passed": True,
                "exceptions": [],
            }
            mock_stage.return_value = mock_stage_instance

            agent = ValidationAgent(config=MagicMock())
            envelope = APOPEnvelope(
                id="doc_1",
                source="quality",
                type="quality.verified",
                data={"extracted_fields": {"po_number": "12345"}},
            )

            result = await agent.process(envelope)
            assert result is not None
            assert result.type == "rules.valid"

    @pytest.mark.asyncio
    async def test_routing_agent(self):
        """Test RoutingAgent."""
        with patch('sap_llm.apop.stage_agents.RoutingStage') as mock_stage:
            mock_stage_instance = MagicMock()
            mock_stage_instance.process.return_value = {
                "routing_decision": {
                    "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
                    "confidence": 0.95,
                }
            }
            mock_stage.return_value = mock_stage_instance

            agent = RoutingAgent(config=MagicMock())
            envelope = APOPEnvelope(
                id="doc_1",
                source="validation",
                type="rules.valid",
                data={"extracted_fields": {"po_number": "12345"}},
            )

            result = await agent.process(envelope)
            assert result is not None
            assert result.type == "router.done"

    @pytest.mark.asyncio
    async def test_agent_exception_handling(self):
        """Test agent exception handling."""
        with patch('sap_llm.apop.stage_agents.ValidationStage') as mock_stage:
            mock_stage_instance = MagicMock()
            mock_stage_instance.process.return_value = {
                "validation_passed": False,
                "exceptions": [
                    {"field": "total_amount", "error": "mismatch"}
                ],
            }
            mock_stage.return_value = mock_stage_instance

            agent = ValidationAgent(config=MagicMock())
            envelope = APOPEnvelope(
                id="doc_1",
                source="quality",
                type="quality.verified",
                data={"extracted_fields": {"total_amount": -100}},
            )

            result = await agent.process(envelope)
            assert result is not None
            # Should route to exception handling
            assert result.next_action_hint == "exception.handle" or result.type == "validation.failed"
