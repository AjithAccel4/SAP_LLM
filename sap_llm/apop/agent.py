"""
Base Agent for APOP Orchestration

All pipeline stages can be implemented as agents that subscribe to
APOP envelopes and publish results.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from sap_llm.apop.envelope import APOPEnvelope, create_envelope
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """
    Base class for APOP agents.

    Agents:
    - Subscribe to specific event types
    - Process events autonomously
    - Publish results as new events
    - Support async operation
    """

    def __init__(
        self,
        agent_name: str,
        subscribes_to: List[str],
        publishes: List[str],
    ):
        """
        Initialize agent.

        Args:
            agent_name: Unique agent identifier
            subscribes_to: List of event types this agent subscribes to
            publishes: List of event types this agent can publish
        """
        self.agent_name = agent_name
        self.subscribes_to = subscribes_to
        self.publishes = publishes

        logger.info(
            f"Agent '{agent_name}' initialized "
            f"(subscribes={subscribes_to}, publishes={publishes})"
        )

    @abstractmethod
    async def process_event(self, envelope: APOPEnvelope) -> APOPEnvelope:
        """
        Process incoming event and return result envelope.

        Args:
            envelope: Incoming APOP envelope

        Returns:
            Result envelope to publish

        Raises:
            Exception: If processing fails
        """
        pass

    def can_handle(self, envelope: APOPEnvelope) -> bool:
        """
        Check if this agent can handle the envelope.

        Args:
            envelope: APOP envelope

        Returns:
            True if agent subscribes to this event type
        """
        return envelope.type in self.subscribes_to

    async def handle_envelope(self, envelope: APOPEnvelope) -> Optional[APOPEnvelope]:
        """
        Handle incoming envelope.

        Args:
            envelope: APOP envelope

        Returns:
            Result envelope or None if cannot handle
        """
        if not self.can_handle(envelope):
            logger.debug(
                f"Agent '{self.agent_name}' cannot handle event type: {envelope.type}"
            )
            return None

        try:
            logger.info(
                f"Agent '{self.agent_name}' processing event: {envelope.type} "
                f"(id={envelope.id})"
            )

            # Process event
            result_envelope = await self.process_event(envelope)

            # Validate result
            if not result_envelope.validate():
                logger.error(f"Invalid result envelope from agent '{self.agent_name}'")
                return None

            # Set correlation ID
            if result_envelope.correlation_id is None:
                result_envelope.correlation_id = envelope.correlation_id

            logger.info(
                f"Agent '{self.agent_name}' completed: {result_envelope.type}"
            )

            return result_envelope

        except Exception as e:
            logger.error(
                f"Agent '{self.agent_name}' failed to process event: {e}",
                exc_info=True,
            )

            # Create error envelope
            error_envelope = create_envelope(
                source=self.agent_name,
                event_type="error",
                data={
                    "error": str(e),
                    "original_event": envelope.to_dict(),
                },
                correlation_id=envelope.correlation_id,
            )

            return error_envelope

    def create_result_envelope(
        self,
        event_type: str,
        data: Dict[str, Any],
        next_action_hint: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> APOPEnvelope:
        """
        Create result envelope.

        Args:
            event_type: Result event type
            data: Result data
            next_action_hint: Optional routing hint
            correlation_id: Optional correlation ID

        Returns:
            APOP envelope
        """
        envelope = create_envelope(
            source=self.agent_name,
            event_type=event_type,
            data=data,
            next_action_hint=next_action_hint,
            correlation_id=correlation_id,
        )

        return envelope


class AgentRegistry:
    """
    Registry for managing agents.

    Keeps track of all registered agents and their capabilities.
    """

    def __init__(self):
        """Initialize agent registry."""
        self.agents: Dict[str, BaseAgent] = {}
        self.event_handlers: Dict[str, List[BaseAgent]] = {}

        logger.info("Agent registry initialized")

    def register(self, agent: BaseAgent) -> None:
        """
        Register an agent.

        Args:
            agent: Agent to register
        """
        # Add to agents dict
        self.agents[agent.agent_name] = agent

        # Add to event handlers
        for event_type in agent.subscribes_to:
            if event_type not in self.event_handlers:
                self.event_handlers[event_type] = []

            self.event_handlers[event_type].append(agent)

        logger.info(f"Registered agent: {agent.agent_name}")

    def unregister(self, agent_name: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_name: Name of agent to unregister
        """
        if agent_name in self.agents:
            agent = self.agents[agent_name]

            # Remove from event handlers
            for event_type in agent.subscribes_to:
                if event_type in self.event_handlers:
                    self.event_handlers[event_type].remove(agent)

            # Remove from agents dict
            del self.agents[agent_name]

            logger.info(f"Unregistered agent: {agent_name}")

    def get_handlers(self, event_type: str) -> List[BaseAgent]:
        """
        Get agents that can handle an event type.

        Args:
            event_type: Event type

        Returns:
            List of agents
        """
        return self.event_handlers.get(event_type, [])

    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get agent by name.

        Args:
            agent_name: Agent name

        Returns:
            Agent or None if not found
        """
        return self.agents.get(agent_name)

    def list_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return list(self.agents.keys())
