"""
Agentic Orchestrator - Routes APOP envelopes between agents

Implements autonomous workflow orchestration using next_action_hint
and agent capabilities.
"""

import asyncio
from typing import Any, Dict, List, Optional

from sap_llm.apop.agent import AgentRegistry, BaseAgent
from sap_llm.apop.envelope import APOPEnvelope
from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class AgenticOrchestrator:
    """
    Autonomous workflow orchestrator using APOP.

    Routes envelopes between agents based on:
    - next_action_hint in envelope
    - Agent subscriptions
    - PMG historical patterns
    - Default flow rules
    """

    def __init__(
        self,
        pmg: Optional[ProcessMemoryGraph] = None,
        max_hops: int = 20,
    ):
        """
        Initialize orchestrator.

        Args:
            pmg: Optional Process Memory Graph for intelligent routing
            max_hops: Maximum number of hops to prevent infinite loops
        """
        self.registry = AgentRegistry()
        self.pmg = pmg
        self.max_hops = max_hops

        # Default routing flow
        self.default_flow = {
            "inbox.routed": "preproc.ocr",
            "preproc.ready": "classify.detect",
            "classify.done": "extract.fields",
            "extract.done": "quality.check",
            "quality.verified": "rules.validate",
            "rules.valid": "router.post",
            "router.done": "complete",
        }

        logger.info("Agentic Orchestrator initialized")

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent.

        Args:
            agent: Agent to register
        """
        self.registry.register(agent)

    def unregister_agent(self, agent_name: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_name: Agent name
        """
        self.registry.unregister(agent_name)

    async def process_envelope(
        self,
        envelope: APOPEnvelope,
        max_hops: Optional[int] = None,
    ) -> List[APOPEnvelope]:
        """
        Process envelope through agent chain.

        Args:
            envelope: Initial envelope
            max_hops: Optional max hops override

        Returns:
            List of all envelopes produced in the chain
        """
        max_hops = max_hops or self.max_hops
        envelopes = [envelope]
        current_envelope = envelope
        hop_count = 0

        while hop_count < max_hops:
            # Determine next action
            next_action = self._determine_next_action(current_envelope)

            if next_action == "complete" or next_action is None:
                logger.info(f"Workflow complete after {hop_count} hops")
                break

            # Route to next agent
            next_envelope = await self._route_to_agent(current_envelope, next_action)

            if next_envelope is None:
                logger.warning(f"No agent could handle action: {next_action}")
                break

            # Add to chain
            envelopes.append(next_envelope)
            current_envelope = next_envelope
            hop_count += 1

            # Check for errors
            if next_envelope.type == "error":
                logger.error("Error in workflow, stopping")
                break

        if hop_count >= max_hops:
            logger.warning(f"Max hops ({max_hops}) reached, stopping workflow")

        return envelopes

    def _determine_next_action(self, envelope: APOPEnvelope) -> Optional[str]:
        """
        Determine next action from envelope.

        Priority:
        1. next_action_hint in envelope
        2. PMG historical patterns
        3. Default flow rules
        4. Agent capabilities

        Args:
            envelope: Current envelope

        Returns:
            Next action identifier or None
        """
        # Priority 1: Explicit hint
        if envelope.next_action_hint:
            logger.debug(f"Using next_action_hint: {envelope.next_action_hint}")
            return envelope.next_action_hint

        # Priority 2: PMG patterns
        if self.pmg and envelope.data:
            pmg_action = self._get_pmg_suggested_action(envelope)
            if pmg_action:
                logger.debug(f"Using PMG suggested action: {pmg_action}")
                return pmg_action

        # Priority 3: Default flow
        default_action = self.default_flow.get(envelope.type)
        if default_action:
            logger.debug(f"Using default flow action: {default_action}")
            return default_action

        # No action found
        logger.warning(f"No next action found for event type: {envelope.type}")
        return None

    def _get_pmg_suggested_action(self, envelope: APOPEnvelope) -> Optional[str]:
        """
        Query PMG for suggested next action.

        Args:
            envelope: Current envelope

        Returns:
            Suggested action or None
        """
        # TODO: Query PMG for similar workflows
        # For now, return None
        return None

    async def _route_to_agent(
        self,
        envelope: APOPEnvelope,
        action: str,
    ) -> Optional[APOPEnvelope]:
        """
        Route envelope to agent based on action.

        Args:
            envelope: Envelope to route
            action: Action identifier

        Returns:
            Result envelope or None
        """
        # Parse action (format: "service.action" or just "action")
        if "." in action:
            service_name, _ = action.split(".", 1)
        else:
            service_name = action

        # Find agent by name or event type
        agent = self.registry.get_agent(service_name)

        if agent is None:
            # Try to find by event type
            handlers = self.registry.get_handlers(envelope.type)
            if handlers:
                agent = handlers[0]  # Use first matching handler

        if agent is None:
            logger.warning(f"No agent found for action: {action}")
            return None

        # Process envelope
        result_envelope = await agent.handle_envelope(envelope)

        return result_envelope

    async def broadcast_envelope(
        self,
        envelope: APOPEnvelope,
    ) -> List[APOPEnvelope]:
        """
        Broadcast envelope to all capable agents.

        Args:
            envelope: Envelope to broadcast

        Returns:
            List of result envelopes
        """
        # Get all handlers for this event type
        handlers = self.registry.get_handlers(envelope.type)

        if not handlers:
            logger.warning(f"No handlers for event type: {envelope.type}")
            return []

        # Process in parallel
        tasks = [
            agent.handle_envelope(envelope)
            for agent in handlers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        valid_results = []
        for result in results:
            if isinstance(result, APOPEnvelope):
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Agent failed: {result}")

        return valid_results

    def get_workflow_status(
        self,
        correlation_id: str,
    ) -> Dict[str, Any]:
        """
        Get status of workflow by correlation ID.

        Args:
            correlation_id: Correlation ID

        Returns:
            Workflow status
        """
        # TODO: Query PMG for workflow status
        return {
            "correlation_id": correlation_id,
            "status": "unknown",
            "steps_completed": 0,
            "current_step": None,
        }

    def list_registered_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return self.registry.list_agents()
