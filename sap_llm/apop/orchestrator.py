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
        Query PMG for suggested next action based on similar workflows.

        Uses historical workflow data from PMG to intelligently suggest
        the next action based on similar successful routing decisions.

        Args:
            envelope: Current envelope

        Returns:
            Suggested action or None
        """
        try:
            # Extract document metadata from envelope
            data = envelope.data or {}
            doc_type = data.get("doc_type")
            supplier_id = data.get("supplier_id")
            company_code = data.get("company_code")

            # Need at least doc_type to query PMG
            if not doc_type:
                logger.debug("No doc_type in envelope data, cannot query PMG")
                return None

            # Query PMG for similar successful routing decisions
            logger.debug(
                f"Querying PMG for similar workflows: doc_type={doc_type}, "
                f"supplier={supplier_id}, company_code={company_code}"
            )

            similar_routings = self.pmg.get_similar_routing(
                doc_type=doc_type,
                supplier=supplier_id,
                company_code=company_code,
                limit=20,
            )

            if not similar_routings:
                logger.debug("No similar routing decisions found in PMG")
                return None

            # Analyze historical patterns to suggest next action
            # Map current event type to next action based on historical data
            next_actions = []
            for routing in similar_routings:
                # Extract endpoint and convert to action format
                endpoint = routing.get("endpoint", "")
                if endpoint:
                    # Convert endpoint to action hint (e.g., "/api/extract" -> "extract")
                    action = self._endpoint_to_action(endpoint)
                    if action:
                        next_actions.append(action)

            if next_actions:
                # Find most common next action
                from collections import Counter

                action_counts = Counter(next_actions)
                most_common_action, count = action_counts.most_common(1)[0]

                logger.info(
                    f"PMG suggests action '{most_common_action}' "
                    f"(based on {count}/{len(similar_routings)} similar workflows)"
                )

                return most_common_action

            return None

        except Exception as e:
            logger.error(f"Error querying PMG for suggested action: {e}")
            # Fallback to None on error - orchestrator will use default flow
            return None

    def _endpoint_to_action(self, endpoint: str) -> Optional[str]:
        """
        Convert API endpoint to action hint.

        Maps endpoints like '/api/v1/extract' to action hints like 'extract.fields'.

        Args:
            endpoint: API endpoint

        Returns:
            Action hint or None
        """
        # Simple mapping - can be enhanced based on actual endpoint structure
        endpoint_map = {
            "extract": "extract.fields",
            "classify": "classify.detect",
            "validate": "rules.validate",
            "quality": "quality.check",
            "route": "router.post",
            "ocr": "preproc.ocr",
        }

        # Extract action from endpoint
        endpoint_lower = endpoint.lower()
        for key, action in endpoint_map.items():
            if key in endpoint_lower:
                return action

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

        Queries PMG for complete workflow history including all steps,
        routing decisions, responses, and exceptions.

        Args:
            correlation_id: Correlation ID

        Returns:
            Workflow status dictionary with:
            - correlation_id: The correlation ID
            - status: Workflow status (active, completed, failed, unknown)
            - steps_completed: Number of completed steps
            - current_step: Current step endpoint or None
            - steps: List of all workflow steps
            - exceptions: List of exceptions raised
            - success: Whether workflow completed successfully
        """
        # Default response if PMG not available
        if not self.pmg:
            logger.warning("PMG not available, cannot query workflow status")
            return {
                "correlation_id": correlation_id,
                "status": "unknown",
                "steps_completed": 0,
                "current_step": None,
                "steps": [],
                "exceptions": [],
                "success": None,
            }

        try:
            # Query PMG for workflow data
            logger.debug(f"Querying PMG for workflow status: {correlation_id}")

            workflow = self.pmg.get_workflow_by_correlation_id(correlation_id)

            if not workflow:
                logger.warning(f"No workflow found for correlation_id: {correlation_id}")
                return {
                    "correlation_id": correlation_id,
                    "status": "not_found",
                    "steps_completed": 0,
                    "current_step": None,
                    "steps": [],
                    "exceptions": [],
                    "success": None,
                }

            # Get workflow steps
            steps = self.pmg.get_workflow_steps(correlation_id)
            steps_completed = len(steps)

            # Determine current step (last step if workflow still active)
            current_step = None
            if steps:
                current_step = steps[-1].get("endpoint")

            # Get exceptions
            exceptions = workflow.get("exceptions", [])
            has_exceptions = len(exceptions) > 0

            # Determine workflow status
            responses = workflow.get("responses", [])

            # Check if workflow completed successfully
            success = None
            if responses:
                last_response = responses[-1]
                success = last_response.get("success", False)

            # Determine overall status
            if success is True:
                status = "completed"
            elif success is False or has_exceptions:
                status = "failed"
            elif steps_completed > 0:
                status = "active"
            else:
                status = "unknown"

            logger.info(
                f"Workflow {correlation_id}: status={status}, "
                f"steps={steps_completed}, exceptions={len(exceptions)}"
            )

            return {
                "correlation_id": correlation_id,
                "status": status,
                "steps_completed": steps_completed,
                "current_step": current_step,
                "steps": steps,
                "exceptions": exceptions,
                "success": success,
                "document": workflow.get("document", {}),
            }

        except Exception as e:
            logger.error(f"Error querying workflow status: {e}")
            # Fallback to unknown status on error
            return {
                "correlation_id": correlation_id,
                "status": "error",
                "steps_completed": 0,
                "current_step": None,
                "steps": [],
                "exceptions": [],
                "success": None,
                "error": str(e),
            }

    def list_registered_agents(self) -> List[str]:
        """Get list of registered agent names."""
        return self.registry.list_agents()
