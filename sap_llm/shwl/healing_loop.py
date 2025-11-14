"""
Self-Healing Workflow Loop - Main orchestration

Runs periodic jobs to:
1. Cluster exceptions from PMG
2. Generate rule fixes
3. Submit for human approval
4. Deploy approved fixes
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.shwl.clusterer import ExceptionClusterer
from sap_llm.shwl.rule_generator import RuleGenerator
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class SelfHealingWorkflowLoop:
    """
    Self-Healing Workflow Loop orchestrator.

    Runs on a schedule (typically nightly) to:
    - Fetch recent exceptions from PMG
    - Cluster similar exceptions
    - Generate rule fix proposals
    - Submit for approval
    - Track deployment
    """

    def __init__(
        self,
        pmg: ProcessMemoryGraph,
        reasoning_engine: Optional[ReasoningEngine] = None,
        config: Optional[Any] = None,
    ):
        """
        Initialize SHWL.

        Args:
            pmg: Process Memory Graph instance
            reasoning_engine: Optional reasoning engine
            config: SHWL configuration
        """
        self.pmg = pmg
        self.reasoning_engine = reasoning_engine

        # Initialize components
        self.clusterer = ExceptionClusterer(
            min_cluster_size=getattr(
                getattr(config, "clustering", None),
                "min_cluster_size",
                15,
            ) if config else 15,
        )

        self.rule_generator = RuleGenerator(
            reasoning_engine=reasoning_engine,
            confidence_threshold=getattr(
                getattr(config, "rule_generation", None),
                "confidence_threshold",
                0.90,
            ) if config else 0.90,
        )

        # Configuration
        self.lookback_days = 7
        self.auto_approve_threshold = 0.95
        self.human_review_required = True

        if config and hasattr(config, "schedule"):
            schedule_config = config.schedule
            self.lookback_days = getattr(schedule_config, "lookback_days", 7)

        if config and hasattr(config, "approval"):
            approval_config = config.approval
            self.auto_approve_threshold = getattr(
                approval_config,
                "auto_approve_threshold",
                0.95,
            )
            self.human_review_required = getattr(
                approval_config,
                "human_review_required",
                True,
            )

        # Track proposals
        self.pending_proposals: List[Dict[str, Any]] = []
        self.approved_proposals: List[Dict[str, Any]] = []
        self.rejected_proposals: List[Dict[str, Any]] = []

        logger.info(
            f"SHWL initialized (lookback={self.lookback_days}d, "
            f"auto_approve_threshold={self.auto_approve_threshold})"
        )

    def run_healing_cycle(self) -> Dict[str, Any]:
        """
        Run one complete healing cycle.

        Returns:
            Cycle statistics
        """
        logger.info("Starting self-healing cycle...")

        start_time = datetime.now()

        # Step 1: Fetch exceptions
        exceptions = self._fetch_exceptions()
        logger.info(f"Fetched {len(exceptions)} exceptions")

        if not exceptions:
            logger.info("No exceptions to process")
            return {
                "start_time": start_time.isoformat(),
                "exceptions_fetched": 0,
                "clusters_found": 0,
                "proposals_generated": 0,
                "proposals_approved": 0,
            }

        # Step 2: Cluster exceptions
        clusters = self._cluster_exceptions(exceptions)
        logger.info(f"Found {len(clusters)} clusters")

        # Step 3: Generate fix proposals
        proposals = self._generate_proposals(clusters)
        logger.info(f"Generated {len(proposals)} fix proposals")

        # Step 4: Process approvals
        approved_count = self._process_approvals(proposals)
        logger.info(f"Approved {approved_count} proposals")

        # Step 5: Deploy approved fixes
        deployed_count = self._deploy_approved_fixes()
        logger.info(f"Deployed {deployed_count} fixes")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "exceptions_fetched": len(exceptions),
            "clusters_found": len(clusters),
            "proposals_generated": len(proposals),
            "proposals_approved": approved_count,
            "fixes_deployed": deployed_count,
        }

    def _fetch_exceptions(self) -> List[Dict[str, Any]]:
        """Fetch recent exceptions from PMG."""
        try:
            exceptions = self.pmg.query_exceptions(days=self.lookback_days)
            return exceptions
        except Exception as e:
            logger.error(f"Failed to fetch exceptions: {e}")
            return []

    def _cluster_exceptions(
        self,
        exceptions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Cluster exceptions by similarity."""
        try:
            clusters = self.clusterer.cluster(exceptions)

            # Filter clusters by size and severity
            significant_clusters = [
                c for c in clusters
                if c["size"] >= 15 and c["severity"] in ["HIGH", "MEDIUM"]
            ]

            return significant_clusters

        except Exception as e:
            logger.error(f"Failed to cluster exceptions: {e}")
            return []

    def _generate_proposals(
        self,
        clusters: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate fix proposals for clusters."""
        proposals = []

        # Get existing rules (TODO: load from configuration)
        existing_rules = []

        for cluster in clusters:
            try:
                # Generate fix proposal
                proposal = self.rule_generator.generate_fix(
                    cluster,
                    existing_rules,
                )

                # Add metadata
                proposal["cluster_id"] = cluster["id"]
                proposal["cluster_size"] = cluster["size"]
                proposal["timestamp"] = datetime.now().isoformat()
                proposal["status"] = "pending"

                proposals.append(proposal)

            except Exception as e:
                logger.error(
                    f"Failed to generate proposal for cluster {cluster['id']}: {e}"
                )

        return proposals

    def _process_approvals(self, proposals: List[Dict[str, Any]]) -> int:
        """
        Process approval for fix proposals.

        Args:
            proposals: List of proposals

        Returns:
            Number of approved proposals
        """
        approved_count = 0

        for proposal in proposals:
            # Auto-approve if confidence is very high and risk is low
            if (
                not self.human_review_required
                and proposal.get("confidence", 0) >= self.auto_approve_threshold
                and proposal.get("risk_level") == "low"
            ):
                proposal["status"] = "approved"
                proposal["approval_method"] = "automatic"
                proposal["approved_at"] = datetime.now().isoformat()

                self.approved_proposals.append(proposal)
                approved_count += 1

                logger.info(
                    f"Auto-approved proposal {proposal.get('rule_id')} "
                    f"(confidence={proposal.get('confidence'):.2f})"
                )

            else:
                # Queue for human review
                proposal["status"] = "pending_review"
                self.pending_proposals.append(proposal)

                logger.info(
                    f"Proposal {proposal.get('rule_id')} queued for human review"
                )

        return approved_count

    def _deploy_approved_fixes(self) -> int:
        """
        Deploy approved fixes.

        In production, this would:
        - Update business rule configuration
        - Deploy to staging environment
        - Run validation tests
        - Progressive rollout to production
        - Monitor for regressions

        Returns:
            Number of deployed fixes
        """
        deployed_count = 0

        for proposal in self.approved_proposals:
            if proposal.get("status") == "approved":
                try:
                    # TODO: Implement actual deployment
                    logger.info(
                        f"[MOCK] Deploying fix for rule {proposal.get('rule_id')}"
                    )

                    proposal["status"] = "deployed"
                    proposal["deployed_at"] = datetime.now().isoformat()

                    deployed_count += 1

                except Exception as e:
                    logger.error(f"Failed to deploy fix: {e}")
                    proposal["status"] = "deployment_failed"
                    proposal["error"] = str(e)

        return deployed_count

    def approve_proposal(self, proposal_id: str) -> bool:
        """
        Manually approve a proposal.

        Args:
            proposal_id: Proposal ID or cluster ID

        Returns:
            True if approved successfully
        """
        # Find proposal
        proposal = None
        for p in self.pending_proposals:
            if (
                p.get("cluster_id") == proposal_id
                or p.get("rule_id") == proposal_id
            ):
                proposal = p
                break

        if proposal is None:
            logger.error(f"Proposal not found: {proposal_id}")
            return False

        # Approve
        proposal["status"] = "approved"
        proposal["approval_method"] = "manual"
        proposal["approved_at"] = datetime.now().isoformat()

        # Move to approved list
        self.pending_proposals.remove(proposal)
        self.approved_proposals.append(proposal)

        logger.info(f"Manually approved proposal: {proposal_id}")

        return True

    def reject_proposal(self, proposal_id: str, reason: str) -> bool:
        """
        Reject a proposal.

        Args:
            proposal_id: Proposal ID
            reason: Rejection reason

        Returns:
            True if rejected successfully
        """
        # Find proposal
        proposal = None
        for p in self.pending_proposals:
            if (
                p.get("cluster_id") == proposal_id
                or p.get("rule_id") == proposal_id
            ):
                proposal = p
                break

        if proposal is None:
            logger.error(f"Proposal not found: {proposal_id}")
            return False

        # Reject
        proposal["status"] = "rejected"
        proposal["rejection_reason"] = reason
        proposal["rejected_at"] = datetime.now().isoformat()

        # Move to rejected list
        self.pending_proposals.remove(proposal)
        self.rejected_proposals.append(proposal)

        logger.info(f"Rejected proposal: {proposal_id} (reason: {reason})")

        return True

    def get_pending_proposals(self) -> List[Dict[str, Any]]:
        """Get list of pending proposals awaiting review."""
        return self.pending_proposals.copy()

    def get_proposal_status(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific proposal."""
        # Search all lists
        for proposal_list in [
            self.pending_proposals,
            self.approved_proposals,
            self.rejected_proposals,
        ]:
            for p in proposal_list:
                if (
                    p.get("cluster_id") == proposal_id
                    or p.get("rule_id") == proposal_id
                ):
                    return p

        return None
