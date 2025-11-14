"""
Rule Generator - Generates business rule fixes using reasoning model

Analyzes exception clusters and proposes rule modifications to prevent
future occurrences.
"""

import json
import re
from typing import Any, Dict, List, Optional

from sap_llm.models.reasoning_engine import ReasoningEngine
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class RuleGenerator:
    """
    Generate rule fixes for exception clusters.

    Uses Reasoning Engine (Mixtral) to:
    - Analyze exception patterns
    - Propose rule modifications
    - Estimate impact and risk
    - Generate human-readable explanations
    """

    def __init__(
        self,
        reasoning_engine: Optional[ReasoningEngine] = None,
        confidence_threshold: float = 0.90,
    ):
        """
        Initialize rule generator.

        Args:
            reasoning_engine: Reasoning engine instance
            confidence_threshold: Minimum confidence for auto-apply
        """
        self.reasoning_engine = reasoning_engine
        self.confidence_threshold = confidence_threshold

        if reasoning_engine is None:
            logger.warning("Reasoning engine not provided, using mock mode")
            self.mock_mode = True
        else:
            self.mock_mode = False

        logger.info("RuleGenerator initialized")

    def generate_fix(
        self,
        cluster: Dict[str, Any],
        existing_rules: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate rule fix for exception cluster.

        Args:
            cluster: Exception cluster
            existing_rules: Current business rules

        Returns:
            Rule fix proposal with:
            - rule_diff: Proposed change
            - confidence: Confidence score
            - reasoning: Explanation
            - risk_level: Risk assessment
            - estimated_impact: Number of exceptions resolved
        """
        if self.mock_mode:
            return self._generate_mock_fix(cluster)

        # Analyze cluster pattern
        pattern = self._analyze_cluster_pattern(cluster)

        # Find relevant existing rules
        relevant_rules = self._find_relevant_rules(cluster, existing_rules)

        # Generate fix using reasoning engine
        prompt = self._build_rule_generation_prompt(
            cluster,
            pattern,
            relevant_rules,
        )

        response = self.reasoning_engine.generate(prompt)

        # Parse response
        fix_proposal = self._parse_fix_proposal(response)

        # Validate proposal
        if not self._validate_proposal(fix_proposal):
            logger.warning("Generated proposal failed validation")
            fix_proposal["confidence"] = 0.0

        # Estimate impact
        fix_proposal["estimated_impact"] = len(cluster["exceptions"])

        return fix_proposal

    def _analyze_cluster_pattern(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze exception cluster to identify common pattern.

        Args:
            cluster: Exception cluster

        Returns:
            Pattern dictionary
        """
        exceptions = cluster["exceptions"]

        # Extract common fields
        fields = [e.get("field") for e in exceptions if "field" in e]
        messages = [e.get("message") for e in exceptions if "message" in e]
        categories = [e.get("category") for e in exceptions if "category" in e]

        # Find most common
        from collections import Counter

        pattern = {
            "size": len(exceptions),
            "category": cluster["category"],
            "severity": cluster["severity"],
            "common_field": max(set(fields), key=fields.count) if fields else None,
            "common_message": max(set(messages), key=messages.count) if messages else None,
            "field_distribution": dict(Counter(fields)),
            "example_exceptions": exceptions[:5],  # Sample
        }

        return pattern

    def _find_relevant_rules(
        self,
        cluster: Dict[str, Any],
        existing_rules: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Find existing rules relevant to this cluster.

        Args:
            cluster: Exception cluster
            existing_rules: All existing rules

        Returns:
            List of relevant rules
        """
        relevant = []

        category = cluster["category"]

        for rule in existing_rules:
            # Check if rule matches category or affects same fields
            if category in rule.get("categories", []):
                relevant.append(rule)

        return relevant[:5]  # Top 5

    def _build_rule_generation_prompt(
        self,
        cluster: Dict[str, Any],
        pattern: Dict[str, Any],
        relevant_rules: List[Dict[str, Any]],
    ) -> str:
        """
        Build prompt for rule generation.

        Args:
            cluster: Exception cluster
            pattern: Analyzed pattern
            relevant_rules: Existing related rules

        Returns:
            Prompt string
        """
        prompt = f"""You are an expert at business rule engineering for document processing systems.

**Exception Cluster Analysis:**
Cluster ID: {cluster['id']}
Number of Exceptions: {pattern['size']}
Category: {pattern['category']}
Severity: {pattern['severity']}
Common Field: {pattern.get('common_field', 'N/A')}
Common Message: {pattern.get('common_message', 'N/A')}

**Sample Exceptions:**
{json.dumps(pattern['example_exceptions'], indent=2)[:1000]}

**Current Related Rules:**
{json.dumps(relevant_rules, indent=2)[:1000] if relevant_rules else 'No related rules found'}

**Task:**
Analyze this exception pattern and propose a rule modification that will:
1. Prevent these exceptions in the future
2. Maintain data integrity
3. Not introduce new issues

Output a JSON response with this structure:
{{
  "modification_type": "adjust_threshold|add_condition|add_exception_handler|create_new_rule",
  "rule_id": "existing_rule_id_or_NEW",
  "changes": {{
    "description": "What changes to make",
    "old_value": "current value if modifying existing rule",
    "new_value": "proposed new value"
  }},
  "reasoning": "Detailed explanation of why this fix is appropriate and safe",
  "confidence": 0.95,
  "risk_level": "low|medium|high",
  "side_effects": "Potential unintended consequences"
}}

**Rule Proposal:**
"""
        return prompt

    def _parse_fix_proposal(self, response: str) -> Dict[str, Any]:
        """
        Parse fix proposal from model response.

        Args:
            response: Model response text

        Returns:
            Parsed proposal dictionary
        """
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)

        if json_match:
            try:
                proposal = json.loads(json_match.group())
                return proposal
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")

        # Return default proposal
        return {
            "modification_type": "unknown",
            "rule_id": "UNKNOWN",
            "changes": {},
            "reasoning": response[:500],
            "confidence": 0.0,
            "risk_level": "high",
        }

    def _validate_proposal(self, proposal: Dict[str, Any]) -> bool:
        """
        Validate fix proposal.

        Args:
            proposal: Proposed fix

        Returns:
            True if valid
        """
        required_fields = ["modification_type", "rule_id", "changes", "reasoning"]

        for field in required_fields:
            if field not in proposal:
                logger.warning(f"Proposal missing required field: {field}")
                return False

        # Check confidence
        if proposal.get("confidence", 0) < 0.5:
            logger.warning(f"Proposal confidence too low: {proposal.get('confidence')}")
            return False

        return True

    def _generate_mock_fix(self, cluster: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock fix proposal for testing."""
        return {
            "modification_type": "adjust_threshold",
            "rule_id": "VAL_001",
            "changes": {
                "description": "Increase tolerance threshold",
                "old_value": "0.03",
                "new_value": "0.05",
            },
            "reasoning": (
                f"Mock fix for cluster with {len(cluster['exceptions'])} exceptions. "
                f"Category: {cluster['category']}"
            ),
            "confidence": 0.85,
            "risk_level": "low",
            "estimated_impact": len(cluster["exceptions"]),
        }
