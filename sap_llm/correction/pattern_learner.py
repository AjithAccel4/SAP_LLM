"""
Error Pattern Learning System.

Learns from corrections to improve future correction accuracy:
1. Identifies what changed in corrections
2. Extracts patterns from successful corrections
3. Stores patterns for future reference
4. Recommends best strategies based on historical success
"""

import json
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ErrorPatternLearner:
    """
    Learns from correction history to improve future corrections.

    Capabilities:
    1. Learn from successful corrections
    2. Identify common error patterns
    3. Track strategy effectiveness
    4. Suggest best correction strategies
    5. Provide historical patterns for context
    """

    def __init__(self, pmg=None, storage_path: Optional[str] = None):
        """
        Initialize error pattern learner.

        Args:
            pmg: Optional ProcessMemoryGraph for storing patterns
            storage_path: Optional path to store patterns locally
        """
        self.pmg = pmg
        self.storage_path = storage_path

        # In-memory pattern storage
        self.error_patterns = []
        self.strategy_effectiveness = defaultdict(lambda: {"attempts": 0, "successes": 0})
        self.field_error_patterns = defaultdict(list)

        # Load existing patterns if storage path provided
        if storage_path:
            self._load_patterns()

        logger.info("ErrorPatternLearner initialized")

    def learn_from_correction(
        self,
        original_prediction: Dict[str, Any],
        corrected_prediction: Dict[str, Any],
        correction_strategy: str,
        context: Dict[str, Any]
    ):
        """
        Learn from a successful correction.

        Args:
            original_prediction: Original prediction with errors
            corrected_prediction: Corrected prediction
            correction_strategy: Strategy that made the correction
            context: Context information
        """
        logger.info(f"Learning from correction using strategy: {correction_strategy}")

        try:
            # Identify what changed
            changes = self._diff_predictions(original_prediction, corrected_prediction)

            if not changes:
                logger.info("No changes detected in correction")
                return

            # Classify error type
            error_type = self._classify_error(changes, original_prediction)

            # Extract context features
            context_features = self._extract_features(context, original_prediction)

            # Create pattern
            pattern = {
                "id": self._generate_pattern_id(),
                "timestamp": datetime.now().isoformat(),
                "document_type": context.get("document_type", "UNKNOWN"),
                "vendor": context.get("vendor") or original_prediction.get("vendor_name", {}).get("value"),
                "error_type": error_type,
                "correction": {
                    "fields_changed": list(changes.keys()),
                    "strategy": correction_strategy,
                    "changes": changes
                },
                "context_features": context_features,
                "success": True
            }

            # Store pattern
            self.error_patterns.append(pattern)

            # Update per-field patterns
            for field in changes.keys():
                self.field_error_patterns[field].append(pattern)

            # Update strategy effectiveness
            self._update_strategy_effectiveness(correction_strategy, success=True)

            # Store in PMG if available
            if self.pmg:
                try:
                    self.pmg.add_error_pattern(pattern)
                except Exception as e:
                    logger.error(f"Failed to store pattern in PMG: {e}")

            # Save to local storage
            if self.storage_path:
                self._save_patterns()

            logger.info(
                f"Learned pattern: {error_type}, "
                f"fields={list(changes.keys())}, strategy={correction_strategy}"
            )

        except Exception as e:
            logger.error(f"Failed to learn from correction: {e}", exc_info=True)

    def get_relevant_patterns(
        self,
        context: Dict[str, Any],
        error_fields: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get relevant error patterns for current context.

        Args:
            context: Current context (document_type, vendor, etc.)
            error_fields: Optional list of fields with errors
            limit: Maximum patterns to return

        Returns:
            List of relevant patterns
        """
        logger.info("Retrieving relevant error patterns")

        patterns = []

        # Query PMG if available
        if self.pmg:
            try:
                patterns = self.pmg.query_error_patterns(
                    document_type=context.get("document_type"),
                    vendor=context.get("vendor"),
                    limit=limit
                )
            except Exception as e:
                logger.error(f"Failed to query PMG for patterns: {e}")

        # Fallback to in-memory patterns
        if not patterns:
            patterns = self._search_local_patterns(context, error_fields, limit)

        logger.info(f"Found {len(patterns)} relevant patterns")
        return patterns

    def suggest_correction_strategy(
        self,
        error_type: str,
        context: Dict[str, Any],
        error_fields: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Suggest the most effective correction strategy.

        Args:
            error_type: Type of error detected
            context: Current context
            error_fields: Fields with errors

        Returns:
            Recommended strategy name or None
        """
        logger.info(f"Suggesting correction strategy for error_type={error_type}")

        # Get relevant patterns
        patterns = self.get_relevant_patterns(context, error_fields)

        if not patterns:
            logger.info("No historical patterns found, using default strategy order")
            return None

        # Count strategy successes for this error type
        strategy_scores = Counter()

        for pattern in patterns:
            if pattern.get("error_type") == error_type:
                strategy = pattern["correction"]["strategy"]
                strategy_scores[strategy] += 1

        if not strategy_scores:
            logger.info(f"No patterns for error type {error_type}")
            return None

        # Return most successful strategy
        best_strategy, count = strategy_scores.most_common(1)[0]

        logger.info(
            f"Recommended strategy: {best_strategy} "
            f"(successful {count}/{len(patterns)} times)"
        )

        return best_strategy

    def get_strategy_effectiveness(
        self,
        strategy: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get effectiveness statistics for strategies.

        Args:
            strategy: Optional specific strategy, or all if None

        Returns:
            Effectiveness statistics
        """
        if strategy:
            stats = self.strategy_effectiveness.get(strategy, {"attempts": 0, "successes": 0})
            success_rate = (
                stats["successes"] / stats["attempts"]
                if stats["attempts"] > 0
                else 0.0
            )

            return {
                "strategy": strategy,
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "success_rate": success_rate
            }

        # Return all strategies
        all_stats = []
        for strat, stats in self.strategy_effectiveness.items():
            success_rate = (
                stats["successes"] / stats["attempts"]
                if stats["attempts"] > 0
                else 0.0
            )
            all_stats.append({
                "strategy": strat,
                "attempts": stats["attempts"],
                "successes": stats["successes"],
                "success_rate": success_rate
            })

        # Sort by success rate
        all_stats.sort(key=lambda x: x["success_rate"], reverse=True)

        return {
            "strategies": all_stats,
            "total_patterns": len(self.error_patterns)
        }

    def _diff_predictions(
        self,
        original: Dict[str, Any],
        corrected: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Identify differences between original and corrected predictions.

        Returns:
            Dictionary of changes per field
        """
        changes = {}

        # Check all fields in corrected prediction
        for field, corrected_value in corrected.items():
            # Skip metadata fields
            if field in ["correction_metadata", "requires_human_review", "status"]:
                continue

            original_value = original.get(field)

            # Detect changes
            if corrected_value != original_value:
                change = {
                    "original": original_value,
                    "corrected": corrected_value
                }

                # Extract actual values if dict format
                if isinstance(original_value, dict):
                    change["original_value"] = original_value.get("value")
                    change["original_confidence"] = original_value.get("confidence")

                if isinstance(corrected_value, dict):
                    change["corrected_value"] = corrected_value.get("value")
                    change["corrected_confidence"] = corrected_value.get("confidence")
                    change["correction_strategy"] = corrected_value.get("correction_strategy")

                changes[field] = change

        return changes

    def _classify_error(
        self,
        changes: Dict[str, Any],
        original: Dict[str, Any]
    ) -> str:
        """
        Classify the type of error that was corrected.

        Args:
            changes: Detected changes
            original: Original prediction

        Returns:
            Error type classification
        """
        # Check for calculation errors
        if "total_amount" in changes:
            if "subtotal" in original or "line_items" in original:
                return "calculation_error"

        # Check for format errors
        for field in changes.keys():
            if "date" in field.lower():
                return "date_format_error"
            if "amount" in field.lower() or "total" in field.lower():
                return "amount_error"

        # Check for missing field errors
        if any(
            changes[f].get("original_value") is None
            for f in changes.keys()
        ):
            return "missing_field_error"

        # Check for low confidence
        low_confidence_count = sum(
            1 for f in changes.keys()
            if changes[f].get("original_confidence", 1.0) < 0.75
        )

        if low_confidence_count > 0:
            return "low_confidence_error"

        return "unknown_error"

    def _extract_features(
        self,
        context: Dict[str, Any],
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract relevant features from context and prediction."""
        features = {
            "document_type": context.get("document_type"),
            "vendor": context.get("vendor") or prediction.get("vendor_name", {}).get("value"),
            "has_line_items": "line_items" in prediction,
            "num_fields": len(prediction),
            "field_names": list(prediction.keys())
        }

        # Add confidence distribution
        confidences = [
            v.get("confidence", 1.0)
            for v in prediction.values()
            if isinstance(v, dict)
        ]

        if confidences:
            features["avg_confidence"] = sum(confidences) / len(confidences)
            features["min_confidence"] = min(confidences)
            features["max_confidence"] = max(confidences)

        return features

    def _update_strategy_effectiveness(
        self,
        strategy: str,
        success: bool
    ):
        """Update strategy effectiveness tracking."""
        self.strategy_effectiveness[strategy]["attempts"] += 1
        if success:
            self.strategy_effectiveness[strategy]["successes"] += 1

        logger.debug(
            f"Strategy {strategy}: "
            f"{self.strategy_effectiveness[strategy]['successes']}/"
            f"{self.strategy_effectiveness[strategy]['attempts']} successful"
        )

    def _search_local_patterns(
        self,
        context: Dict[str, Any],
        error_fields: Optional[List[str]],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Search local in-memory patterns."""
        matching_patterns = []

        doc_type = context.get("document_type")
        vendor = context.get("vendor")

        for pattern in self.error_patterns:
            score = 0

            # Score by document type match
            if pattern.get("document_type") == doc_type:
                score += 2

            # Score by vendor match
            if vendor and pattern.get("vendor") == vendor:
                score += 2

            # Score by field overlap
            if error_fields:
                pattern_fields = set(pattern["correction"]["fields_changed"])
                error_fields_set = set(error_fields)
                overlap = len(pattern_fields & error_fields_set)
                score += overlap

            if score > 0:
                matching_patterns.append((score, pattern))

        # Sort by score and return top matches
        matching_patterns.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in matching_patterns[:limit]]

    def _generate_pattern_id(self) -> str:
        """Generate unique pattern ID."""
        import uuid
        return f"pattern_{uuid.uuid4().hex[:12]}"

    def _save_patterns(self):
        """Save patterns to local storage."""
        if not self.storage_path:
            return

        try:
            storage_file = Path(self.storage_path) / "error_patterns.json"
            storage_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "patterns": self.error_patterns,
                "strategy_effectiveness": dict(self.strategy_effectiveness),
                "last_updated": datetime.now().isoformat()
            }

            with open(storage_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.error_patterns)} patterns to {storage_file}")

        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")

    def _load_patterns(self):
        """Load patterns from local storage."""
        if not self.storage_path:
            return

        try:
            storage_file = Path(self.storage_path) / "error_patterns.json"

            if not storage_file.exists():
                logger.info("No existing patterns file found")
                return

            with open(storage_file, 'r') as f:
                data = json.load(f)

            self.error_patterns = data.get("patterns", [])

            # Rebuild strategy effectiveness
            for strategy, stats in data.get("strategy_effectiveness", {}).items():
                self.strategy_effectiveness[strategy] = stats

            # Rebuild field patterns
            for pattern in self.error_patterns:
                for field in pattern["correction"]["fields_changed"]:
                    self.field_error_patterns[field].append(pattern)

            logger.info(
                f"Loaded {len(self.error_patterns)} patterns from {storage_file}"
            )

        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
