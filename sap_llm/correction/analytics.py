"""
Correction Analytics and Monitoring.

Provides comprehensive analytics and reporting on the self-correction system:
1. Correction success rates
2. Strategy effectiveness
3. Error pattern analysis
4. Performance metrics
5. Human escalation analysis
"""

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class CorrectionAnalytics:
    """
    Analytics and monitoring for the self-correction system.

    Tracks:
    - Correction attempts and success rates
    - Strategy effectiveness
    - Error patterns and trends
    - Human escalation rates
    - Performance metrics
    """

    def __init__(
        self,
        correction_engine=None,
        pattern_learner=None,
        escalation_manager=None,
        storage_path: Optional[str] = None
    ):
        """
        Initialize correction analytics.

        Args:
            correction_engine: SelfCorrectionEngine instance
            pattern_learner: ErrorPatternLearner instance
            escalation_manager: EscalationManager instance
            storage_path: Optional path to store analytics data
        """
        self.correction_engine = correction_engine
        self.pattern_learner = pattern_learner
        self.escalation_manager = escalation_manager
        self.storage_path = storage_path

        # Event storage
        self.correction_events = []

        # Load historical data if available
        if storage_path:
            self._load_events()

        logger.info("CorrectionAnalytics initialized")

    def record_correction_event(
        self,
        prediction: Dict[str, Any],
        correction_metadata: Dict[str, Any],
        context: Dict[str, Any]
    ):
        """
        Record a correction event for analytics.

        Args:
            prediction: Corrected prediction
            correction_metadata: Metadata from correction process
            context: Context information
        """
        event = {
            "id": self._generate_event_id(),
            "timestamp": datetime.now().isoformat(),
            "document_id": context.get("document_id"),
            "document_type": context.get("document_type"),
            "vendor": context.get("vendor"),
            "correction_attempted": correction_metadata.get("total_attempts", 0) > 0,
            "success": correction_metadata.get("success", False),
            "total_attempts": correction_metadata.get("total_attempts", 0),
            "strategies_tried": correction_metadata.get("strategies_tried", []),
            "required_human_review": correction_metadata.get("required_human_review", False),
            "final_confidence": correction_metadata.get("final_confidence"),
            "duration_seconds": correction_metadata.get("duration_seconds"),
            "initial_quality": correction_metadata.get("initial_quality"),
        }

        # Extract fields corrected
        if "attempts" in correction_metadata:
            all_fields_corrected = []
            for attempt in correction_metadata["attempts"]:
                all_fields_corrected.extend(attempt.get("fields_corrected", []))
            event["fields_corrected"] = list(set(all_fields_corrected))

        self.correction_events.append(event)

        # Save to storage
        if self.storage_path:
            self._save_events()

        logger.debug(f"Recorded correction event: {event['id']}")

    def generate_correction_report(
        self,
        period_days: int = 7,
        include_details: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive correction report.

        Args:
            period_days: Number of days to include in report
            include_details: Include detailed breakdowns

        Returns:
            Correction report
        """
        logger.info(f"Generating correction report for last {period_days} days")

        # Filter events by time period
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_events = [
            e for e in self.correction_events
            if datetime.fromisoformat(e["timestamp"]) > cutoff_date
        ]

        if not recent_events:
            logger.warning("No correction events in specified period")
            return {
                "period_days": period_days,
                "total_corrections": 0,
                "message": "No correction events in specified period"
            }

        # Basic statistics
        total_corrections = len(recent_events)
        successful = sum(1 for e in recent_events if e.get("success"))
        human_escalations = sum(
            1 for e in recent_events
            if e.get("required_human_review")
        )

        report = {
            "period_days": period_days,
            "period_start": cutoff_date.isoformat(),
            "period_end": datetime.now().isoformat(),
            "total_corrections": total_corrections,
            "successful_corrections": successful,
            "success_rate": successful / total_corrections if total_corrections > 0 else 0,
            "human_escalations": human_escalations,
            "escalation_rate": human_escalations / total_corrections if total_corrections > 0 else 0,
        }

        # Average attempts and duration
        attempts = [e.get("total_attempts", 0) for e in recent_events if e.get("total_attempts")]
        durations = [e.get("duration_seconds", 0) for e in recent_events if e.get("duration_seconds")]

        if attempts:
            report["avg_attempts"] = sum(attempts) / len(attempts)
            report["max_attempts"] = max(attempts)

        if durations:
            report["avg_duration_seconds"] = sum(durations) / len(durations)
            report["max_duration_seconds"] = max(durations)

        # Strategy effectiveness
        if include_details:
            report["strategy_effectiveness"] = self._analyze_strategy_effectiveness(recent_events)
            report["most_common_errors"] = self._identify_common_errors(recent_events)
            report["correction_trends"] = self._analyze_trends(recent_events)
            report["document_type_breakdown"] = self._analyze_by_document_type(recent_events)

        # Add external component stats if available
        if self.pattern_learner:
            report["pattern_learning"] = self.pattern_learner.get_strategy_effectiveness()

        if self.escalation_manager:
            report["escalation_stats"] = self.escalation_manager.get_escalation_stats()

        logger.info(
            f"Report generated: {total_corrections} corrections, "
            f"{report['success_rate']:.1%} success rate"
        )

        return report

    def _analyze_strategy_effectiveness(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze effectiveness of each correction strategy."""
        strategy_stats = defaultdict(lambda: {"used": 0, "successful": 0})

        for event in events:
            strategies = event.get("strategies_tried", [])
            success = event.get("success", False)

            for strategy in strategies:
                strategy_stats[strategy]["used"] += 1
                if success:
                    strategy_stats[strategy]["successful"] += 1

        # Calculate success rates
        result = {}
        for strategy, stats in strategy_stats.items():
            result[strategy] = {
                "times_used": stats["used"],
                "successful": stats["successful"],
                "success_rate": stats["successful"] / stats["used"] if stats["used"] > 0 else 0
            }

        # Sort by success rate
        result = dict(
            sorted(result.items(), key=lambda x: x[1]["success_rate"], reverse=True)
        )

        return result

    def _identify_common_errors(
        self,
        events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify most common error patterns."""
        # Count fields that needed correction
        field_corrections = Counter()

        for event in events:
            fields = event.get("fields_corrected", [])
            for field in fields:
                field_corrections[field] += 1

        # Get top 10 most commonly corrected fields
        common_errors = [
            {
                "field": field,
                "count": count,
                "percentage": count / len(events) if len(events) > 0 else 0
            }
            for field, count in field_corrections.most_common(10)
        ]

        return common_errors

    def _analyze_trends(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze correction trends over time."""
        # Group by day
        daily_stats = defaultdict(lambda: {"total": 0, "successful": 0, "human_escalations": 0})

        for event in events:
            date = datetime.fromisoformat(event["timestamp"]).date().isoformat()

            daily_stats[date]["total"] += 1
            if event.get("success"):
                daily_stats[date]["successful"] += 1
            if event.get("required_human_review"):
                daily_stats[date]["human_escalations"] += 1

        # Calculate daily success rates
        trends = {}
        for date, stats in sorted(daily_stats.items()):
            trends[date] = {
                "total": stats["total"],
                "successful": stats["successful"],
                "success_rate": stats["successful"] / stats["total"] if stats["total"] > 0 else 0,
                "human_escalations": stats["human_escalations"],
                "escalation_rate": stats["human_escalations"] / stats["total"] if stats["total"] > 0 else 0
            }

        return trends

    def _analyze_by_document_type(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze corrections by document type."""
        doc_type_stats = defaultdict(lambda: {"total": 0, "successful": 0, "human_escalations": 0})

        for event in events:
            doc_type = event.get("document_type", "UNKNOWN")

            doc_type_stats[doc_type]["total"] += 1
            if event.get("success"):
                doc_type_stats[doc_type]["successful"] += 1
            if event.get("required_human_review"):
                doc_type_stats[doc_type]["human_escalations"] += 1

        # Calculate rates
        result = {}
        for doc_type, stats in doc_type_stats.items():
            result[doc_type] = {
                "total": stats["total"],
                "successful": stats["successful"],
                "success_rate": stats["successful"] / stats["total"] if stats["total"] > 0 else 0,
                "human_escalations": stats["human_escalations"],
                "escalation_rate": stats["human_escalations"] / stats["total"] if stats["total"] > 0 else 0
            }

        # Sort by total
        result = dict(
            sorted(result.items(), key=lambda x: x[1]["total"], reverse=True)
        )

        return result

    def export_report(
        self,
        report: Dict[str, Any],
        output_path: str,
        format: str = "json"
    ):
        """
        Export report to file.

        Args:
            report: Report data
            output_path: Output file path
            format: Export format (json, csv, html)
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report exported to {output_file}")

        elif format == "html":
            html = self._generate_html_report(report)
            with open(output_file, 'w') as f:
                f.write(html)
            logger.info(f"HTML report exported to {output_file}")

        else:
            logger.warning(f"Unsupported export format: {format}")

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Self-Correction Analytics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        .metric {{ background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
    </style>
</head>
<body>
    <h1>Self-Correction System Analytics Report</h1>

    <div class="metric">
        <h2>Summary ({report.get('period_days', 0)} days)</h2>
        <p><strong>Total Corrections:</strong> {report.get('total_corrections', 0)}</p>
        <p><strong>Success Rate:</strong> <span class="{'success' if report.get('success_rate', 0) > 0.8 else 'warning'}">{report.get('success_rate', 0):.1%}</span></p>
        <p><strong>Escalation Rate:</strong> <span class="{'success' if report.get('escalation_rate', 0) < 0.1 else 'warning'}">{report.get('escalation_rate', 0):.1%}</span></p>
        <p><strong>Avg Attempts:</strong> {report.get('avg_attempts', 0):.1f}</p>
        <p><strong>Avg Duration:</strong> {report.get('avg_duration_seconds', 0):.1f}s</p>
    </div>

    <h2>Strategy Effectiveness</h2>
    <table>
        <tr>
            <th>Strategy</th>
            <th>Times Used</th>
            <th>Successful</th>
            <th>Success Rate</th>
        </tr>
"""

        for strategy, stats in report.get('strategy_effectiveness', {}).items():
            html += f"""
        <tr>
            <td>{strategy}</td>
            <td>{stats['times_used']}</td>
            <td>{stats['successful']}</td>
            <td>{stats['success_rate']:.1%}</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""
        return html

    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return f"event_{uuid.uuid4().hex[:12]}"

    def _save_events(self):
        """Save events to storage."""
        if not self.storage_path:
            return

        try:
            storage_file = Path(self.storage_path) / "correction_events.json"
            storage_file.parent.mkdir(parents=True, exist_ok=True)

            with open(storage_file, 'w') as f:
                json.dump({
                    "events": self.correction_events,
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)

            logger.debug(f"Saved {len(self.correction_events)} events to {storage_file}")

        except Exception as e:
            logger.error(f"Failed to save events: {e}")

    def _load_events(self):
        """Load events from storage."""
        if not self.storage_path:
            return

        try:
            storage_file = Path(self.storage_path) / "correction_events.json"

            if not storage_file.exists():
                logger.info("No existing events file found")
                return

            with open(storage_file, 'r') as f:
                data = json.load(f)

            self.correction_events = data.get("events", [])

            logger.info(f"Loaded {len(self.correction_events)} events from {storage_file}")

        except Exception as e:
            logger.error(f"Failed to load events: {e}")
