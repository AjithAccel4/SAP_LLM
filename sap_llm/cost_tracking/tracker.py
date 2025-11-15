"""
Real-Time Cost Tracking

Tracks processing cost per document in real-time:
- GPU time measurement (milliseconds)
- Token counting (input + output)
- Storage calculations (bytes)
- API call logging
- Cost per stage breakdown
- Per-customer billing

Target Cost: < $0.005 per document
- Cloud: $0.0047/doc
- On-prem: $0.0016/doc
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class StageCost:
    """Cost breakdown for a pipeline stage."""
    stage_name: str
    gpu_time_ms: float
    cpu_time_ms: float
    tokens_processed: int
    storage_bytes: int
    api_calls: int
    cost_usd: float


@dataclass
class DocumentCost:
    """Complete cost breakdown for document."""
    document_id: str
    doc_type: str
    timestamp: str

    # Stage costs
    classification_cost: StageCost
    extraction_cost: StageCost
    validation_cost: StageCost
    quality_cost: StageCost
    routing_cost: StageCost
    sap_posting_cost: StageCost
    pmg_storage_cost: StageCost

    # Totals
    total_gpu_time_ms: float
    total_cpu_time_ms: float
    total_tokens: int
    total_storage_bytes: int
    total_api_calls: int
    total_cost_usd: float

    # Deployment type
    deployment: str  # "cloud" or "on-prem"

    # Customer billing
    customer_id: Optional[str] = None
    tenant_id: Optional[str] = None


class CostTracker:
    """
    Real-time cost tracking for SAP_LLM.

    Pricing Model (Cloud):
    - GPU time: $2.50 / GPU-hour
    - Storage: $0.023 / GB-month
    - API calls: $0.0001 per call
    - Tokens: Included (model is self-hosted)

    Pricing Model (On-Prem):
    - GPU amortization: $0.50 / GPU-hour (over 3 years)
    - Storage: $0.005 / GB-month
    - API calls: $0.0001 per call
    - No external API costs
    """

    # Pricing constants (Cloud)
    GPU_COST_PER_HOUR_CLOUD = 2.50  # A10 GPU
    STORAGE_COST_PER_GB_MONTH_CLOUD = 0.023
    API_CALL_COST = 0.0001

    # Pricing constants (On-Prem)
    GPU_COST_PER_HOUR_ONPREM = 0.50  # Amortized over 3 years
    STORAGE_COST_PER_GB_MONTH_ONPREM = 0.005

    # Average processing times (milliseconds)
    AVG_CLASSIFICATION_GPU_MS = 50
    AVG_EXTRACTION_GPU_MS = 600
    AVG_VALIDATION_CPU_MS = 50
    AVG_QUALITY_CPU_MS = 30
    AVG_ROUTING_CPU_MS = 40
    AVG_SAP_API_MS = 200
    AVG_PMG_CPU_MS = 100

    def __init__(self, deployment: str = "cloud"):
        """
        Initialize cost tracker.

        Args:
            deployment: "cloud" or "on-prem"
        """
        self.deployment = deployment
        self.cost_history: List[DocumentCost] = []

        # Select pricing model
        if deployment == "cloud":
            self.gpu_cost_per_hour = self.GPU_COST_PER_HOUR_CLOUD
            self.storage_cost_per_gb_month = self.STORAGE_COST_PER_GB_MONTH_CLOUD
        else:
            self.gpu_cost_per_hour = self.GPU_COST_PER_HOUR_ONPREM
            self.storage_cost_per_gb_month = self.STORAGE_COST_PER_GB_MONTH_ONPREM

        logger.info(f"CostTracker initialized: {deployment}")

    def calculate_stage_cost(
        self,
        stage_name: str,
        gpu_time_ms: float,
        cpu_time_ms: float,
        tokens_processed: int,
        storage_bytes: int,
        api_calls: int
    ) -> StageCost:
        """Calculate cost for a pipeline stage."""
        # GPU cost
        gpu_cost = (gpu_time_ms / 1000 / 3600) * self.gpu_cost_per_hour

        # Storage cost (monthly, prorated to per-document)
        storage_gb = storage_bytes / (1024 ** 3)
        storage_cost = storage_gb * self.storage_cost_per_gb_month / 30 / 1000  # Per-document proration

        # API call cost
        api_cost = api_calls * self.API_CALL_COST

        # Total stage cost
        total_cost = gpu_cost + storage_cost + api_cost

        return StageCost(
            stage_name=stage_name,
            gpu_time_ms=gpu_time_ms,
            cpu_time_ms=cpu_time_ms,
            tokens_processed=tokens_processed,
            storage_bytes=storage_bytes,
            api_calls=api_calls,
            cost_usd=total_cost
        )

    def calculate_document_cost(
        self,
        document_id: str,
        doc_type: str,
        actual_times: Optional[Dict[str, float]] = None,
        customer_id: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> DocumentCost:
        """
        Calculate complete cost for document processing.

        Args:
            document_id: Document identifier
            doc_type: Document type
            actual_times: Actual processing times (optional, uses averages if not provided)
            customer_id: Customer ID for billing
            tenant_id: Tenant ID for multi-tenancy

        Returns:
            Complete cost breakdown
        """
        # Use actual times or averages
        times = actual_times or {}

        # Classification stage
        classification_cost = self.calculate_stage_cost(
            stage_name="classification",
            gpu_time_ms=times.get("classification_gpu_ms", self.AVG_CLASSIFICATION_GPU_MS),
            cpu_time_ms=times.get("classification_cpu_ms", 10),
            tokens_processed=times.get("classification_tokens", 500),
            storage_bytes=times.get("classification_storage", 1024),
            api_calls=0
        )

        # Extraction stage
        extraction_cost = self.calculate_stage_cost(
            stage_name="extraction",
            gpu_time_ms=times.get("extraction_gpu_ms", self.AVG_EXTRACTION_GPU_MS),
            cpu_time_ms=times.get("extraction_cpu_ms", 20),
            tokens_processed=times.get("extraction_tokens", 2000),
            storage_bytes=times.get("extraction_storage", 5120),
            api_calls=0
        )

        # Validation stage (CPU only)
        validation_cost = self.calculate_stage_cost(
            stage_name="validation",
            gpu_time_ms=0,
            cpu_time_ms=times.get("validation_cpu_ms", self.AVG_VALIDATION_CPU_MS),
            tokens_processed=0,
            storage_bytes=times.get("validation_storage", 2048),
            api_calls=0
        )

        # Quality check stage (CPU only)
        quality_cost = self.calculate_stage_cost(
            stage_name="quality_check",
            gpu_time_ms=0,
            cpu_time_ms=times.get("quality_cpu_ms", self.AVG_QUALITY_CPU_MS),
            tokens_processed=0,
            storage_bytes=times.get("quality_storage", 1024),
            api_calls=0
        )

        # Routing stage
        routing_cost = self.calculate_stage_cost(
            stage_name="routing",
            gpu_time_ms=times.get("routing_gpu_ms", 30),
            cpu_time_ms=times.get("routing_cpu_ms", self.AVG_ROUTING_CPU_MS),
            tokens_processed=times.get("routing_tokens", 800),
            storage_bytes=times.get("routing_storage", 2048),
            api_calls=0
        )

        # SAP posting stage (API call)
        sap_posting_cost = self.calculate_stage_cost(
            stage_name="sap_posting",
            gpu_time_ms=0,
            cpu_time_ms=times.get("sap_cpu_ms", self.AVG_SAP_API_MS),
            tokens_processed=0,
            storage_bytes=times.get("sap_storage", 3072),
            api_calls=1  # One SAP API call
        )

        # PMG storage stage
        pmg_storage_cost = self.calculate_stage_cost(
            stage_name="pmg_storage",
            gpu_time_ms=0,
            cpu_time_ms=times.get("pmg_cpu_ms", self.AVG_PMG_CPU_MS),
            tokens_processed=0,
            storage_bytes=times.get("pmg_storage", 10240),  # 10KB per doc
            api_calls=2  # Cosmos DB writes
        )

        # Calculate totals
        total_gpu_time_ms = (
            classification_cost.gpu_time_ms +
            extraction_cost.gpu_time_ms +
            routing_cost.gpu_time_ms
        )

        total_cpu_time_ms = (
            classification_cost.cpu_time_ms +
            extraction_cost.cpu_time_ms +
            validation_cost.cpu_time_ms +
            quality_cost.cpu_time_ms +
            routing_cost.cpu_time_ms +
            sap_posting_cost.cpu_time_ms +
            pmg_storage_cost.cpu_time_ms
        )

        total_tokens = (
            classification_cost.tokens_processed +
            extraction_cost.tokens_processed +
            routing_cost.tokens_processed
        )

        total_storage_bytes = (
            classification_cost.storage_bytes +
            extraction_cost.storage_bytes +
            validation_cost.storage_bytes +
            quality_cost.storage_bytes +
            routing_cost.storage_bytes +
            sap_posting_cost.storage_bytes +
            pmg_storage_cost.storage_bytes
        )

        total_api_calls = (
            sap_posting_cost.api_calls +
            pmg_storage_cost.api_calls
        )

        total_cost_usd = (
            classification_cost.cost_usd +
            extraction_cost.cost_usd +
            validation_cost.cost_usd +
            quality_cost.cost_usd +
            routing_cost.cost_usd +
            sap_posting_cost.cost_usd +
            pmg_storage_cost.cost_usd
        )

        # Create document cost
        doc_cost = DocumentCost(
            document_id=document_id,
            doc_type=doc_type,
            timestamp=datetime.now().isoformat(),
            classification_cost=classification_cost,
            extraction_cost=extraction_cost,
            validation_cost=validation_cost,
            quality_cost=quality_cost,
            routing_cost=routing_cost,
            sap_posting_cost=sap_posting_cost,
            pmg_storage_cost=pmg_storage_cost,
            total_gpu_time_ms=total_gpu_time_ms,
            total_cpu_time_ms=total_cpu_time_ms,
            total_tokens=total_tokens,
            total_storage_bytes=total_storage_bytes,
            total_api_calls=total_api_calls,
            total_cost_usd=total_cost_usd,
            deployment=self.deployment,
            customer_id=customer_id,
            tenant_id=tenant_id
        )

        # Store in history
        self.cost_history.append(doc_cost)

        return doc_cost

    def calculate_total_cost(self, result: Dict[str, Any]) -> float:
        """Calculate total cost from pipeline result."""
        # Extract actual times from result
        actual_times = {
            "classification_gpu_ms": result.get("classification", {}).get("processing_time_ms", 50),
            "extraction_gpu_ms": result.get("extraction", {}).get("processing_time_ms", 600),
            "validation_cpu_ms": result.get("validation", {}).get("processing_time_ms", 50),
        }

        doc_cost = self.calculate_document_cost(
            document_id=result.get("document_id", "unknown"),
            doc_type=result.get("classification", {}).get("doc_type", "unknown"),
            actual_times=actual_times
        )

        return doc_cost.total_cost_usd

    def calculate_routing_cost(self, routing_result: Dict[str, Any]) -> float:
        """Calculate routing stage cost."""
        stage_cost = self.calculate_stage_cost(
            stage_name="routing",
            gpu_time_ms=30,
            cpu_time_ms=routing_result.get("processing_time_ms", 40),
            tokens_processed=800,
            storage_bytes=2048,
            api_calls=0
        )

        return stage_cost.cost_usd

    def get_statistics(
        self,
        customer_id: Optional[str] = None,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Get cost statistics.

        Args:
            customer_id: Filter by customer (optional)
            period_days: Period in days

        Returns:
            Cost statistics and trends
        """
        # Filter cost history
        if customer_id:
            costs = [c for c in self.cost_history if c.customer_id == customer_id]
        else:
            costs = self.cost_history

        if not costs:
            return {"total_documents": 0}

        total_docs = len(costs)
        total_cost = sum(c.total_cost_usd for c in costs)
        avg_cost = total_cost / total_docs

        # Cost by document type
        cost_by_type = {}
        for cost in costs:
            if cost.doc_type not in cost_by_type:
                cost_by_type[cost.doc_type] = {"count": 0, "total_cost": 0.0}

            cost_by_type[cost.doc_type]["count"] += 1
            cost_by_type[cost.doc_type]["total_cost"] += cost.total_cost_usd

        for doc_type, stats in cost_by_type.items():
            stats["avg_cost"] = stats["total_cost"] / stats["count"]

        return {
            "total_documents": total_docs,
            "total_cost_usd": total_cost,
            "avg_cost_per_document": avg_cost,
            "deployment": self.deployment,
            "cost_by_document_type": cost_by_type,
            "target_cost": 0.005,
            "target_met": avg_cost <= 0.005,
            "cost_breakdown": {
                "classification": sum(c.classification_cost.cost_usd for c in costs) / total_docs,
                "extraction": sum(c.extraction_cost.cost_usd for c in costs) / total_docs,
                "validation": sum(c.validation_cost.cost_usd for c in costs) / total_docs,
                "quality": sum(c.quality_cost.cost_usd for c in costs) / total_docs,
                "routing": sum(c.routing_cost.cost_usd for c in costs) / total_docs,
                "sap_posting": sum(c.sap_posting_cost.cost_usd for c in costs) / total_docs,
                "pmg_storage": sum(c.pmg_storage_cost.cost_usd for c in costs) / total_docs
            }
        }

    def export_billing_report(
        self,
        customer_id: str,
        period_start: str,
        period_end: str
    ) -> str:
        """Export billing report for customer."""
        # Filter costs for customer and period
        costs = [
            c for c in self.cost_history
            if c.customer_id == customer_id and
            period_start <= c.timestamp <= period_end
        ]

        report = {
            "customer_id": customer_id,
            "period_start": period_start,
            "period_end": period_end,
            "deployment": self.deployment,
            "total_documents": len(costs),
            "total_cost_usd": sum(c.total_cost_usd for c in costs),
            "avg_cost_per_document": sum(c.total_cost_usd for c in costs) / len(costs) if costs else 0,
            "documents": [asdict(c) for c in costs]
        }

        return json.dumps(report, indent=2)


# Singleton instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker(deployment: str = "cloud") -> CostTracker:
    """Get singleton cost tracker."""
    global _cost_tracker

    if _cost_tracker is None:
        _cost_tracker = CostTracker(deployment=deployment)

    return _cost_tracker


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test cost tracking
    tracker = get_cost_tracker(deployment="on-prem")

    # Calculate cost for a document
    doc_cost = tracker.calculate_document_cost(
        document_id="DOC-001",
        doc_type="SUPPLIER_INVOICE",
        customer_id="CUST-001",
        tenant_id="TENANT-001"
    )

    print(f"\nDocument Cost Breakdown:")
    print(f"Total Cost: ${doc_cost.total_cost_usd:.6f}")
    print(f"Target: $0.005")
    print(f"Target Met: {doc_cost.total_cost_usd <= 0.005}")
    print(f"\nStage Costs:")
    print(f"  Classification: ${doc_cost.classification_cost.cost_usd:.6f}")
    print(f"  Extraction: ${doc_cost.extraction_cost.cost_usd:.6f}")
    print(f"  Validation: ${doc_cost.validation_cost.cost_usd:.6f}")
    print(f"  Quality: ${doc_cost.quality_cost.cost_usd:.6f}")
    print(f"  Routing: ${doc_cost.routing_cost.cost_usd:.6f}")
    print(f"  SAP Posting: ${doc_cost.sap_posting_cost.cost_usd:.6f}")
    print(f"  PMG Storage: ${doc_cost.pmg_storage_cost.cost_usd:.6f}")

    print(f"\nResource Usage:")
    print(f"  GPU Time: {doc_cost.total_gpu_time_ms:.0f}ms")
    print(f"  CPU Time: {doc_cost.total_cpu_time_ms:.0f}ms")
    print(f"  Tokens: {doc_cost.total_tokens:,}")
    print(f"  Storage: {doc_cost.total_storage_bytes / 1024:.1f} KB")
    print(f"  API Calls: {doc_cost.total_api_calls}")
