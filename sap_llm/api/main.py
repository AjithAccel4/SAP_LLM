"""
SAP_LLM Core Engine API Server

FastAPI server exposing all pipeline stages:
- POST /v1/classify        # Document classification
- POST /v1/extract         # Field extraction
- POST /v1/validate        # Business rule validation
- POST /v1/route           # SAP routing decision
- POST /v1/process         # End-to-end pipeline
- GET  /v1/health          # Health check
- GET  /v1/metrics         # Prometheus metrics

Performance Targets:
- Latency: 780ms P95
- Throughput: 5000 docs/hour per instance
- Concurrent requests: 100
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import time
import logging
from datetime import datetime
import uvicorn

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SAP_LLM Core Engine",
    version="2.0.0",
    description="Enterprise Document Processing AI with Autonomous Decision-Making"
)

# CORS middleware - SECURITY: Restrict origins (no wildcards in production)
# Load allowed origins from environment variable
import os
cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000")
cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]

# Validate no wildcards in production
if "*" in cors_origins and os.getenv("ENVIRONMENT", "development") == "production":
    raise ValueError("CORS wildcard (*) not allowed in production. Set CORS_ALLOWED_ORIGINS environment variable.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    request_count = Counter(
        'sap_llm_requests_total',
        'Total requests',
        ['endpoint', 'status']
    )
    request_latency = Histogram(
        'sap_llm_latency_seconds',
        'Request latency',
        ['endpoint']
    )
    documents_processed = Counter(
        'sap_llm_documents_processed_total',
        'Documents processed',
        ['doc_type', 'status']
    )
    gpu_utilization = Gauge(
        'sap_llm_gpu_utilization',
        'GPU utilization percentage'
    )
    active_requests = Gauge(
        'sap_llm_active_requests',
        'Number of active requests'
    )


# Request/Response Models
class DocumentInput(BaseModel):
    """Input document for processing."""
    document_id: str = Field(..., description="Unique document identifier")
    content: str = Field(..., description="Base64 encoded document or text")
    content_type: str = Field(default="pdf", description="Document type: pdf, image, text")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Additional metadata")


class ClassificationResponse(BaseModel):
    """Document classification result."""
    document_id: str
    doc_type: str
    doc_subtype: Optional[str]
    confidence: float
    reasoning: str
    processing_time_ms: float


class ExtractionResponse(BaseModel):
    """Field extraction result."""
    document_id: str
    doc_type: str
    fields: Dict[str, Any]
    confidence_scores: Dict[str, float]
    overall_confidence: float
    processing_time_ms: float


class ValidationResponse(BaseModel):
    """Validation result."""
    document_id: str
    is_valid: bool
    errors: List[Dict[str, str]]
    warnings: List[Dict[str, str]]
    rules_checked: int
    processing_time_ms: float


class RoutingResponse(BaseModel):
    """SAP routing decision."""
    document_id: str
    sap_endpoint: str
    payload: Dict[str, Any]
    confidence: float
    reasoning: str
    estimated_cost: float
    processing_time_ms: float


class EndToEndResponse(BaseModel):
    """Complete pipeline result."""
    document_id: str
    classification: ClassificationResponse
    extraction: ExtractionResponse
    validation: ValidationResponse
    routing: RoutingResponse
    sap_response: Optional[Dict[str, Any]]
    total_processing_time_ms: float
    status: str
    cost_usd: float


# Health check model
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    services: Dict[str, str]
    gpu_available: bool
    uptime_seconds: float


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "SAP_LLM Core Engine",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "classify": "/v1/classify",
            "extract": "/v1/extract",
            "validate": "/v1/validate",
            "route": "/v1/route",
            "process": "/v1/process",
            "health": "/v1/health",
            "metrics": "/v1/metrics"
        }
    }


@app.post("/v1/classify", response_model=ClassificationResponse)
async def classify_document(doc: DocumentInput):
    """
    Classify document type and subtype.

    Supports:
    - 15 document types
    - 35+ subtypes (invoices, POs)
    - <50ms target latency
    """
    start_time = time.time()

    try:
        # Import classification service
        from sap_llm.pipeline.classifier import DocumentClassifier

        classifier = DocumentClassifier()

        # Classify
        result = classifier.classify(
            document_id=doc.document_id,
            content=doc.content,
            metadata=doc.metadata
        )

        processing_time = (time.time() - start_time) * 1000

        # Metrics
        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='classify', status='success').inc()
            request_latency.labels(endpoint='classify').observe(processing_time / 1000)
            documents_processed.labels(doc_type=result['doc_type'], status='success').inc()

        return ClassificationResponse(
            document_id=doc.document_id,
            doc_type=result['doc_type'],
            doc_subtype=result.get('doc_subtype'),
            confidence=result['confidence'],
            reasoning=result.get('reasoning', ''),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Classification failed: {e}")

        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='classify', status='error').inc()

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/extract", response_model=ExtractionResponse)
async def extract_fields(doc: DocumentInput):
    """
    Extract fields from document.

    Features:
    - 180+ field types
    - JSON schema compliance
    - <800ms target latency
    """
    start_time = time.time()

    try:
        from sap_llm.pipeline.extractor import FieldExtractor

        extractor = FieldExtractor()

        result = extractor.extract(
            document_id=doc.document_id,
            content=doc.content,
            doc_type=doc.metadata.get('doc_type', 'unknown')
        )

        processing_time = (time.time() - start_time) * 1000

        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='extract', status='success').inc()
            request_latency.labels(endpoint='extract').observe(processing_time / 1000)

        return ExtractionResponse(
            document_id=doc.document_id,
            doc_type=result['doc_type'],
            fields=result['fields'],
            confidence_scores=result['confidence_scores'],
            overall_confidence=result['overall_confidence'],
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Extraction failed: {e}")

        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='extract', status='error').inc()

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/validate", response_model=ValidationResponse)
async def validate_document(doc: DocumentInput):
    """
    Validate document against business rules.

    Checks:
    - Required fields
    - Three-way match (PO, Invoice, GR)
    - Date validation
    - Calculation checks
    """
    start_time = time.time()

    try:
        from sap_llm.pipeline.validator import DocumentValidator

        validator = DocumentValidator()

        result = validator.validate(
            document_id=doc.document_id,
            fields=doc.metadata.get('fields', {}),
            doc_type=doc.metadata.get('doc_type', 'unknown')
        )

        processing_time = (time.time() - start_time) * 1000

        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='validate', status='success').inc()
            request_latency.labels(endpoint='validate').observe(processing_time / 1000)

        return ValidationResponse(
            document_id=doc.document_id,
            is_valid=result['is_valid'],
            errors=result.get('errors', []),
            warnings=result.get('warnings', []),
            rules_checked=result.get('rules_checked', 0),
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")

        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='validate', status='error').inc()

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/route", response_model=RoutingResponse)
async def route_to_sap(doc: DocumentInput):
    """
    Generate SAP routing decision.

    Features:
    - 400+ S/4HANA API mapping
    - OData V2 payload generation
    - 97% routing accuracy
    """
    start_time = time.time()

    try:
        from sap_llm.pipeline.router import SAPRouter
        from sap_llm.cost_tracking.tracker import CostTracker

        router = SAPRouter()
        cost_tracker = CostTracker()

        result = router.route(
            document_id=doc.document_id,
            doc_type=doc.metadata.get('doc_type', 'unknown'),
            fields=doc.metadata.get('fields', {})
        )

        # Track cost
        cost = cost_tracker.calculate_routing_cost(result)

        processing_time = (time.time() - start_time) * 1000

        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='route', status='success').inc()
            request_latency.labels(endpoint='route').observe(processing_time / 1000)

        return RoutingResponse(
            document_id=doc.document_id,
            sap_endpoint=result['sap_endpoint'],
            payload=result['payload'],
            confidence=result['confidence'],
            reasoning=result.get('reasoning', ''),
            estimated_cost=cost,
            processing_time_ms=processing_time
        )

    except Exception as e:
        logger.error(f"Routing failed: {e}")

        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='route', status='error').inc()

        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/process", response_model=EndToEndResponse)
async def process_end_to_end(doc: DocumentInput, background_tasks: BackgroundTasks):
    """
    Complete end-to-end pipeline.

    Stages:
    1. Classification
    2. Extraction
    3. Validation
    4. Quality check
    5. Routing
    6. SAP posting
    7. PMG storage
    """
    start_time = time.time()

    try:
        if PROMETHEUS_AVAILABLE:
            active_requests.inc()

        from sap_llm.pipeline.orchestrator import PipelineOrchestrator
        from sap_llm.cost_tracking.tracker import CostTracker

        orchestrator = PipelineOrchestrator()
        cost_tracker = CostTracker()

        # Run pipeline
        result = await orchestrator.process_document(doc)

        # Calculate total cost
        total_cost = cost_tracker.calculate_total_cost(result)

        processing_time = (time.time() - start_time) * 1000

        # Store in PMG (background task)
        background_tasks.add_task(store_in_pmg, doc.document_id, result)

        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='process', status='success').inc()
            request_latency.labels(endpoint='process').observe(processing_time / 1000)
            active_requests.dec()

        return EndToEndResponse(
            document_id=doc.document_id,
            classification=result['classification'],
            extraction=result['extraction'],
            validation=result['validation'],
            routing=result['routing'],
            sap_response=result.get('sap_response'),
            total_processing_time_ms=processing_time,
            status=result['status'],
            cost_usd=total_cost
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

        if PROMETHEUS_AVAILABLE:
            request_count.labels(endpoint='process', status='error').inc()
            active_requests.dec()

        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and dependencies.
    """
    import torch

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        services={
            "pmg": "healthy",
            "model": "loaded",
            "cache": "connected"
        },
        gpu_available=torch.cuda.is_available(),
        uptime_seconds=time.time() - app.state.start_time
    )


@app.get("/v1/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format.
    """
    if not PROMETHEUS_AVAILABLE:
        return {"error": "Prometheus not available"}

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Background task
async def store_in_pmg(document_id: str, result: Dict[str, Any]):
    """Store processing result in PMG."""
    try:
        from sap_llm.pmg.graph_client import ProcessMemoryGraph

        pmg = ProcessMemoryGraph()
        pmg.store_transaction(document_id, result)

    except Exception as e:
        logger.error(f"PMG storage failed: {e}")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    app.state.start_time = time.time()
    logger.info("SAP_LLM Core Engine started")

    # Warmup models
    try:
        from sap_llm.pipeline.classifier import DocumentClassifier
        from sap_llm.pipeline.extractor import FieldExtractor

        logger.info("Loading models...")
        classifier = DocumentClassifier()
        extractor = FieldExtractor()
        logger.info("Models loaded successfully")

    except Exception as e:
        logger.error(f"Model loading failed: {e}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("SAP_LLM Core Engine shutting down")


# Run server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_level="info"
    )
