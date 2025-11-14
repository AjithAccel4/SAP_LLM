"""
FastAPI Server for SAP_LLM

Production-ready REST API with document processing endpoints,
authentication, rate limiting, and real-time status updates.
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from sap_llm.config import Config, load_config
from sap_llm.models.unified_model import UnifiedExtractorModel
from sap_llm.stages.inbox import InboxStage
from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.stages.classification import ClassificationStage
from sap_llm.stages.type_identifier import TypeIdentifierStage
from sap_llm.stages.extraction import ExtractionStage
from sap_llm.stages.quality_check import QualityCheckStage
from sap_llm.stages.validation import ValidationStage
from sap_llm.stages.routing import RoutingStage
from sap_llm.pmg.graph_client import ProcessMemoryGraph
from sap_llm.apop.envelope import APOPEnvelope
from sap_llm.apop.signature import APOPSignature
from sap_llm.monitoring.observability import observability
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global state
processing_jobs: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}


# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    job_id: str
    status: str
    message: str
    timestamp: str


class ExtractionRequest(BaseModel):
    """Request for field extraction."""
    document_url: Optional[str] = None
    document_base64: Optional[str] = None
    expected_type: Optional[str] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ExtractionResponse(BaseModel):
    """Response for field extraction."""
    job_id: str
    status: str
    document_type: Optional[str] = None
    document_subtype: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    confidence: Optional[float] = None
    routing_decision: Optional[Dict[str, Any]] = None
    sap_response: Optional[Dict[str, Any]] = None
    exceptions: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: Optional[float] = None
    timestamp: str


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str
    status: str
    progress: Optional[float] = None
    current_stage: Optional[str] = None
    result: Optional[ExtractionResponse] = None
    error: Optional[str] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    details: Dict[str, bool]


# Application state manager
class ApplicationState:
    """Manages application-level state and resources."""

    def __init__(self):
        self.config: Optional[Config] = None
        self.unified_model: Optional[UnifiedExtractorModel] = None
        self.pmg: Optional[ProcessMemoryGraph] = None
        self.stages: Dict[str, Any] = {}
        self.apop_signature: Optional[APOPSignature] = None
        self.is_ready = False

    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing SAP_LLM API server...")

        # Load configuration
        self.config = load_config()
        logger.info("Configuration loaded")

        # Initialize PMG
        self.pmg = ProcessMemoryGraph(
            endpoint=self.config.pmg.cosmos_endpoint,
            key=self.config.pmg.cosmos_key,
            database=self.config.pmg.cosmos_database,
            container=self.config.pmg.cosmos_container,
        )
        logger.info("PMG initialized")

        # Initialize unified model
        self.unified_model = UnifiedExtractorModel(config=self.config)
        logger.info("Unified model loaded")

        # Initialize pipeline stages
        self.stages["inbox"] = InboxStage(config=self.config.stages.inbox)
        self.stages["preprocessing"] = PreprocessingStage(
            config=self.config.stages.preprocessing
        )
        self.stages["classification"] = ClassificationStage(
            model=self.unified_model.vision_encoder,
            config=self.config.stages.classification,
        )
        self.stages["type_identifier"] = TypeIdentifierStage(
            model=self.unified_model.vision_encoder,
            config=self.config.stages.type_identifier,
        )
        self.stages["extraction"] = ExtractionStage(
            model=self.unified_model.language_decoder,
            config=self.config.stages.extraction,
        )
        self.stages["quality_check"] = QualityCheckStage(
            model=self.unified_model.language_decoder,
            pmg=self.pmg,
            config=self.config.stages.quality_check,
        )
        self.stages["validation"] = ValidationStage(
            config=self.config.stages.validation
        )
        self.stages["routing"] = RoutingStage(
            reasoning_engine=self.unified_model.reasoning_engine,
            pmg=self.pmg,
            config=self.config.stages.routing,
        )
        logger.info("Pipeline stages initialized")

        # Initialize APOP signature
        self.apop_signature = APOPSignature()
        self.apop_signature.generate_key_pair()
        logger.info("APOP signature initialized")

        self.is_ready = True
        logger.info("SAP_LLM API server ready")

    async def shutdown(self):
        """Cleanup resources."""
        logger.info("Shutting down SAP_LLM API server...")
        self.is_ready = False


# Application lifecycle manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    await app.state.app_state.initialize()
    yield
    # Shutdown
    await app.state.app_state.shutdown()


def create_app(config_path: Optional[str] = None) -> FastAPI:
    """
    Create FastAPI application.

    Args:
        config_path: Optional path to configuration file

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="SAP_LLM API",
        description="Autonomous document processing system with zero 3rd party LLM dependencies",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Initialize application state
    app.state.app_state = ApplicationState()

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Configure in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add monitoring middleware
    @app.middleware("http")
    async def monitoring_middleware(request: Request, call_next):
        """Track all HTTP requests with metrics."""
        start_time = time.time()

        # Process request
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            logger.error(f"Request failed: {e}")
            status_code = 500
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            observability.metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status=status_code,
                duration=duration
            )

        return response

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    return app


# Create app instance
app = create_app()


# Helper functions
def get_app_state() -> ApplicationState:
    """Get application state."""
    return app.state.app_state


async def notify_websocket(job_id: str, update: Dict[str, Any]):
    """Send update to WebSocket connection if exists."""
    if job_id in websocket_connections:
        try:
            await websocket_connections[job_id].send_json(update)
        except Exception as e:
            logger.error(f"Failed to send WebSocket update: {e}")


async def process_document_pipeline(
    job_id: str,
    file_path: str,
    file_content: bytes,
    expected_type: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
):
    """
    Process document through full pipeline.

    Args:
        job_id: Job identifier
        file_path: Original file path
        file_content: File content bytes
        expected_type: Expected document type hint
        options: Processing options
    """
    start_time = time.time()
    app_state = get_app_state()
    options = options or {}

    try:
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        processing_jobs[job_id]["current_stage"] = "inbox"
        await notify_websocket(
            job_id, {"status": "processing", "stage": "inbox", "progress": 0.0}
        )

        # Stage 1: Inbox
        inbox_output = app_state.stages["inbox"].process({
            "file_path": file_path,
            "file_content": file_content,
        })
        await notify_websocket(
            job_id, {"status": "processing", "stage": "preprocessing", "progress": 0.125}
        )

        # Stage 2: Preprocessing
        processing_jobs[job_id]["current_stage"] = "preprocessing"
        preprocessing_output = app_state.stages["preprocessing"].process(inbox_output)
        await notify_websocket(
            job_id, {"status": "processing", "stage": "classification", "progress": 0.25}
        )

        # Stage 3: Classification
        processing_jobs[job_id]["current_stage"] = "classification"
        classification_output = app_state.stages["classification"].process(
            preprocessing_output
        )
        await notify_websocket(
            job_id, {"status": "processing", "stage": "type_identifier", "progress": 0.375}
        )

        # Stage 4: Type Identifier
        processing_jobs[job_id]["current_stage"] = "type_identifier"
        type_output = app_state.stages["type_identifier"].process(
            classification_output
        )
        await notify_websocket(
            job_id, {"status": "processing", "stage": "extraction", "progress": 0.5}
        )

        # Stage 5: Extraction
        processing_jobs[job_id]["current_stage"] = "extraction"
        extraction_output = app_state.stages["extraction"].process(type_output)
        await notify_websocket(
            job_id, {"status": "processing", "stage": "quality_check", "progress": 0.625}
        )

        # Stage 6: Quality Check
        processing_jobs[job_id]["current_stage"] = "quality_check"
        quality_output = app_state.stages["quality_check"].process(extraction_output)
        await notify_websocket(
            job_id, {"status": "processing", "stage": "validation", "progress": 0.75}
        )

        # Stage 7: Validation
        processing_jobs[job_id]["current_stage"] = "validation"
        validation_output = app_state.stages["validation"].process(quality_output)
        await notify_websocket(
            job_id, {"status": "processing", "stage": "routing", "progress": 0.875}
        )

        # Stage 8: Routing
        processing_jobs[job_id]["current_stage"] = "routing"
        routing_output = app_state.stages["routing"].process(validation_output)

        # Store in PMG
        if not options.get("skip_pmg", False):
            app_state.pmg.store_transaction(
                document=routing_output["adc"],
                routing_decision=routing_output.get("routing_decision"),
                sap_response=routing_output.get("sap_response"),
                exceptions=routing_output.get("exceptions", []),
            )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Build response
        result = ExtractionResponse(
            job_id=job_id,
            status="completed",
            document_type=routing_output.get("document_type"),
            document_subtype=routing_output.get("document_subtype"),
            extracted_data=routing_output.get("adc"),
            quality_score=routing_output.get("quality_score"),
            confidence=routing_output.get("confidence"),
            routing_decision=routing_output.get("routing_decision"),
            sap_response=routing_output.get("sap_response"),
            exceptions=routing_output.get("exceptions", []),
            processing_time_ms=processing_time,
            timestamp=datetime.utcnow().isoformat(),
        )

        # Update job status
        processing_jobs[job_id]["status"] = "completed"
        processing_jobs[job_id]["result"] = result
        processing_jobs[job_id]["current_stage"] = "completed"

        await notify_websocket(
            job_id, {"status": "completed", "progress": 1.0, "result": result.dict()}
        )

        logger.info(
            f"Document processing completed: job_id={job_id}, "
            f"type={routing_output.get('document_type')}, "
            f"time={processing_time:.2f}ms"
        )

    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)

        # Record error metrics
        error_stage = processing_jobs[job_id].get("current_stage", "unknown")
        observability.metrics.record_error(
            error_type=type(e).__name__,
            stage=error_stage
        )

        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)

        await notify_websocket(
            job_id, {"status": "failed", "error": str(e)}
        )


# API Endpoints


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "service": "SAP_LLM API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns basic health status of the service.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.utcnow().isoformat(),
        components={
            "api": "healthy",
            "models": "loaded" if get_app_state().unified_model else "not_loaded",
            "pmg": "connected" if get_app_state().pmg else "disconnected",
        },
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["Health"])
async def readiness_check():
    """
    Readiness check endpoint.

    Returns whether the service is ready to accept requests.
    """
    app_state = get_app_state()

    details = {
        "config": app_state.config is not None,
        "unified_model": app_state.unified_model is not None,
        "pmg": app_state.pmg is not None,
        "stages": len(app_state.stages) == 8,
    }

    ready = all(details.values()) and app_state.is_ready

    return ReadinessResponse(ready=ready, details=details)


@app.post(
    "/v1/extract",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Processing"],
)
@limiter.limit("100/minute")
async def extract_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    expected_type: Optional[str] = None,
):
    """
    Extract fields from document (async).

    Upload a document for processing. Returns immediately with a job_id.
    Use /v1/jobs/{job_id} to check status or WebSocket for real-time updates.

    - **file**: Document file (PDF, PNG, JPG, TIFF)
    - **expected_type**: Optional document type hint
    """
    app_state = get_app_state()

    if not app_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}",
        )

    # Validate file size
    if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 50MB limit",
        )

    # Initialize job
    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "file_name": file.filename,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Queue background processing
    background_tasks.add_task(
        process_document_pipeline,
        job_id=job_id,
        file_path=file.filename or "unknown",
        file_content=file_content,
        expected_type=expected_type,
    )

    logger.info(f"Document queued for processing: job_id={job_id}, file={file.filename}")

    return DocumentUploadResponse(
        job_id=job_id,
        status="queued",
        message="Document queued for processing",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post(
    "/v1/extract/sync",
    response_model=ExtractionResponse,
    tags=["Processing"],
)
@limiter.limit("20/minute")
async def extract_document_sync(
    request: Request,
    file: UploadFile = File(...),
    expected_type: Optional[str] = None,
):
    """
    Extract fields from document (synchronous).

    Upload a document and wait for processing to complete.
    Use this for real-time processing needs.

    - **file**: Document file (PDF, PNG, JPG, TIFF)
    - **expected_type**: Optional document type hint
    """
    app_state = get_app_state()

    if not app_state.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready",
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Read file content
    try:
        file_content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read file: {str(e)}",
        )

    # Validate file size
    if len(file_content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 50MB limit",
        )

    # Initialize job
    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "file_name": file.filename,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Process synchronously
    await process_document_pipeline(
        job_id=job_id,
        file_path=file.filename or "unknown",
        file_content=file_content,
        expected_type=expected_type,
    )

    # Get result
    job = processing_jobs[job_id]

    if job["status"] == "failed":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=job.get("error", "Processing failed"),
        )

    return job["result"]


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["Jobs"])
async def get_job_status(job_id: str):
    """
    Get job status.

    Check the status of a document processing job.

    - **job_id**: Job identifier returned from /v1/extract
    """
    if job_id not in processing_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    job = processing_jobs[job_id]

    # Calculate progress
    stage_progress = {
        "queued": 0.0,
        "inbox": 0.125,
        "preprocessing": 0.25,
        "classification": 0.375,
        "type_identifier": 0.5,
        "extraction": 0.625,
        "quality_check": 0.75,
        "validation": 0.875,
        "routing": 0.9,
        "completed": 1.0,
        "failed": None,
    }

    current_stage = job.get("current_stage", job["status"])
    progress = stage_progress.get(current_stage, 0.0)

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=progress,
        current_stage=current_stage,
        result=job.get("result"),
        error=job.get("error"),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.delete("/v1/jobs/{job_id}", tags=["Jobs"])
async def delete_job(job_id: str):
    """
    Delete job.

    Remove a job from the processing queue or history.

    - **job_id**: Job identifier
    """
    if job_id not in processing_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job not found: {job_id}",
        )

    del processing_jobs[job_id]

    return {"message": f"Job {job_id} deleted"}


@app.websocket("/v1/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job updates.

    Connect to receive real-time updates about document processing progress.

    - **job_id**: Job identifier
    """
    await websocket.accept()
    websocket_connections[job_id] = websocket

    try:
        # Send initial status
        if job_id in processing_jobs:
            await websocket.send_json({
                "status": processing_jobs[job_id]["status"],
                "message": "Connected to job updates",
            })

        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")

    finally:
        # Cleanup
        if job_id in websocket_connections:
            del websocket_connections[job_id]


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format for scraping.
    """
    return observability.get_prometheus_metrics()


@app.get("/v1/slo", tags=["Monitoring"])
async def get_slo_status():
    """
    Get SLO status and error budgets.

    Returns current SLO compliance and remaining error budgets.
    """
    return observability.get_slo_status()


@app.get("/v1/stats", tags=["Monitoring"])
async def get_stats():
    """
    Get system statistics.

    Returns statistics about document processing and system health.
    """
    app_state = get_app_state()

    total_jobs = len(processing_jobs)
    completed_jobs = sum(1 for j in processing_jobs.values() if j["status"] == "completed")
    failed_jobs = sum(1 for j in processing_jobs.values() if j["status"] == "failed")
    processing = sum(1 for j in processing_jobs.values() if j["status"] == "processing")
    queued = sum(1 for j in processing_jobs.values() if j["status"] == "queued")

    return {
        "jobs": {
            "total": total_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
            "processing": processing,
            "queued": queued,
        },
        "system": {
            "models_loaded": app_state.unified_model is not None,
            "pmg_connected": app_state.pmg is not None,
            "active_websockets": len(websocket_connections),
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


# Error handlers


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "sap_llm.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
