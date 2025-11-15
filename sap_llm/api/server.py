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
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, HttpUrl
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
from sap_llm.api.auth import User, get_current_active_user, require_admin

logger = get_logger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Global state
processing_jobs: Dict[str, Dict[str, Any]] = {}
websocket_connections: Dict[str, WebSocket] = {}


# Pydantic models for API
class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    job_id: str = Field(..., description="Unique job identifier", example="550e8400-e29b-41d4-a716-446655440000")
    status: str = Field(..., description="Job status", example="queued")
    message: str = Field(..., description="Human-readable message", example="Document queued for processing")
    timestamp: str = Field(..., description="ISO 8601 timestamp", example="2025-11-14T10:30:00Z")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "message": "Document queued for processing",
                "timestamp": "2025-11-14T10:30:00Z"
            }
        }


class ExtractionRequest(BaseModel):
    """Request for field extraction."""
    document_url: Optional[str] = Field(None, description="URL to document file", example="https://example.com/invoice.pdf")
    document_base64: Optional[str] = Field(None, description="Base64-encoded document content")
    expected_type: Optional[str] = Field(None, description="Expected document type hint", example="invoice")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing options")

    class Config:
        schema_extra = {
            "example": {
                "document_url": "https://example.com/invoice.pdf",
                "expected_type": "invoice",
                "options": {
                    "skip_pmg": False,
                    "priority": "normal"
                }
            }
        }


class ExtractionResponse(BaseModel):
    """Response for field extraction."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: queued, processing, completed, failed")
    document_type: Optional[str] = Field(None, description="Identified document type", example="invoice")
    document_subtype: Optional[str] = Field(None, description="Document subtype", example="standard_invoice")
    extracted_data: Optional[Dict[str, Any]] = Field(None, description="Extracted field data")
    quality_score: Optional[float] = Field(None, description="Quality score (0-1)", ge=0, le=1, example=0.95)
    confidence: Optional[float] = Field(None, description="Extraction confidence (0-1)", ge=0, le=1, example=0.92)
    routing_decision: Optional[Dict[str, Any]] = Field(None, description="SAP routing decision")
    sap_response: Optional[Dict[str, Any]] = Field(None, description="SAP system response")
    exceptions: Optional[List[Dict[str, Any]]] = Field(None, description="Processing exceptions")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds", example=1234.56)
    timestamp: str = Field(..., description="ISO 8601 timestamp")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "document_type": "invoice",
                "document_subtype": "standard_invoice",
                "extracted_data": {
                    "invoice_number": "INV-2025-001",
                    "invoice_date": "2025-11-14",
                    "total_amount": 1250.00,
                    "currency": "USD",
                    "vendor_name": "Acme Corporation"
                },
                "quality_score": 0.95,
                "confidence": 0.92,
                "routing_decision": {
                    "action": "post",
                    "target_system": "SAP_S4HANA",
                    "priority": "normal"
                },
                "processing_time_ms": 1234.56,
                "timestamp": "2025-11-14T10:30:15Z"
            }
        }


class JobStatusResponse(BaseModel):
    """Response for job status query."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: queued, processing, completed, failed")
    progress: Optional[float] = Field(None, description="Processing progress (0-1)", ge=0, le=1, example=0.75)
    current_stage: Optional[str] = Field(None, description="Current pipeline stage", example="extraction")
    result: Optional[ExtractionResponse] = Field(None, description="Processing result (when completed)")
    error: Optional[str] = Field(None, description="Error message (when failed)")
    timestamp: str = Field(..., description="ISO 8601 timestamp")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "progress": 0.75,
                "current_stage": "validation",
                "timestamp": "2025-11-14T10:30:10Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status", example="healthy")
    version: str = Field(..., description="API version", example="0.1.0")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    components: Dict[str, str] = Field(..., description="Component health status")

    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "0.1.0",
                "timestamp": "2025-11-14T10:30:00Z",
                "components": {
                    "api": "healthy",
                    "models": "loaded",
                    "pmg": "connected"
                }
            }
        }


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    ready: bool = Field(..., description="Whether service is ready to accept requests")
    details: Dict[str, bool] = Field(..., description="Readiness status of each component")

    class Config:
        schema_extra = {
            "example": {
                "ready": True,
                "details": {
                    "config": True,
                    "unified_model": True,
                    "pmg": True,
                    "stages": True
                }
            }
        }


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
        description="""
# SAP_LLM Document Processing API

Autonomous document processing system with zero 3rd party LLM dependencies.

## Features

- **Intelligent Document Processing**: Automatically extract fields from invoices, purchase orders, receipts, and more
- **Real-time Processing**: Track document processing status in real-time via WebSocket
- **High Accuracy**: Vision-language models trained specifically for business documents
- **SAP Integration**: Seamless routing to SAP S/4HANA, SAP Ariba, and other SAP systems
- **Quality Assurance**: Built-in quality checks and validation
- **Production Ready**: Rate limiting, monitoring, and observability built-in

## Processing Pipeline

1. **Inbox**: Document ingestion and initial validation
2. **Preprocessing**: Image enhancement and normalization
3. **Classification**: Document category identification
4. **Type Identification**: Specific document type detection
5. **Extraction**: Field extraction using unified model
6. **Quality Check**: Confidence scoring and validation
7. **Validation**: Business rules and data validation
8. **Routing**: SAP system routing and posting

## Authentication

Currently, the API uses rate limiting for protection. API key authentication can be configured via environment variables.

## Rate Limits

- **Async Processing**: 100 requests/minute per IP
- **Sync Processing**: 20 requests/minute per IP
- **Status Queries**: Unlimited

## Support

For issues and questions, please refer to the documentation or contact the development team.
        """,
        version="0.1.0",
        lifespan=lifespan,
        contact={
            "name": "SAP_LLM Team",
            "url": "https://github.com/your-org/SAP_LLM",
            "email": "support@example.com"
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        openapi_tags=[
            {
                "name": "General",
                "description": "General API information and root endpoints"
            },
            {
                "name": "Health",
                "description": "Health check and readiness endpoints for monitoring"
            },
            {
                "name": "Processing",
                "description": "Document processing endpoints - both async and synchronous"
            },
            {
                "name": "Jobs",
                "description": "Job management and status tracking"
            },
            {
                "name": "Monitoring",
                "description": "Metrics, SLO tracking, and system statistics"
            }
        ],
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "displayRequestDuration": True,
            "filter": True,
            "showExtensions": True,
            "syntaxHighlight.theme": "obsidian"
        }
    )

    # Initialize application state
    app.state.app_state = ApplicationState()

    # Load config for CORS settings
    config = load_config(config_path)

    # Parse CORS origins from environment variable (comma-separated)
    cors_origins = []
    if config.api.cors.get("origins"):
        for origin in config.api.cors["origins"]:
            # Handle comma-separated origins from env vars
            if "," in origin:
                cors_origins.extend([o.strip() for o in origin.split(",")])
            else:
                cors_origins.append(origin)

    # Add middleware with proper CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins else ["http://localhost:3000"],
        allow_credentials=config.api.cors.get("credentials", True),
        allow_methods=config.api.cors.get("methods", ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
        allow_headers=config.api.cors.get("headers", ["*"]),
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
    """
    Root endpoint providing API information.

    Returns basic information about the API service including version,
    status, and links to documentation.

    ## Response

    Returns service metadata and documentation links.
    """
    return {
        "service": "SAP_LLM API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.

    Returns the overall health status of the service and its components.
    This endpoint is designed for health monitoring systems and load balancers.

    ## Use Cases

    - **Load Balancer Health Checks**: Use this endpoint to determine if the service should receive traffic
    - **Monitoring Systems**: Track service availability over time
    - **Alerting**: Trigger alerts when health status changes

    ## Response

    Returns health status of all major components:
    - API server status
    - ML models loading status
    - Process Memory Graph connectivity

    ## Status Values

    - `healthy`: All systems operational
    - `degraded`: Service operational but some components have issues
    - `unhealthy`: Service experiencing critical issues
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
    Readiness check endpoint for Kubernetes and orchestration systems.

    Returns whether the service is fully initialized and ready to accept
    processing requests. Unlike /health, this endpoint checks if all
    components are loaded and configured.

    ## Use Cases

    - **Kubernetes Readiness Probes**: Determine if pod should receive traffic
    - **Startup Validation**: Verify all components initialized successfully
    - **Rolling Deployments**: Ensure new instances are ready before routing traffic

    ## Response

    Returns detailed readiness status:
    - Overall readiness boolean
    - Individual component readiness status

    ## Component Checks

    - **config**: Configuration loaded successfully
    - **unified_model**: ML models loaded into memory
    - **pmg**: Process Memory Graph connected
    - **stages**: All 8 pipeline stages initialized

    Returns `ready: false` if any component is not initialized.
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
    current_user: User = Depends(get_current_active_user),
    file: UploadFile = File(..., description="Document file to process (PDF, PNG, JPG, TIFF)"),
    expected_type: Optional[str] = None,
):
    """
    Upload document for asynchronous processing.

    This endpoint accepts a document and queues it for processing through the
    full 8-stage pipeline. Returns immediately with a job ID that can be used
    to track processing status.

    ## Processing Flow

    1. Document is uploaded and validated
    2. Job is created and queued for background processing
    3. Response with job_id is returned immediately (HTTP 202)
    4. Use `/v1/jobs/{job_id}` to poll status
    5. Use WebSocket `/v1/ws/{job_id}` for real-time updates

    ## Supported Formats

    - **PDF**: Multi-page PDF documents
    - **PNG**: PNG images
    - **JPG/JPEG**: JPEG images
    - **TIFF**: TIFF images (including multi-page)

    ## File Size Limits

    - Maximum file size: 50MB
    - Recommended: < 10MB for optimal performance

    ## Document Types

    Supported document types (optional hint via `expected_type`):
    - `invoice`: Commercial invoices
    - `purchase_order`: Purchase orders
    - `receipt`: Receipts
    - `delivery_note`: Delivery notes
    - `credit_note`: Credit notes
    - `contract`: Contracts

    ## Rate Limiting

    - 100 requests per minute per IP address
    - Returns HTTP 429 if limit exceeded

    ## Example Usage

    ```bash
    curl -X POST "http://localhost:8000/v1/extract" \\
      -H "Content-Type: multipart/form-data" \\
      -F "file=@invoice.pdf" \\
      -F "expected_type=invoice"
    ```

    ## Response

    Returns job details including:
    - `job_id`: Unique identifier for tracking
    - `status`: Current job status (queued)
    - `message`: Human-readable message
    - `timestamp`: Job creation timestamp
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
    current_user: User = Depends(get_current_active_user),
    file: UploadFile = File(..., description="Document file to process (PDF, PNG, JPG, TIFF)"),
    expected_type: Optional[str] = None,
):
    """
    Upload document for synchronous processing.

    This endpoint processes the document immediately and waits for completion
    before returning results. Use this for real-time, interactive scenarios
    where you need immediate results.

    ## Processing Flow

    1. Document is uploaded and validated
    2. Processing begins immediately
    3. Request blocks until processing completes
    4. Complete results returned in response (HTTP 200)

    ## When to Use

    **Use Sync Endpoint When:**
    - You need immediate results
    - Processing latency is acceptable (2-5 seconds typical)
    - Building interactive user interfaces
    - Implementing real-time validation

    **Use Async Endpoint When:**
    - Processing large batches of documents
    - Building background processing systems
    - Latency must be minimized for API response
    - Implementing queue-based architectures

    ## Performance Considerations

    - **Typical Processing Time**: 2-5 seconds per document
    - **Request Timeout**: 30 seconds
    - **Concurrent Requests**: Limited by server resources

    ## Rate Limiting

    - 20 requests per minute per IP address (lower than async)
    - Returns HTTP 429 if limit exceeded

    ## Error Handling

    Returns HTTP 500 if processing fails. Error details included in response.

    ## Example Usage

    ```bash
    curl -X POST "http://localhost:8000/v1/extract/sync" \\
      -H "Content-Type: multipart/form-data" \\
      -F "file=@invoice.pdf" \\
      -F "expected_type=invoice"
    ```

    ## Response

    Returns complete extraction results including:
    - Document type and subtype
    - Extracted field data
    - Quality and confidence scores
    - Routing decisions
    - SAP system responses
    - Processing metrics
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
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Get processing job status and results.

    Query the current status of a document processing job. Returns job metadata,
    progress information, and results when processing is complete.

    ## Polling Strategy

    For optimal performance, use the following polling strategy:

    1. **Initial Poll**: Wait 1 second after job creation
    2. **Active Processing**: Poll every 2-3 seconds
    3. **Backoff**: Increase to 5-10 seconds if job is queued
    4. **WebSocket Alternative**: Use WebSocket endpoint for real-time updates

    ## Job Lifecycle

    ```
    queued → processing → completed
                       ↓
                     failed
    ```

    ## Status Values

    - **queued**: Job waiting to be processed
    - **processing**: Currently being processed
    - **completed**: Processing finished successfully
    - **failed**: Processing encountered an error

    ## Pipeline Stages

    When status is "processing", `current_stage` indicates the active stage:

    1. `inbox` - Document ingestion (12.5% complete)
    2. `preprocessing` - Image enhancement (25%)
    3. `classification` - Category detection (37.5%)
    4. `type_identifier` - Type identification (50%)
    5. `extraction` - Field extraction (62.5%)
    6. `quality_check` - Quality validation (75%)
    7. `validation` - Business rules (87.5%)
    8. `routing` - SAP routing (90%)

    ## Response Fields

    - `job_id`: Unique job identifier
    - `status`: Current job status
    - `progress`: Processing progress (0.0 to 1.0)
    - `current_stage`: Active pipeline stage
    - `result`: Complete extraction results (when completed)
    - `error`: Error message (when failed)
    - `timestamp`: Response timestamp

    ## Example Usage

    ```bash
    # Poll job status
    curl http://localhost:8000/v1/jobs/550e8400-e29b-41d4-a716-446655440000
    ```

    ## Error Responses

    - **404 Not Found**: Job ID does not exist or has been deleted
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
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete a processing job.

    Remove a job from the server's memory. This cleans up job history and
    frees resources. Note that this does not delete data from the Process
    Memory Graph (PMG).

    ## When to Use

    - Clean up completed jobs to free memory
    - Remove failed jobs after handling errors
    - Implement job retention policies
    - Cancel jobs that are no longer needed

    ## Important Notes

    - **Cannot Cancel**: This does not cancel active processing
    - **PMG Data Preserved**: Historical data in PMG is not affected
    - **Idempotent**: Safe to call multiple times

    ## Example Usage

    ```bash
    curl -X DELETE http://localhost:8000/v1/jobs/550e8400-e29b-41d4-a716-446655440000
    ```

    ## Response

    Returns confirmation message with deleted job ID.

    ## Error Responses

    - **404 Not Found**: Job ID does not exist
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

    Establish a WebSocket connection to receive real-time updates about
    document processing progress. More efficient than polling for status.

    ## Connection Flow

    1. Client connects to WebSocket endpoint with job_id
    2. Server accepts connection and sends initial status
    3. Server sends updates as processing progresses through pipeline
    4. Connection remains open until processing completes or client disconnects

    ## Message Format

    All messages are JSON objects with the following possible fields:

    ```json
    {
      "status": "processing",
      "stage": "extraction",
      "progress": 0.625,
      "message": "Extracting fields...",
      "result": {...}  // Only when completed
    }
    ```

    ## Status Updates

    You will receive updates at key points:
    - Job starts processing
    - Each pipeline stage completes (8 updates total)
    - Processing completes successfully
    - Processing fails with error

    ## Keep-Alive

    Send "ping" messages to keep connection alive:

    ```
    Client → "ping"
    Server → "pong"
    ```

    ## Example Usage (JavaScript)

    ```javascript
    const ws = new WebSocket('ws://localhost:8000/v1/ws/550e8400-e29b-41d4-a716-446655440000');

    ws.onmessage = (event) => {
      const update = JSON.parse(event.data);
      console.log(`Status: ${update.status}, Progress: ${update.progress * 100}%`);

      if (update.status === 'completed') {
        console.log('Results:', update.result);
        ws.close();
      }
    };

    // Keep-alive
    setInterval(() => ws.send('ping'), 30000);
    ```

    ## Example Usage (Python)

    ```python
    import asyncio
    import websockets
    import json

    async def track_job(job_id):
        uri = f"ws://localhost:8000/v1/ws/{job_id}"
        async with websockets.connect(uri) as websocket:
            while True:
                message = await websocket.recv()
                update = json.loads(message)
                print(f"Status: {update['status']}")

                if update['status'] in ['completed', 'failed']:
                    break

    asyncio.run(track_job('550e8400-e29b-41d4-a716-446655440000'))
    ```

    ## Connection Lifecycle

    - **Automatic Cleanup**: Connection closed automatically when job completes
    - **Client Disconnect**: Server handles disconnections gracefully
    - **Multiple Connections**: Multiple clients can track the same job

    ## Error Handling

    - Connection automatically closes if job does not exist
    - Error messages sent before closing on processing failures
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
async def metrics(
    current_user: User = Depends(require_admin),
):
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus exposition format for scraping by monitoring systems.

    ## Metrics Exposed

    ### Request Metrics
    - `http_requests_total`: Total HTTP requests by method, endpoint, and status
    - `http_request_duration_seconds`: Request duration histogram
    - `http_requests_in_progress`: Currently processing HTTP requests

    ### Processing Metrics
    - `document_processing_total`: Total documents processed by type and status
    - `document_processing_duration_seconds`: Processing time histogram
    - `pipeline_stage_duration_seconds`: Duration by pipeline stage

    ### System Metrics
    - `model_inference_duration_seconds`: ML model inference time
    - `pmg_operations_total`: PMG operations by type
    - `active_websocket_connections`: Current WebSocket connections

    ### Error Metrics
    - `processing_errors_total`: Errors by type and stage
    - `rate_limit_exceeded_total`: Rate limit violations

    ## Prometheus Configuration

    Add this job to your `prometheus.yml`:

    ```yaml
    scrape_configs:
      - job_name: 'sap_llm_api'
        scrape_interval: 15s
        static_configs:
          - targets: ['localhost:8000']
        metrics_path: '/metrics'
    ```

    ## Example Usage

    ```bash
    curl http://localhost:8000/metrics
    ```

    ## Grafana Dashboards

    Use these queries for Grafana dashboards:

    - **Request Rate**: `rate(http_requests_total[5m])`
    - **Error Rate**: `rate(http_requests_total{status=~"5.."}[5m])`
    - **P95 Latency**: `histogram_quantile(0.95, http_request_duration_seconds_bucket)`
    - **Processing Throughput**: `rate(document_processing_total{status="completed"}[5m])`
    """
    return observability.get_prometheus_metrics()


@app.get("/v1/slo", tags=["Monitoring"])
async def get_slo_status(
    current_user: User = Depends(require_admin),
):
    """
    Get SLO (Service Level Objective) status and error budgets.

    Returns current SLO compliance metrics and remaining error budgets for
    monitoring service reliability.

    ## SLO Targets

    ### Availability SLO
    - **Target**: 99.9% uptime (43 minutes downtime per month)
    - **Measurement**: Successful requests / Total requests

    ### Latency SLO
    - **Target**: 95th percentile < 5 seconds
    - **Measurement**: Processing time from upload to completion

    ### Accuracy SLO
    - **Target**: 95% extraction accuracy
    - **Measurement**: Fields extracted correctly / Total fields

    ## Error Budget

    The error budget represents how many errors you can have before violating SLO:

    - **Formula**: (1 - SLO Target) × Total Requests
    - **Example**: With 99.9% SLO and 1M requests, error budget is 1,000 errors
    - **Consumption**: Tracks errors against budget

    ## Response Format

    ```json
    {
      "availability": {
        "target": 0.999,
        "current": 0.9995,
        "error_budget_remaining": 0.5,
        "status": "healthy"
      },
      "latency_p95": {
        "target_ms": 5000,
        "current_ms": 3450,
        "status": "healthy"
      }
    }
    ```

    ## Status Values

    - **healthy**: Meeting SLO targets with budget remaining
    - **warning**: Approaching SLO threshold (< 20% budget remaining)
    - **critical**: SLO violated or budget exhausted

    ## Use Cases

    - **SLO Dashboards**: Monitor service reliability
    - **Incident Response**: Assess impact on SLOs
    - **Capacity Planning**: Understand reliability trends
    - **Feature Rollouts**: Validate deployments maintain SLOs
    """
    return observability.get_slo_status()


@app.get("/v1/stats", tags=["Monitoring"])
async def get_stats(
    current_user: User = Depends(require_admin),
):
    """
    Get system statistics and operational metrics.

    Returns real-time statistics about document processing, job queue status,
    and system health. Useful for dashboards and operational monitoring.

    ## Statistics Provided

    ### Job Statistics
    - **total**: Total jobs in system memory
    - **completed**: Successfully completed jobs
    - **failed**: Failed jobs with errors
    - **processing**: Currently processing jobs
    - **queued**: Jobs waiting to be processed

    ### System Health
    - **models_loaded**: Whether ML models are loaded
    - **pmg_connected**: Process Memory Graph connection status
    - **active_websockets**: Number of active WebSocket connections

    ## Example Response

    ```json
    {
      "jobs": {
        "total": 150,
        "completed": 142,
        "failed": 3,
        "processing": 2,
        "queued": 3
      },
      "system": {
        "models_loaded": true,
        "pmg_connected": true,
        "active_websockets": 5
      },
      "timestamp": "2025-11-14T10:30:00Z"
    }
    ```

    ## Use Cases

    - **Operations Dashboards**: Display system state
    - **Queue Monitoring**: Track job backlog
    - **Capacity Planning**: Understand system utilization
    - **Health Checks**: Quick system overview

    ## Refresh Rate

    - **Real-time**: Statistics reflect current server state
    - **Recommended Polling**: Every 5-30 seconds
    - **No Rate Limiting**: Unlimited queries

    ## Example Usage

    ```bash
    # Get current statistics
    curl http://localhost:8000/v1/stats

    # Watch statistics (refresh every 5 seconds)
    watch -n 5 curl -s http://localhost:8000/v1/stats | jq
    ```
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
