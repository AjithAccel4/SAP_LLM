"""
Autonomous Stage Agents for SAP_LLM Document Processing Pipeline.

Implements CloudEvents-based agents for each processing stage:
- Document ingestion
- Preprocessing (OCR)
- Classification
- Field extraction
- Quality control
- Business rules validation
- Post-processing

Each agent:
- Subscribes to specific CloudEvents
- Processes documents autonomously
- Publishes results as new CloudEvents
- Supports error handling and retry logic
- Integrates with PMG for continuous learning
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path

from sap_llm.apop.agent import BaseAgent
from sap_llm.apop.envelope import APOPEnvelope, create_envelope
from sap_llm.apop.cloudevents_bus import CloudEvent, CloudEventsBus

logger = logging.getLogger(__name__)


class PreprocessingAgent(BaseAgent):
    """
    Preprocessing agent: OCR and image enhancement.

    Subscribes to: com.sap.document.received
    Publishes: com.sap.document.preprocessed
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize preprocessing agent.

        Args:
            model_path: Optional path to OCR model
        """
        super().__init__(
            agent_name="preprocessing_agent",
            subscribes_to=["com.sap.document.received", "inbox.routed"],
            publishes=["com.sap.document.preprocessed", "preproc.ready"]
        )

        self.model_path = model_path
        self.ocr_engine = None  # Lazy load

        logger.info("Preprocessing agent initialized")

    async def process_event(self, envelope: APOPEnvelope) -> APOPEnvelope:
        """
        Process document: OCR and preprocessing.

        Args:
            envelope: Incoming envelope with document path

        Returns:
            Result envelope with OCR text and preprocessed image
        """
        document_path = envelope.data.get("document_path")
        document_id = envelope.data.get("document_id")

        if not document_path:
            raise ValueError("Missing document_path in envelope")

        logger.info(f"Preprocessing document: {document_id}")

        # Load document
        # In production: Use actual OCR engine (Tesseract, PaddleOCR, etc.)
        ocr_text = self._perform_ocr(document_path)

        # Image preprocessing
        preprocessed_image_path = self._preprocess_image(document_path)

        # Create result envelope
        result = self.create_result_envelope(
            event_type="com.sap.document.preprocessed",
            data={
                "document_id": document_id,
                "document_path": document_path,
                "preprocessed_image_path": preprocessed_image_path,
                "ocr_text": ocr_text,
                "ocr_confidence": 0.95,  # Mock confidence
                "preprocessing_timestamp": str(asyncio.get_event_loop().time()),
            },
            next_action_hint="classify.detect",
            correlation_id=envelope.correlation_id,
        )

        return result

    def _perform_ocr(self, document_path: str) -> str:
        """
        Perform OCR on document.

        Args:
            document_path: Path to document

        Returns:
            Extracted text
        """
        # Mock OCR - in production, use actual OCR engine
        logger.info(f"Performing OCR on {document_path}")

        # Simulate OCR
        return f"Mock OCR text from {Path(document_path).name}"

    def _preprocess_image(self, document_path: str) -> str:
        """
        Preprocess document image.

        Args:
            document_path: Path to document

        Returns:
            Path to preprocessed image
        """
        # Mock preprocessing - in production, apply image enhancements
        logger.info(f"Preprocessing image {document_path}")

        preprocessed_path = document_path.replace(".pdf", "_preprocessed.png")
        return preprocessed_path


class ClassificationAgent(BaseAgent):
    """
    Classification agent: Detect document type.

    Subscribes to: com.sap.document.preprocessed
    Publishes: com.sap.document.classified
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize classification agent.

        Args:
            model_path: Path to classification model
        """
        super().__init__(
            agent_name="classification_agent",
            subscribes_to=["com.sap.document.preprocessed", "preproc.ready"],
            publishes=["com.sap.document.classified", "classify.done"]
        )

        self.model_path = model_path
        self.model = None  # Lazy load

        logger.info("Classification agent initialized")

    async def process_event(self, envelope: APOPEnvelope) -> APOPEnvelope:
        """
        Classify document type.

        Args:
            envelope: Envelope with preprocessed document

        Returns:
            Result envelope with document type
        """
        document_id = envelope.data.get("document_id")
        preprocessed_image_path = envelope.data.get("preprocessed_image_path")
        ocr_text = envelope.data.get("ocr_text", "")

        logger.info(f"Classifying document: {document_id}")

        # Classify document
        # In production: Use actual model (Qwen2.5-VL, LayoutLM, etc.)
        document_type, confidence = self._classify_document(
            preprocessed_image_path,
            ocr_text
        )

        # Create result envelope
        result = self.create_result_envelope(
            event_type="com.sap.document.classified",
            data={
                "document_id": document_id,
                "document_path": envelope.data.get("document_path"),
                "preprocessed_image_path": preprocessed_image_path,
                "ocr_text": ocr_text,
                "document_type": document_type,
                "classification_confidence": confidence,
                "classification_timestamp": str(asyncio.get_event_loop().time()),
            },
            next_action_hint="extract.fields",
            correlation_id=envelope.correlation_id,
        )

        return result

    def _classify_document(self, image_path: str, ocr_text: str) -> tuple:
        """
        Classify document type.

        Args:
            image_path: Path to preprocessed image
            ocr_text: OCR text

        Returns:
            (document_type, confidence)
        """
        # Mock classification - in production, use actual model
        logger.info(f"Classifying {image_path}")

        # Simple keyword-based classification (mock)
        if "invoice" in ocr_text.lower():
            return "invoice", 0.98
        elif "purchase order" in ocr_text.lower():
            return "purchase_order", 0.95
        elif "delivery" in ocr_text.lower():
            return "delivery_note", 0.92
        else:
            return "unknown", 0.50


class ExtractionAgent(BaseAgent):
    """
    Extraction agent: Extract fields from document.

    Subscribes to: com.sap.document.classified
    Publishes: com.sap.document.extracted
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize extraction agent.

        Args:
            model_path: Path to extraction model
        """
        super().__init__(
            agent_name="extraction_agent",
            subscribes_to=["com.sap.document.classified", "classify.done"],
            publishes=["com.sap.document.extracted", "extract.done"]
        )

        self.model_path = model_path
        self.model = None  # Lazy load

        # Field schemas per document type
        self.field_schemas = {
            "invoice": [
                "invoice_number", "invoice_date", "due_date",
                "vendor_name", "vendor_address", "total_amount",
                "tax_amount", "line_items"
            ],
            "purchase_order": [
                "po_number", "po_date", "vendor_name",
                "delivery_address", "line_items", "total_amount"
            ],
            "delivery_note": [
                "delivery_number", "delivery_date", "sender",
                "recipient", "items", "tracking_number"
            ],
        }

        logger.info("Extraction agent initialized")

    async def process_event(self, envelope: APOPEnvelope) -> APOPEnvelope:
        """
        Extract fields from document.

        Args:
            envelope: Envelope with classified document

        Returns:
            Result envelope with extracted fields
        """
        document_id = envelope.data.get("document_id")
        document_type = envelope.data.get("document_type")
        preprocessed_image_path = envelope.data.get("preprocessed_image_path")
        ocr_text = envelope.data.get("ocr_text", "")

        logger.info(f"Extracting fields from {document_type}: {document_id}")

        # Extract fields
        # In production: Use actual model (Qwen2.5-VL with field extraction)
        extracted_fields = self._extract_fields(
            document_type,
            preprocessed_image_path,
            ocr_text
        )

        # Create result envelope
        result = self.create_result_envelope(
            event_type="com.sap.document.extracted",
            data={
                "document_id": document_id,
                "document_path": envelope.data.get("document_path"),
                "document_type": document_type,
                "extracted_fields": extracted_fields,
                "extraction_confidence": 0.93,  # Mock
                "extraction_timestamp": str(asyncio.get_event_loop().time()),
            },
            next_action_hint="quality.check",
            correlation_id=envelope.correlation_id,
        )

        return result

    def _extract_fields(self,
                        document_type: str,
                        image_path: str,
                        ocr_text: str) -> Dict[str, Any]:
        """
        Extract fields from document.

        Args:
            document_type: Type of document
            image_path: Path to preprocessed image
            ocr_text: OCR text

        Returns:
            Extracted fields
        """
        # Mock extraction - in production, use actual model
        logger.info(f"Extracting fields for {document_type}")

        # Get schema
        schema = self.field_schemas.get(document_type, [])

        # Mock extracted data
        extracted = {}
        if document_type == "invoice":
            extracted = {
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-01-15",
                "due_date": "2024-02-15",
                "vendor_name": "ABC Corporation",
                "vendor_address": "123 Business St, City, Country",
                "total_amount": "1250.00",
                "tax_amount": "250.00",
                "line_items": [
                    {"description": "Product A", "quantity": 10, "unit_price": 100.00},
                    {"description": "Product B", "quantity": 5, "unit_price": 50.00},
                ],
            }

        return extracted


class QualityControlAgent(BaseAgent):
    """
    Quality control agent: Validate extracted data.

    Subscribes to: com.sap.document.extracted
    Publishes: com.sap.document.quality_checked
    """

    def __init__(self, quality_threshold: float = 0.8):
        """
        Initialize quality control agent.

        Args:
            quality_threshold: Minimum quality score to pass
        """
        super().__init__(
            agent_name="quality_control_agent",
            subscribes_to=["com.sap.document.extracted", "extract.done"],
            publishes=["com.sap.document.quality_checked", "quality.verified"]
        )

        self.quality_threshold = quality_threshold

        logger.info(f"Quality control agent initialized: threshold={quality_threshold}")

    async def process_event(self, envelope: APOPEnvelope) -> APOPEnvelope:
        """
        Perform quality checks on extracted data.

        Args:
            envelope: Envelope with extracted fields

        Returns:
            Result envelope with quality assessment
        """
        document_id = envelope.data.get("document_id")
        extracted_fields = envelope.data.get("extracted_fields", {})

        logger.info(f"Quality checking document: {document_id}")

        # Perform quality checks
        quality_score, issues = self._check_quality(extracted_fields)

        # Determine status
        status = "passed" if quality_score >= self.quality_threshold else "failed"

        # Create result envelope
        result = self.create_result_envelope(
            event_type="com.sap.document.quality_checked",
            data={
                "document_id": document_id,
                "document_path": envelope.data.get("document_path"),
                "document_type": envelope.data.get("document_type"),
                "extracted_fields": extracted_fields,
                "quality_score": quality_score,
                "quality_status": status,
                "quality_issues": issues,
                "quality_timestamp": str(asyncio.get_event_loop().time()),
            },
            next_action_hint="rules.validate" if status == "passed" else "manual.review",
            correlation_id=envelope.correlation_id,
        )

        return result

    def _check_quality(self, extracted_fields: Dict[str, Any]) -> tuple:
        """
        Check quality of extracted fields.

        Args:
            extracted_fields: Extracted fields

        Returns:
            (quality_score, issues)
        """
        # Mock quality checks - in production, use actual validation
        logger.info("Performing quality checks")

        issues = []
        score = 1.0

        # Check for missing required fields
        if not extracted_fields.get("invoice_number"):
            issues.append("Missing invoice_number")
            score -= 0.3

        if not extracted_fields.get("total_amount"):
            issues.append("Missing total_amount")
            score -= 0.3

        # Check for format issues
        total_amount = extracted_fields.get("total_amount", "")
        if total_amount and not isinstance(total_amount, (int, float, str)):
            issues.append("Invalid total_amount format")
            score -= 0.2

        return max(score, 0.0), issues


class BusinessRulesAgent(BaseAgent):
    """
    Business rules validation agent.

    Subscribes to: com.sap.document.quality_checked
    Publishes: com.sap.document.rules_validated
    """

    def __init__(self, rules_config: Optional[Dict[str, Any]] = None):
        """
        Initialize business rules agent.

        Args:
            rules_config: Business rules configuration
        """
        super().__init__(
            agent_name="business_rules_agent",
            subscribes_to=["com.sap.document.quality_checked", "quality.verified"],
            publishes=["com.sap.document.rules_validated", "rules.valid"]
        )

        self.rules_config = rules_config or {}

        logger.info("Business rules agent initialized")

    async def process_event(self, envelope: APOPEnvelope) -> APOPEnvelope:
        """
        Validate document against business rules.

        Args:
            envelope: Envelope with quality-checked document

        Returns:
            Result envelope with validation results
        """
        document_id = envelope.data.get("document_id")
        document_type = envelope.data.get("document_type")
        extracted_fields = envelope.data.get("extracted_fields", {})

        logger.info(f"Validating business rules: {document_id}")

        # Validate against rules
        is_valid, violations = self._validate_rules(
            document_type,
            extracted_fields
        )

        # Create result envelope
        result = self.create_result_envelope(
            event_type="com.sap.document.rules_validated",
            data={
                "document_id": document_id,
                "document_path": envelope.data.get("document_path"),
                "document_type": document_type,
                "extracted_fields": extracted_fields,
                "rules_valid": is_valid,
                "rule_violations": violations,
                "validation_timestamp": str(asyncio.get_event_loop().time()),
            },
            next_action_hint="router.post" if is_valid else "manual.review",
            correlation_id=envelope.correlation_id,
        )

        return result

    def _validate_rules(self,
                        document_type: str,
                        extracted_fields: Dict[str, Any]) -> tuple:
        """
        Validate against business rules.

        Args:
            document_type: Type of document
            extracted_fields: Extracted fields

        Returns:
            (is_valid, violations)
        """
        # Mock validation - in production, use actual business rules engine
        logger.info(f"Validating rules for {document_type}")

        violations = []

        # Example rule: Invoice total must be > 0
        if document_type == "invoice":
            total_amount = extracted_fields.get("total_amount", "0")
            try:
                if float(total_amount) <= 0:
                    violations.append("Invoice total must be greater than 0")
            except (ValueError, TypeError):
                violations.append("Invalid total_amount")

            # Example rule: Invoice must have due date
            if not extracted_fields.get("due_date"):
                violations.append("Invoice missing due_date")

        is_valid = len(violations) == 0

        return is_valid, violations


class PostProcessingAgent(BaseAgent):
    """
    Post-processing agent: Format and route to destination.

    Subscribes to: com.sap.document.rules_validated
    Publishes: com.sap.document.completed
    """

    def __init__(self, output_config: Optional[Dict[str, Any]] = None):
        """
        Initialize post-processing agent.

        Args:
            output_config: Output configuration
        """
        super().__init__(
            agent_name="postprocessing_agent",
            subscribes_to=["com.sap.document.rules_validated", "rules.valid"],
            publishes=["com.sap.document.completed", "router.done"]
        )

        self.output_config = output_config or {}

        logger.info("Post-processing agent initialized")

    async def process_event(self, envelope: APOPEnvelope) -> APOPEnvelope:
        """
        Post-process and route document.

        Args:
            envelope: Envelope with validated document

        Returns:
            Result envelope with completion status
        """
        document_id = envelope.data.get("document_id")
        document_type = envelope.data.get("document_type")
        extracted_fields = envelope.data.get("extracted_fields", {})

        logger.info(f"Post-processing document: {document_id}")

        # Format output
        formatted_output = self._format_output(
            document_type,
            extracted_fields
        )

        # Route to destination
        destination = self._determine_destination(document_type)

        # Create result envelope
        result = self.create_result_envelope(
            event_type="com.sap.document.completed",
            data={
                "document_id": document_id,
                "document_path": envelope.data.get("document_path"),
                "document_type": document_type,
                "extracted_fields": extracted_fields,
                "formatted_output": formatted_output,
                "destination": destination,
                "completion_timestamp": str(asyncio.get_event_loop().time()),
                "status": "success",
            },
            next_action_hint="complete",
            correlation_id=envelope.correlation_id,
        )

        return result

    def _format_output(self,
                       document_type: str,
                       extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format extracted data for output.

        Args:
            document_type: Type of document
            extracted_fields: Extracted fields

        Returns:
            Formatted output
        """
        # Mock formatting - in production, format per destination requirements
        logger.info(f"Formatting output for {document_type}")

        return {
            "document_type": document_type,
            "fields": extracted_fields,
            "format": "json",
            "version": "1.0",
        }

    def _determine_destination(self, document_type: str) -> str:
        """
        Determine routing destination.

        Args:
            document_type: Type of document

        Returns:
            Destination identifier
        """
        # Mock routing - in production, use actual routing rules
        destinations = {
            "invoice": "sap.finance.invoices",
            "purchase_order": "sap.procurement.orders",
            "delivery_note": "sap.logistics.deliveries",
        }

        return destinations.get(document_type, "sap.archive")


# Example usage
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)

    # Create agents
    preprocessing_agent = PreprocessingAgent()
    classification_agent = ClassificationAgent()
    extraction_agent = ExtractionAgent()
    quality_agent = QualityControlAgent(quality_threshold=0.8)
    rules_agent = BusinessRulesAgent()
    postprocessing_agent = PostProcessingAgent()

    print("Stage agents loaded successfully")
    print(f"Agents: {[
        preprocessing_agent.agent_name,
        classification_agent.agent_name,
        extraction_agent.agent_name,
        quality_agent.agent_name,
        rules_agent.agent_name,
        postprocessing_agent.agent_name,
    ]}")
