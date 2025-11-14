"""
Pipeline stages for SAP_LLM.

This module contains implementations for all 8 processing stages:
1. Inbox - Document ingestion & routing
2. Preprocessing - OCR, image enhancement, text extraction
3. Classification - Document type identification
4. Type Identifier - 35+ invoice/PO subtypes
5. Extraction - Field-level data extraction (180+ fields)
6. Quality Check - Confidence scoring & validation
7. Validation - Business rules & tolerance checks
8. Routing - SAP API endpoint selection & payload generation
"""

from sap_llm.stages.inbox import InboxStage
from sap_llm.stages.preprocessing import PreprocessingStage
from sap_llm.stages.classification import ClassificationStage
from sap_llm.stages.type_identifier import TypeIdentifierStage
from sap_llm.stages.extraction import ExtractionStage
from sap_llm.stages.quality_check import QualityCheckStage
from sap_llm.stages.validation import ValidationStage
from sap_llm.stages.routing import RoutingStage

__all__ = [
    "InboxStage",
    "PreprocessingStage",
    "ClassificationStage",
    "TypeIdentifierStage",
    "ExtractionStage",
    "QualityCheckStage",
    "ValidationStage",
    "RoutingStage",
]
