"""
Test fixtures for SAP_LLM.
"""

from tests.fixtures.sample_documents import (
    create_sample_purchase_order,
    create_sample_supplier_invoice,
    create_sample_sales_order,
    create_sample_document_image,
    create_sample_ocr_output,
    create_sample_exception_cluster,
    create_sample_api_schemas,
    create_sample_business_rules,
)

from tests.fixtures.mock_data import (
    generate_mock_adc,
    generate_mock_exception,
    generate_mock_pmg_transaction,
    generate_batch_documents,
)

__all__ = [
    # Sample documents
    "create_sample_purchase_order",
    "create_sample_supplier_invoice",
    "create_sample_sales_order",
    "create_sample_document_image",
    "create_sample_ocr_output",
    "create_sample_exception_cluster",
    "create_sample_api_schemas",
    "create_sample_business_rules",
    # Mock data generators
    "generate_mock_adc",
    "generate_mock_exception",
    "generate_mock_pmg_transaction",
    "generate_batch_documents",
]
