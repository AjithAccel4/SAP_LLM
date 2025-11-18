"""
Training Data Preparation for Reasoning Engine.

Creates 200K+ routing examples from SAP transaction logs with:
- Chain-of-thought reasoning traces
- PMG context (similar document routings)
- Success/failure feedback for RLHF
- Field transformations and payload generation examples
"""

import json
import random
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RoutingExample:
    """Single routing training example."""
    doc_id: str
    doc_type: str
    adc_json: Dict[str, Any]
    api_schemas: List[Dict[str, Any]]
    similar_cases: List[Dict[str, Any]]
    target_endpoint: str
    target_payload: Dict[str, Any]
    reasoning_trace: str
    confidence: float
    success: bool  # Whether the routing was successful
    feedback: Optional[str] = None


class SAPRoutingDatasetBuilder:
    """
    Build training dataset from SAP transaction logs.

    Creates chain-of-thought examples with:
    - Input: ADC JSON + PMG context + API schemas
    - Output: Routing decision + reasoning + payload
    """

    def __init__(
        self,
        transaction_log_path: str,
        api_schemas_path: str,
        pmg_data_path: str,
        output_dir: str,
    ):
        """
        Initialize dataset builder.

        Args:
            transaction_log_path: Path to SAP transaction logs (JSON)
            api_schemas_path: Path to SAP API schemas
            pmg_data_path: Path to PMG historical data
            output_dir: Output directory for training data
        """
        self.transaction_log_path = Path(transaction_log_path)
        self.api_schemas_path = Path(api_schemas_path)
        self.pmg_data_path = Path(pmg_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load schemas and PMG data
        self.api_schemas = self._load_api_schemas()
        self.pmg_data = self._load_pmg_data()

        logger.info(f"Loaded {len(self.api_schemas)} API schemas")
        logger.info(f"Loaded {len(self.pmg_data)} PMG routing records")

    def _load_api_schemas(self) -> Dict[str, Any]:
        """Load SAP API schemas."""
        if not self.api_schemas_path.exists():
            logger.warning(f"API schemas not found at {self.api_schemas_path}, using mock data")
            return self._generate_mock_api_schemas()

        with open(self.api_schemas_path) as f:
            return json.load(f)

    def _load_pmg_data(self) -> List[Dict[str, Any]]:
        """Load PMG historical routing data."""
        if not self.pmg_data_path.exists():
            logger.warning(f"PMG data not found at {self.pmg_data_path}, using mock data")
            return self._generate_mock_pmg_data()

        with open(self.pmg_data_path) as f:
            return json.load(f)

    def _generate_mock_api_schemas(self) -> Dict[str, Any]:
        """Generate mock API schemas for testing."""
        return {
            "API_PURCHASEORDER_PROCESS_SRV": {
                "name": "API_PURCHASEORDER_PROCESS_SRV",
                "entity": "A_PurchaseOrder",
                "method": "POST",
                "fields": {
                    "PurchaseOrder": "string",
                    "CompanyCode": "string",
                    "PurchaseOrderType": "string",
                    "Supplier": "string",
                    "DocumentCurrency": "string",
                },
            },
            "API_SUPPLIERINVOICE_PROCESS_SRV": {
                "name": "API_SUPPLIERINVOICE_PROCESS_SRV",
                "entity": "A_SupplierInvoice",
                "method": "POST",
                "fields": {
                    "SupplierInvoice": "string",
                    "FiscalYear": "string",
                    "CompanyCode": "string",
                    "DocumentDate": "date",
                    "InvoicingParty": "string",
                },
            },
            "API_SALES_ORDER_SRV": {
                "name": "API_SALES_ORDER_SRV",
                "entity": "A_SalesOrder",
                "method": "POST",
                "fields": {
                    "SalesOrderType": "string",
                    "SalesOrganization": "string",
                    "DistributionChannel": "string",
                    "OrganizationDivision": "string",
                    "SoldToParty": "string",
                },
            },
        }

    def _generate_mock_pmg_data(self) -> List[Dict[str, Any]]:
        """Generate mock PMG data for testing."""
        return [
            {
                "doc_id": f"DOC_{i:06d}",
                "doc_type": random.choice(["PURCHASE_ORDER", "SUPPLIER_INVOICE", "SALES_ORDER"]),
                "endpoint": random.choice(list(self.api_schemas.keys())),
                "success": random.random() > 0.05,  # 95% success rate
                "timestamp": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            }
            for i in range(10000)
        ]

    def build_dataset(
        self,
        num_examples: int = 200000,
        train_split: float = 0.8,
        val_split: float = 0.1,
    ) -> Tuple[List[RoutingExample], List[RoutingExample], List[RoutingExample]]:
        """
        Build complete training dataset.

        Args:
            num_examples: Number of examples to generate
            train_split: Training set ratio
            val_split: Validation set ratio

        Returns:
            (train_examples, val_examples, test_examples)
        """
        logger.info(f"Building dataset with {num_examples} examples...")

        examples = []

        # Load transaction logs
        if self.transaction_log_path.exists():
            examples = self._load_from_transaction_logs(num_examples)
        else:
            logger.warning("Transaction logs not found, generating synthetic examples")
            examples = self._generate_synthetic_examples(num_examples)

        # Shuffle
        random.shuffle(examples)

        # Split
        n_train = int(num_examples * train_split)
        n_val = int(num_examples * val_split)

        train_examples = examples[:n_train]
        val_examples = examples[n_train:n_train + n_val]
        test_examples = examples[n_train + n_val:]

        logger.info(f"Dataset split: {len(train_examples)} train, {len(val_examples)} val, {len(test_examples)} test")

        # Save to disk
        self._save_dataset(train_examples, "train")
        self._save_dataset(val_examples, "val")
        self._save_dataset(test_examples, "test")

        return train_examples, val_examples, test_examples

    def _load_from_transaction_logs(self, num_examples: int) -> List[RoutingExample]:
        """Load and process real transaction logs."""
        examples = []

        with open(self.transaction_log_path) as f:
            for i, line in enumerate(f):
                if i >= num_examples:
                    break

                try:
                    log_entry = json.loads(line)
                    example = self._create_example_from_log(log_entry)
                    if example:
                        examples.append(example)
                except Exception as e:
                    logger.warning(f"Failed to process log entry {i}: {e}")
                    continue

        return examples

    def _create_example_from_log(self, log_entry: Dict[str, Any]) -> Optional[RoutingExample]:
        """Create training example from transaction log entry."""
        # Extract information
        doc_id = log_entry.get("document_id")
        doc_type = log_entry.get("document_type")
        adc_json = log_entry.get("extracted_data", {})
        endpoint = log_entry.get("routed_endpoint")
        payload = log_entry.get("sap_payload", {})
        success = log_entry.get("success", False)

        if not all([doc_id, doc_type, adc_json, endpoint]):
            return None

        # Get similar cases from PMG
        similar_cases = self._get_similar_cases(doc_type, adc_json)

        # Generate reasoning trace
        reasoning_trace = self._generate_reasoning_trace(
            doc_type, adc_json, endpoint, similar_cases, success
        )

        # Compute confidence
        confidence = self._compute_confidence(similar_cases, success)

        return RoutingExample(
            doc_id=doc_id,
            doc_type=doc_type,
            adc_json=adc_json,
            api_schemas=list(self.api_schemas.values()),
            similar_cases=similar_cases,
            target_endpoint=endpoint,
            target_payload=payload,
            reasoning_trace=reasoning_trace,
            confidence=confidence,
            success=success,
            feedback=log_entry.get("feedback"),
        )

    def _generate_synthetic_examples(self, num_examples: int) -> List[RoutingExample]:
        """Generate synthetic training examples."""
        logger.info(f"Generating {num_examples} synthetic examples...")

        examples = []
        doc_types = ["PURCHASE_ORDER", "SUPPLIER_INVOICE", "SALES_ORDER", "CUSTOMER_INVOICE", "GOODS_RECEIPT"]

        for i in range(num_examples):
            doc_type = random.choice(doc_types)

            # Generate synthetic ADC
            adc_json = self._generate_synthetic_adc(doc_type, i)

            # Select appropriate endpoint
            endpoint = self._select_endpoint_for_doc_type(doc_type)

            # Get similar cases
            similar_cases = self._get_similar_cases(doc_type, adc_json)

            # Generate payload
            payload = self._generate_synthetic_payload(endpoint, adc_json)

            # Generate reasoning
            reasoning_trace = self._generate_reasoning_trace(
                doc_type, adc_json, endpoint, similar_cases, success=True
            )

            # Success rate: 95%
            success = random.random() > 0.05
            confidence = random.uniform(0.85, 0.99) if success else random.uniform(0.50, 0.85)

            examples.append(
                RoutingExample(
                    doc_id=f"SYNTH_{i:08d}",
                    doc_type=doc_type,
                    adc_json=adc_json,
                    api_schemas=list(self.api_schemas.values()),
                    similar_cases=similar_cases[:3],  # Top 3 similar
                    target_endpoint=endpoint,
                    target_payload=payload,
                    reasoning_trace=reasoning_trace,
                    confidence=confidence,
                    success=success,
                )
            )

            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1}/{num_examples} examples")

        return examples

    def _generate_synthetic_adc(self, doc_type: str, index: int) -> Dict[str, Any]:
        """Generate synthetic ADC JSON."""
        base_adc = {
            "doc_id": f"DOC_{index:08d}",
            "doc_type": doc_type,
            "supplier_name": f"Supplier_{random.randint(1, 100):03d}",
            "supplier_id": f"SUP{random.randint(1000, 9999)}",
            "company_code": random.choice(["1000", "2000", "3000", "4000"]),
            "currency": random.choice(["USD", "EUR", "GBP"]),
            "total_amount": round(random.uniform(100, 100000), 2),
            "document_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        }

        # Add type-specific fields
        if doc_type == "PURCHASE_ORDER":
            base_adc.update({
                "po_number": f"PO{random.randint(100000, 999999)}",
                "items": [
                    {
                        "material": f"MAT{random.randint(1000, 9999)}",
                        "quantity": random.randint(1, 100),
                        "unit_price": round(random.uniform(10, 1000), 2),
                    }
                    for _ in range(random.randint(1, 5))
                ],
            })
        elif doc_type == "SUPPLIER_INVOICE":
            base_adc.update({
                "invoice_number": f"INV{random.randint(100000, 999999)}",
                "invoice_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "payment_terms": random.choice(["NET30", "NET60", "COD"]),
            })

        return base_adc

    def _select_endpoint_for_doc_type(self, doc_type: str) -> str:
        """Select appropriate SAP endpoint for document type."""
        mapping = {
            "PURCHASE_ORDER": "API_PURCHASEORDER_PROCESS_SRV",
            "SUPPLIER_INVOICE": "API_SUPPLIERINVOICE_PROCESS_SRV",
            "SALES_ORDER": "API_SALES_ORDER_SRV",
            "CUSTOMER_INVOICE": "API_BILLING_DOCUMENT_SRV",
            "GOODS_RECEIPT": "API_MATERIAL_DOCUMENT_SRV",
        }
        return mapping.get(doc_type, "API_PURCHASEORDER_PROCESS_SRV")

    def _generate_synthetic_payload(self, endpoint: str, adc_json: Dict[str, Any]) -> Dict[str, Any]:
        """Generate synthetic SAP payload."""
        # Simplified payload generation
        return {
            "d": {
                "CompanyCode": adc_json.get("company_code"),
                "DocumentDate": adc_json.get("document_date"),
                "Supplier": adc_json.get("supplier_id"),
                "Currency": adc_json.get("currency"),
                "TotalAmount": adc_json.get("total_amount"),
            }
        }

    def _get_similar_cases(self, doc_type: str, adc_json: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get similar routing cases from PMG."""
        # Filter PMG data by doc_type
        similar = [
            case for case in self.pmg_data
            if case.get("doc_type") == doc_type
        ]

        # Sort by success and recency (simplified)
        similar.sort(key=lambda x: (x.get("success", False), x.get("timestamp", "")), reverse=True)

        return similar[:5]

    def _generate_reasoning_trace(
        self,
        doc_type: str,
        adc_json: Dict[str, Any],
        endpoint: str,
        similar_cases: List[Dict[str, Any]],
        success: bool,
    ) -> str:
        """Generate chain-of-thought reasoning trace."""
        # Calculate success rate from similar cases
        if similar_cases:
            success_rate = sum(1 for c in similar_cases if c.get("success", False)) / len(similar_cases)
        else:
            success_rate = 0.5

        reasoning = f"""**Step 1: Document Analysis**
Document Type: {doc_type}
Supplier: {adc_json.get('supplier_name', 'Unknown')} ({adc_json.get('supplier_id', 'N/A')})
Company Code: {adc_json.get('company_code', 'Unknown')}
Total Amount: {adc_json.get('total_amount', 0)} {adc_json.get('currency', 'USD')}

**Step 2: Historical Context**
Similar past routings: {len(similar_cases)} cases found
Success rate for similar cases: {success_rate * 100:.1f}%
Most common endpoint: {endpoint}

**Step 3: API Selection**
Based on document type '{doc_type}' and historical data, the appropriate endpoint is: {endpoint}
This endpoint handles {doc_type.replace('_', ' ').lower()} transactions for company code {adc_json.get('company_code')}.

**Step 4: Field Validation**
All required fields are present:
- Company Code: {adc_json.get('company_code')} ✓
- Supplier: {adc_json.get('supplier_id')} ✓
- Currency: {adc_json.get('currency')} ✓
- Amount: {adc_json.get('total_amount')} ✓

**Step 5: Confidence Assessment**
Confidence Score: {success_rate * 100:.1f}%
Reasoning: {'High confidence based on successful similar routings' if success else 'Medium confidence, some similar cases failed'}

**Decision: Route to {endpoint}**
"""
        return reasoning

    def _compute_confidence(self, similar_cases: List[Dict[str, Any]], success: bool) -> float:
        """Compute confidence score based on similar cases."""
        if not similar_cases:
            return 0.5

        success_rate = sum(1 for c in similar_cases if c.get("success", False)) / len(similar_cases)

        # Adjust based on actual success
        if success:
            return min(0.99, success_rate + random.uniform(0, 0.1))
        else:
            return max(0.4, success_rate - random.uniform(0.1, 0.3))

    def _save_dataset(self, examples: List[RoutingExample], split: str) -> None:
        """Save dataset split to disk."""
        output_file = self.output_dir / f"{split}_routing_examples.jsonl"

        with open(output_file, "w") as f:
            for example in examples:
                f.write(json.dumps({
                    "doc_id": example.doc_id,
                    "doc_type": example.doc_type,
                    "adc_json": example.adc_json,
                    "api_schemas": example.api_schemas,
                    "similar_cases": example.similar_cases,
                    "target_endpoint": example.target_endpoint,
                    "target_payload": example.target_payload,
                    "reasoning_trace": example.reasoning_trace,
                    "confidence": example.confidence,
                    "success": example.success,
                    "feedback": example.feedback,
                }) + "\n")

        logger.info(f"Saved {len(examples)} examples to {output_file}")

    def create_preference_pairs(
        self,
        examples: List[RoutingExample],
    ) -> List[Dict[str, Any]]:
        """
        Create preference pairs for RLHF reward model training.

        Format: {chosen: successful_routing, rejected: failed_routing}
        """
        logger.info("Creating preference pairs for RLHF...")

        # Separate successful and failed examples
        successful = [e for e in examples if e.success]
        failed = [e for e in examples if not e.success]

        logger.info(f"Found {len(successful)} successful and {len(failed)} failed examples")

        preference_pairs = []

        # Create pairs: match failed with similar successful
        for failed_ex in failed:
            # Find similar successful example
            similar_successful = [
                s for s in successful
                if s.doc_type == failed_ex.doc_type
                and s.target_endpoint == failed_ex.target_endpoint
            ]

            if similar_successful:
                chosen = random.choice(similar_successful)

                preference_pairs.append({
                    "chosen": {
                        "doc_id": chosen.doc_id,
                        "adc_json": chosen.adc_json,
                        "endpoint": chosen.target_endpoint,
                        "payload": chosen.target_payload,
                        "reasoning": chosen.reasoning_trace,
                        "confidence": chosen.confidence,
                    },
                    "rejected": {
                        "doc_id": failed_ex.doc_id,
                        "adc_json": failed_ex.adc_json,
                        "endpoint": failed_ex.target_endpoint,
                        "payload": failed_ex.target_payload,
                        "reasoning": failed_ex.reasoning_trace,
                        "confidence": failed_ex.confidence,
                    },
                    "margin": 1.0,
                })

        logger.info(f"Created {len(preference_pairs)} preference pairs")

        # Save preference pairs
        output_file = self.output_dir / "preference_pairs.jsonl"
        with open(output_file, "w") as f:
            for pair in preference_pairs:
                f.write(json.dumps(pair) + "\n")

        logger.info(f"Saved preference pairs to {output_file}")

        return preference_pairs


if __name__ == "__main__":
    # Example usage
    builder = SAPRoutingDatasetBuilder(
        transaction_log_path="data/sap_transactions.jsonl",
        api_schemas_path="data/sap_api_schemas.json",
        pmg_data_path="data/pmg_routing_history.json",
        output_dir="data/training/reasoning_engine",
    )

    # Build dataset
    train, val, test = builder.build_dataset(num_examples=200000)

    # Create preference pairs for RLHF
    preference_pairs = builder.create_preference_pairs(train)

    print(f"✓ Dataset created: {len(train)} train, {len(val)} val, {len(test)} test")
    print(f"✓ Preference pairs: {len(preference_pairs)}")
