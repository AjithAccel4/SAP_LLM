"""
Reasoning Engine for SAP_LLM.

Based on Mixtral-8x7B for autonomous decision-making and routing.
Handles business logic, exception handling, and SAP endpoint selection.
"""

import json
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningEngine(nn.Module):
    """
    Reasoning engine based on Mixtral-8x7B.

    This engine makes autonomous decisions for routing, exception handling,
    and business rule application using chain-of-thought reasoning.

    Architecture:
    - Base: Mixtral-8x7B (47B total, 6B active)
    - Mixture of Experts (8 experts, 2 active per token)
    - Input: Extracted ADC + PMG context
    - Output: APOP envelope with next_action_hint

    Args:
        model_name: HuggingFace model name or path
        device: Device to run model on (cuda/cpu)
        precision: Model precision (fp32/fp16/int8)
        max_length: Maximum generation length
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mixtral-8x7B-v0.1",
        device: str = "cuda",
        precision: str = "int8",
        max_length: int = 4096,
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.max_length = max_length

        logger.info(f"Initializing ReasoningEngine: {model_name}")
        logger.info(f"Device: {device}, Precision: {precision}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with quantization
        if precision == "int8":
            logger.info("Loading model with 8-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif precision == "int4":
            logger.info("Loading model with 4-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if precision == "fp16" else torch.float32,
            )
            self.model.to(device)

        # Generation config for reasoning
        self.generation_config = GenerationConfig(
            max_new_tokens=500,
            do_sample=True,
            temperature=0.1,  # Low temp for consistent reasoning
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        self.model.eval()

        logger.info(f"ReasoningEngine initialized: {self._count_parameters():,} parameters")

    def _count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def create_routing_prompt(
        self,
        adc_json: Dict[str, Any],
        doc_type: str,
        api_schemas: List[Dict[str, Any]],
        similar_cases: List[Dict[str, Any]],
    ) -> str:
        """
        Create chain-of-thought prompt for routing decision.

        Args:
            adc_json: Extracted document data
            doc_type: Document type
            api_schemas: Available SAP API schemas
            similar_cases: Similar routing decisions from PMG

        Returns:
            Formatted prompt
        """
        prompt = f"""You are an SAP routing expert. Given a document, determine the correct SAP API endpoint and generate the payload.

**Document Information:**
Type: {doc_type}
Supplier: {adc_json.get('supplier_name', 'Unknown')}
Company Code: {adc_json.get('company_code', 'Unknown')}
Total Amount: {adc_json.get('total_amount', 0)} {adc_json.get('currency', 'USD')}

**Extracted Data (ADC):**
{json.dumps(adc_json, indent=2)[:1500]}  # Truncate if too long

**Available SAP APIs:**
{json.dumps([api['name'] for api in api_schemas], indent=2)}

**Similar Past Routings:**
{json.dumps(similar_cases[:3], indent=2)[:1000] if similar_cases else 'No similar cases found'}

**Task:**
1. Analyze the document type and extracted data
2. Consider similar past routing decisions
3. Select the appropriate SAP API endpoint
4. Explain your reasoning
5. Provide a confidence score

Output JSON format:
{{
  "endpoint": "API_NAME",
  "method": "POST",
  "entity": "EntityName",
  "confidence": 0.95,
  "reasoning": "Explanation of why this endpoint is appropriate"
}}

**Decision:**
"""
        return prompt

    def create_exception_handling_prompt(
        self,
        exception: Dict[str, Any],
        similar_exceptions: List[Dict[str, Any]],
    ) -> str:
        """
        Create prompt for exception handling decision.

        Args:
            exception: Exception details
            similar_exceptions: Similar exceptions from PMG

        Returns:
            Formatted prompt
        """
        prompt = f"""You are an exception handling expert for document processing.

**Exception:**
Category: {exception.get('category')}
Severity: {exception.get('severity')}
Field: {exception.get('field')}
Expected: {exception.get('expected')}
Actual: {exception.get('value')}
Message: {exception.get('message')}

**Similar Past Exceptions:**
{json.dumps(similar_exceptions[:5], indent=2)[:1000]}

**Task:**
Analyze this exception and recommend an action:
1. AUTO_CORRECT - Can be fixed automatically
2. ESCALATE - Requires human review
3. REJECT - Document should be rejected
4. APPLY_RULE - Apply a business rule

Provide reasoning and suggested correction if AUTO_CORRECT.

Output JSON:
{{
  "action": "AUTO_CORRECT|ESCALATE|REJECT|APPLY_RULE",
  "reasoning": "Explanation",
  "correction": {{"field": "new_value"}},
  "confidence": 0.90
}}

**Decision:**
"""
        return prompt

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate reasoning output from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens

        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config,
        )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # Remove prompt
        generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def decide_routing(
        self,
        adc_json: Dict[str, Any],
        doc_type: str,
        api_schemas: List[Dict[str, Any]],
        similar_cases: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make routing decision using chain-of-thought reasoning.

        Args:
            adc_json: Extracted document data
            doc_type: Document type
            api_schemas: Available API schemas
            similar_cases: Similar routing decisions

        Returns:
            Routing decision with reasoning
        """
        if similar_cases is None:
            similar_cases = []

        # Create prompt
        prompt = self.create_routing_prompt(
            adc_json,
            doc_type,
            api_schemas,
            similar_cases,
        )

        # Generate decision
        response = self.generate(prompt)

        # Parse JSON response
        decision = self._parse_json_response(response)

        # Validate decision
        if not decision or "endpoint" not in decision:
            logger.warning("Invalid routing decision, using fallback")
            decision = self._fallback_routing(doc_type, api_schemas)

        return decision

    def handle_exception(
        self,
        exception: Dict[str, Any],
        similar_exceptions: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make exception handling decision.

        Args:
            exception: Exception details
            similar_exceptions: Similar past exceptions

        Returns:
            Exception handling decision
        """
        if similar_exceptions is None:
            similar_exceptions = []

        # Create prompt
        prompt = self.create_exception_handling_prompt(
            exception,
            similar_exceptions,
        )

        # Generate decision
        response = self.generate(prompt)

        # Parse response
        decision = self._parse_json_response(response)

        if not decision or "action" not in decision:
            logger.warning("Invalid exception handling decision, escalating")
            decision = {
                "action": "ESCALATE",
                "reasoning": "Failed to generate valid decision",
                "confidence": 0.0,
            }

        return decision

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Parse JSON from model response."""
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', text, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Text: {text}")

        return {}

    def _fallback_routing(
        self,
        doc_type: str,
        api_schemas: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fallback routing logic based on document type."""
        # Simple rule-based fallback
        type_to_api = {
            "PURCHASE_ORDER": "API_PURCHASEORDER_PROCESS_SRV",
            "SUPPLIER_INVOICE": "API_SUPPLIERINVOICE_PROCESS_SRV",
            "SALES_ORDER": "API_SALES_ORDER_SRV",
            "CUSTOMER_INVOICE": "API_BILLING_DOCUMENT_SRV",
            "GOODS_RECEIPT": "API_MATERIAL_DOCUMENT_SRV",
        }

        endpoint = type_to_api.get(doc_type, "UNKNOWN")

        return {
            "endpoint": endpoint,
            "method": "POST",
            "confidence": 0.5,  # Low confidence for fallback
            "reasoning": "Fallback routing based on document type",
        }

    def save(self, output_path: str) -> None:
        """Save model and tokenizer."""
        logger.info(f"Saving ReasoningEngine to {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = "cuda",
        precision: str = "int8",
    ) -> "ReasoningEngine":
        """Load model from path."""
        logger.info(f"Loading ReasoningEngine from {model_path}")
        return cls(
            model_name=model_path,
            device=device,
            precision=precision,
        )
