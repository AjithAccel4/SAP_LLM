"""
Language Decoder for SAP_LLM.

Based on LLaMA-2-7B for text understanding and structured JSON generation.
Handles ADC (Adaptive Document Contract) generation with constrained decoding.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class JSONSchemaConstraintProcessor(LogitsProcessor):
    """
    Constrained decoding processor for JSON schema compliance.

    Implements constrained decoding by:
    1. Converting JSON schema to allowed token sets
    2. Masking invalid tokens during generation
    3. Ensuring structural validity (braces, quotes, commas)

    Based on 2025 best practices from vLLM and transformers-cfg.
    """

    def __init__(
        self,
        schema: Dict[str, Any],
        tokenizer: AutoTokenizer,
        required_fields: Optional[List[str]] = None,
    ):
        self.schema = schema
        self.tokenizer = tokenizer
        self.required_fields = required_fields or []

        # Build allowed token sets for JSON structure
        self._build_token_sets()

    def _build_token_sets(self) -> None:
        """Build sets of allowed tokens for different JSON contexts."""
        vocab = self.tokenizer.get_vocab()

        # Structural tokens
        self.json_structural = {
            vocab.get(t, -1) for t in ["{", "}", "[", "]", ":", ",", '"']
        } - {-1}

        # Boolean tokens
        self.json_bool = {
            vocab.get(t, -1) for t in ["true", "false", "True", "False"]
        } - {-1}

        # Null tokens
        self.json_null = {
            vocab.get(t, -1) for t in ["null", "None"]
        } - {-1}

        # Number tokens (0-9, ., -, +, e, E)
        self.json_number = {
            vocab.get(str(i), -1) for i in range(10)
        } | {
            vocab.get(t, -1) for t in [".", "-", "+", "e", "E"]
        } - {-1}

        # Whitespace tokens
        self.json_whitespace = {
            vocab.get(t, -1) for t in [" ", "\n", "\t", "\r"]
        } - {-1}

        # Field name tokens (from schema)
        self.field_tokens = set()
        if "properties" in self.schema:
            for field_name in self.schema["properties"].keys():
                # Tokenize field name and add token IDs
                field_tokens = self.tokenizer.encode(f'"{field_name}"', add_special_tokens=False)
                self.field_tokens.update(field_tokens)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Process logits to enforce JSON schema constraints.

        Args:
            input_ids: Generated token IDs so far [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]

        Returns:
            Modified scores with invalid tokens masked
        """
        batch_size = scores.shape[0]

        for batch_idx in range(batch_size):
            # Get current generation context
            current_text = self.tokenizer.decode(
                input_ids[batch_idx],
                skip_special_tokens=False,
            )

            # Determine allowed tokens based on context
            allowed_tokens = self._get_allowed_tokens(current_text)

            # Mask invalid tokens (set to -inf)
            mask = torch.ones_like(scores[batch_idx], dtype=torch.bool)
            mask[list(allowed_tokens)] = False
            scores[batch_idx] = scores[batch_idx].masked_fill(mask, float("-inf"))

        return scores

    def _get_allowed_tokens(self, current_text: str) -> Set[int]:
        """
        Determine allowed tokens based on current generation state.

        Args:
            current_text: Currently generated text

        Returns:
            Set of allowed token IDs
        """
        # Count braces to determine context
        open_braces = current_text.count("{") - current_text.count("}")
        open_brackets = current_text.count("[") - current_text.count("]")

        # Check if we're inside quotes
        in_string = current_text.count('"') % 2 == 1

        # Default: allow structural tokens
        allowed = self.json_structural | self.json_whitespace

        if open_braces == 0 and open_brackets == 0:
            # Start of JSON: only allow opening brace
            allowed = {self.tokenizer.convert_tokens_to_ids("{")}

        elif in_string:
            # Inside string: allow alphanumeric and field names
            allowed = self.field_tokens | {
                self.tokenizer.convert_tokens_to_ids('"')  # Allow closing quote
            }

        elif current_text.rstrip().endswith(":"):
            # After colon: allow values (string, number, bool, null, object, array)
            allowed = (
                {self.tokenizer.convert_tokens_to_ids('"')} |  # String start
                self.json_number |  # Number
                self.json_bool |  # Boolean
                self.json_null |  # Null
                {self.tokenizer.convert_tokens_to_ids("{")} |  # Object start
                {self.tokenizer.convert_tokens_to_ids("[")} |  # Array start
                self.json_whitespace
            )

        elif current_text.rstrip().endswith(","):
            # After comma: allow field names
            allowed = self.field_tokens | {
                self.tokenizer.convert_tokens_to_ids('"')
            } | self.json_whitespace

        else:
            # General case: allow most JSON tokens
            allowed = (
                self.json_structural |
                self.json_whitespace |
                self.json_bool |
                self.json_null |
                self.json_number |
                self.field_tokens
            )

        # Always allow EOS token for completion
        allowed.add(self.tokenizer.eos_token_id)

        return allowed


class LanguageDecoder(nn.Module):
    """
    Language decoder based on LLaMA-2-7B.

    This decoder generates structured JSON outputs (ADC format) from
    visual-text features extracted by the vision encoder.

    Architecture:
    - Base: LLaMA-2-7B (7B parameters)
    - Input: Text embeddings + visual features
    - Output: JSON-formatted ADC

    Args:
        model_name: HuggingFace model name or path
        device: Device to run model on (cuda/cpu)
        precision: Model precision (fp32/fp16/int8)
        max_length: Maximum generation length
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        device: str = "cuda",
        precision: str = "int8",
        max_length: int = 2048,
        enable_constrained_decoding: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.max_length = max_length
        self.enable_constrained_decoding = enable_constrained_decoding
        self.current_schema = None  # Will be set during generation

        logger.info(f"Initializing LanguageDecoder: {model_name}")
        logger.info(f"Device: {device}, Precision: {precision}")
        logger.info(f"Constrained Decoding: {'Enabled' if enable_constrained_decoding else 'Disabled'}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="left",  # For generation
        )

        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        if precision == "int8":
            # 8-bit quantization
            logger.info("Loading model with 8-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif precision == "int4":
            # 4-bit quantization
            logger.info("Loading model with 4-bit quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            # Full precision or FP16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if precision == "fp16" else torch.float32,
            )
            self.model.to(device)

        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=max_length,
            do_sample=False,  # Deterministic for structured output
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Set to eval mode
        self.model.eval()

        logger.info(f"LanguageDecoder initialized: {self._count_parameters():,} parameters")

    def _count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.model.parameters())

    def create_extraction_prompt(
        self,
        ocr_text: str,
        doc_type: str,
        schema: Dict[str, Any],
    ) -> str:
        """
        Create prompt for field extraction.

        Args:
            ocr_text: OCR extracted text
            doc_type: Document type
            schema: ADC schema for this document type

        Returns:
            Formatted prompt string
        """
        # Format schema for prompt
        fields_desc = "\n".join([
            f"  - {field}: {info.get('description', info.get('type'))}"
            for field, info in schema.get("properties", {}).items()
        ])

        prompt = f"""Extract structured data from this {doc_type} document and output as JSON.

OCR Text:
{ocr_text[:2000]}  # Truncate if too long

Required JSON Schema:
{{
{fields_desc}
}}

Output valid JSON only, no explanation:
"""
        return prompt

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through language decoder.

        Args:
            input_ids: Token IDs [B, seq_len]
            attention_mask: Attention mask [B, seq_len]

        Returns:
            Dictionary with logits and hidden states
        """
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )

        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        constrained_vocab: Optional[List[int]] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate text from prompt with optional schema constraints.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            constrained_vocab: List of allowed token IDs (deprecated, use schema)
            schema: JSON schema for constrained decoding

        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Update generation config
        gen_config = self.generation_config
        if max_new_tokens is not None:
            gen_config.max_new_tokens = max_new_tokens

        # Constrained Decoding Logic (2025 best practices)
        logits_processor = None
        if self.enable_constrained_decoding and schema is not None:
            logger.info("Enabling JSON schema constrained decoding")

            # Create constraint processor
            constraint_processor = JSONSchemaConstraintProcessor(
                schema=schema,
                tokenizer=self.tokenizer,
                required_fields=schema.get("required", []),
            )

            # Add to logits processor list
            logits_processor = LogitsProcessorList([constraint_processor])

        # Generate with constraints
        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config,
            logits_processor=logits_processor,
        )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # Remove prompt from output
        generated_text = generated_text[len(prompt):].strip()

        return generated_text

    def extract_fields(
        self,
        ocr_text: str,
        doc_type: str,
        schema: Dict[str, Any],
        visual_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Extract structured fields from document with schema-constrained generation.

        Args:
            ocr_text: OCR extracted text
            doc_type: Document type
            schema: ADC schema for constrained decoding
            visual_features: Optional visual features from vision encoder

        Returns:
            Extracted fields as dictionary
        """
        # Store schema for constrained decoding
        self.current_schema = schema

        # Create prompt
        prompt = self.create_extraction_prompt(ocr_text, doc_type, schema)

        # Generate JSON with schema constraints
        generated_json = self.generate(
            prompt,
            max_new_tokens=1024,
            schema=schema,  # Enable constrained decoding
        )

        # Parse JSON
        try:
            # Extract JSON from text (may have markdown formatting)
            json_text = self._extract_json_from_text(generated_json)
            extracted_data = json.loads(json_text)

            # Validate against schema
            self._validate_schema(extracted_data, schema)

            return extracted_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Generated text: {generated_json}")

            # Attempt self-correction
            corrected_json = self._self_correct_json(generated_json, schema)
            return corrected_json

    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from potentially markdown-formatted text."""
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1]
            text = text.split("```")[0]
        elif "```" in text:
            text = text.split("```")[1]
            text = text.split("```")[0]

        # Find JSON object
        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1

        if start_idx != -1 and end_idx != 0:
            return text[start_idx:end_idx]

        return text

    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> None:
        """Validate extracted data against schema."""
        from jsonschema import validate, ValidationError

        try:
            validate(instance=data, schema=schema)
        except ValidationError as e:
            logger.warning(f"Schema validation failed: {e}")
            raise

    def _self_correct_json(self, text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to correct malformed JSON."""
        # Simple corrections
        corrections = [
            (r"'", '"'),  # Single to double quotes
            (r",\s*}", "}"),  # Trailing commas
            (r",\s*]", "]"),  # Trailing commas in arrays
        ]

        import re
        corrected = text
        for pattern, replacement in corrections:
            corrected = re.sub(pattern, replacement, corrected)

        try:
            return json.loads(corrected)
        except json.JSONDecodeError:
            logger.error("Self-correction failed, returning empty dict")
            return {}

    def save(self, output_path: str) -> None:
        """Save model and tokenizer."""
        logger.info(f"Saving LanguageDecoder to {output_path}")
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = "cuda",
        precision: str = "int8",
    ) -> "LanguageDecoder":
        """Load model from path."""
        logger.info(f"Loading LanguageDecoder from {model_path}")
        return cls(
            model_name=model_path,
            device=device,
            precision=precision,
        )
