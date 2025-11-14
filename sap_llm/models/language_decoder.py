"""
Language Decoder for SAP_LLM.

Based on LLaMA-2-7B for text understanding and structured JSON generation.
Handles ADC (Adaptive Document Contract) generation with constrained decoding.
"""

import json
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)

from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


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
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.max_length = max_length

        logger.info(f"Initializing LanguageDecoder: {model_name}")
        logger.info(f"Device: {device}, Precision: {precision}")

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
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            constrained_vocab: List of allowed token IDs

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

        # Generate
        outputs = self.model.generate(
            **inputs,
            generation_config=gen_config,
            # TODO: Add constrained decoding logic
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
        Extract structured fields from document.

        Args:
            ocr_text: OCR extracted text
            doc_type: Document type
            schema: ADC schema
            visual_features: Optional visual features from vision encoder

        Returns:
            Extracted fields as dictionary
        """
        # Create prompt
        prompt = self.create_extraction_prompt(ocr_text, doc_type, schema)

        # Generate JSON
        generated_json = self.generate(
            prompt,
            max_new_tokens=1024,
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
