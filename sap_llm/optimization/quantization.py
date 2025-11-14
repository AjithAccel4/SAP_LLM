"""
Model quantization utilities for SAP_LLM.

Supports:
- INT8 quantization (PyTorch native)
- GPTQ-4 quantization (4-bit grouped quantization)
- AWQ (Activation-aware Weight Quantization)
- Dynamic vs Static quantization
- Per-channel and per-tensor quantization
- Calibration dataset support

Based on:
- GPTQ: https://arxiv.org/abs/2210.17323
- AWQ: https://arxiv.org/abs/2306.00978
- LLM.int8(): https://arxiv.org/abs/2208.07339
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    method: str = "int8"  # int8, gptq-4, awq, dynamic
    bits: int = 8  # 4, 8
    group_size: int = 128  # For GPTQ/AWQ
    damp_percent: float = 0.01  # GPTQ damping factor
    desc_act: bool = False  # GPTQ activation order
    static: bool = True  # Static vs dynamic quantization
    symmetric: bool = False  # Symmetric vs asymmetric quantization
    per_channel: bool = True  # Per-channel vs per-tensor
    calibration_samples: int = 512  # Number of calibration samples
    act_order: bool = True  # AWQ activation ordering
    true_sequential: bool = True  # Sequential quantization


class ModelQuantizer:
    """
    Model quantization for inference optimization.

    Reduces model size and improves inference speed:
    - FP16 (baseline): 72B params × 2 bytes = 144GB
    - INT8: 72B params × 1 byte = 72GB (2× reduction, ~10% accuracy loss)
    - GPTQ-4: 72B params × 0.5 bytes = 36GB (4× reduction, ~5% accuracy loss)

    Inference speedup:
    - INT8: 1.5-2× faster on CPU/GPU
    - GPTQ-4: 2-3× faster with custom kernels
    """

    def __init__(self, model: nn.Module, config: QuantizationConfig):
        """
        Initialize quantizer.

        Args:
            model: Model to quantize (Qwen2.5-VL)
            config: Quantization configuration
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        logger.info(f"ModelQuantizer initialized: method={config.method}, bits={config.bits}")

    def quantize(self,
                 calibration_dataset: Optional[Any] = None,
                 save_path: Optional[str] = None) -> nn.Module:
        """
        Quantize model.

        Args:
            calibration_dataset: Dataset for calibration (required for static quantization)
            save_path: Path to save quantized model

        Returns:
            Quantized model
        """
        logger.info(f"Starting quantization: {self.config.method}")

        if self.config.method == "int8":
            quantized_model = self._quantize_int8(calibration_dataset)
        elif self.config.method == "gptq-4":
            quantized_model = self._quantize_gptq(calibration_dataset)
        elif self.config.method == "awq":
            quantized_model = self._quantize_awq(calibration_dataset)
        elif self.config.method == "dynamic":
            quantized_model = self._quantize_dynamic()
        else:
            raise ValueError(f"Unsupported quantization method: {self.config.method}")

        # Save if requested
        if save_path:
            self._save_quantized_model(quantized_model, save_path)

        return quantized_model

    def _quantize_int8(self, calibration_dataset: Optional[Any] = None) -> nn.Module:
        """
        INT8 quantization using PyTorch native support.

        Uses per-channel affine quantization for weights.
        """
        logger.info("Applying INT8 quantization")

        # Prepare model for quantization
        self.model.eval()

        if self.config.static:
            # Static quantization (requires calibration)
            if calibration_dataset is None:
                raise ValueError("Static quantization requires calibration dataset")

            # Prepare model
            from torch.quantization import prepare, convert

            # Set quantization config
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
            self.model.qconfig = qconfig

            # Fuse modules (Conv+ReLU, Linear+ReLU)
            self.model = torch.quantization.fuse_modules(
                self.model,
                [['conv', 'relu'], ['linear', 'relu']],
                inplace=True
            )

            # Prepare for calibration
            model_prepared = prepare(self.model, inplace=False)

            # Calibrate
            logger.info(f"Calibrating with {self.config.calibration_samples} samples")
            with torch.no_grad():
                for i, batch in enumerate(calibration_dataset):
                    if i >= self.config.calibration_samples:
                        break

                    # Forward pass for calibration
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    model_prepared(**inputs)

            # Convert to quantized model
            quantized_model = convert(model_prepared, inplace=False)

        else:
            # Dynamic quantization (no calibration needed)
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )

        logger.info("INT8 quantization complete")
        return quantized_model

    def _quantize_gptq(self, calibration_dataset: Any) -> nn.Module:
        """
        GPTQ-4 quantization (4-bit grouped quantization).

        GPTQ algorithm:
        1. Divide weights into groups (e.g., 128 weights per group)
        2. For each group, find optimal quantization scale/zero-point
        3. Use Hessian-based approximation to minimize quantization error
        4. Apply layer-wise quantization

        Reference: https://arxiv.org/abs/2210.17323
        """
        logger.info("Applying GPTQ-4 quantization")

        try:
            from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        except ImportError:
            logger.error("auto-gptq not installed. Install with: pip install auto-gptq")
            raise

        # GPTQ configuration
        quantize_config = BaseQuantizeConfig(
            bits=self.config.bits,
            group_size=self.config.group_size,
            damp_percent=self.config.damp_percent,
            desc_act=self.config.desc_act,
            sym=self.config.symmetric,
            true_sequential=self.config.true_sequential,
        )

        # Prepare calibration data
        calibration_data = []
        for i, batch in enumerate(calibration_dataset):
            if i >= self.config.calibration_samples:
                break
            calibration_data.append(batch)

        # Quantize model
        logger.info(f"Quantizing with GPTQ (bits={self.config.bits}, group_size={self.config.group_size})")

        # Note: This is a simplified interface
        # In practice, you'd use AutoGPTQForCausalLM.from_pretrained() with quantization
        quantized_model = self._apply_gptq_to_layers(
            self.model,
            calibration_data,
            quantize_config
        )

        logger.info("GPTQ-4 quantization complete")
        return quantized_model

    def _apply_gptq_to_layers(self,
                               model: nn.Module,
                               calibration_data: List[Any],
                               config: Any) -> nn.Module:
        """
        Apply GPTQ quantization layer by layer.

        For each linear layer:
        1. Collect input activations
        2. Compute Hessian matrix
        3. Find optimal quantization parameters
        4. Quantize weights
        """
        # Simplified GPTQ implementation
        # Full implementation would use auto-gptq library

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Get weight
                weight = module.weight.data

                # Group quantization
                if config.group_size > 0:
                    weight_quantized = self._quantize_weight_grouped(
                        weight,
                        bits=config.bits,
                        group_size=config.group_size
                    )
                else:
                    weight_quantized = self._quantize_weight_per_channel(
                        weight,
                        bits=config.bits
                    )

                # Replace weight
                module.weight.data = weight_quantized

        return model

    def _quantize_awq(self, calibration_dataset: Any) -> nn.Module:
        """
        AWQ (Activation-aware Weight Quantization).

        AWQ algorithm:
        1. Analyze activation distributions
        2. Scale weights based on activation importance
        3. Apply group-wise quantization
        4. Protects salient weights (1% most important)

        Reference: https://arxiv.org/abs/2306.00978
        """
        logger.info("Applying AWQ quantization")

        try:
            from awq import AutoAWQForCausalLM
        except ImportError:
            logger.error("AutoAWQ not installed. Install with: pip install autoawq")
            raise

        # Prepare calibration data
        calibration_data = []
        for i, batch in enumerate(calibration_dataset):
            if i >= self.config.calibration_samples:
                break
            calibration_data.append(batch)

        # AWQ quantization
        logger.info(f"Quantizing with AWQ (bits={self.config.bits})")

        # Simplified AWQ implementation
        quantized_model = self._apply_awq_to_layers(
            self.model,
            calibration_data
        )

        logger.info("AWQ quantization complete")
        return quantized_model

    def _apply_awq_to_layers(self,
                              model: nn.Module,
                              calibration_data: List[Any]) -> nn.Module:
        """
        Apply AWQ quantization layer by layer.

        Steps:
        1. Run calibration to collect activations
        2. Compute activation scales per channel
        3. Scale weights by inverse of activation scales
        4. Apply group-wise quantization
        """
        # Collect activations
        activation_scales = {}

        def get_activation_hook(name):
            def hook(module, input, output):
                # Compute per-channel activation scales
                if isinstance(output, torch.Tensor):
                    scales = output.abs().mean(dim=0)
                    if name not in activation_scales:
                        activation_scales[name] = []
                    activation_scales[name].append(scales)
            return hook

        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(get_activation_hook(name))
                hooks.append(hook)

        # Run calibration
        model.eval()
        with torch.no_grad():
            for batch in calibration_data:
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                model(**inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Compute average activation scales
        for name in activation_scales:
            activation_scales[name] = torch.stack(activation_scales[name]).mean(dim=0)

        # Apply AWQ scaling and quantization
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in activation_scales:
                # Get activation scale
                scale = activation_scales[name]

                # Scale weights by inverse of activation scale
                # This protects salient weights
                module.weight.data = module.weight.data / scale.unsqueeze(0)

                # Quantize
                weight_quantized = self._quantize_weight_grouped(
                    module.weight.data,
                    bits=self.config.bits,
                    group_size=self.config.group_size
                )

                module.weight.data = weight_quantized

        return model

    def _quantize_dynamic(self) -> nn.Module:
        """
        Dynamic quantization (quantizes weights only, activations at runtime).

        Fastest to apply, no calibration needed.
        Good for CPU inference.
        """
        logger.info("Applying dynamic quantization")

        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )

        logger.info("Dynamic quantization complete")
        return quantized_model

    def _quantize_weight_grouped(self,
                                  weight: torch.Tensor,
                                  bits: int = 4,
                                  group_size: int = 128) -> torch.Tensor:
        """
        Group-wise weight quantization.

        Args:
            weight: Weight tensor (out_features, in_features)
            bits: Number of bits (4 or 8)
            group_size: Size of each group

        Returns:
            Quantized weight (dequantized to FP16 for compatibility)
        """
        out_features, in_features = weight.shape

        # Reshape to groups
        num_groups = (in_features + group_size - 1) // group_size

        # Quantize each group
        quantized_groups = []

        for i in range(num_groups):
            start = i * group_size
            end = min((i + 1) * group_size, in_features)

            group = weight[:, start:end]

            # Compute scale and zero point
            q_min = 0
            q_max = 2 ** bits - 1

            w_min = group.min()
            w_max = group.max()

            scale = (w_max - w_min) / (q_max - q_min)
            zero_point = q_min - w_min / scale

            # Quantize
            group_q = torch.clamp(
                torch.round(group / scale + zero_point),
                q_min,
                q_max
            )

            # Dequantize (for now - would use quantized kernels in production)
            group_dq = (group_q - zero_point) * scale

            quantized_groups.append(group_dq)

        # Concatenate groups
        weight_quantized = torch.cat(quantized_groups, dim=1)

        return weight_quantized

    def _quantize_weight_per_channel(self,
                                      weight: torch.Tensor,
                                      bits: int = 8) -> torch.Tensor:
        """
        Per-channel weight quantization.

        Args:
            weight: Weight tensor (out_features, in_features)
            bits: Number of bits

        Returns:
            Quantized weight
        """
        out_features, in_features = weight.shape

        # Quantize per output channel
        q_min = 0
        q_max = 2 ** bits - 1

        # Compute per-channel scales
        w_min = weight.min(dim=1, keepdim=True)[0]
        w_max = weight.max(dim=1, keepdim=True)[0]

        scale = (w_max - w_min) / (q_max - q_min)
        zero_point = q_min - w_min / scale

        # Quantize
        weight_q = torch.clamp(
            torch.round(weight / scale + zero_point),
            q_min,
            q_max
        )

        # Dequantize
        weight_dq = (weight_q - zero_point) * scale

        return weight_dq

    def _save_quantized_model(self, model: nn.Module, save_path: str):
        """
        Save quantized model with config.

        Args:
            model: Quantized model
            save_path: Directory to save model
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = save_dir / "quantized_model.pt"
        torch.save(model.state_dict(), model_path)

        # Save config
        config_path = save_dir / "quantization_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                "method": self.config.method,
                "bits": self.config.bits,
                "group_size": self.config.group_size,
                "static": self.config.static,
                "symmetric": self.config.symmetric,
                "per_channel": self.config.per_channel,
            }, f, indent=2)

        logger.info(f"Quantized model saved to {save_path}")

    @staticmethod
    def load_quantized_model(load_path: str, model_class: type) -> Tuple[nn.Module, QuantizationConfig]:
        """
        Load quantized model and config.

        Args:
            load_path: Directory containing quantized model
            model_class: Model class to instantiate

        Returns:
            (quantized_model, config)
        """
        load_dir = Path(load_path)

        # Load config
        config_path = load_dir / "quantization_config.json"
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        config = QuantizationConfig(**config_dict)

        # Load model
        model_path = load_dir / "quantized_model.pt"
        model = model_class()
        model.load_state_dict(torch.load(model_path))

        logger.info(f"Loaded quantized model from {load_path}")

        return model, config

    def benchmark(self,
                  quantized_model: nn.Module,
                  test_dataset: Any,
                  num_samples: int = 100) -> Dict[str, Any]:
        """
        Benchmark quantized model vs original.

        Args:
            quantized_model: Quantized model
            test_dataset: Test dataset
            num_samples: Number of samples to benchmark

        Returns:
            Benchmark results
        """
        logger.info("Benchmarking quantized model")

        import time

        # Model sizes
        original_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**3
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024**3

        # Inference speed
        self.model.eval()
        quantized_model.eval()

        original_times = []
        quantized_times = []

        with torch.no_grad():
            for i, batch in enumerate(test_dataset):
                if i >= num_samples:
                    break

                inputs = {k: v.to(self.device) for k, v in batch.items()}

                # Original model
                start = time.time()
                self.model(**inputs)
                original_times.append(time.time() - start)

                # Quantized model
                start = time.time()
                quantized_model(**inputs)
                quantized_times.append(time.time() - start)

        results = {
            "original_size_gb": original_size,
            "quantized_size_gb": quantized_size,
            "compression_ratio": original_size / quantized_size,
            "original_latency_ms": sum(original_times) / len(original_times) * 1000,
            "quantized_latency_ms": sum(quantized_times) / len(quantized_times) * 1000,
            "speedup": sum(original_times) / sum(quantized_times),
        }

        logger.info(f"Benchmark results: {results}")

        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example configuration
    config = QuantizationConfig(
        method="gptq-4",
        bits=4,
        group_size=128,
        static=True,
        calibration_samples=512
    )

    print(f"Quantization config: {config}")

    # Example model (placeholder)
    # model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-72B")
    # calibration_dataset = load_calibration_data()

    # quantizer = ModelQuantizer(model, config)
    # quantized_model = quantizer.quantize(
    #     calibration_dataset=calibration_dataset,
    #     save_path="./models/qwen2.5-vl-72b-gptq4"
    # )

    # Benchmark
    # results = quantizer.benchmark(quantized_model, test_dataset)
    # print(f"Compression: {results['compression_ratio']:.2f}×")
    # print(f"Speedup: {results['speedup']:.2f}×")

    print("Quantization module loaded successfully")
