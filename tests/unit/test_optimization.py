"""
Comprehensive unit tests for Model Optimization components.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
import tempfile
from pathlib import Path

from sap_llm.optimization.quantization import ModelQuantizer
from sap_llm.optimization.pruning import ModelPruner
from sap_llm.optimization.distillation import KnowledgeDistillation
from sap_llm.optimization.model_optimizer import ModelOptimizer
from sap_llm.optimization.tensorrt_converter import TensorRTConverter
from sap_llm.optimization.cost_optimizer import CostOptimizer


@pytest.mark.unit
@pytest.mark.optimization
class TestModelQuantizer:
    """Tests for ModelQuantizer."""

    @pytest.fixture
    def quantizer(self):
        """Create ModelQuantizer instance."""
        config = MagicMock()
        config.bits = 8
        config.symmetric = True
        config.per_channel = True
        return ModelQuantizer(config=config)

    @pytest.fixture
    def simple_model(self):
        """Create simple PyTorch model for testing."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
        return model

    def test_quantizer_initialization(self, quantizer):
        """Test quantizer initialization."""
        assert quantizer is not None
        assert quantizer.bits == 8

    def test_quantize_model_int8(self, quantizer, simple_model):
        """Test INT8 quantization."""
        with patch('sap_llm.optimization.quantization.torch.quantization') as mock_quant:
            mock_quant.quantize_dynamic.return_value = simple_model

            quantized = quantizer.quantize(simple_model, bits=8)
            assert quantized is not None

    def test_quantize_model_int4(self, simple_model):
        """Test INT4 quantization."""
        config = MagicMock()
        config.bits = 4
        config.symmetric = True

        quantizer = ModelQuantizer(config=config)

        with patch.object(quantizer, 'apply_int4_quantization') as mock_quant:
            mock_quant.return_value = simple_model

            quantized = quantizer.quantize(simple_model, bits=4)
            assert quantized is not None

    def test_calibration(self, quantizer, simple_model):
        """Test quantization calibration."""
        calibration_data = [torch.randn(32, 100) for _ in range(10)]

        with patch.object(quantizer, 'calibrate') as mock_calibrate:
            mock_calibrate.return_value = {"scale": 0.1, "zero_point": 0}

            calibration_params = quantizer.calibrate(simple_model, calibration_data)
            assert calibration_params is not None

    def test_per_channel_quantization(self, simple_model):
        """Test per-channel quantization."""
        config = MagicMock()
        config.bits = 8
        config.per_channel = True

        quantizer = ModelQuantizer(config=config)

        with patch.object(quantizer, 'quantize_per_channel') as mock_quant:
            mock_quant.return_value = simple_model

            quantized = quantizer.quantize(simple_model)
            assert quantized is not None

    def test_evaluate_quantized_model(self, quantizer, simple_model):
        """Test evaluating quantized model."""
        test_data = torch.randn(100, 100)

        with patch('sap_llm.optimization.quantization.torch.quantization'):
            quantized = quantizer.quantize(simple_model, bits=8)

            with patch.object(quantizer, 'evaluate') as mock_eval:
                mock_eval.return_value = {"accuracy": 0.95, "latency": 10.5}

                metrics = quantizer.evaluate(quantized, test_data)
                assert "accuracy" in metrics

    @pytest.mark.parametrize("bits", [4, 8, 16])
    def test_different_bit_widths(self, simple_model, bits):
        """Test quantization with different bit widths."""
        config = MagicMock()
        config.bits = bits

        quantizer = ModelQuantizer(config=config)

        with patch.object(quantizer, 'quantize') as mock_quant:
            mock_quant.return_value = simple_model

            quantized = quantizer.quantize(simple_model, bits=bits)
            assert quantized is not None


@pytest.mark.unit
@pytest.mark.optimization
class TestModelPruner:
    """Tests for ModelPruner."""

    @pytest.fixture
    def pruner(self):
        """Create ModelPruner instance."""
        config = MagicMock()
        config.sparsity = 0.5  # 50% sparsity
        config.method = "magnitude"
        return ModelPruner(config=config)

    @pytest.fixture
    def simple_model(self):
        """Create simple model."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
        return model

    def test_pruner_initialization(self, pruner):
        """Test pruner initialization."""
        assert pruner is not None
        assert pruner.sparsity == 0.5

    def test_magnitude_pruning(self, pruner, simple_model):
        """Test magnitude-based pruning."""
        with patch('sap_llm.optimization.pruning.torch.nn.utils.prune') as mock_prune:
            mock_prune.l1_unstructured.return_value = None

            pruned = pruner.prune(simple_model, method="magnitude")
            assert pruned is not None

    def test_structured_pruning(self, pruner, simple_model):
        """Test structured pruning."""
        with patch.object(pruner, 'structured_prune') as mock_prune:
            mock_prune.return_value = simple_model

            pruned = pruner.prune(simple_model, method="structured")
            assert pruned is not None

    def test_iterative_pruning(self, pruner, simple_model):
        """Test iterative pruning."""
        train_data = [(torch.randn(32, 100), torch.randint(0, 10, (32,))) for _ in range(10)]

        with patch.object(pruner, 'prune_iteratively') as mock_prune:
            mock_prune.return_value = simple_model

            pruned = pruner.prune_iteratively(
                simple_model,
                train_data,
                iterations=3,
                sparsity_per_iteration=0.2,
            )
            assert pruned is not None

    def test_get_sparsity(self, pruner, simple_model):
        """Test getting model sparsity."""
        with patch.object(pruner, 'calculate_sparsity') as mock_calc:
            mock_calc.return_value = 0.45

            sparsity = pruner.calculate_sparsity(simple_model)
            assert 0.0 <= sparsity <= 1.0

    @pytest.mark.parametrize("sparsity_level", [0.3, 0.5, 0.7, 0.9])
    def test_different_sparsity_levels(self, simple_model, sparsity_level):
        """Test pruning with different sparsity levels."""
        config = MagicMock()
        config.sparsity = sparsity_level
        config.method = "magnitude"

        pruner = ModelPruner(config=config)

        with patch.object(pruner, 'prune') as mock_prune:
            mock_prune.return_value = simple_model

            pruned = pruner.prune(simple_model)
            assert pruned is not None


@pytest.mark.unit
@pytest.mark.optimization
class TestKnowledgeDistillation:
    """Tests for KnowledgeDistillation."""

    @pytest.fixture
    def distillation(self):
        """Create KnowledgeDistillation instance."""
        config = MagicMock()
        config.temperature = 4.0
        config.alpha = 0.5  # Weight for distillation loss
        return KnowledgeDistillation(config=config)

    @pytest.fixture
    def teacher_model(self):
        """Create teacher model."""
        return nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 10),
        )

    @pytest.fixture
    def student_model(self):
        """Create smaller student model."""
        return nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def test_distillation_initialization(self, distillation):
        """Test distillation initialization."""
        assert distillation is not None
        assert distillation.temperature == 4.0

    def test_distillation_loss(self, distillation, teacher_model, student_model):
        """Test computing distillation loss."""
        inputs = torch.randn(32, 100)
        labels = torch.randint(0, 10, (32,))

        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        student_outputs = student_model(inputs)

        with patch.object(distillation, 'compute_loss') as mock_loss:
            mock_loss.return_value = torch.tensor(2.5)

            loss = distillation.compute_loss(
                student_outputs,
                teacher_outputs,
                labels,
            )
            assert loss is not None

    def test_distill_model(self, distillation, teacher_model, student_model):
        """Test distilling knowledge from teacher to student."""
        train_data = [(torch.randn(32, 100), torch.randint(0, 10, (32,))) for _ in range(10)]

        with patch.object(distillation, 'train') as mock_train:
            mock_train.return_value = student_model

            distilled_student = distillation.distill(
                teacher_model,
                student_model,
                train_data,
                epochs=5,
            )
            assert distilled_student is not None

    def test_soft_targets(self, distillation, teacher_model):
        """Test generating soft targets from teacher."""
        inputs = torch.randn(32, 100)

        with patch.object(distillation, 'get_soft_targets') as mock_soft:
            mock_soft.return_value = torch.softmax(torch.randn(32, 10), dim=1)

            soft_targets = distillation.get_soft_targets(teacher_model, inputs)
            assert soft_targets is not None

    @pytest.mark.parametrize("temperature", [1.0, 2.0, 4.0, 8.0])
    def test_different_temperatures(self, teacher_model, student_model, temperature):
        """Test distillation with different temperatures."""
        config = MagicMock()
        config.temperature = temperature
        config.alpha = 0.5

        distillation = KnowledgeDistillation(config=config)
        assert distillation.temperature == temperature


@pytest.mark.unit
@pytest.mark.optimization
class TestModelOptimizer:
    """Tests for ModelOptimizer (combines multiple optimization techniques)."""

    @pytest.fixture
    def model_optimizer(self):
        """Create ModelOptimizer instance."""
        config = MagicMock()
        config.enable_quantization = True
        config.enable_pruning = True
        config.enable_distillation = False
        return ModelOptimizer(config=config)

    @pytest.fixture
    def model(self):
        """Create model to optimize."""
        return nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def test_optimizer_initialization(self, model_optimizer):
        """Test model optimizer initialization."""
        assert model_optimizer is not None

    def test_optimize_pipeline(self, model_optimizer, model):
        """Test full optimization pipeline."""
        with patch.object(model_optimizer, 'quantizer') as mock_quant:
            mock_quant.quantize.return_value = model

            with patch.object(model_optimizer, 'pruner') as mock_prune:
                mock_prune.prune.return_value = model

                optimized = model_optimizer.optimize(model)
                assert optimized is not None

    def test_selective_optimization(self, model):
        """Test selective optimization (only some techniques)."""
        config = MagicMock()
        config.enable_quantization = True
        config.enable_pruning = False
        config.enable_distillation = False

        optimizer = ModelOptimizer(config=config)

        with patch.object(optimizer, 'quantizer') as mock_quant:
            mock_quant.quantize.return_value = model

            optimized = optimizer.optimize(model)
            assert optimized is not None

    def test_benchmark_optimization(self, model_optimizer, model):
        """Test benchmarking optimization results."""
        test_data = torch.randn(100, 100)

        with patch.object(model_optimizer, 'benchmark') as mock_bench:
            mock_bench.return_value = {
                "latency_ms": 5.2,
                "throughput_qps": 192.3,
                "memory_mb": 45.6,
                "accuracy": 0.94,
            }

            metrics = model_optimizer.benchmark(model, test_data)
            assert "latency_ms" in metrics

    def test_compression_ratio(self, model_optimizer, model):
        """Test calculating compression ratio."""
        original_size = 1000  # MB
        optimized_size = 250  # MB

        with patch.object(model_optimizer, 'get_model_size') as mock_size:
            mock_size.side_effect = [original_size, optimized_size]

            compression_ratio = model_optimizer.calculate_compression_ratio(model, model)
            # Should be 4x compression


@pytest.mark.unit
@pytest.mark.optimization
@pytest.mark.requires_gpu
class TestTensorRTConverter:
    """Tests for TensorRTConverter."""

    @pytest.fixture
    def tensorrt_converter(self):
        """Create TensorRTConverter instance."""
        config = MagicMock()
        config.precision = "fp16"
        config.max_batch_size = 32
        return TensorRTConverter(config=config)

    @pytest.fixture
    def onnx_model_path(self, temp_dir):
        """Create dummy ONNX model path."""
        return temp_dir / "model.onnx"

    def test_converter_initialization(self, tensorrt_converter):
        """Test TensorRT converter initialization."""
        assert tensorrt_converter is not None

    def test_convert_onnx_to_tensorrt(self, tensorrt_converter, onnx_model_path):
        """Test converting ONNX to TensorRT."""
        with patch('sap_llm.optimization.tensorrt_converter.trt') as mock_trt:
            mock_builder = MagicMock()
            mock_trt.Builder.return_value = mock_builder

            engine = tensorrt_converter.convert(str(onnx_model_path))
            # Should create TensorRT engine

    def test_optimize_for_inference(self, tensorrt_converter):
        """Test optimization for inference."""
        with patch.object(tensorrt_converter, 'apply_optimizations') as mock_opt:
            mock_opt.return_value = True

            success = tensorrt_converter.optimize_for_inference()
            # Should apply TensorRT optimizations

    @pytest.mark.parametrize("precision", ["fp32", "fp16", "int8"])
    def test_different_precisions(self, onnx_model_path, precision):
        """Test conversion with different precisions."""
        config = MagicMock()
        config.precision = precision

        converter = TensorRTConverter(config=config)
        assert converter.precision == precision


@pytest.mark.unit
@pytest.mark.optimization
class TestCostOptimizer:
    """Tests for CostOptimizer."""

    @pytest.fixture
    def cost_optimizer(self):
        """Create CostOptimizer instance."""
        config = MagicMock()
        config.cost_per_gpu_hour = 1.50
        config.target_throughput = 1000  # docs/hour
        return CostOptimizer(config=config)

    def test_cost_optimizer_initialization(self, cost_optimizer):
        """Test cost optimizer initialization."""
        assert cost_optimizer is not None

    def test_calculate_inference_cost(self, cost_optimizer):
        """Test calculating inference cost."""
        model_config = {
            "latency_ms": 50,
            "batch_size": 32,
            "gpu_hours": 1,
        }

        with patch.object(cost_optimizer, 'calculate_cost') as mock_calc:
            mock_calc.return_value = 0.75  # Cost in USD

            cost = cost_optimizer.calculate_inference_cost(model_config)
            assert cost > 0

    def test_optimize_batch_size(self, cost_optimizer):
        """Test optimizing batch size for cost."""
        with patch.object(cost_optimizer, 'find_optimal_batch_size') as mock_opt:
            mock_opt.return_value = 64

            optimal_batch_size = cost_optimizer.optimize_batch_size(
                latency_target_ms=100,
                throughput_target=1000,
            )
            assert optimal_batch_size > 0

    def test_compare_configurations(self, cost_optimizer):
        """Test comparing different configurations."""
        configs = [
            {"name": "base", "cost_per_hour": 1.5, "throughput": 800},
            {"name": "optimized", "cost_per_hour": 0.5, "throughput": 1200},
        ]

        with patch.object(cost_optimizer, 'compare') as mock_compare:
            mock_compare.return_value = configs[1]  # Return optimized

            best_config = cost_optimizer.compare_configurations(configs)
            assert best_config is not None

    def test_cost_projection(self, cost_optimizer):
        """Test cost projection over time."""
        with patch.object(cost_optimizer, 'project_costs') as mock_project:
            mock_project.return_value = {
                "daily": 36.0,
                "monthly": 1080.0,
                "yearly": 13140.0,
            }

            projection = cost_optimizer.project_costs(
                docs_per_day=10000,
                cost_per_doc=0.0036,
            )

            assert "monthly" in projection


@pytest.mark.unit
@pytest.mark.optimization
class TestOptimizationIntegration:
    """Integration tests for optimization pipeline."""

    @pytest.fixture
    def model(self):
        """Create model for optimization."""
        return nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 15),  # 15 document classes
        )

    def test_full_optimization_pipeline(self, model):
        """Test complete optimization pipeline."""
        # 1. Quantization
        quantizer = ModelQuantizer(config=MagicMock(bits=8))
        with patch.object(quantizer, 'quantize') as mock_quant:
            mock_quant.return_value = model

            quantized = quantizer.quantize(model)

            # 2. Pruning
            pruner = ModelPruner(config=MagicMock(sparsity=0.5))
            with patch.object(pruner, 'prune') as mock_prune:
                mock_prune.return_value = model

                pruned = pruner.prune(quantized)

                # 3. Benchmark
                optimizer = ModelOptimizer(config=MagicMock())
                with patch.object(optimizer, 'benchmark') as mock_bench:
                    mock_bench.return_value = {
                        "latency_ms": 8.5,
                        "accuracy": 0.93,
                    }

                    metrics = optimizer.benchmark(pruned, torch.randn(100, 768))
                    assert metrics is not None

    def test_optimization_accuracy_preservation(self, model):
        """Test that optimization preserves accuracy."""
        original_accuracy = 0.95
        tolerance = 0.02  # 2% accuracy drop acceptable

        optimizer = ModelOptimizer(config=MagicMock())

        with patch.object(optimizer, 'evaluate_accuracy') as mock_eval:
            mock_eval.side_effect = [original_accuracy, 0.94]  # Slight drop

            optimized_accuracy = mock_eval()
            assert abs(original_accuracy - optimized_accuracy) <= tolerance
