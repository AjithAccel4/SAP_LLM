#!/bin/bash
#
# Quick Start Script for Vision Encoder Training
#
# This script sets up the environment and runs a quick training test
# with synthetic data to verify everything is working correctly.
#

set -e  # Exit on error

echo "=================================================="
echo "Vision Encoder Quick Start"
echo "=================================================="
echo ""

# Check if running from project root
if [ ! -f "setup.py" ] && [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the SAP_LLM project root"
    exit 1
fi

# Create directories
echo "Creating directories..."
mkdir -p data/vision_encoder_test/{train,val,test}
mkdir -p models/vision_encoder_test
mkdir -p logs

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if virtual environment is recommended
if [ -z "$VIRTUAL_ENV" ]; then
    echo ""
    echo "Warning: No virtual environment detected."
    echo "It's recommended to use a virtual environment:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q torch torchvision transformers pillow scikit-learn tqdm

# Generate synthetic data
echo ""
echo "Generating synthetic test data..."
python3 << 'EOF'
from sap_llm.training.vision_dataset import create_synthetic_dataset

print("Creating training data (100 samples)...")
create_synthetic_dataset(
    output_dir="./data/vision_encoder_test/train",
    num_samples=100,
    split="train"
)

print("Creating validation data (20 samples)...")
create_synthetic_dataset(
    output_dir="./data/vision_encoder_test/val",
    num_samples=20,
    split="val"
)

print("Creating test data (20 samples)...")
create_synthetic_dataset(
    output_dir="./data/vision_encoder_test/test",
    num_samples=20,
    split="test"
)

print("Synthetic data created successfully!")
EOF

# Check for GPU
echo ""
echo "Checking for GPU availability..."
python3 << 'EOF'
import torch
if torch.cuda.is_available():
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
else:
    print("✗ No GPU available. Training will use CPU (much slower).")
EOF

# Run quick training test
echo ""
echo "Running quick training test (1000 steps)..."
echo "This will take approximately 5-10 minutes depending on your hardware."
echo ""

python3 scripts/train_vision_encoder.py \
    --data_dir ./data/vision_encoder_test/train \
    --val_data_dir ./data/vision_encoder_test/val \
    --output_dir ./models/vision_encoder_test \
    --batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --max_steps 1000 \
    --logging_steps 50 \
    --eval_steps 500 \
    --save_steps 500 \
    --num_workers 2

# Run evaluation
echo ""
echo "Running evaluation on test set..."
python3 scripts/evaluate_vision_encoder.py \
    --model_path ./models/vision_encoder_test/best \
    --data_dir ./data/vision_encoder_test/test \
    --output_dir ./evaluation_results \
    --batch_size 4 \
    --benchmark

# Display results
echo ""
echo "=================================================="
echo "Quick Start Complete!"
echo "=================================================="
echo ""
echo "Results:"
echo "  - Model checkpoint: ./models/vision_encoder_test/best"
echo "  - Evaluation results: ./evaluation_results"
echo ""
echo "To view evaluation metrics:"
echo "  cat ./evaluation_results/evaluation_results.json"
echo ""
echo "To train on your own data:"
echo "  1. Prepare your data in the required format (see docs/VISION_ENCODER_TRAINING.md)"
echo "  2. Run: python scripts/train_vision_encoder.py --data_dir <your_data_dir> ..."
echo ""
echo "For detailed training instructions, see: docs/VISION_ENCODER_TRAINING.md"
echo ""
