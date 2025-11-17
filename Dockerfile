# Multi-stage Dockerfile for SAP_LLM
# Optimized for production deployment with GPU support

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Stage 2: Dependencies
FROM base AS dependencies

WORKDIR /tmp

# Copy requirements
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM base AS application

# Create app user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app /data /models && \
    chown -R appuser:appuser /app /data /models

# Copy installed dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Install SAP_LLM package
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "sap_llm.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Stage 4: Development image (with additional tools)
FROM application AS development

USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tmux \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    ipython \
    ipdb \
    pytest-watch \
    black \
    flake8 \
    mypy

USER appuser

CMD ["uvicorn", "sap_llm.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Stage 5: Training image (with Jupyter)
FROM application AS training

USER root

# Install Jupyter and training tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    tensorboard \
    wandb \
    matplotlib \
    seaborn

USER appuser

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
