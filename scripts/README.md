# SAP_LLM Infrastructure Scripts

This directory contains production-ready infrastructure setup and management scripts for SAP_LLM.

## Overview

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `setup_infrastructure.sh` | Main setup script - orchestrates complete infrastructure setup | bash, python3 |
| `download_models.py` | Download HuggingFace models with retry logic and progress tracking | transformers, huggingface_hub, rich |
| `init_databases.py` | Initialize Cosmos DB, MongoDB, and Redis | azure-cosmos, pymongo, redis, rich |
| `health_check.py` | System health verification and readiness check | All database clients, psutil, rich |
| `build_knowledge_base.py` | Build SAP knowledge base (existing) | Various |
| `run_tests.sh` | Run test suite (existing) | pytest |

---

## 1. setup_infrastructure.sh

**Complete infrastructure setup orchestration script.**

### Features

- ✅ Dependency verification (Python, Git, Docker, Kubernetes)
- ✅ Virtual environment creation
- ✅ Python package installation
- ✅ Model downloading
- ✅ Database initialization
- ✅ Kubernetes deployment
- ✅ Secret generation (API keys, APOP keys)
- ✅ Health check
- ✅ Comprehensive logging

### Usage

```bash
# Interactive setup (recommended for first-time setup)
./scripts/setup_infrastructure.sh

# Non-interactive setup (for CI/CD)
./scripts/setup_infrastructure.sh --non-interactive

# Skip model downloads (download later)
./scripts/setup_infrastructure.sh --skip-models

# Skip Kubernetes setup
./scripts/setup_infrastructure.sh --skip-k8s

# View help
./scripts/setup_infrastructure.sh --help
```

### What It Does

1. **Dependency Check**: Verifies all required tools are installed
2. **Virtual Environment**: Creates and configures Python virtual environment
3. **Package Installation**: Installs all Python dependencies
4. **Model Download**: Downloads required HuggingFace models (~100GB)
5. **Database Setup**: Initializes Cosmos DB, MongoDB, and Redis
6. **Kubernetes**: Deploys to Kubernetes cluster (if available)
7. **Secrets**: Generates API keys and APOP cryptographic keys
8. **Health Check**: Verifies all systems are operational

### Output

- Creates virtual environment at `./venv/`
- Downloads models to `/models/` (configurable via `MODELS_DIR`)
- Creates `.env` file from `.env.example`
- Generates keys in `./keys/`
- Logs everything to `./setup.log`

---

## 2. download_models.py

**Download HuggingFace models with advanced features.**

### Features

- ✅ Progress bars with transfer speed and ETA
- ✅ Automatic retry logic with exponential backoff
- ✅ Disk space verification
- ✅ Skip already downloaded models
- ✅ Parallel downloads
- ✅ Comprehensive error handling
- ✅ Support for gated models with authentication

### Models Downloaded

| Model | Repository | Size | Purpose |
|-------|-----------|------|---------|
| LayoutLMv3 | microsoft/layoutlmv3-base | 1.2 GB | Vision Encoder |
| LLaMA-2-7B | meta-llama/Llama-2-7b-hf | 13.5 GB | Language Decoder |
| Mixtral-8x7B | mistralai/Mixtral-8x7B-v0.1 | 87.0 GB | Reasoning Engine |
| TrOCR | microsoft/trocr-base-handwritten | 0.5 GB | OCR (optional) |

### Usage

```bash
# Download all models
python scripts/download_models.py

# Download to specific directory
python scripts/download_models.py --cache-dir /path/to/models

# Download specific models only
python scripts/download_models.py --models vision_encoder language_decoder

# With HuggingFace token (for gated models like LLaMA)
python scripts/download_models.py --token YOUR_HF_TOKEN

# List available models
python scripts/download_models.py --list

# Force re-download existing models
python scripts/download_models.py --no-skip-existing

# Custom retry configuration
python scripts/download_models.py --max-retries 5 --retry-delay 10
```

### Environment Variables

```bash
# HuggingFace token for authentication
export HF_TOKEN="your_token_here"

# Default cache directory
export HF_CACHE_DIR="/models/huggingface_cache"
```

### Example Output

```
╭─────────────────────────────────────────────╮
│ SAP_LLM Model Downloader                    │
│ Download HuggingFace models with progress   │
╰─────────────────────────────────────────────╯

┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Model                 ┃ Repository        ┃ Size    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Vision Encoder        │ microsoft/...     │ 1.2 GB  │
│ Language Decoder      │ meta-llama/...    │ 13.5 GB │
│ Reasoning Engine      │ mistralai/...     │ 87.0 GB │
└───────────────────────┴───────────────────┴─────────┘

⠋ Downloading Vision Encoder (LayoutLMv3)... ━━━━━━━━━╸━━━━━━━━━━ 65% 25.3 MB/s 00:15
```

---

## 3. init_databases.py

**Initialize all databases with validation and error handling.**

### Features

- ✅ Cosmos DB (Gremlin API) initialization for PMG
- ✅ MongoDB collections and indexes
- ✅ Redis cache structure
- ✅ Connection validation
- ✅ Schema validation
- ✅ Automatic index creation
- ✅ Dry-run mode for testing

### Databases Initialized

#### Cosmos DB (Process Memory Graph)
- Database: `qorsync`
- Container: `pmg`
- Graph structure for document relationships
- Configured with Gremlin API

#### MongoDB (Document Storage)
- Database: `sap_llm`
- Collections:
  - `documents` - Document storage
  - `results` - Processing results
  - `exceptions` - Exception tracking
  - `audit_log` - Audit trail
  - `pmg_cache` - PMG cache

#### Redis (Caching)
- Namespaces:
  - `sap_llm:models` - Model cache
  - `sap_llm:documents` - Document cache
  - `sap_llm:results` - Result cache
  - `sap_llm:pmg` - PMG cache
  - `sap_llm:sessions` - Session cache
  - `sap_llm:ratelimit` - Rate limiting

### Usage

```bash
# Initialize all databases
python scripts/init_databases.py

# Dry run (validate connections without changes)
python scripts/init_databases.py --dry-run

# Initialize specific database only
python scripts/init_databases.py --database mongodb
python scripts/init_databases.py --database redis
python scripts/init_databases.py --database cosmos

# Use custom config file
python scripts/init_databases.py --config /path/to/config.yaml
```

### Prerequisites

1. **Environment Variables**: Configure `.env` file with database credentials:
   ```bash
   # Cosmos DB
   COSMOS_ENDPOINT=https://your-account.documents.azure.com:443/
   COSMOS_KEY=your-key-here

   # MongoDB
   MONGODB_URI=mongodb://localhost:27017

   # Redis
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_PASSWORD=your-password  # optional
   ```

2. **Running Databases**:
   ```bash
   # Start MongoDB
   docker run -d -p 27017:27017 --name mongodb mongo:latest

   # Start Redis
   docker run -d -p 6379:6379 --name redis redis:latest
   ```

### Example Output

```
╭─────────────────────────────────────────────╮
│ SAP_LLM Database Initializer                │
│ Initialize Cosmos DB, MongoDB, and Redis    │
╰─────────────────────────────────────────────╯

Initializing Cosmos DB (PMG)...
  Connecting to https://your-account.documents.azure.com:443/...
  ✓ Connected to Cosmos DB account
  ✓ Database created: qorsync
  ✓ Container created: pmg
✓ Cosmos DB initialization complete

Initializing MongoDB...
  ✓ Connected to MongoDB
    Version: 7.0.4
  ✓ Database: sap_llm
    Collections: 5
✓ MongoDB initialization complete

┏━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Database   ┃ Status    ┃ Details                  ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Cosmos DB  │ ✓ Success │ Database: qorsync        │
│ MongoDB    │ ✓ Success │ Collections: 5           │
│ Redis      │ ✓ Success │ Host: localhost:6379     │
└────────────┴───────────┴──────────────────────────┘
```

---

## 4. health_check.py

**Comprehensive system health verification.**

### Features

- ✅ GPU availability and CUDA check
- ✅ Model file verification
- ✅ Database connection tests
- ✅ API service health
- ✅ System resource monitoring
- ✅ Detailed health reports
- ✅ JSON export capability

### Checks Performed

1. **GPU Availability**
   - CUDA availability
   - GPU device count
   - Memory capacity
   - Driver version

2. **Model Files**
   - Model directory existence
   - File size verification
   - Model file integrity

3. **Cosmos DB**
   - Connection test
   - Database existence
   - Container accessibility

4. **MongoDB**
   - Connection test
   - Database existence
   - Collection verification
   - Index validation

5. **Redis**
   - Connection test
   - Memory usage
   - Key count

6. **API Service**
   - Service availability
   - Health endpoint response
   - Version verification

7. **System Resources**
   - CPU usage
   - Memory usage
   - Disk space
   - Resource thresholds

### Usage

```bash
# Run all health checks
python scripts/health_check.py

# Verbose output
python scripts/health_check.py --verbose

# Check specific component
python scripts/health_check.py --component gpu
python scripts/health_check.py --component models
python scripts/health_check.py --component mongodb

# Export report to JSON
python scripts/health_check.py --export health_report.json

# Use custom config
python scripts/health_check.py --config /path/to/config.yaml
```

### Exit Codes

- `0` - All systems healthy
- `1` - Warnings detected
- `2` - Critical errors detected

### Example Output

```
╭─────────────────────────────────────────────╮
│ SAP_LLM Health Check                        │
│ System health and readiness verification    │
╰─────────────────────────────────────────────╯

Checking GPU Availability...
  ✓ GPU 0: NVIDIA A100-SXM4-40GB
    Capability: (8, 0)
    Memory: 40.0 GB

Checking Model Files...
  ✓ vision_encoder
    Path: /models/vision_encoder
    Size: 1.23 GB
  ✓ language_decoder
    Path: /models/language_decoder
    Size: 13.48 GB

┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Component      ┃ Status     ┃ Details             ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ GPU            │ ✓ Healthy  │ Found 1 GPU(s)      │
│ Models         │ ✓ Healthy  │ All 3 models found  │
│ Cosmos DB      │ ✓ Healthy  │ Connected           │
│ MongoDB        │ ✓ Healthy  │ 5 collections       │
│ Redis          │ ✓ Healthy  │ Connected           │
│ API            │ ⚠ Warning  │ Service not running │
│ System         │ ✓ Healthy  │ Resources OK        │
└────────────────┴────────────┴─────────────────────┘

Overall Status: ⚠ Some warnings detected
```

---

## Common Workflows

### First-Time Setup

```bash
# 1. Run complete setup
./scripts/setup_infrastructure.sh

# 2. Activate virtual environment
source venv/bin/activate

# 3. Verify health
python scripts/health_check.py
```

### Download Models Only

```bash
# Activate virtual environment
source venv/bin/activate

# Download models
python scripts/download_models.py --cache-dir /models
```

### Initialize Databases Only

```bash
# Configure .env file first
nano .env

# Activate virtual environment
source venv/bin/activate

# Initialize databases
python scripts/init_databases.py
```

### CI/CD Pipeline

```bash
# Non-interactive setup without models
./scripts/setup_infrastructure.sh --non-interactive --skip-models

# Run health check
python scripts/health_check.py --export health_report.json

# Check exit code
if [ $? -eq 0 ]; then
  echo "System healthy"
else
  echo "System has issues"
  cat health_report.json
fi
```

### Development Environment

```bash
# 1. Setup without Kubernetes
./scripts/setup_infrastructure.sh --skip-k8s

# 2. Start local services
docker-compose up -d

# 3. Initialize databases
python scripts/init_databases.py

# 4. Health check
python scripts/health_check.py --verbose
```

---

## Troubleshooting

### Models Not Downloading

**Problem**: Model download fails with authentication error

**Solution**:
```bash
# Get HuggingFace token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"

# For gated models (like LLaMA), request access first at:
# https://huggingface.co/meta-llama/Llama-2-7b-hf

# Retry download
python scripts/download_models.py --token $HF_TOKEN
```

### Database Connection Fails

**Problem**: Cannot connect to databases

**Solution**:
```bash
# Check if services are running
docker ps

# Start services if needed
docker-compose up -d

# Test connections manually
# MongoDB
mongosh mongodb://localhost:27017

# Redis
redis-cli ping

# Re-run initialization
python scripts/init_databases.py --dry-run
```

### Insufficient Disk Space

**Problem**: Not enough space for models

**Solution**:
```bash
# Check available space
df -h

# Download to external drive
mkdir -p /mnt/external/models
python scripts/download_models.py --cache-dir /mnt/external/models

# Update .env
echo "HF_CACHE_DIR=/mnt/external/models" >> .env
```

### GPU Not Detected

**Problem**: CUDA not available

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Configuration

All scripts use the same configuration sources:

1. **YAML Config**: `configs/default_config.yaml`
2. **Environment Variables**: `.env` file
3. **Command-line Arguments**: Override defaults

### Priority Order

Command-line args > Environment variables > Config file > Defaults

---

## Logging

All scripts log to console and files:

- **Main setup**: `./setup.log`
- **Model downloads**: Progress bars + logs
- **Database init**: Detailed operation logs
- **Health checks**: Summary + JSON export

---

## Requirements

### Python Packages

All required packages are in `requirements.txt`:
- `transformers` - Model downloads
- `azure-cosmos` - Cosmos DB
- `pymongo` - MongoDB
- `redis` - Redis
- `rich` - Beautiful console output
- `psutil` - System monitoring
- `requests` - HTTP requests
- `pyyaml` - Config parsing

### System Tools

- Python 3.8+
- Git
- Docker (optional)
- Kubernetes/kubectl (optional)
- OpenSSL (for key generation)

---

## Support

For issues or questions:

1. Check `setup.log` for detailed error messages
2. Run health check: `python scripts/health_check.py --verbose`
3. Review documentation in `/docs`
4. Create GitHub issue with logs

---

## License

Part of SAP_LLM project. See main repository LICENSE.
