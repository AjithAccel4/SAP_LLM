# 🚀 SAP_LLM Infrastructure Quick Start

**Complete setup in 3 commands!**

---

## ⚡ Super Quick Start

```bash
# 1. Run setup script
cd /home/user/SAP_LLM
./scripts/setup_infrastructure.sh

# 2. Activate environment
source venv/bin/activate

# 3. Verify system
python scripts/health_check.py
```

**That's it!** The system is ready to use. 🎉

---

## 📋 What Gets Installed

- ✅ Virtual environment with all dependencies
- ✅ 4 HuggingFace models (~102 GB)
  - LayoutLMv3 (Vision Encoder)
  - LLaMA-2-7B (Language Decoder)
  - Mixtral-8x7B (Reasoning Engine)
  - TrOCR (OCR)
- ✅ Databases initialized
  - Cosmos DB (Process Memory Graph)
  - MongoDB (Document Storage)
  - Redis (Caching)
- ✅ API keys and APOP cryptographic keys generated
- ✅ Kubernetes resources deployed (optional)

---

## 🎯 Individual Scripts

### Download Models Only
```bash
source venv/bin/activate
python scripts/download_models.py --cache-dir /models
```

### Initialize Databases Only
```bash
source venv/bin/activate
python scripts/init_databases.py
```

### Health Check Only
```bash
source venv/bin/activate
python scripts/health_check.py --verbose
```

---

## 🔧 Configuration

Before running, configure environment:

```bash
# Copy example config
cp .env.example .env

# Edit with your credentials
nano .env
```

Required settings:
- `COSMOS_ENDPOINT` - Azure Cosmos DB endpoint
- `COSMOS_KEY` - Azure Cosmos DB key
- `MONGODB_URI` - MongoDB connection string
- `REDIS_HOST` - Redis host
- `HF_TOKEN` - HuggingFace token (for LLaMA)

---

## 🐳 Docker Quick Start

```bash
# Start services
docker-compose up -d

# Initialize databases
python scripts/init_databases.py

# Check health
python scripts/health_check.py
```

---

## 🤖 CI/CD Quick Start

```bash
# Non-interactive setup
./scripts/setup_infrastructure.sh --non-interactive --skip-models

# Health check with exit code
python scripts/health_check.py
```

---

## 🆘 Troubleshooting

### Models won't download?
```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_token"
python scripts/download_models.py --token $HF_TOKEN
```

### Database connection fails?
```bash
# Check services are running
docker ps

# Start if needed
docker-compose up -d

# Test connection
python scripts/init_databases.py --dry-run
```

### Need more help?
```bash
# View detailed logs
cat setup.log

# Check system health
python scripts/health_check.py --verbose --export report.json

# Read full documentation
cat scripts/README.md
```

---

## 📚 Full Documentation

- **Detailed Guide**: `scripts/README.md`
- **Implementation Summary**: `INFRASTRUCTURE_SCRIPTS_SUMMARY.md`
- **Main README**: `README.md`
- **Deployment Guide**: `DEPLOYMENT.md`

---

## 💡 Pro Tips

1. **Skip model downloads** during initial setup (saves time):
   ```bash
   ./scripts/setup_infrastructure.sh --skip-models
   # Download later with: python scripts/download_models.py
   ```

2. **Use verbose mode** for debugging:
   ```bash
   python scripts/health_check.py --verbose
   ```

3. **Export health reports** for monitoring:
   ```bash
   python scripts/health_check.py --export health.json
   ```

4. **Check specific components**:
   ```bash
   python scripts/health_check.py --component gpu
   python scripts/health_check.py --component mongodb
   ```

---

**Need help?** Check `scripts/README.md` for comprehensive documentation!
