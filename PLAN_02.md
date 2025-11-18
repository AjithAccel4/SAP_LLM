# SAP_LLM Complete Development Plan From Scratch
## 100% Autonomous Document Processing System - Zero 3rd Party Dependencies

**Version:** 1.0  
**Date:** November 14, 2025  
**Objective:** Build a fully custom, self-hosted LLM system that handles all 8 QorSync pipeline stages autonomously without any external LLM APIs (no GPT-4o, Claude, or commercial services)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Phase 1: Foundation Models & Infrastructure](#phase-1-foundation-models--infrastructure)
4. [Phase 2: Training Data Pipeline](#phase-2-training-data-pipeline)
5. [Phase 3: SAP Knowledge Base Construction](#phase-3-sap-knowledge-base-construction)
6. [Phase 4: Model Development - 8 Stage Components](#phase-4-model-development---8-stage-components)
7. [Phase 5: Process Memory Graph Integration](#phase-5-process-memory-graph-integration)
8. [Phase 6: APOP Integration & Agentic Orchestration](#phase-6-apop-integration--agentic-orchestration)
9. [Phase 7: Self-Healing Workflow Loop](#phase-7-self-healing-workflow-loop)
10. [Phase 8: Deployment & Infrastructure](#phase-8-deployment--infrastructure)
11. [Phase 9: Testing & Validation](#phase-9-testing--validation)
12. [Phase 10: Performance Optimization](#phase-10-performance-optimization)
13. [Technical Specifications](#technical-specifications)
14. [Cost Analysis](#cost-analysis)
15. [Risk Mitigation](#risk-mitigation)
16. [Success Metrics](#success-metrics)

---

## Executive Summary

### Goal
Develop **SAP_LLM**: A proprietary, fully autonomous document processing system that:
- Processes ALL 8 QorSync pipeline stages end-to-end
- Learns continuously from Process Memory Graph (PMG)
- Makes autonomous decisions via APOP orchestration
- Has ZERO dependency on 3rd party LLMs or commercial APIs
- Achieves ≥95% classification accuracy and ≥85% touchless rate
- Costs <$0.004 per document (100% self-hosted)

### Core Capabilities

**Pipeline Stages Handled:**
1. **Inbox** - Document ingestion & routing
2. **Preprocessing** - OCR, image enhancement, text extraction
3. **Classification** - Document type identification
4. **Type Identifier** - 35+ invoice/PO subtypes
5. **Extraction** - Field-level data extraction (180+ fields)
6. **Data Quality Check** - Confidence scoring & validation
7. **Validation** - Business rules & tolerance checks
8. **Routing** - SAP API endpoint selection & payload generation

### Key Differentiators
- **Self-Distilled Architecture**: Custom 13-B parameter model distilled from specialized document understanding models
- **Hybrid Neural-Symbolic**: Combines deep learning with SAP business rule reasoning
- **Continuous Learning**: PMG-powered feedback loop with automatic retraining
- **Agentic Decision-Making**: APOP-compliant autonomous workflow orchestration
- **Air-Gap Ready**: Fully on-premises deployment with no external dependencies

---

## System Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         SAP_LLM CORE                            │
│                    (Unified 13-B Model)                         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │   Vision     │  │   Language   │  │   Reasoning  │        │
│  │  Encoder     │→ │   Decoder    │→ │   Engine     │        │
│  │ (LayoutLMv3) │  │  (LLaMA-2)   │  │  (Mixtral)   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
│         ↓                  ↓                  ↓                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         SAP Knowledge Base (Local)                       │ │
│  │  • 400+ S/4HANA API schemas                             │ │
│  │  • 13 document type mappings                            │ │
│  │  • Field transformation rules                           │ │
│  │  • Validation business logic                            │ │
│  └──────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│               STAGE-SPECIFIC MICRO-MODELS                       │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐       │
│  │Stage1│ │Stage2│ │Stage3│ │Stage4│ │Stage5│ │Stage6│ ...   │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘       │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                INTEGRATION LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │     PMG      │  │     APOP     │  │     SHWL     │        │
│  │  (Learning)  │  │(Orchestration)│  │(Self-Heal)   │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Component Architecture

**1. Core SAP_LLM (13-B Unified Model)**
   - Vision Encoder: 300M params (LayoutLMv3-based)
   - Language Decoder: 7B params (LLaMA-2-7B backbone)
   - Reasoning Engine: 6B params (Mixtral MoE - 8 experts)
   - Total: ~13B parameters, 26GB VRAM

**2. Stage-Specific Task Heads**
   - 8 specialized adapter layers (LoRA/QLoRA)
   - Shared backbone, task-specific outputs
   - 100M params each (total 800M additional)

**3. Knowledge Integration**
   - Vector DB: 1M+ SAP field embeddings
   - Graph DB: Business rule relationships
   - Cache: Redis for inference optimization

---

## Phase 1: Foundation Models & Infrastructure

### 1.1 Model Selection & Architecture Design

#### Base Model Components

**Vision Encoder (Document Understanding)**
```yaml
Model: LayoutLMv3-base (fine-tuned)
Purpose: Visual-text feature extraction
Size: 300M parameters
Input: PDF pages → 1024x1024 images + OCR tokens
Output: 768-dim embeddings per page region
Training: Fine-tune on 180k invoice/PO pages
Accuracy Target: ≥94% field-level F1
```

**Language Decoder (Text Generation)**
```yaml
Model: LLaMA-2-7B (custom trained)
Purpose: Text understanding & generation
Size: 7B parameters
Input: Extracted text + embeddings
Output: JSON ADC format (Adaptive Document Contract)
Training: Supervised fine-tuning on 500k labeled docs
Quality Target: JSON schema compliance ≥99%
```

**Reasoning Engine (Decision Making)**
```yaml
Model: Mixtral-8x7B (Mixture of Experts)
Purpose: Autonomous decision-making & routing
Size: 6B active parameters (47B total, 8 experts)
Input: Extracted ADC + PMG context
Output: APOP envelope with next_action_hint
Training: Reinforcement learning from PMG feedback
Decision Accuracy: ≥97% correct routing
```

#### Integration Architecture

**Unified Inference Pipeline:**
```python
# Pseudocode for SAP_LLM inference
def process_document(pdf_path, pmg_context):
    # Stage 1: Inbox - Document intake
    doc_hash = compute_hash(pdf_path)
    if doc_hash in redis_cache:
        return cached_result
    
    # Stage 2: Preprocessing
    pages = pdf_to_images(pdf_path)
    ocr_text, word_boxes = ocr_extract(pages)
    
    # Stage 3: Classification
    doc_type, confidence = vision_encoder.classify(pages, ocr_text)
    
    # Stage 4: Type Identifier (35+ subtypes)
    subtype = type_identifier.predict(doc_type, ocr_text)
    
    # Stage 5: Extraction
    visual_features = vision_encoder.encode(pages, word_boxes)
    adc_json = language_decoder.extract(
        visual_features, 
        ocr_text,
        schema=get_schema(doc_type)
    )
    
    # Stage 6: Data Quality Check
    quality_score = validate_extraction_quality(adc_json)
    if quality_score < 0.90:
        adc_json = self_correct_extraction(adc_json, pages)
    
    # Stage 7: Validation
    business_rules = get_rules_from_pmg(doc_type, subtype)
    violations = validate_business_rules(adc_json, business_rules)
    
    if violations:
        # SHWL: Log exception, cluster, propose fix
        shwl_response = handle_exception(violations, pmg_context)
    
    # Stage 8: Routing
    sap_endpoint, payload = reasoning_engine.route(
        adc_json,
        doc_type,
        subtype,
        pmg_context
    )
    
    # Create APOP envelope
    apop_envelope = create_envelope(
        data=payload,
        next_action_hint=f"router.post.{sap_endpoint}",
        confidence=quality_score,
        trace_id=generate_trace_id()
    )
    
    # Store in PMG for learning
    pmg.store_transaction(
        document=adc_json,
        routing_decision=apop_envelope,
        outcome=None  # Updated after SAP response
    )
    
    return apop_envelope
```

### 1.2 Infrastructure Requirements

#### Hardware Specifications

**Training Cluster:**
```yaml
GPU Nodes: 4x A100 80GB (or 8x A10 40GB)
CPU: 2x AMD EPYC 7763 (128 cores total)
RAM: 1TB per node
Storage: 50TB NVMe SSD (model checkpoints + datasets)
Network: 400 Gbps InfiniBand for multi-node training
Estimated Cost: $80k-$120k for on-prem
Cloud Alternative: Azure NC A100 v4 instances
```

**Inference Cluster (Production):**
```yaml
GPU Nodes: 2x A10 24GB per instance
CPU: 16 cores AMD EPYC
RAM: 128GB
Storage: 2TB NVMe SSD
Throughput: 50k docs/hour per node
Latency Target: P95 < 1.5s per document
Cost per Node: $15k-$20k
```

#### Software Stack

```yaml
OS: Ubuntu 22.04 LTS
Container: Docker + Kubernetes (AKS or on-prem)
Orchestration: Dapr sidecars
Message Bus: Kafka + Service Bus
Databases:
  - Cosmos DB (PMG - graph)
  - Neo4j (classification patterns - legacy)
  - Redis (caching + feature flags)
  - MongoDB (document storage)
AI Framework:
  - PyTorch 2.1+ (mixed precision training)
  - HuggingFace Transformers
  - DeepSpeed (distributed training)
  - ONNX Runtime (inference optimization)
```

### 1.3 Development Environment Setup

#### Step-by-Step Infrastructure Bootstrap

**1. Set up GPU Training Environment**
```bash
# Install CUDA 12.1 + cuDNN
sudo apt update
sudo apt install nvidia-driver-535 nvidia-cuda-toolkit

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace ecosystem
pip3 install transformers datasets accelerate evaluate
pip3 install deepspeed bitsandbytes peft

# Install document processing tools
pip3 install pdf2image pytesseract opencv-python pillow
pip3 install layoutparser detectron2
```

**2. Clone Base Models**
```bash
# Download LayoutLMv3 base model
git lfs install
git clone https://huggingface.co/microsoft/layoutlmv3-base

# Download LLaMA-2-7B (requires Meta license)
git clone https://huggingface.co/meta-llama/Llama-2-7b-hf

# Download Mixtral-8x7B
git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
```

**3. Set up Databases**
```bash
# Cosmos DB (PMG) - Azure or local emulator
docker run -p 8081:8081 -p 10251:10251 -p 10252:10252 -p 10253:10253 -p 10254:10254 \
    mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator

# Neo4j (classification patterns)
docker run -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

# Redis (caching)
docker run -p 6379:6379 redis:latest

# MongoDB (document storage)
docker run -p 27017:27017 mongo:latest
```

---

## Phase 2: Training Data Pipeline

### 2.1 Data Collection Strategy

#### Sources

**1. Customer Design Partner Data (Primary)**
```yaml
Volume: 500k documents
Types: 
  - Purchase Orders: 180k
  - Supplier Invoices: 220k
  - Sales Orders: 50k
  - Customer Invoices: 30k
  - Other (GR, ASN, etc.): 20k
Format: PDF + JSON ground truth (ADC format)
Quality: GxP-cleaned, NDA-protected
Label Accuracy: Cohen's kappa > 0.92
```

**2. Public Datasets (Secondary)**
```yaml
Sources:
  - Kaggle invoice datasets: 50k
  - SROIE dataset: 1k receipts/invoices
  - RVL-CDIP: 400k document images (classification)
  - DocBank: 500k document layout annotations
Usage: Augmentation + pre-training
```

**3. Synthetic Data Generation (Tertiary)**
```yaml
Tool: Custom LaTeX + FakerJS generator
Volume: 200k synthetic documents
Coverage:
  - Edge cases (RTL languages, rare currencies)
  - Adversarial examples (low quality scans)
  - Rare PO types (consignment, drop-ship)
  - Multi-page complex documents
Quality: Programmatically perfect labels
```

### 2.2 Data Labeling & QA

#### Annotation Pipeline

**Phase 1: Initial Labeling (BPO Chennai)**
```yaml
Team Size: 20 annotators
Throughput: 100 docs/annotator/day
Tools: Custom annotation UI
Output: JSON ADC format with bounding boxes
QC: 10% random sample reviewed by QA lead
Timeline: 8 weeks for 500k documents
```

**Phase 2: Triple-Check (Internal QA Bucharest)**
```yaml
Team Size: 5 senior annotators
Sample: 10% stratified by doc type & confidence
Method: Independent re-annotation
Agreement Metric: Cohen's kappa ≥ 0.92
Conflict Resolution: Third annotator arbitration
```

**Phase 3: Active Learning Loop**
```yaml
Process:
  1. Train initial model on 100k labeled docs
  2. Run inference on unlabeled 400k docs
  3. Flag low-confidence predictions (<0.85)
  4. Human review of flagged samples
  5. Add corrected samples to training set
  6. Retrain model
  7. Repeat until accuracy plateaus
Benefit: Reduces labeling cost by 40%
```

### 2.3 Data Engineering Pipeline

#### Preprocessing Workflow

**1. PDF to Training Format**
```python
def preprocess_document(pdf_path, annotations):
    """
    Convert PDF + annotations to model-ready format
    """
    # Extract pages as images
    pages = pdf2image.convert_from_path(pdf_path, dpi=300)
    
    # Run OCR (Tesseract or EasyOCR)
    ocr_results = []
    for page in pages:
        words, boxes, confidences = run_ocr(page)
        ocr_results.append({
            'words': words,
            'boxes': boxes,  # [x1, y1, x2, y2] normalized
            'confidences': confidences
        })
    
    # Extract ground truth from annotations
    adc_json = annotations['adc']
    field_boxes = annotations['bounding_boxes']
    
    # Create LayoutLMv3 format
    training_sample = {
        'image': pages,
        'words': ocr_results,
        'labels': convert_to_token_labels(adc_json, ocr_results),
        'boxes': field_boxes,
        'adc_gt': adc_json  # Ground truth ADC
    }
    
    return training_sample
```

**2. Dataset Splits**
```yaml
Training Set: 70% (350k documents)
Validation Set: 15% (75k documents)
Test Set: 15% (75k documents)

Stratification:
  - By document type (proportional)
  - By supplier geography (balanced)
  - By quality (20% low-quality scans)
  - By complexity (30% multi-page)
```

**3. Data Augmentation**
```python
def augment_training_data(sample):
    """
    Apply augmentation to increase robustness
    """
    augmentations = [
        # Image-level
        RandomRotation(degrees=2),
        RandomBrightness(factor=0.3),
        AddGaussianNoise(sigma=5),
        JPEGCompression(quality=75),
        
        # Document-level
        SimulatePhotocopy(),
        SimulateFax(),
        AddWatermark(),
        
        # Text-level
        RandomOCRErrors(rate=0.02),
        SwapSimilarChars('O' → '0', 'l' → '1'),
    ]
    
    return apply_augmentations(sample, augmentations)
```

---

## Phase 3: SAP Knowledge Base Construction

### 3.1 SAP API Schema Extraction

#### Automated Schema Crawler

```python
class SAPSchemaExtractor:
    """
    Crawl SAP Business Accelerator Hub and extract API schemas
    """
    def __init__(self):
        self.hub_url = "https://api.sap.com"
        self.schemas = {}
    
    def extract_all_schemas(self):
        """
        Extract 400+ S/4HANA API schemas
        """
        apis = [
            'API_SALES_ORDER_SRV',
            'API_PURCHASEORDER_PROCESS_SRV',
            'API_BUSINESS_PARTNER',
            'API_INBOUND_DELIVERY_SRV',
            'API_OUTBOUND_DELIVERY_SRV',
            # ... 400+ more
        ]
        
        for api_name in apis:
            schema = self.fetch_odata_metadata(api_name)
            self.schemas[api_name] = self.parse_edmx(schema)
    
    def parse_edmx(self, edmx_xml):
        """
        Parse OData $metadata EDMX to extract:
        - Entity types
        - Property names & types
        - Navigation properties
        - Function imports
        """
        entities = {}
        for entity_type in edmx_xml.findall('.//EntityType'):
            name = entity_type.get('Name')
            properties = {}
            
            for prop in entity_type.findall('.//Property'):
                prop_name = prop.get('Name')
                prop_type = prop.get('Type')
                nullable = prop.get('Nullable', 'true')
                max_length = prop.get('MaxLength')
                
                properties[prop_name] = {
                    'type': prop_type,
                    'required': nullable == 'false',
                    'max_length': max_length
                }
            
            entities[name] = properties
        
        return entities
```

#### Knowledge Base Structure

**1. Field Mapping Database**
```json
{
  "document_type": "PURCHASE_ORDER",
  "sap_api": "API_PURCHASEORDER_PROCESS_SRV",
  "entity": "A_PurchaseOrder",
  "field_mappings": {
    "po_number": {
      "adc_field": "po_number",
      "sap_field": "PurchaseOrder",
      "data_type": "Edm.String",
      "max_length": 10,
      "required": true,
      "transformation": "direct_mapping",
      "validation_rules": [
        "regex: ^[0-9]{10}$",
        "unique: true"
      ]
    },
    "vendor_id": {
      "adc_field": "vendor_id",
      "sap_field": "Supplier",
      "data_type": "Edm.String",
      "max_length": 10,
      "required": true,
      "transformation": "lookup_mapping",
      "lookup_source": "business_partner_master",
      "validation_rules": [
        "exists_in: BP_SUPPLIER",
        "status: ACTIVE"
      ]
    }
    // ... 180+ more fields
  }
}
```

**2. Business Rule Database**
```json
{
  "rule_id": "VAL_001",
  "rule_name": "Three-Way Match Tolerance",
  "document_types": ["SUPPLIER_INVOICE"],
  "condition": {
    "when": "adc.total_amount != po.total_amount",
    "threshold": "abs((adc.total_amount - po.total_amount) / po.total_amount) > 0.03"
  },
  "action": {
    "result": "EXCEPTION",
    "severity": "MEDIUM",
    "message": "Price variance exceeds 3% tolerance",
    "next_action_hint": "approval.manager.required",
    "approval_workflow": "financial_controller"
  },
  "pmg_learning": {
    "cluster_similar": true,
    "auto_adjust_threshold": true,
    "confidence_requirement": 0.95
  }
}
```

**3. Transformation Function Library**
```python
class TransformationFunctions:
    """
    SAP-specific data transformations
    """
    @staticmethod
    def direct_mapping(value, schema):
        """Simple 1:1 field mapping"""
        return value
    
    @staticmethod
    def lookup_mapping(value, lookup_table, fallback=None):
        """Lookup from master data"""
        result = lookup_table.get(value)
        if result is None and fallback:
            return fallback
        return result
    
    @staticmethod
    def date_format(value, input_format, output_format='YYYYMMDD'):
        """Convert date formats"""
        from datetime import datetime
        dt = datetime.strptime(value, input_format)
        return dt.strftime(output_format)
    
    @staticmethod
    def currency_conversion(amount, from_currency, to_currency, rate_table):
        """Convert currency"""
        rate = rate_table.get(f"{from_currency}_{to_currency}")
        return amount * rate
    
    @staticmethod
    def address_mapping(address_dict, sap_format='STRUCTURED'):
        """Convert address to SAP format"""
        if sap_format == 'STRUCTURED':
            return {
                'Street': address_dict.get('street'),
                'City': address_dict.get('city'),
                'PostalCode': address_dict.get('postal_code'),
                'Country': address_dict.get('country')
            }
        elif sap_format == 'SINGLE_LINE':
            return ', '.join(filter(None, [
                address_dict.get('street'),
                address_dict.get('city'),
                address_dict.get('postal_code'),
                address_dict.get('country')
            ]))
```

### 3.2 Embedding the Knowledge Base

#### Vector Store Creation

```python
from sentence_transformers import SentenceTransformer
import faiss

class SAPKnowledgeVectorStore:
    """
    Semantic search over SAP knowledge base
    """
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.knowledge_items = []
    
    def build_index(self, knowledge_base):
        """
        Create embeddings for all SAP fields and rules
        """
        for doc_type, mappings in knowledge_base.items():
            for field_name, field_info in mappings.items():
                # Create rich text description
                text = self.create_field_description(
                    doc_type,
                    field_name,
                    field_info
                )
                
                # Generate embedding
                embedding = self.model.encode(text)
                
                # Add to index
                self.index.add(embedding.reshape(1, -1))
                self.knowledge_items.append({
                    'doc_type': doc_type,
                    'field': field_name,
                    'info': field_info,
                    'text': text
                })
    
    def create_field_description(self, doc_type, field_name, field_info):
        """
        Generate searchable text description
        """
        parts = [
            f"Document type: {doc_type}",
            f"Field: {field_name}",
            f"SAP field: {field_info['sap_field']}",
            f"Type: {field_info['data_type']}",
            f"Description: {field_info.get('description', '')}",
            f"Transformation: {field_info.get('transformation', '')}",
            f"Validation: {', '.join(field_info.get('validation_rules', []))}"
        ]
        return ' | '.join(parts)
    
    def search(self, query, k=5):
        """
        Find most relevant SAP fields/rules
        """
        query_embedding = self.model.encode(query)
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1),
            k
        )
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'item': self.knowledge_items[idx],
                'similarity': 1 / (1 + dist)
            })
        
        return results
```

---

## Phase 4: Model Development - 8 Stage Components

### 4.1 Stage 1: Inbox (Document Ingestion)

#### Model Architecture
```yaml
Type: Lightweight classifier
Base: ResNet-18 (image-based) + BERT-tiny (text-based)
Size: 50M parameters
Input: Document thumbnail (256x256) + first 128 tokens
Output: Document category (INVOICE, PO, SO, GR, etc.)
Latency: <50ms
Accuracy Target: 99.5%
```

#### Training Process

```python
import torch
import torch.nn as nn
from transformers import BertModel
from torchvision.models import resnet18

class InboxClassifier(nn.Module):
    """
    Fast document category classifier for inbox routing
    """
    def __init__(self, num_classes=15):
        super().__init__()
        
        # Visual branch
        self.visual_encoder = resnet18(pretrained=True)
        self.visual_encoder.fc = nn.Linear(512, 256)
        
        # Text branch
        self.text_encoder = BertModel.from_pretrained('prajjwal1/bert-tiny')
        self.text_projection = nn.Linear(128, 256)
        
        # Fusion & classification
        self.fusion = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, image, text_ids, text_mask):
        # Visual features
        visual_features = self.visual_encoder(image)
        
        # Text features
        text_output = self.text_encoder(
            input_ids=text_ids,
            attention_mask=text_mask
        )
        text_features = self.text_projection(
            text_output.pooler_output
        )
        
        # Concatenate and classify
        combined = torch.cat([visual_features, text_features], dim=1)
        fused = self.fusion(combined)
        fused = self.dropout(fused)
        logits = self.classifier(fused)
        
        return logits

# Training script
def train_inbox_classifier():
    model = InboxClassifier(num_classes=15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(10):
        for batch in train_dataloader:
            images = batch['thumbnail']
            text_ids = batch['first_page_tokens']
            text_mask = batch['attention_mask']
            labels = batch['category']
            
            logits = model(images, text_ids, text_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model
```

### 4.2 Stage 2: Preprocessing (OCR & Enhancement)

#### OCR Engine Selection

**Option 1: Tesseract (Lightweight)**
```yaml
Pros: Free, CPU-only, good for high-quality scans
Cons: Lower accuracy on poor quality documents
Speed: 300ms per page
Accuracy: 92% character-level
```

**Option 2: EasyOCR (Neural)**
```yaml
Pros: Better accuracy, multi-language
Cons: Requires GPU, slower
Speed: 800ms per page (GPU)
Accuracy: 97% character-level
```

**Option 3: Custom OCR (Recommended)**
```yaml
Base: TrOCR (Transformer-based OCR)
Fine-tuned on: Invoice/PO dataset (500k pages)
Hardware: A10 GPU
Speed: 500ms per page
Accuracy: 98.5% character-level
```

#### Image Enhancement Pipeline

```python
import cv2
import numpy as np
from PIL import Image

class DocumentPreprocessor:
    """
    Image enhancement for better OCR accuracy
    """
    def __init__(self):
        self.target_dpi = 300
    
    def enhance(self, image):
        """
        Apply enhancement pipeline
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Deskew
        gray = self.deskew(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Binarization (adaptive thresholding)
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Remove borders/stamps
        cleaned = self.remove_borders(binary)
        
        # Upscale if needed
        if self.estimate_dpi(image) < 200:
            cleaned = self.upscale_image(cleaned)
        
        return cleaned
    
    def deskew(self, image):
        """
        Correct rotation
        """
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def remove_borders(self, image, threshold=0.1):
        """
        Remove document borders and stamps
        """
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        mask = np.zeros_like(image)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > image.size * threshold:
                cv2.drawContours(mask, [cnt], -1, 255, -1)
        
        return cv2.bitwise_and(image, mask)
```

### 4.3 Stage 3: Classification (Document Type)

#### LayoutLMv3 Fine-Tuning

```python
from transformers import LayoutLMv3ForSequenceClassification, LayoutLMv3Processor
import torch

class DocumentClassifier:
    """
    Fine-tuned LayoutLMv3 for document classification
    """
    def __init__(self, num_classes=15):
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
            "microsoft/layoutlmv3-base",
            num_labels=num_classes
        )
    
    def train(self, train_dataloader, num_epochs=5):
        """
        Fine-tune on labeled documents
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=5e-5
        )
        
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    bbox=batch['bbox'],
                    pixel_values=batch['pixel_values'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def predict(self, image, words, boxes):
        """
        Classify document type
        """
        # Prepare inputs
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt"
        )
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
            confidence = torch.softmax(logits, dim=-1).max().item()
        
        return predicted_class, confidence
```

**Training Configuration:**
```yaml
Model: LayoutLMv3-base
Dataset: 350k training documents
Batch Size: 4 per GPU
Gradient Accumulation: 8 steps
Learning Rate: 5e-5
Warmup Steps: 1000
Max Steps: 50000
Mixed Precision: fp16
Hardware: 4x A100 80GB
Training Time: ~36 hours
Checkpoint Size: 440MB
```

### 4.4 Stage 4: Type Identifier (35+ Subtypes)

#### Hierarchical Classification

```python
class DocumentTypeIdentifier(nn.Module):
    """
    Hierarchical classifier for 35+ document subtypes
    """
    def __init__(self, backbone_model):
        super().__init__()
        
        self.backbone = backbone_model  # Pre-trained LayoutLMv3
        hidden_size = self.backbone.config.hidden_size
        
        # Level 1: Major category (5 classes)
        self.major_classifier = nn.Linear(hidden_size, 5)
        
        # Level 2: Subcategory (35 classes total)
        self.subtype_classifiers = nn.ModuleDict({
            'PURCHASE_ORDER': nn.Linear(hidden_size, 10),
            'SUPPLIER_INVOICE': nn.Linear(hidden_size, 8),
            'SALES_ORDER': nn.Linear(hidden_size, 7),
            'CUSTOMER_INVOICE': nn.Linear(hidden_size, 6),
            'GOODS_RECEIPT': nn.Linear(hidden_size, 4)
        })
    
    def forward(self, **inputs):
        # Get backbone features
        outputs = self.backbone(**inputs, output_hidden_states=True)
        pooled_output = outputs.pooler_output
        
        # Level 1: Major category
        major_logits = self.major_classifier(pooled_output)
        major_pred = torch.argmax(major_logits, dim=-1)
        
        # Level 2: Subtype based on major category
        subtype_logits = []
        for i, maj_pred in enumerate(major_pred):
            category = self.idx_to_category[maj_pred.item()]
            subtype_head = self.subtype_classifiers[category]
            sub_logits = subtype_head(pooled_output[i])
            subtype_logits.append(sub_logits)
        
        subtype_logits = torch.stack(subtype_logits)
        
        return {
            'major_logits': major_logits,
            'subtype_logits': subtype_logits
        }
```

**PO Subtypes Supported:**
```yaml
Standard PO: General purchase order
Blanket PO: Long-term agreement with release schedule
Contract PO: Master agreement reference
Service PO: Time & materials or fixed-price services
Subcontract PO: Manufacturing subcontracting
Consignment PO: Vendor-managed inventory
Stock Transfer PO: Inter-plant transfers
Limit PO: Unknown items within budget limit
Drop Ship PO: Direct ship to customer
CapEx PO: Capital expenditure purchases
```

### 4.5 Stage 5: Extraction (Field-Level Data)

#### Unified Extraction Model

This is the most critical component. We'll use a custom architecture combining:
1. LayoutLMv3 for visual-text understanding
2. LLaMA-2-7B decoder for structured output generation
3. Constrained decoding for JSON schema compliance

```python
import torch
import torch.nn as nn
from transformers import LayoutLMv3Model, LlamaForCausalLM

class UnifiedExtractorModel(nn.Module):
    """
    Combined vision-language model for field extraction
    """
    def __init__(self):
        super().__init__()
        
        # Vision encoder (LayoutLMv3)
        self.vision_encoder = LayoutLMv3Model.from_pretrained(
            "microsoft/layoutlmv3-base"
        )
        
        # Language decoder (LLaMA-2-7B)
        self.language_decoder = LlamaForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf"
        )
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=4096,  # LLaMA hidden size
                num_heads=32
            )
            for _ in range(4)
        ])
        
        # Projection layer
        self.vision_projection = nn.Linear(
            self.vision_encoder.config.hidden_size,  # 768
            4096  # LLaMA hidden size
        )
    
    def forward(
        self,
        pixel_values,
        input_ids_ocr,
        bbox,
        decoder_input_ids,
        decoder_attention_mask
    ):
        # Encode visual-text features
        vision_outputs = self.vision_encoder(
            pixel_values=pixel_values,
            input_ids=input_ids_ocr,
            bbox=bbox
        )
        
        # Project vision features to decoder dimension
        vision_features = self.vision_projection(
            vision_outputs.last_hidden_state
        )
        
        # Get decoder embeddings
        decoder_outputs = self.language_decoder.model(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            output_hidden_states=True
        )
        
        decoder_hidden = decoder_outputs.last_hidden_state
        
        # Apply cross-attention
        for cross_attn in self.cross_attention:
            decoder_hidden, _ = cross_attn(
                query=decoder_hidden,
                key=vision_features,
                value=vision_features
            )
        
        # Generate logits
        lm_logits = self.language_decoder.lm_head(decoder_hidden)
        
        return lm_logits

class ConstrainedDecoder:
    """
    Force decoder to output valid JSON matching ADC schema
    """
    def __init__(self, schema, tokenizer):
        self.schema = schema
        self.tokenizer = tokenizer
        self.state_machine = self.build_state_machine()
    
    def build_state_machine(self):
        """
        Create FSM from JSON schema
        """
        # State machine for JSON structure
        states = {
            'START': ['{'],
            'FIELD_NAME': ['\"'] + list(self.schema.keys()),
            'COLON': [':'],
            'FIELD_VALUE': ['\"', 'null', '0-9'],
            'COMMA': [',', '}']
        }
        return states
    
    def filter_vocab(self, logits, current_state, partial_json):
        """
        Mask invalid tokens based on current state
        """
        valid_tokens = self.get_valid_tokens(current_state, partial_json)
        
        # Create mask
        mask = torch.ones_like(logits) * float('-inf')
        for token_id in valid_tokens:
            mask[token_id] = 0
        
        # Apply mask
        filtered_logits = logits + mask
        
        return filtered_logits
    
    def get_valid_tokens(self, state, partial):
        """
        Determine which tokens are valid in current state
        """
        # Parse partial JSON to determine context
        # Return list of valid token IDs
        pass

def extract_fields(model, image, ocr_text, boxes, schema):
    """
    Extract structured data from document
    """
    # Prepare vision inputs
    vision_inputs = processor(
        image,
        ocr_text,
        boxes=boxes,
        return_tensors="pt"
    )
    
    # Start decoding
    decoder_input = tokenizer.encode(
        '{"',
        return_tensors="pt"
    )
    
    generated_json = '{'
    current_state = 'FIELD_NAME'
    
    # Constrained generation
    for step in range(max_length):
        # Forward pass
        logits = model(
            pixel_values=vision_inputs['pixel_values'],
            input_ids_ocr=vision_inputs['input_ids'],
            bbox=vision_inputs['bbox'],
            decoder_input_ids=decoder_input,
            decoder_attention_mask=None
        )
        
        # Get next token logits
        next_token_logits = logits[0, -1, :]
        
        # Apply constraints
        filtered_logits = constrained_decoder.filter_vocab(
            next_token_logits,
            current_state,
            generated_json
        )
        
        # Sample next token
        next_token = torch.argmax(filtered_logits).item()
        next_char = tokenizer.decode([next_token])
        
        # Update state
        generated_json += next_char
        current_state = update_state(current_state, next_char)
        
        # Check if complete
        if generated_json.endswith('}'):
            break
        
        # Append to decoder input
        decoder_input = torch.cat([
            decoder_input,
            torch.tensor([[next_token]])
        ], dim=1)
    
    # Parse and validate
    try:
        adc_json = json.loads(generated_json)
        validate_against_schema(adc_json, schema)
        return adc_json
    except json.JSONDecodeError:
        # Self-correction attempt
        return self_correct_json(generated_json, schema)
```

**Training Process:**
```python
def train_extractor_model():
    """
    Fine-tune unified extractor on labeled documents
    """
    model = UnifiedExtractorModel()
    
    # Freeze vision encoder initially
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    
    # Train decoder first (10k steps)
    optimizer = torch.optim.AdamW(
        model.language_decoder.parameters(),
        lr=1e-4
    )
    
    for epoch in range(3):
        for batch in train_dataloader:
            # Forward pass
            logits = model(
                pixel_values=batch['pixel_values'],
                input_ids_ocr=batch['input_ids_ocr'],
                bbox=batch['bbox'],
                decoder_input_ids=batch['decoder_input'],
                decoder_attention_mask=batch['decoder_mask']
            )
            
            # Compute loss (cross-entropy on next token)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch['labels'].view(-1)
            )
            
            # Add schema compliance loss
            predicted_json = greedy_decode(logits)
            compliance_loss = schema_compliance_penalty(
                predicted_json,
                batch['schema']
            )
            
            total_loss = loss + 0.1 * compliance_loss
            
            # Backprop
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Unfreeze vision encoder for fine-tuning (5k steps)
    for param in model.vision_encoder.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-6  # Lower LR for full model
    )
    
    # Continue training...
    
    return model
```

**Expected Performance:**
```yaml
Field-Level Accuracy:
  - Header Fields: 97.4% F1
  - Line Items: 92.1% F1
  - Monetary Values: 96.6% F1
  - Dates: 97.2% F1
  - Addresses: 89.3% F1

Latency: 780ms per document (A10 GPU)
Cost: $0.0036 per document
Schema Compliance: 99.2%
```

### 4.6 Stage 6: Data Quality Check

#### Confidence Scoring Model

```python
class QualityChecker:
    """
    Assess extraction quality and trigger self-correction
    """
    def __init__(self, threshold=0.90):
        self.threshold = threshold
        self.field_importance = self.load_field_weights()
    
    def compute_quality_score(self, extracted_adc, extraction_metadata):
        """
        Compute overall quality score
        """
        scores = []
        
        # 1. Individual field confidence
        for field, value in extracted_adc.items():
            field_confidence = extraction_metadata['confidences'].get(field, 0)
            field_weight = self.field_importance.get(field, 1.0)
            scores.append(field_confidence * field_weight)
        
        # 2. Required fields completeness
        required_fields = extraction_metadata['required_fields']
        completeness = sum(
            1 for f in required_fields if f in extracted_adc
        ) / len(required_fields)
        
        # 3. Schema compliance
        schema_valid = validate_against_schema(
            extracted_adc,
            extraction_metadata['schema']
        )
        
        # 4. Business rule checks
        business_rules_pass = validate_business_rules(
            extracted_adc,
            extraction_metadata['doc_type']
        )
        
        # Weighted average
        quality_score = (
            0.5 * np.mean(scores) +           # Field confidence
            0.2 * completeness +                # Completeness
            0.2 * (1 if schema_valid else 0) + # Schema
            0.1 * business_rules_pass           # Business rules
        )
        
        return quality_score
    
    def identify_low_confidence_fields(self, extracted_adc, metadata):
        """
        Flag fields that need re-extraction
        """
        low_confidence = []
        
        for field, value in extracted_adc.items():
            confidence = metadata['confidences'].get(field, 0)
            
            if confidence < 0.85:
                low_confidence.append({
                    'field': field,
                    'value': value,
                    'confidence': confidence,
                    'reason': 'low_confidence'
                })
        
        # Check for missing required fields
        for field in metadata['required_fields']:
            if field not in extracted_adc:
                low_confidence.append({
                    'field': field,
                    'value': None,
                    'confidence': 0.0,
                    'reason': 'missing_required'
                })
        
        return low_confidence
    
    def self_correct(self, adc, low_confidence_fields, document_context):
        """
        Attempt to correct low-confidence extractions
        """
        corrected_adc = adc.copy()
        
        for field_info in low_confidence_fields:
            field = field_info['field']
            
            # Strategy 1: Re-extract from specific region
            if 'bbox' in field_info:
                region = crop_image(
                    document_context['image'],
                    field_info['bbox']
                )
                new_value = targeted_extraction(region, field)
                
                if new_value['confidence'] > field_info['confidence']:
                    corrected_adc[field] = new_value['value']
            
            # Strategy 2: Use PMG similar documents
            similar_docs = pmg.find_similar_documents(
                document_context['doc_type'],
                document_context['supplier_id']
            )
            
            if similar_docs:
                # Use mode/median from similar docs
                similar_values = [
                    doc[field] for doc in similar_docs
                    if field in doc
                ]
                
                if similar_values:
                    corrected_value = most_common(similar_values)
                    corrected_adc[field] = corrected_value
            
            # Strategy 3: Apply heuristic rules
            if field in ['total_amount', 'subtotal', 'tax_amount']:
                # Check if total = subtotal + tax
                if 'subtotal' in corrected_adc and 'tax_amount' in corrected_adc:
                    expected_total = (
                        corrected_adc['subtotal'] +
                        corrected_adc['tax_amount']
                    )
                    
                    if abs(expected_total - adc.get('total_amount', 0)) < 0.01:
                        corrected_adc['total_amount'] = expected_total
        
        # Re-score
        new_score = self.compute_quality_score(
            corrected_adc,
            document_context
        )
        
        return corrected_adc, new_score
```

### 4.7 Stage 7: Validation (Business Rules)

#### Rule Engine

```python
class BusinessRuleValidator:
    """
    Validate extracted data against business rules
    """
    def __init__(self):
        self.rules = self.load_rules_from_pmg()
        self.rule_engine = RuleEngine()
    
    def validate(self, adc_json, doc_type, pmg_context):
        """
        Run all applicable business rules
        """
        violations = []
        
        # Get rules for this document type
        applicable_rules = [
            rule for rule in self.rules
            if doc_type in rule['document_types']
        ]
        
        for rule in applicable_rules:
            result = self.rule_engine.evaluate(
                rule,
                adc_json,
                pmg_context
            )
            
            if not result['passed']:
                violations.append({
                    'rule_id': rule['rule_id'],
                    'rule_name': rule['rule_name'],
                    'severity': rule['action']['severity'],
                    'message': rule['action']['message'],
                    'field': result['field'],
                    'value': result['value'],
                    'expected': result['expected'],
                    'next_action_hint': rule['action']['next_action_hint']
                })
        
        return violations
    
    def load_rules_from_pmg(self):
        """
        Fetch latest business rules from Process Memory Graph
        """
        rules = pmg.query("""
            MATCH (r:Rule)-[:APPLIES_TO]->(dt:DocumentType)
            WHERE r.status = 'ACTIVE'
            RETURN r, dt
        """)
        
        return rules

class RuleEngine:
    """
    Execute business rule logic
    """
    def evaluate(self, rule, data, context):
        """
        Evaluate rule condition
        """
        condition = rule['condition']
        
        # Parse condition
        if condition['when'] == 'field_comparison':
            return self.evaluate_comparison(
                condition,
                data,
                context
            )
        elif condition['when'] == 'pmg_lookup':
            return self.evaluate_pmg_lookup(
                condition,
                data,
                context
            )
        elif condition['when'] == 'calculation':
            return self.evaluate_calculation(
                condition,
                data
            )
    
    def evaluate_comparison(self, condition, data, context):
        """
        Compare field values
        """
        field = condition['field']
        operator = condition['operator']
        threshold = condition['threshold']
        
        value = data.get(field)
        
        if operator == '>':
            passed = value > threshold
        elif operator == '<':
            passed = value < threshold
        elif operator == '!=':
            passed = value != threshold
        # ... more operators
        
        return {
            'passed': passed,
            'field': field,
            'value': value,
            'expected': threshold
        }
    
    def evaluate_pmg_lookup(self, condition, data, context):
        """
        Validate against PMG historical data
        """
        # Example: Three-way match
        po_number = data.get('po_number')
        
        # Fetch PO from PMG
        po_data = pmg.get_document(
            doc_type='PURCHASE_ORDER',
            doc_id=po_number
        )
        
        if not po_data:
            return {
                'passed': False,
                'field': 'po_number',
                'value': po_number,
                'expected': 'Valid PO',
                'reason': 'PO not found in system'
            }
        
        # Check price variance
        invoice_total = data.get('total_amount')
        po_total = po_data.get('total_amount')
        
        variance = abs(invoice_total - po_total) / po_total
        tolerance = condition['threshold']
        
        return {
            'passed': variance <= tolerance,
            'field': 'total_amount',
            'value': invoice_total,
            'expected': f"Within {tolerance*100}% of PO total",
            'reason': f"Price variance: {variance*100:.2f}%"
        }
```

**Example Business Rules:**
```yaml
# Three-Way Match
rule_id: VAL_001
name: Three-Way Match Price Variance
doc_types: [SUPPLIER_INVOICE]
condition:
  when: pmg_lookup
  fields: [total_amount, po_number]
  threshold: 0.03  # 3% tolerance
action:
  if_fail: EXCEPTION
  severity: MEDIUM
  next_action_hint: approval.manager.required

# Duplicate Invoice Check
rule_id: VAL_002
name: Duplicate Invoice Detection
doc_types: [SUPPLIER_INVOICE]
condition:
  when: pmg_lookup
  fields: [supplier_invoice_number, supplier_id]
  check: unique_within_90_days
action:
  if_fail: REJECT
  severity: HIGH
  next_action_hint: rules.exception.duplicate

# Date Validation
rule_id: VAL_003
name: Invoice Date Reasonableness
doc_types: [SUPPLIER_INVOICE]
condition:
  when: calculation
  field: invoice_date
  rules:
    - not_future_dated
    - within_365_days_past
    - before_due_date
action:
  if_fail: WARNING
  severity: LOW
  next_action_hint: review.date_correction
```

### 4.8 Stage 8: Routing (SAP API Selection)

#### Intelligent Router with Reasoning

```python
class IntelligentRouter:
    """
    Autonomous routing decisions using reasoning model
    """
    def __init__(self):
        self.reasoning_model = self.load_reasoning_model()
        self.sap_knowledge = SAPKnowledgeVectorStore()
        self.pmg = ProcessMemoryGraph()
    
    def route(self, adc_json, doc_type, subtype, pmg_context):
        """
        Decide SAP endpoint and generate payload
        """
        # 1. Get SAP API schema for this document type
        api_schema = self.get_api_schema(doc_type, subtype)
        
        # 2. Query PMG for similar routing decisions
        similar_routings = self.pmg.get_similar_routing(
            doc_type=doc_type,
            supplier=adc_json.get('supplier_id'),
            company_code=adc_json.get('company_code')
        )
        
        # 3. Use reasoning model to make decision
        routing_decision = self.reasoning_model.decide(
            adc_json=adc_json,
            api_schema=api_schema,
            similar_cases=similar_routings,
            business_context=pmg_context
        )
        
        # 4. Generate SAP payload
        sap_payload = self.generate_sap_payload(
            adc_json,
            api_schema,
            routing_decision
        )
        
        # 5. Validate payload against schema
        validation_result = self.validate_sap_payload(
            sap_payload,
            api_schema
        )
        
        if not validation_result['valid']:
            # Self-correction
            sap_payload = self.fix_payload(
                sap_payload,
                validation_result['errors']
            )
        
        return {
            'endpoint': routing_decision['endpoint'],
            'method': routing_decision['method'],
            'payload': sap_payload,
            'confidence': routing_decision['confidence'],
            'reasoning': routing_decision['explanation']
        }
    
    def load_reasoning_model(self):
        """
        Load Mixtral-8x7B for reasoning
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1",
            device_map="auto",
            load_in_8bit=True  # Quantization for memory efficiency
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mixtral-8x7B-v0.1"
        )
        
        return ReasoningWrapper(model, tokenizer)

class ReasoningWrapper:
    """
    Wrapper for reasoning model with structured prompting
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def decide(self, adc_json, api_schema, similar_cases, business_context):
        """
        Make routing decision using chain-of-thought reasoning
        """
        # Construct prompt
        prompt = self.build_prompt(
            adc_json,
            api_schema,
            similar_cases,
            business_context
        )
        
        # Generate reasoning
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.1,  # Low temp for consistency
            do_sample=True,
            top_p=0.95
        )
        
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Parse structured response
        decision = self.parse_decision(response)
        
        return decision
    
    def build_prompt(self, adc, schema, similar, context):
        """
        Create chain-of-thought prompt
        """
        prompt = f"""
You are an SAP routing expert. Given a document, determine the correct SAP API endpoint and generate the payload.

**Document Information:**
Type: {context['doc_type']}
Subtype: {context['subtype']}
Supplier: {adc.get('supplier_name')}
Company Code: {context.get('company_code')}
Total Amount: {adc.get('total_amount')} {adc.get('currency')}

**Extracted Data (ADC):**
{json.dumps(adc, indent=2)}

**Available SAP APIs:**
{json.dumps(schema['available_endpoints'], indent=2)}

**Similar Past Routings:**
{json.dumps(similar, indent=2)}

**Task:**
1. Analyze the document type and extracted data
2. Consider similar past routing decisions
3. Select the appropriate SAP API endpoint
4. Explain your reasoning
5. Output a JSON decision in this format:
{{
  "endpoint": "API_SALES_ORDER_SRV",
  "method": "POST",
  "entity": "A_SalesOrder",
  "confidence": 0.95,
  "reasoning": "This is a standard sales order with all required fields present. Similar documents from this supplier were routed to the same endpoint with 98% success rate."
}}

**Decision:**
"""
        return prompt
    
    def parse_decision(self, response):
        """
        Extract structured decision from model output
        """
        # Find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        
        if json_match:
            try:
                decision = json.loads(json_match.group())
                return decision
            except json.JSONDecodeError:
                # Fallback parsing
                pass
        
        # If parsing fails, use heuristics
        return self.fallback_parsing(response)
```

#### SAP Payload Generator

```python
class SAPPayloadGenerator:
    """
    Generate SAP OData V2 compliant payloads
    """
    def __init__(self):
        self.transformations = TransformationFunctions()
        self.field_mappings = self.load_field_mappings()
    
    def generate(self, adc_json, api_schema):
        """
        Transform ADC to SAP payload
        """
        payload = {}
        
        # Header fields
        for adc_field, sap_mapping in api_schema['field_mappings'].items():
            if adc_field in adc_json:
                value = adc_json[adc_field]
                
                # Apply transformation
                transform_fn = sap_mapping.get('transformation')
                if transform_fn:
                    value = self.transformations.apply(
                        transform_fn,
                        value,
                        sap_mapping
                    )
                
                payload[sap_mapping['sap_field']] = value
        
        # Line items
        if 'items' in adc_json:
            payload['to_SalesOrderItem'] = []
            
            for item in adc_json['items']:
                sap_item = self.transform_line_item(
                    item,
                    api_schema['item_mappings']
                )
                payload['to_SalesOrderItem'].append(sap_item)
        
        # Add default values
        payload = self.add_defaults(payload, api_schema)
        
        # Format for OData
        odata_payload = self.format_for_odata(payload)
        
        return odata_payload
    
    def transform_line_item(self, item, item_schema):
        """
        Transform individual line item
        """
        sap_item = {}
        
        for adc_field, sap_mapping in item_schema.items():
            if adc_field in item:
                value = item[adc_field]
                
                # Transform
                transform_fn = sap_mapping.get('transformation')
                if transform_fn:
                    value = self.transformations.apply(
                        transform_fn,
                        value,
                        sap_mapping
                    )
                
                sap_item[sap_mapping['sap_field']] = value
        
        return sap_item
    
    def format_for_odata(self, payload):
        """
        Format payload for OData V2 POST request
        """
        return {
            'd': payload
        }
```

---

## Phase 5: Process Memory Graph Integration

### 5.1 PMG Architecture

#### Graph Schema

```cypher
// Document nodes
CREATE (:Document {
  doc_id: string,
  doc_type: string,
  doc_subtype: string,
  supplier_id: string,
  company_code: string,
  total_amount: float,
  currency: string,
  ingestion_timestamp: datetime,
  adc_json: json,
  embedding: vector[768]
})

// Rule nodes
CREATE (:Rule {
  rule_id: string,
  rule_name: string,
  version: string,
  status: enum['ACTIVE', 'DEPRECATED'],
  condition: json,
  action: json,
  valid_from: datetime,
  valid_to: datetime
})

// Exception nodes
CREATE (:Exception {
  exception_id: string,
  category: string,
  severity: enum['LOW', 'MEDIUM', 'HIGH'],
  field: string,
  expected: string,
  actual: string,
  explanation: string,
  embedding: vector[768]
})

// Routing decision nodes
CREATE (:RoutingDecision {
  decision_id: string,
  sap_endpoint: string,
  sap_method: string,
  payload: json,
  confidence: float,
  reasoning: string,
  timestamp: datetime
})

// SAP response nodes
CREATE (:SAPResponse {
  response_id: string,
  status_code: int,
  success: boolean,
  response_body: json,
  timestamp: datetime
})

// Relationships
CREATE (d:Document)-[:CLASSIFIED_AS]->(t:DocumentType)
CREATE (d:Document)-[:VALIDATED_BY]->(r:Rule)
CREATE (d:Document)-[:RAISED_EXCEPTION]->(e:Exception)
CREATE (d:Document)-[:ROUTED_TO]->(rd:RoutingDecision)
CREATE (rd:RoutingDecision)-[:GOT_RESPONSE]->(sr:SAPResponse)
CREATE (d:Document)-[:SIMILAR_TO {similarity: float}]->(d2:Document)
CREATE (e:Exception)-[:CLUSTERED_WITH]->(e2:Exception)
CREATE (r:Rule)-[:SUPERSEDED_BY]->(r2:Rule)
```

#### PMG Interface

```python
from gremlin_python.driver import client, serializer
from azure.cosmos import CosmosClient

class ProcessMemoryGraph:
    """
    Interface to Cosmos DB Gremlin API for PMG
    """
    def __init__(self):
        self.cosmos_client = CosmosClient(
            os.getenv('COSMOS_ENDPOINT'),
            os.getenv('COSMOS_KEY')
        )
        
        self.gremlin_client = client.Client(
            f"{os.getenv('COSMOS_ENDPOINT')}:443/",
            'g',
            username=f"/dbs/qorsync/colls/pmg",
            password=os.getenv('COSMOS_KEY'),
            message_serializer=serializer.GraphSONSerializersV2d0()
        )
        
        self.vector_search = self.init_vector_search()
    
    def store_transaction(
        self,
        document,
        routing_decision,
        outcome=None
    ):
        """
        Store complete transaction in PMG
        """
        # Create document vertex
        doc_id = str(uuid.uuid4())
        embedding = self.compute_embedding(document)
        
        query = f"""
        g.addV('Document')
          .property('id', '{doc_id}')
          .property('doc_type', '{document['doc_type']}')
          .property('doc_subtype', '{document.get('doc_subtype', '')}')
          .property('supplier_id', '{document.get('supplier_id', '')}')
          .property('company_code', '{document.get('company_code', '')}')
          .property('total_amount', {document.get('total_amount', 0)})
          .property('currency', '{document.get('currency', '')}')
          .property('ingestion_timestamp', '{datetime.now().isoformat()}')
          .property('adc_json', '{json.dumps(document)}')
          .property('embedding', {embedding.tolist()})
        """
        
        self.gremlin_client.submitAsync(query).result()
        
        # Create routing decision vertex
        if routing_decision:
            decision_id = str(uuid.uuid4())
            
            query = f"""
            g.addV('RoutingDecision')
              .property('id', '{decision_id}')
              .property('sap_endpoint', '{routing_decision['endpoint']}')
              .property('sap_method', '{routing_decision['method']}')
              .property('payload', '{json.dumps(routing_decision['payload'])}')
              .property('confidence', {routing_decision['confidence']})
              .property('reasoning', '{routing_decision.get('reasoning', '')}')
              .property('timestamp', '{datetime.now().isoformat()}')
            """
            
            self.gremlin_client.submitAsync(query).result()
            
            # Create edge
            query = f"""
            g.V('{doc_id}').addE('ROUTED_TO').to(g.V('{decision_id}'))
            """
            
            self.gremlin_client.submitAsync(query).result()
        
        # Store outcome if available
        if outcome:
            self.store_sap_response(decision_id, outcome)
        
        return doc_id
    
    def find_similar_documents(
        self,
        doc_type,
        supplier_id=None,
        limit=10
    ):
        """
        Find similar documents from PMG
        """
        query = f"""
        g.V()
          .has('Document', 'doc_type', '{doc_type}')
        """
        
        if supplier_id:
            query += f".has('supplier_id', '{supplier_id}')"
        
        query += f".limit({limit}).valueMap()"
        
        result = self.gremlin_client.submitAsync(query).result()
        
        documents = []
        for item in result:
            documents.append(self.parse_vertex(item))
        
        return documents
    
    def get_similar_routing(
        self,
        doc_type,
        supplier=None,
        company_code=None
    ):
        """
        Query PMG for similar routing decisions
        """
        query = f"""
        g.V()
          .has('Document', 'doc_type', '{doc_type}')
        """
        
        if supplier:
            query += f".has('supplier_id', '{supplier}')"
        
        if company_code:
            query += f".has('company_code', '{company_code}')"
        
        query += """
          .out('ROUTED_TO')
          .as('routing')
          .out('GOT_RESPONSE')
          .has('success', true)
          .select('routing')
          .limit(20)
          .valueMap()
        """
        
        result = self.gremlin_client.submitAsync(query).result()
        
        routings = []
        for item in result:
            routings.append(self.parse_vertex(item))
        
        return routings
    
    def compute_embedding(self, document):
        """
        Generate embedding for document
        """
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create text representation
        text = self.document_to_text(document)
        
        # Generate embedding
        embedding = model.encode(text)
        
        return embedding
    
    def document_to_text(self, document):
        """
        Convert document to text for embedding
        """
        parts = []
        
        parts.append(f"Type: {document.get('doc_type', '')}")
        parts.append(f"Supplier: {document.get('supplier_name', '')}")
        parts.append(f"Amount: {document.get('total_amount', 0)} {document.get('currency', '')}")
        
        # Add key fields
        for field in ['po_number', 'invoice_number', 'description']:
            if field in document:
                parts.append(f"{field}: {document[field]}")
        
        return ' | '.join(parts)
```

### 5.2 Continuous Learning Loop

```python
class ContinuousLearner:
    """
    Learn from PMG to improve model accuracy
    """
    def __init__(self):
        self.pmg = ProcessMemoryGraph()
        self.model_trainer = ModelTrainer()
    
    def learn_from_feedback(self):
        """
        Nightly job to retrain models on new data
        """
        # 1. Query PMG for recent documents with outcomes
        recent_docs = self.pmg.query_recent_transactions(
            days=7,
            with_outcomes=True
        )
        
        # 2. Filter for high-confidence successful cases
        training_samples = []
        for doc in recent_docs:
            if doc['routing_decision']['confidence'] > 0.95 and \
               doc['sap_response']['success']:
                training_samples.append(doc)
        
        # 3. Add to training set
        if len(training_samples) > 100:
            self.model_trainer.add_training_data(training_samples)
            
            # 4. Check if retraining is needed
            drift_score = self.detect_drift(training_samples)
            
            if drift_score > 0.25:
                print("Drift detected! Triggering model retraining...")
                self.model_trainer.retrain()
    
    def detect_drift(self, recent_samples):
        """
        Compute Population Stability Index (PSI)
        """
        # Get historical distribution
        historical = self.pmg.get_field_distribution(
            field='total_amount',
            period='last_90_days'
        )
        
        # Get recent distribution
        recent = self.compute_distribution(
            [s['total_amount'] for s in recent_samples]
        )
        
        # Compute PSI
        psi = 0
        for bucket in historical.keys():
            expected = historical[bucket]
            actual = recent.get(bucket, 0)
            
            if expected > 0 and actual > 0:
                psi += (actual - expected) * np.log(actual / expected)
        
        return psi
```

---

## Phase 6: APOP Integration & Agentic Orchestration

### 6.1 APOP Envelope Structure

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json

@dataclass
class APOPEnvelope:
    """
    Agentic Process Orchestration Protocol envelope
    """
    # CloudEvents base fields
    id: str  # UUID
    source: str  # Originating service
    type: str  # Event type (e.g., "classify.done")
    specversion: str = "1.0"
    time: str = None  # ISO-8601 timestamp
    datacontenttype: str = "application/json"
    
    # APOP extensions
    next_action_hint: Optional[str] = None  # e.g., "router.post"
    correlation_id: str = None
    traceparent: str = None  # W3C trace context
    tenant_id: str = None
    signature: str = None  # ECDSA signature
    
    # Payload
    data: Dict[str, Any] = None
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        envelope = {
            'id': self.id,
            'source': self.source,
            'type': self.type,
            'specversion': self.specversion,
            'time': self.time or datetime.now().isoformat(),
            'datacontenttype': self.datacontenttype,
            'next_action_hint': self.next_action_hint,
            'correlation_id': self.correlation_id,
            'traceparent': self.traceparent,
            'tenant_id': self.tenant_id,
            'data': self.data
        }
        
        # Sign envelope
        envelope['signature'] = self.sign(envelope)
        
        return envelope
    
    def sign(self, envelope):
        """
        Generate ECDSA signature
        """
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        
        # Load private key
        private_key = self.load_private_key()
        
        # Serialize envelope (excluding signature field)
        message = json.dumps(envelope, sort_keys=True).encode()
        
        # Sign
        signature = private_key.sign(
            message,
            ec.ECDSA(hashes.SHA256())
        )
        
        return signature.hex()
    
    @classmethod
    def from_dict(cls, data):
        """Parse APOP envelope from dictionary"""
        return cls(**data)
```

### 6.2 Agentic Orchestration

```python
class AgenticOrchestrator:
    """
    Autonomous workflow orchestration using APOP
    """
    def __init__(self):
        self.service_bus = ServiceBusClient()
        self.pmg = ProcessMemoryGraph()
        self.agents = self.register_agents()
    
    def register_agents(self):
        """
        Register available agents and their capabilities
        """
        agents = {
            'inbox': {
                'subscribes_to': ['ingress.document.blob'],
                'publishes': ['inbox.routed'],
                'next_actions': ['preproc.ocr']
            },
            'preprocessor': {
                'subscribes_to': ['inbox.routed'],
                'publishes': ['preproc.ready'],
                'next_actions': ['classify.detect']
            },
            'classifier': {
                'subscribes_to': ['preproc.ready'],
                'publishes': ['classify.done'],
                'next_actions': ['extract.fields', 'reject.unknown']
            },
            'extractor': {
                'subscribes_to': ['classify.done'],
                'publishes': ['adc.extracted'],
                'next_actions': ['quality.check']
            },
            'quality_checker': {
                'subscribes_to': ['adc.extracted'],
                'publishes': ['quality.verified', 'quality.failed'],
                'next_actions': ['rules.validate', 'extract.retry']
            },
            'validator': {
                'subscribes_to': ['quality.verified'],
                'publishes': ['rules.valid', 'rules.exception'],
                'next_actions': ['router.post', 'shwl.exception']
            },
            'router': {
                'subscribes_to': ['rules.valid'],
                'publishes': ['router.done', 'router.exception'],
                'next_actions': ['complete', 'retry']
            }
        }
        
        return agents
    
    def route_envelope(self, envelope):
        """
        Route APOP envelope to next agent
        """
        next_action = envelope.next_action_hint
        
        if not next_action:
            # No hint, use default flow
            next_action = self.determine_next_action(envelope)
        
        # Find agent that handles this action
        target_agent = None
        for agent_name, agent_info in self.agents.items():
            if any(next_action.startswith(action) 
                   for action in agent_info['next_actions']):
                target_agent = agent_name
                break
        
        if target_agent:
            # Publish to agent's topic
            topic = f"qorsync.{target_agent}"
            self.service_bus.publish(topic, envelope.to_dict())
        else:
            # Unknown action, raise exception
            self.handle_routing_failure(envelope, next_action)
    
    def determine_next_action(self, envelope):
        """
        Use PMG to determine next action when hint is missing
        """
        # Query PMG for similar documents
        similar_docs = self.pmg.find_similar_documents(
            doc_type=envelope.data.get('doc_type'),
            supplier_id=envelope.data.get('supplier_id')
        )
        
        # Get most common next action from similar docs
        next_actions = [
            doc['next_action'] for doc in similar_docs
            if 'next_action' in doc
        ]
        
        if next_actions:
            from collections import Counter
            most_common = Counter(next_actions).most_common(1)[0][0]
            return most_common
        
        # Fallback to default flow
        return self.default_flow_next_action(envelope.type)
```

---

## Phase 7: Self-Healing Workflow Loop

### 7.1 Exception Clustering

```python
class SelfHealingWorkflowLoop:
    """
    SHWL: Automatically detect, cluster, and fix exceptions
    """
    def __init__(self):
        self.pmg = ProcessMemoryGraph()
        self.clusterer = ExceptionClusterer()
        self.rule_generator = RuleGenerator()
    
    def run_nightly_healing(self):
        """
        Nightly batch job to cluster exceptions and propose fixes
        """
        # 1. Fetch all exceptions from last 7 days
        exceptions = self.pmg.query_exceptions(days=7)
        
        # 2. Cluster by similarity
        clusters = self.clusterer.cluster(exceptions)
        
        # 3. For each cluster, analyze and propose fix
        proposals = []
        for cluster in clusters:
            if len(cluster['exceptions']) >= 15:  # Min cluster size
                proposal = self.analyze_cluster(cluster)
                if proposal:
                    proposals.append(proposal)
        
        # 4. Review and approve (human-in-the-loop)
        for proposal in proposals:
            self.submit_for_approval(proposal)
    
    def analyze_cluster(self, cluster):
        """
        Analyze exception cluster and propose rule fix
        """
        # Get common pattern
        pattern = self.extract_pattern(cluster['exceptions'])
        
        # Generate rule diff
        rule_diff = self.rule_generator.generate_fix(
            pattern,
            cluster['category']
        )
        
        # Estimate impact
        impact = self.estimate_impact(rule_diff, cluster)
        
        return {
            'cluster_id': cluster['id'],
            'pattern': pattern,
            'rule_diff': rule_diff,
            'affected_count': len(cluster['exceptions']),
            'estimated_resolution_rate': impact['resolution_rate'],
            'confidence': impact['confidence'],
            'explanation': self.explain_fix(pattern, rule_diff)
        }

class ExceptionClusterer:
    """
    Cluster similar exceptions using embeddings
    """
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def cluster(self, exceptions):
        """
        Use HNSW clustering on exception embeddings
        """
        # Extract embeddings
        embeddings = []
        for exc in exceptions:
            if 'embedding' not in exc:
                # Generate embedding
                text = self.exception_to_text(exc)
                exc['embedding'] = self.embedding_model.encode(text)
            
            embeddings.append(exc['embedding'])
        
        embeddings = np.array(embeddings)
        
        # Cluster using HDBSCAN
        from hdbscan import HDBSCAN
        
        clusterer = HDBSCAN(
            min_cluster_size=15,
            metric='cosine',
            cluster_selection_epsilon=0.3
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Group exceptions by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise
                continue
            
            if label not in clusters:
                clusters[label] = {
                    'id': str(uuid.uuid4()),
                    'exceptions': [],
                    'category': None
                }
            
            clusters[label]['exceptions'].append(exceptions[idx])
        
        # Determine cluster category
        for cluster in clusters.values():
            categories = [e['category'] for e in cluster['exceptions']]
            cluster['category'] = max(set(categories), key=categories.count)
        
        return list(clusters.values())
```

### 7.2 Rule Generation

```python
class RuleGenerator:
    """
    Generate rule fixes using reasoning model
    """
    def __init__(self):
        self.reasoning_model = self.load_reasoning_model()
    
    def generate_fix(self, pattern, category):
        """
        Use LLM to propose rule modification
        """
        prompt = self.build_rule_generation_prompt(pattern, category)
        
        # Generate rule diff
        response = self.reasoning_model.generate(prompt)
        
        # Parse rule diff
        rule_diff = self.parse_rule_diff(response)
        
        return rule_diff
    
    def build_rule_generation_prompt(self, pattern, category):
        """
        Create prompt for rule generation
        """
        prompt = f"""
You are an expert at SAP business rule engineering. You have identified a pattern of exceptions that should be handled automatically.

**Exception Pattern:**
Category: {category}
Common Characteristics:
{json.dumps(pattern, indent=2)}

**Current Rule (causing exceptions):**
{self.get_current_rule(category)}

**Task:**
Generate a rule modification that will handle these exceptions automatically while maintaining data integrity.

Output format (JSON):
{{
  "rule_id": "VAL_XXX",
  "modification_type": "adjust_threshold|add_condition|add_exception_handler",
  "changes": {{
    "old_value": ...,
    "new_value": ...
  }},
  "reasoning": "Explanation of why this fix is appropriate",
  "risk_level": "low|medium|high"
}}
"""
        return prompt
```

---

## Phase 8: Deployment & Infrastructure

### 8.1 Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sap-llm-extractor
  namespace: qorsync
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sap-llm-extractor
  template:
    metadata:
      labels:
        app: sap-llm-extractor
    spec:
      containers:
      - name: extractor
        image: qorsync.azurecr.io/sap-llm:v1.0.0
        resources:
          requests:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: "1"
          limits:
            memory: "64Gi"
            cpu: "16"
            nvidia.com/gpu: "1"
        env:
        - name: MODEL_PATH
          value: "/models/sap-llm-13b"
        - name: COSMOS_ENDPOINT
          valueFrom:
            secretKeyRef:
              name: cosmos-secret
              key: endpoint
        - name: REDIS_HOST
          value: "redis-service:6379"
        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: shared-cache
          mountPath: /cache
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: shared-cache
        emptyDir:
          medium: Memory
          sizeLimit: 16Gi
---
apiVersion: v1
kind: Service
metadata:
  name: sap-llm-extractor
  namespace: qorsync
spec:
  selector:
    app: sap-llm-extractor
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### 8.2 Model Serving Infrastructure

```python
from fastapi import FastAPI, File, UploadFile
from transformers import pipeline
import torch

app = FastAPI()

class ModelServer:
    """
    Production model serving with FastAPI
    """
    def __init__(self):
        self.model = self.load_model()
        self.cache = self.init_cache()
    
    def load_model(self):
        """
        Load SAP_LLM model
        """
        model = UnifiedExtractorModel()
        model.load_state_dict(
            torch.load('/models/sap-llm-13b/pytorch_model.bin')
        )
        model.eval()
        model.to('cuda')
        
        return model
    
    @app.post("/extract")
    async def extract_document(file: UploadFile = File(...)):
        """
        Extract fields from uploaded document
        """
        # Check cache
        file_hash = compute_hash(await file.read())
        cached = redis.get(f"extract:{file_hash}")
        
        if cached:
            return json.loads(cached)
        
        # Process document
        result = await self.process_document(file)
        
        # Cache result
        redis.setex(
            f"extract:{file_hash}",
            3600,  # 1 hour TTL
            json.dumps(result)
        )
        
        return result
    
    async def process_document(self, file):
        """
        Full 8-stage processing pipeline
        """
        # Stage 1-2: Preprocessing
        image, ocr_text, boxes = preprocess(file)
        
        # Stage 3-4: Classification
        doc_type, subtype = self.model.classify(image, ocr_text)
        
        # Stage 5: Extraction
        adc_json = self.model.extract(
            image,
            ocr_text,
            boxes,
            doc_type
        )
        
        # Stage 6: Quality check
        quality_score = quality_checker.check(adc_json)
        
        if quality_score < 0.90:
            adc_json = self.model.self_correct(adc_json, image)
        
        # Stage 7: Validation
        violations = validator.validate(adc_json, doc_type)
        
        # Stage 8: Routing
        if not violations:
            routing = router.route(adc_json, doc_type)
        else:
            routing = {'next_action': 'exception_handling'}
        
        return {
            'doc_type': doc_type,
            'subtype': subtype,
            'extracted_data': adc_json,
            'quality_score': quality_score,
            'violations': violations,
            'routing': routing
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Phase 9: Testing & Validation

### 9.1 Comprehensive Test Suite

```python
import pytest
from pathlib import Path

class TestSAPLLM:
    """
    Comprehensive test suite for SAP_LLM
    """
    
    @pytest.fixture
    def model(self):
        return UnifiedExtractorModel.load('/models/sap-llm-13b')
    
    @pytest.fixture
    def test_dataset(self):
        return load_test_dataset('/data/test_set')
    
    def test_classification_accuracy(self, model, test_dataset):
        """
        Test document classification accuracy
        """
        correct = 0
        total = len(test_dataset)
        
        for sample in test_dataset:
            predicted = model.classify(
                sample['image'],
                sample['ocr_text']
            )
            
            if predicted['doc_type'] == sample['ground_truth']['doc_type']:
                correct += 1
        
        accuracy = correct / total
        assert accuracy >= 0.95, f"Classification accuracy {accuracy} below threshold"
    
    def test_extraction_f1_score(self, model, test_dataset):
        """
        Test field extraction F1 score
        """
        from sklearn.metrics import f1_score
        
        all_predictions = []
        all_ground_truths = []
        
        for sample in test_dataset:
            predicted = model.extract(
                sample['image'],
                sample['ocr_text'],
                sample['boxes'],
                sample['doc_type']
            )
            
            # Convert to binary labels per field
            pred_labels = self.adc_to_labels(predicted)
            gt_labels = self.adc_to_labels(sample['ground_truth'])
            
            all_predictions.extend(pred_labels)
            all_ground_truths.extend(gt_labels)
        
        f1 = f1_score(all_ground_truths, all_predictions, average='weighted')
        assert f1 >= 0.92, f"Extraction F1 {f1} below threshold"
    
    def test_latency_p95(self, model, test_dataset):
        """
        Test P95 latency
        """
        import time
        latencies = []
        
        for sample in test_dataset[:100]:
            start = time.time()
            
            model.process_document(sample['pdf_path'])
            
            latency = time.time() - start
            latencies.append(latency)
        
        p95_latency = np.percentile(latencies, 95)
        assert p95_latency <= 1.5, f"P95 latency {p95_latency}s exceeds 1.5s"
    
    def test_schema_compliance(self, model, test_dataset):
        """
        Test JSON schema compliance
        """
        from jsonschema import validate, ValidationError
        
        failures = 0
        
        for sample in test_dataset:
            extracted = model.extract(
                sample['image'],
                sample['ocr_text'],
                sample['boxes'],
                sample['doc_type']
            )
            
            schema = get_adc_schema(sample['doc_type'])
            
            try:
                validate(instance=extracted, schema=schema)
            except ValidationError:
                failures += 1
        
        compliance_rate = 1 - (failures / len(test_dataset))
        assert compliance_rate >= 0.99, f"Schema compliance {compliance_rate} below 99%"
    
    def test_pmg_integration(self, model):
        """
        Test PMG storage and retrieval
        """
        # Create test document
        test_doc = self.create_test_document()
        
        # Process and store in PMG
        result = model.process_document(test_doc['pdf_path'])
        doc_id = pmg.store_transaction(result)
        
        # Retrieve from PMG
        retrieved = pmg.get_document(doc_id)
        
        assert retrieved is not None
        assert retrieved['doc_type'] == result['doc_type']
    
    def test_apop_envelope_creation(self, model):
        """
        Test APOP envelope creation
        """
        test_doc = self.create_test_document()
        result = model.process_document(test_doc['pdf_path'])
        
        envelope = create_apop_envelope(result)
        
        assert envelope['specversion'] == '1.0'
        assert 'next_action_hint' in envelope
        assert 'signature' in envelope
        assert self.verify_signature(envelope)
    
    def test_self_healing(self):
        """
        Test SHWL exception clustering
        """
        # Create test exceptions
        exceptions = self.create_test_exceptions(count=50)
        
        # Cluster
        clusters = shwl.cluster_exceptions(exceptions)
        
        assert len(clusters) > 0
        
        # Test rule generation
        for cluster in clusters:
            proposal = shwl.analyze_cluster(cluster)
            
            assert 'rule_diff' in proposal
            assert 'confidence' in proposal
```

### 9.2 Performance Benchmarks

```python
def run_performance_benchmarks():
    """
    Comprehensive performance testing
    """
    results = {
        'classification': {},
        'extraction': {},
        'end_to_end': {},
        'cost': {}
    }
    
    # Classification benchmarks
    print("Running classification benchmarks...")
    results['classification'] = benchmark_classification()
    
    # Extraction benchmarks
    print("Running extraction benchmarks...")
    results['extraction'] = benchmark_extraction()
    
    # End-to-end throughput
    print("Running throughput benchmarks...")
    results['end_to_end'] = benchmark_throughput()
    
    # Cost analysis
    print("Running cost analysis...")
    results['cost'] = analyze_cost()
    
    # Generate report
    generate_benchmark_report(results)
    
    return results

def benchmark_classification():
    """Benchmark classification performance"""
    dataset = load_test_dataset(size=10000)
    
    start = time.time()
    
    for doc in dataset:
        _ = classifier.classify(doc['image'], doc['text'])
    
    elapsed = time.time() - start
    
    return {
        'total_docs': len(dataset),
        'elapsed_time': elapsed,
        'throughput': len(dataset) / elapsed,
        'avg_latency_ms': (elapsed / len(dataset)) * 1000
    }
```

---

## Phase 10: Performance Optimization

### 10.1 Model Optimization Techniques

**1. Quantization**
```python
def quantize_model(model, method='int8'):
    """
    Quantize model to reduce size and improve inference speed
    """
    if method == 'int8':
        from torch.quantization import quantize_dynamic
        
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
    
    elif method == 'bitsandbytes':
        from bitsandbytes.nn import Linear8bitLt
        
        # Replace all Linear layers with 8-bit versions
        # (Reduces memory by 4x with minimal accuracy loss)
        pass
    
    return quantized_model
```

**2. ONNX Optimization**
```python
def export_to_onnx(model, output_path):
    """
    Export to ONNX for optimized inference
    """
    dummy_input = {
        'pixel_values': torch.randn(1, 3, 224, 224),
        'input_ids': torch.randint(0, 30000, (1, 512)),
        'bbox': torch.randint(0, 1000, (1, 512, 4))
    }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['pixel_values', 'input_ids', 'bbox'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch'},
            'input_ids': {0: 'batch'},
            'bbox': {0: 'batch'}
        },
        opset_version=14
    )
    
    # Optimize with ONNX Runtime
    import onnxruntime as ort
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        output_path,
        sess_options,
        providers=['CUDAExecutionProvider']
    )
    
    return session
```

**3. Distillation (Further Size Reduction)**
```python
def distill_model(teacher_model, student_size='7B'):
    """
    Knowledge distillation: 13B → 7B model
    """
    # Create smaller student model
    if student_size == '7B':
        student = create_7b_model()
    elif student_size == '3B':
        student = create_3b_model()
    
    # Distillation training
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    
    for batch in train_dataloader:
        # Teacher predictions (soft labels)
        with torch.no_grad():
            teacher_logits = teacher_model(**batch)
        
        # Student predictions
        student_logits = student(**batch)
        
        # Distillation loss
        loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return student
```

### 10.2 Inference Optimization

**1. Batch Processing**
```python
class BatchProcessor:
    """
    Process documents in batches for efficiency
    """
    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size
        self.queue = []
    
    async def process_batch(self, documents):
        """
        Process multiple documents in parallel
        """
        batch = []
        results = []
        
        for doc in documents:
            batch.append(doc)
            
            if len(batch) >= self.batch_size:
                # Process batch
                batch_results = await self.model.process_batch(batch)
                results.extend(batch_results)
                batch = []
        
        # Process remaining
        if batch:
            batch_results = await self.model.process_batch(batch)
            results.extend(batch_results)
        
        return results
```

**2. Caching Strategy**
```python
class MultiLevelCache:
    """
    Multi-level caching for inference results
    """
    def __init__(self):
        self.l1_cache = {}  # In-memory (hot)
        self.l2_cache = redis.Redis()  # Redis (warm)
        self.l3_cache = CosmosDB()  # Cosmos DB (cold)
    
    def get(self, key):
        """
        Get from cache with fallback
        """
        # L1: In-memory
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # L2: Redis
        value = self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = json.loads(value)
            return self.l1_cache[key]
        
        # L3: Cosmos DB
        value = self.l3_cache.get(key)
        if value:
            self.l2_cache.setex(key, 3600, json.dumps(value))
            self.l1_cache[key] = value
            return value
        
        return None
    
    def set(self, key, value, ttl=3600):
        """
        Set in all cache levels
        """
        self.l1_cache[key] = value
        self.l2_cache.setex(key, ttl, json.dumps(value))
        self.l3_cache.set(key, value)
```

---

## Technical Specifications

### Model Architecture Summary

```yaml
SAP_LLM Unified Model:
  Total Parameters: 13.8B
  Components:
    - Vision Encoder: 300M (LayoutLMv3-base)
    - Language Decoder: 7B (LLaMA-2-7b)
    - Reasoning Engine: 6B (Mixtral-8x7B, 4 active experts)
    - Task Adapters: 800M (8x 100M LoRA)
  
  Memory Requirements:
    Training: 320GB (4x A100 80GB)
    Inference: 26GB (1x A10 24GB with quantization)
  
  Performance:
    Latency: 780ms P95 per document
    Throughput: 5000 docs/hour per GPU
    Accuracy: 94.6% field-level F1
    Cost: $0.0036 per document
  
  Formats:
    Input: PDF, PNG, JPEG, TIFF
    Output: JSON (ADC v1.2 schema)
```

### Infrastructure Requirements

```yaml
Production Deployment:
  Compute:
    - GPU: 2x NVIDIA A10 24GB per node
    - CPU: 16 cores AMD EPYC 7003
    - RAM: 128GB per node
    - Storage: 2TB NVMe SSD per node
  
  Networking:
    - Bandwidth: 10 Gbps
    - Latency: <5ms to Cosmos DB
  
  Databases:
    - Cosmos DB (PMG): 10k RU/s
    - Neo4j: 4 cores, 32GB RAM
    - Redis: 16GB memory, AOF persistence
    - MongoDB: 4 cores, 64GB RAM
  
  Message Bus:
    - Kafka/Service Bus: 1000 msg/sec
    - Retention: 7 days
  
  Kubernetes:
    - Nodes: 3 GPU nodes + 2 CPU nodes
    - Ingress: NGINX
    - Monitoring: Prometheus + Grafana
```

---

## Cost Analysis

### Development Costs

```yaml
Phase 1-3 (Foundation + Data): 12 weeks
  - ML Engineers (2x): $40k
  - Data Engineers (1x): $15k
  - Labeling (BPO): $30k
  - Cloud Resources: $15k
  Subtotal: $100k

Phase 4-6 (Model Development): 16 weeks
  - ML Engineers (3x): $80k
  - Software Engineers (2x): $40k
  - Cloud GPU Training: $50k
  - Infrastructure: $20k
  Subtotal: $190k

Phase 7-10 (Integration + Testing): 12 weeks
  - ML Engineers (2x): $40k
  - DevOps Engineers (2x): $30k
  - QA Engineers (1x): $10k
  - Cloud Resources: $25k
  Subtotal: $105k

Total Development Cost: $395k
Timeline: 40 weeks (~9 months)
```

### Operational Costs (Per Month)

```yaml
Infrastructure (On-Prem):
  - GPU Servers (2x A10): Amortized $3k/month
  - CPU Servers: Amortized $1k/month
  - Storage: $500/month
  - Networking: $300/month
  Subtotal: $4.8k/month

Cloud Alternative (Azure):
  - Container Apps: $8k/month
  - Cosmos DB: $3.7k/month
  - Redis Cache: $800/month
  - Service Bus: $500/month
  - Storage: $400/month
  - Monitoring: $300/month
  Subtotal: $13.7k/month

At 415M docs/year:
  Cost per document: $0.0047 (cloud) or $0.0016 (on-prem)
  
Compare to:
  - GPT-4o API: $0.0063/doc
  - Manual processing: $11/doc
  - BPO outsource: $1.44/doc
  - Legacy OCR: $0.44/doc
```

---

## Risk Mitigation

### Technical Risks

**1. Model Accuracy Below Threshold**
```yaml
Risk: Model fails to meet 95% classification / 92% extraction F1
Mitigation:
  - Extensive data labeling (500k+ documents)
  - Active learning to identify edge cases
  - Ensemble approach (multiple model checkpoints)
  - Human-in-the-loop fallback for low confidence
Contingency: Hybrid approach with selective API fallback
```

**2. Inference Latency Too High**
```yaml
Risk: P95 latency exceeds 1.5s target
Mitigation:
  - Model quantization (INT8)
  - ONNX optimization
  - Aggressive caching strategy
  - GPU acceleration mandatory
Contingency: Add more GPU nodes for horizontal scaling
```

**3. Integration Complexity**
```yaml
Risk: PMG/APOP integration delays timeline
Mitigation:
  - Start integration early (Phase 5)
  - Incremental testing
  - Fallback to simpler orchestration
Contingency: Launch without full APOP, add later
```

### Business Risks

**1. Training Data Insufficiency**
```yaml
Risk: Unable to obtain 500k labeled documents
Mitigation:
  - Synthetic data generation (200k)
  - Public datasets (50k)
  - Customer data partnerships
Contingency: Start with smaller model, scale later
```

**2. SAP API Changes**
```yaml
Risk: SAP updates APIs, breaking integrations
Mitigation:
  - Version-aware API mapping
  - Automated schema crawler
  - Backward compatibility layer
Contingency: Manual schema updates via config
```

---

## Success Metrics

### Performance KPIs

```yaml
Classification:
  - Accuracy: ≥95%
  - Latency: <50ms P95
  - Throughput: 10k docs/sec

Extraction:
  - Field-level F1: ≥92%
  - Schema compliance: ≥99%
  - Latency: <800ms P95

End-to-End:
  - Touchless rate: ≥85%
  - Total latency: <1.5s P95
  - Cost per doc: <$0.005

Quality:
  - Self-correction success: ≥70%
  - Exception handling: ≥90% auto-resolved
  - False positive rate: <2%
```

### Business KPIs

```yaml
Operational:
  - System uptime: 99.9%
  - Data loss: 0%
  - Security breaches: 0

Financial:
  - ROI: >300% vs. GPT-4o API
  - TCO: <$0.01 per document
  - Development payback: <18 months

Customer:
  - Document types supported: 13+
  - Languages supported: 10+
  - Air-gap deployment: Available
```

---

## Implementation Checklist

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up GPU training environment
- [ ] Install ML frameworks (PyTorch, HuggingFace)
- [ ] Download base models (LayoutLMv3, LLaMA-2, Mixtral)
- [ ] Set up databases (Cosmos, Neo4j, Redis, MongoDB)
- [ ] Implement basic inference pipeline

### Phase 2: Data Pipeline (Weeks 5-12)
- [ ] Collect 500k training documents
- [ ] Set up BPO labeling pipeline
- [ ] Implement QA process (Cohen's kappa >0.92)
- [ ] Generate 200k synthetic documents
- [ ] Create training/val/test splits
- [ ] Implement data augmentation

### Phase 3: SAP Knowledge Base (Weeks 8-12)
- [ ] Crawl 400+ SAP API schemas
- [ ] Create field mapping database (13 doc types)
- [ ] Implement transformation functions
- [ ] Build validation rule engine
- [ ] Create vector embeddings for knowledge base

### Phase 4: Model Training (Weeks 13-20)
- [ ] Fine-tune LayoutLMv3 (vision encoder)
- [ ] Fine-tune LLaMA-2-7B (language decoder)
- [ ] Train Mixtral reasoning engine
- [ ] Create 8 task-specific adapters (LoRA)
- [ ] Implement constrained decoding
- [ ] Validate on test set (≥92% F1)

### Phase 5: PMG Integration (Weeks 21-24)
- [ ] Implement Cosmos DB Gremlin interface
- [ ] Create graph schema (documents, rules, exceptions)
- [ ] Implement embedding generation
- [ ] Build similarity search (HNSW)
- [ ] Create learning feedback loop

### Phase 6: APOP Implementation (Weeks 25-28)
- [ ] Implement APOP envelope structure
- [ ] Create agentic orchestrator
- [ ] Register 8 pipeline agents
- [ ] Implement self-routing logic
- [ ] Test end-to-end orchestration

### Phase 7: SHWL Development (Weeks 29-32)
- [ ] Implement exception clustering (HDBSCAN)
- [ ] Create rule generation system
- [ ] Build approval workflow
- [ ] Implement progressive deployment
- [ ] Test self-healing cycle

### Phase 8: Deployment (Weeks 33-36)
- [ ] Create Docker containers
- [ ] Write Kubernetes manifests
- [ ] Set up CI/CD pipeline
- [ ] Implement monitoring (Prometheus/Grafana)
- [ ] Deploy to staging environment

### Phase 9: Testing (Weeks 37-39)
- [ ] Run classification benchmarks (≥95% acc)
- [ ] Run extraction benchmarks (≥92% F1)
- [ ] Load testing (5k docs/hour per node)
- [ ] Latency testing (P95 <1.5s)
- [ ] Security penetration testing

### Phase 10: Optimization & Launch (Week 40)
- [ ] Model quantization (INT8/ONNX)
- [ ] Cache optimization
- [ ] Performance tuning
- [ ] Documentation
- [ ] Production deployment

---

## Appendix: Prompts for AI Development

### Master Prompt for AI Developer

```
You are tasked with implementing SAP_LLM, a fully autonomous document processing system for QorSync. This system must handle all 8 pipeline stages (Inbox, Preprocessing, Classification, Type Identification, Extraction, Quality Check, Validation, Routing) without any 3rd party LLM APIs.

**Architecture:**
- Vision Encoder: LayoutLMv3-base (300M params)
- Language Decoder: LLaMA-2-7B
- Reasoning Engine: Mixtral-8x7B
- Total: 13.8B parameters

**Requirements:**
1. Fine-tune LayoutLMv3 on 500k invoice/PO documents
2. Implement constrained decoding for JSON schema compliance
3. Integrate with Process Memory Graph (Cosmos DB Gremlin)
4. Implement APOP-compliant agentic orchestration
5. Build Self-Healing Workflow Loop (SHWL)

**Success Criteria:**
- Classification accuracy ≥95%
- Extraction F1 ≥92%
- Latency P95 ≤1.5s
- Cost <$0.005 per document
- Zero dependency on external LLM APIs

**Development Phases:**
Follow the 10-phase plan in this document:
1. Foundation Models & Infrastructure
2. Training Data Pipeline
3. SAP Knowledge Base Construction
4. Model Development (8 stages)
5. Process Memory Graph Integration
6. APOP Integration
7. Self-Healing Workflow Loop
8. Deployment & Infrastructure
9. Testing & Validation
10. Performance Optimization

Begin with Phase 1, Week 1: Set up GPU training environment.
```

---

## Conclusion

This comprehensive plan provides a complete roadmap for developing **SAP_LLM from scratch** without any dependency on 3rd party LLMs or commercial APIs. The system will:

✅ Handle all 8 QorSync pipeline stages autonomously  
✅ Learn continuously from Process Memory Graph  
✅ Make intelligent routing decisions via APOP  
✅ Self-heal through exception clustering and rule generation  
✅ Achieve enterprise-grade accuracy at <$0.005 per document  
✅ Support air-gapped deployment for maximum security  

**Total Development Timeline:** 40 weeks (~9 months)  
**Total Development Cost:** $395k  
**Operational Cost:** $0.0016/doc (on-prem) or $0.0047/doc (cloud)  
**ROI vs GPT-4o:** 73% cost reduction  
**ROI vs Manual:** 99.95% cost reduction  

The system will be production-ready, scalable to 50k+ documents/hour, and provide a sustainable competitive advantage for QorSync in the enterprise document automation market.