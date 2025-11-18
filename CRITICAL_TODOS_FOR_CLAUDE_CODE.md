# SAP_LLM: 10 Critical TODOs for Production Readiness
## Detailed Prompts for Claude Code Execution

**Date**: November 18, 2025  
**Status**: Infrastructure Complete - ML Training Required  
**Estimated Total Time**: 22-32 weeks (5.5-8 months)

---

## Overview

This document contains 10 critical tasks that must be completed to achieve ultra-enterprise production readiness for SAP_LLM. Each task includes a **detailed prompt** that can be given directly to **Claude Code** to complete autonomously.

**Current State**: 40% Complete  
**Target State**: 100% Production-Ready  
**Critical Blockers**: Models not trained, data not collected, knowledge base empty

---

## TODO #1: Training Data Collection Pipeline Execution 🔴 CRITICAL

**Priority**: P0 - BLOCKER  
**Estimated Time**: 6-8 weeks  
**Dependencies**: None  
**Status**: Not Started (0%)

### Detailed Prompt for Claude Code

```
TASK: Execute the complete training data collection pipeline for SAP_LLM to gather 1M+ labeled documents.

CONTEXT:
You are working on the SAP_LLM project located at /Users/ajithkumarr/Desktop/SAP_LLM. The infrastructure and code are complete, but NO training data has been collected yet. The goal is to collect 1M+ SAP business documents from 4 sources and prepare them for model training.

OBJECTIVE:
Collect and preprocess 1,000,000+ documents from 4 data sources:
1. QorSync PostgreSQL database (target: 300,000 documents)
2. SAP Business Accelerator Hub (target: 200,000 documents)
3. Public datasets: RVL-CDIP, CORD, FUNSD, SROIE (target: 200,000 documents)
4. Synthetic document generation (target: 500,000 documents)

REQUIREMENTS:
1. Configure and execute sap_llm/data_pipeline/corpus_builder.py
2. Set up data sources with proper authentication
3. Download and process public datasets
4. Generate 500K synthetic documents with variability
5. Split data into train/val/test (70%/15%/15%)
6. Verify data quality (no duplicates, proper annotations)
7. Store processed data in /Users/ajithkumarr/Desktop/SAP_LLM/data/
8. Generate data statistics report

TECHNICAL SPECIFICATIONS:
- QorSync PostgreSQL: Extract invoices, POs, receipts from production DB
  - Connection: Use environment variable QORSYNC_DB_URI
  - Tables: documents, extractions, validations
  - Date range: Last 3 years of data
  - Fields: document_id, image_path, ocr_text, document_type, extracted_fields, ground_truth
  
- SAP Business Accelerator Hub:
  - Use sap_llm/knowledge_base/crawler.py
  - Scrape sample documents from SAP Community
  - Rate limit: 10 requests/second
  - Save in data/raw/sap_hub/
  
- Public Datasets:
  - RVL-CDIP: https://www.cs.cmu.edu/~aharley/rvl-cdip/ (400K documents - select 100K most relevant)
  - CORD: https://github.com/clovaai/cord (11K receipts)
  - FUNSD: https://guillaumejaume.github.io/FUNSD/ (200 forms)
  - SROIE: https://rrc.cvc.uab.es/?ch=13 (1K receipts)
  - Download, extract, convert to unified format
  
- Synthetic Generation:
  - Use sap_llm/data_pipeline/synthetic_generator.py
  - Generate 500K documents across 15 document types:
    * 150K Purchase Orders (10 subtypes)
    * 150K Supplier Invoices (8 subtypes)
    * 100K Goods Receipts
    * 50K Sales Orders
    * 50K Customer Invoices
  - Variability: 
    * 5 languages (EN, DE, FR, ES, IT)
    * 10 template variations per type
    * Realistic field values (IBAN, VAT, dates, amounts)
    * Augmentations: rotation (±5°), noise, blur, contrast
    
- Data Storage Structure:
  /data/
    raw/
      qorsync/          # PostgreSQL extracts
      sap_hub/          # SAP scraped docs
      public/           # Public datasets
      synthetic/        # Generated docs
    processed/
      train/            # 700K documents
      val/              # 150K documents
      test/             # 150K documents
    metadata/
      statistics.json   # Data statistics
      splits.json       # Train/val/test split info
      label_distribution.json
      
- Quality Checks:
  1. Deduplication (SHA-256 hash checking)
  2. Image quality validation (resolution ≥300 DPI, not corrupted)
  3. OCR text completeness (≥50 words for documents, ≥5 for forms)
  4. Annotation coverage (≥80% fields labeled)
  5. Label distribution balance (no single class >30%)
  
- Output Format (Unified):
  {
    "id": "doc_12345",
    "source": "qorsync|sap_hub|rvl_cdip|cord|funsd|sroie|synthetic",
    "document_type": "PURCHASE_ORDER",
    "subtype": "STANDARD_PO",
    "image_path": "relative/path/to/image.jpg",
    "ocr_text": "...",
    "words": [...],
    "boxes": [...],
    "ground_truth": {
      "po_number": "4500012345",
      "supplier_name": "Acme Corp",
      ...
    },
    "metadata": {
      "language": "EN",
      "created_date": "2024-11-18",
      "quality_score": 0.95
    }
  }

EXECUTION STEPS:
1. Create directory structure: /data/raw/, /data/processed/, /data/metadata/
2. Configure environment variables (QORSYNC_DB_URI, SAP_API_KEY)
3. Run corpus builder:
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   python -m sap_llm.data_pipeline.corpus_builder \
     --output-dir data/ \
     --target-documents 1000000 \
     --sources qorsync,sap_hub,public,synthetic \
     --workers 16
4. Monitor progress (log to data/logs/collection.log)
5. Run quality checks:
   python -m sap_llm.data_pipeline.validate_corpus --data-dir data/
6. Generate statistics:
   python -m sap_llm.data_pipeline.generate_stats --data-dir data/ --output data/metadata/statistics.json
7. Create train/val/test splits:
   python -m sap_llm.data_pipeline.split_data --data-dir data/ --split-ratio 0.7,0.15,0.15
8. Verify outputs:
   - Check data/processed/train/ has ~700K documents
   - Check data/processed/val/ has ~150K documents
   - Check data/processed/test/ has ~150K documents
   - Verify metadata/statistics.json shows distribution

SUCCESS CRITERIA:
✓ 1,000,000+ documents collected
✓ Train/val/test split: 70%/15%/15%
✓ All 15 document types represented (≥10K each)
✓ No duplicates (0% duplication rate)
✓ Image quality ≥300 DPI
✓ Annotation coverage ≥80%
✓ Label distribution balanced (no class >30%)
✓ Data statistics report generated
✓ All files stored in correct directories

DELIVERABLES:
1. /data/processed/train/ - 700K documents
2. /data/processed/val/ - 150K documents
3. /data/processed/test/ - 150K documents
4. /data/metadata/statistics.json - Data statistics
5. /data/logs/collection.log - Collection logs
6. DATA_COLLECTION_REPORT.md - Summary report with:
   - Total documents collected per source
   - Document type distribution
   - Language distribution
   - Quality metrics
   - Issues encountered and resolutions

NOTES:
- QorSync database credentials must be obtained from production team
- SAP API scraping should respect robots.txt and rate limits
- Public datasets require ~500GB storage
- Synthetic generation is computationally intensive (estimate 72 hours on 16-core CPU)
- Monitor disk space (require ~2TB for all data)

ESTIMATED TIME: 6-8 weeks
- Week 1-2: QorSync data extraction + Public dataset download
- Week 3-4: SAP hub scraping + Initial synthetic generation
- Week 5-6: Bulk synthetic generation (500K docs)
- Week 7-8: Quality validation + Splitting + Documentation
```

---

## TODO #2: SAP Knowledge Base Population 🔴 CRITICAL

**Priority**: P0 - BLOCKER  
**Estimated Time**: 4-6 weeks  
**Dependencies**: None  
**Status**: 2% Complete (8/400+ APIs)

### Detailed Prompt for Claude Code

```
TASK: Populate the SAP Knowledge Base with 400+ S/4HANA API schemas, document type mappings, business rules, and field transformations.

CONTEXT:
You are working on SAP_LLM at /Users/ajithkumarr/Desktop/SAP_LLM. The knowledge base infrastructure (MongoDB + FAISS) is ready, but only 8 out of 400+ SAP APIs have been scraped. The system needs complete API schemas, field mappings, business rules, and transformation functions to route documents to correct SAP endpoints.

OBJECTIVE:
1. Scrape 400+ S/4HANA OData API schemas from SAP Business Accelerator Hub
2. Create document type mappings for 13 document types
3. Build business rule database (~200 rules)
4. Generate field transformation functions (~180 fields)
5. Populate FAISS vector store with embeddings

REQUIREMENTS:
1. Use sap_llm/knowledge_base/crawler.py to scrape SAP Business Accelerator Hub
2. Parse OData $metadata EDMX files
3. Store schemas in MongoDB (sap_llm/knowledge_base/storage.py)
4. Create embeddings with sentence-transformers (all-MiniLM-L6-v2)
5. Build FAISS index for semantic search
6. Generate document type → API mappings
7. Define business rules and validation logic
8. Create field transformation functions

TECHNICAL SPECIFICATIONS:

1. SAP API Scraping:
   - Base URL: https://api.sap.com/api/
   - Target APIs (400+):
     * API_PURCHASEORDER_PROCESS_SRV
     * API_SUPPLIERINVOICE_PROCESS_SRV
     * API_SALES_ORDER_SRV
     * API_BILLING_DOCUMENT_SRV
     * API_MATERIAL_DOCUMENT_SRV
     * API_JOURNALENTRY_CREATE_SRV
     * API_PRODUCT_SRV
     * API_BUSINESS_PARTNER
     * ... (see sap_llm/knowledge_base/api_list.json for full list)
   - For each API:
     * Download $metadata EDMX file
     * Parse entity types, properties, navigation properties
     * Extract field types, nullability, max length
     * Identify required fields
     * Store in MongoDB
   
2. MongoDB Schema:
   {
     "_id": ObjectId,
     "api_name": "API_PURCHASEORDER_PROCESS_SRV",
     "version": "1.0",
     "entity_type": "A_PurchaseOrder",
     "fields": [
       {
         "name": "PurchaseOrder",
         "type": "Edm.String",
         "max_length": 10,
         "nullable": false,
         "description": "Purchase Order Number"
       },
       ...
     ],
     "navigation_properties": [...],
     "relationships": [...],
     "documentation_url": "https://...",
     "scraped_date": "2025-11-18",
     "embedding": [0.123, -0.456, ...] // 384-dim vector
   }
   
3. Document Type Mappings (13 types):
   - PURCHASE_ORDER → API_PURCHASEORDER_PROCESS_SRV
     * Subtypes: STANDARD, BLANKET, CONTRACT, SCHEDULING_AGREEMENT, etc.
   - SUPPLIER_INVOICE → API_SUPPLIERINVOICE_PROCESS_SRV
     * Subtypes: STANDARD, CREDIT_MEMO, DEBIT_MEMO, etc.
   - SALES_ORDER → API_SALES_ORDER_SRV
   - CUSTOMER_INVOICE → API_BILLING_DOCUMENT_SRV
   - GOODS_RECEIPT → API_MATERIAL_DOCUMENT_SRV
   - GOODS_ISSUE → API_MATERIAL_DOCUMENT_SRV
   - JOURNAL_ENTRY → API_JOURNALENTRY_CREATE_SRV
   - PRODUCT_MASTER → API_PRODUCT_SRV
   - VENDOR_MASTER → API_BUSINESS_PARTNER
   - CUSTOMER_MASTER → API_BUSINESS_PARTNER
   - MATERIAL_MASTER → API_PRODUCT_SRV
   - PAYMENT_ADVICE → API_PAYMENT_ADVICE_SRV
   - DELIVERY_NOTE → API_INBOUND_DELIVERY_SRV
   
4. Field Mappings (180+ fields):
   ADC Field → SAP API Field + Transformation
   Example:
   {
     "adc_field": "po_number",
     "sap_field": "PurchaseOrder",
     "document_types": ["PURCHASE_ORDER"],
     "transformation": "pad_left_zeros(10)",
     "validation": "regex: ^[0-9]{10}$",
     "required": true
   }
   
5. Business Rules (~200 rules):
   {
     "rule_id": "BR_001",
     "name": "Three-way match for invoices",
     "description": "Invoice amount must match PO amount within ±5% tolerance",
     "applies_to": ["SUPPLIER_INVOICE"],
     "condition": "invoice.total_amount <= po.total_amount * 1.05 AND invoice.total_amount >= po.total_amount * 0.95",
     "severity": "ERROR",
     "auto_correctable": false
   }
   
   Rule Categories:
   - Amount tolerances (±5%, ±10%)
   - Date validations (invoice_date >= po_date)
   - Mandatory field checks
   - Format validations (IBAN, VAT, date formats)
   - Cross-document consistency (PO vs Invoice vs GR)
   - Duplicate detection
   
6. Transformation Functions (~180):
   - pad_left_zeros(length) - "123" → "0000000123"
   - parse_date(format) - "18.11.2025" → "2025-11-18"
   - clean_iban() - "DE 89 3704 0044 0532 0130 00" → "DE89370400440532013000"
   - extract_currency_code() - "1,234.56 USD" → "USD"
   - normalize_vat_number() - "DE123456789" → "DE123456789"
   - calculate_line_total() - qty * unit_price
   - convert_units() - "KG" → "EA" with conversion factor
   
7. FAISS Index:
   - Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
   - Index type: HNSW (Hierarchical Navigable Small World)
   - Distance metric: Cosine similarity
   - Index all:
     * API descriptions
     * Entity type names
     * Field names + descriptions
     * Business rule descriptions
   - Save to: /Users/ajithkumarr/Desktop/SAP_LLM/data/knowledge_base/faiss_index.bin

EXECUTION STEPS:
1. Set up environment:
   export SAP_API_KEY="your_sap_api_key"
   export MONGO_URI="mongodb://localhost:27017"
   export MONGO_DATABASE="sap_llm_kb"
   
2. Run SAP API crawler:
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   python -m sap_llm.knowledge_base.crawler \
     --api-list sap_llm/knowledge_base/api_list.json \
     --output-dir data/knowledge_base/schemas/ \
     --rate-limit 10 \
     --workers 5
   
3. Parse and store schemas:
   python -m sap_llm.knowledge_base.parse_schemas \
     --input-dir data/knowledge_base/schemas/ \
     --mongodb-uri $MONGO_URI \
     --database $MONGO_DATABASE
   
4. Create document type mappings:
   python -m sap_llm.knowledge_base.create_mappings \
     --config configs/document_types.yaml \
     --output data/knowledge_base/mappings.json
   
5. Define business rules:
   python -m sap_llm.knowledge_base.define_rules \
     --input sap_llm/knowledge_base/rules_template.yaml \
     --output data/knowledge_base/business_rules.json
   
6. Generate transformation functions:
   python -m sap_llm.knowledge_base.generate_transformations \
     --mappings data/knowledge_base/mappings.json \
     --output sap_llm/knowledge_base/transformations.py
   
7. Build FAISS index:
   python -m sap_llm.knowledge_base.build_vector_index \
     --mongodb-uri $MONGO_URI \
     --database $MONGO_DATABASE \
     --output data/knowledge_base/faiss_index.bin \
     --model sentence-transformers/all-MiniLM-L6-v2
   
8. Validate knowledge base:
   python -m sap_llm.knowledge_base.validate \
     --mongodb-uri $MONGO_URI \
     --database $MONGO_DATABASE \
     --faiss-index data/knowledge_base/faiss_index.bin

SUCCESS CRITERIA:
✓ 400+ SAP API schemas scraped and stored in MongoDB
✓ 13 document type mappings created
✓ 35+ PO subtypes mapped
✓ 15+ invoice subtypes mapped
✓ ~200 business rules defined
✓ ~180 field transformation functions generated
✓ FAISS index built with 400+ API embeddings
✓ Vector search returns relevant APIs (P@5 > 90%)
✓ All mappings validated (0 broken references)
✓ Knowledge base query latency < 100ms (P95)

DELIVERABLES:
1. MongoDB database populated with 400+ API schemas
2. data/knowledge_base/mappings.json - Document type mappings
3. data/knowledge_base/business_rules.json - Business rules
4. sap_llm/knowledge_base/transformations.py - Transformation functions
5. data/knowledge_base/faiss_index.bin - Vector index
6. KNOWLEDGE_BASE_REPORT.md - Report with:
   - APIs scraped per category
   - Mapping coverage
   - Business rule summary
   - Vector search validation results
   - Known gaps and limitations

NOTES:
- SAP Business Accelerator Hub requires free account (api.sap.com)
- Respect rate limits: 10 requests/second, 1000/day free tier
- Some APIs may be deprecated - flag and skip
- EDMX parsing may fail for complex schemas - implement error handling
- Store raw EDMX files for manual review if automated parsing fails
- Business rules should be reviewed by SAP functional consultant

ESTIMATED TIME: 4-6 weeks
- Week 1-2: API scraping (400+ APIs at 10/sec = 40+ hours + parsing)
- Week 3: Document type mappings + Field mappings
- Week 4: Business rules definition (requires domain knowledge)
- Week 5: Transformation function generation
- Week 6: FAISS index building + Validation + Documentation
```

---

## TODO #3: Vision Encoder Training (LayoutLMv3 → SAP-Specific) 🔴 CRITICAL

**Priority**: P0 - BLOCKER  
**Estimated Time**: 3-4 weeks  
**Dependencies**: TODO #1 (training data), TODO #2 (knowledge base)  
**Status**: Not Started (0%)

### Detailed Prompt for Claude Code

```
TASK: Fine-tune the LayoutLMv3-base vision encoder on SAP documents for document classification, subtype identification, and field detection.

CONTEXT:
You are working on SAP_LLM at /Users/ajithkumarr/Desktop/SAP_LLM. The vision encoder (sap_llm/models/vision_encoder.py) currently loads the base microsoft/layoutlmv3-base model, which has NOT been fine-tuned for SAP documents. You need to train 3 task-specific heads:
1. Document classification (15 types)
2. Subtype classification (35+ PO subtypes, 15+ invoice subtypes)
3. Field detection (180+ fields)

OBJECTIVE:
Train the vision encoder to achieve:
- ≥95% document classification accuracy
- ≥92% subtype classification accuracy
- ≥94% field detection F1 score
- <500ms inference latency per page

REQUIREMENTS:
1. Use training data from TODO #1 (1M+ documents)
2. Implement 3-stage training:
   - Stage 1: Document classification (15 types)
   - Stage 2: Hierarchical subtype classification (50+ subtypes)
   - Stage 3: Token classification for field detection (180+ fields)
3. Use distributed training (FSDP or DeepSpeed) on 4-8 GPUs
4. Apply data augmentations (rotation, blur, noise)
5. Use mixed precision training (FP16)
6. Save best checkpoints based on validation metrics
7. Export to ONNX for production inference

TECHNICAL SPECIFICATIONS:

1. Hardware Requirements:
   - 4-8x NVIDIA A100 80GB or H100 80GB
   - 512GB RAM
   - 10TB NVMe SSD for datasets
   - 400 Gbps InfiniBand (for multi-node training)
   
2. Training Configuration:
   ```yaml
   # configs/training/vision_encoder.yaml
   model:
     base: microsoft/layoutlmv3-base
     parameters: 300_000_000
     precision: fp16
     
   training:
     batch_size: 32  # Per GPU
     gradient_accumulation_steps: 4
     effective_batch_size: 1024  # 32 * 4 * 8 GPUs
     learning_rate: 5e-5
     warmup_steps: 1000
     max_steps: 50000
     weight_decay: 0.01
     max_grad_norm: 1.0
     
   optimization:
     optimizer: adamw
     scheduler: cosine_with_warmup
     mixed_precision: fp16
     distributed_backend: fsdp  # or deepspeed
     
   data:
     train_dir: /data/processed/train/
     val_dir: /data/processed/val/
     test_dir: /data/processed/test/
     num_workers: 16
     pin_memory: true
     
   augmentation:
     rotation_range: [-5, 5]  # degrees
     brightness_range: [0.8, 1.2]
     contrast_range: [0.8, 1.2]
     noise_std: 0.01
     blur_kernel: [3, 5, 7]
     probability: 0.3
   ```
   
3. Stage 1: Document Classification (15 types)
   - Task: Sequence classification
   - Input: Document image + OCR tokens + bounding boxes
   - Output: Document type (15 classes)
   - Loss: CrossEntropyLoss
   - Metrics: Accuracy, F1-macro, Confusion matrix
   - Target: ≥95% accuracy on validation set
   
   Document Types (15):
   1. PURCHASE_ORDER
   2. SUPPLIER_INVOICE
   3. SALES_ORDER
   4. CUSTOMER_INVOICE
   5. GOODS_RECEIPT
   6. GOODS_ISSUE
   7. JOURNAL_ENTRY
   8. PRODUCT_MASTER
   9. VENDOR_MASTER
   10. CUSTOMER_MASTER
   11. MATERIAL_MASTER
   12. PAYMENT_ADVICE
   13. DELIVERY_NOTE
   14. CREDIT_MEMO
   15. UNKNOWN
   
4. Stage 2: Hierarchical Subtype Classification (50+ subtypes)
   - Task: Conditional sequence classification
   - Input: Document image + OCR + Document type (from Stage 1)
   - Output: Subtype within document type
   - Loss: CrossEntropyLoss
   - Metrics: Accuracy, F1-macro per document type
   - Target: ≥92% accuracy
   
   PO Subtypes (35+):
   - STANDARD_PO, BLANKET_PO, CONTRACT_PO, SCHEDULING_AGREEMENT, FRAMEWORK_ORDER, ...
   
   Invoice Subtypes (15+):
   - STANDARD_INVOICE, CREDIT_MEMO, DEBIT_MEMO, PROFORMA_INVOICE, COMMERCIAL_INVOICE, ...
   
5. Stage 3: Token Classification for Field Detection (180+ fields)
   - Task: Token classification
   - Input: Document image + OCR tokens + bounding boxes
   - Output: BIO tags for each token (B-field_name, I-field_name, O)
   - Loss: CrossEntropyLoss with class weights (handle class imbalance)
   - Metrics: Precision, Recall, F1 per field, Overall F1
   - Target: ≥94% F1 score
   
   Field Categories (180+ fields):
   - Header fields: po_number, invoice_number, date, due_date, ...
   - Supplier/Customer: name, address, tax_id, ...
   - Line items: item_number, description, quantity, unit_price, amount, ...
   - Totals: subtotal, tax_amount, total_amount, ...
   - Payment: payment_terms, bank_account, iban, ...
   
6. Data Loading:
   ```python
   # Use sap_llm/data_pipeline/dataset.py
   from sap_llm.data_pipeline.dataset import SAPDocumentDataset
   from torch.utils.data import DataLoader
   
   train_dataset = SAPDocumentDataset(
       data_dir="data/processed/train/",
       task="classification",  # or "subtype" or "field_detection"
       processor=vision_encoder.processor,
       augmentations=augmentation_config,
   )
   
   train_loader = DataLoader(
       train_dataset,
       batch_size=32,
       shuffle=True,
       num_workers=16,
       pin_memory=True,
       collate_fn=vision_encoder.collate_fn,
   )
   ```
   
7. Training Loop:
   ```python
   # Use sap_llm/training/trainer.py
   from sap_llm.training.trainer import DistributedTrainer
   
   trainer = DistributedTrainer(
       model=vision_encoder,
       train_dataset=train_dataset,
       val_dataset=val_dataset,
       output_dir="models/vision_encoder/",
       config=training_config,
       use_deepspeed=True,  # or use_fsdp=True
   )
   
   trainer.train()
   ```
   
8. Evaluation:
   ```python
   # Evaluate on test set
   test_metrics = trainer.evaluate(test_dataset)
   
   # Expected metrics:
   # {
   #   "classification_accuracy": 0.957,
   #   "classification_f1_macro": 0.951,
   #   "subtype_accuracy": 0.923,
   #   "field_detection_f1": 0.943,
   #   "inference_latency_p95": 0.478  # seconds
   # }
   ```

EXECUTION STEPS:

1. Prepare training environment:
   ```bash
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   
   # Set up distributed training
   export MASTER_ADDR=localhost
   export MASTER_PORT=29500
   export WORLD_SIZE=8  # 8 GPUs
   
   # Verify data
   ls -lh data/processed/train/ | wc -l  # Should show ~700K files
   ```

2. Stage 1: Train document classifier:
   ```bash
   torchrun --nproc_per_node=8 \
     -m sap_llm.training.train_vision_encoder \
     --config configs/training/vision_encoder_stage1.yaml \
     --task classification \
     --output-dir models/vision_encoder/stage1/ \
     --num-epochs 10 \
     --eval-steps 500 \
     --save-steps 1000
   ```
   
3. Stage 2: Train subtype classifier:
   ```bash
   torchrun --nproc_per_node=8 \
     -m sap_llm.training.train_vision_encoder \
     --config configs/training/vision_encoder_stage2.yaml \
     --task subtype_classification \
     --pretrained models/vision_encoder/stage1/best_model/ \
     --output-dir models/vision_encoder/stage2/ \
     --num-epochs 5 \
     --eval-steps 500 \
     --save-steps 1000
   ```
   
4. Stage 3: Train field detector:
   ```bash
   torchrun --nproc_per_node=8 \
     -m sap_llm.training.train_vision_encoder \
     --config configs/training/vision_encoder_stage3.yaml \
     --task field_detection \
     --pretrained models/vision_encoder/stage2/best_model/ \
     --output-dir models/vision_encoder/stage3/ \
     --num-epochs 8 \
     --eval-steps 500 \
     --save-steps 1000
   ```
   
5. Evaluate on test set:
   ```bash
   python -m sap_llm.training.evaluate_vision_encoder \
     --model models/vision_encoder/stage3/best_model/ \
     --test-data data/processed/test/ \
     --output results/vision_encoder_evaluation.json
   ```
   
6. Export to ONNX:
   ```bash
   python -m sap_llm.optimization.export_onnx \
     --model models/vision_encoder/stage3/best_model/ \
     --output models/vision_encoder/vision_encoder.onnx \
     --opset-version 14
   ```
   
7. Quantize for production:
   ```bash
   python -m sap_llm.optimization.quantize \
     --model models/vision_encoder/stage3/best_model/ \
     --output models/vision_encoder/vision_encoder_int8.onnx \
     --quantization-mode int8
   ```

SUCCESS CRITERIA:
✓ Stage 1: Document classification accuracy ≥95% on test set
✓ Stage 2: Subtype classification accuracy ≥92% on test set
✓ Stage 3: Field detection F1 score ≥94% on test set
✓ Inference latency P95 < 500ms per page (on A10 GPU)
✓ Model size ≤ 600MB (INT8 quantized)
✓ No overfitting (train-val gap < 3%)
✓ Confusion matrix shows balanced performance across all classes
✓ Checkpoint saved and validated

DELIVERABLES:
1. models/vision_encoder/stage1/ - Document classifier
2. models/vision_encoder/stage2/ - Subtype classifier
3. models/vision_encoder/stage3/ - Field detector (best model)
4. models/vision_encoder/vision_encoder.onnx - Production model
5. results/vision_encoder_evaluation.json - Test metrics
6. VISION_ENCODER_TRAINING_REPORT.md - Report with:
   - Training curves (loss, accuracy over time)
   - Confusion matrices for all 3 stages
   - Per-class performance breakdown
   - Failure analysis (hardest examples)
   - Latency benchmarks
   - Recommendations for improvement

NOTES:
- Training Stage 1 takes ~36 hours on 8x A100 GPUs
- Training Stage 2 takes ~18 hours
- Training Stage 3 takes ~48 hours
- Total training time: ~4 days of continuous GPU usage
- Monitor GPU memory (should use ~60GB per A100)
- Use gradient checkpointing if OOM errors occur
- Save checkpoints every 1000 steps (disk space: ~500MB per checkpoint)
- Keep top-3 checkpoints by validation metric
- Use early stopping with patience=3 epochs

ESTIMATED TIME: 3-4 weeks
- Week 1: Stage 1 training + hyperparameter tuning
- Week 2: Stage 2 training + validation
- Week 3: Stage 3 training + optimization
- Week 4: Evaluation + ONNX export + Documentation
```

---

## TODO #4: Language Decoder Training (LLaMA-2-7B → ADC JSON Generation) 🔴 CRITICAL

**Priority**: P0 - BLOCKER  
**Estimated Time**: 3-4 weeks  
**Dependencies**: TODO #1 (training data), TODO #2 (knowledge base), TODO #3 (vision encoder)  
**Status**: Not Started (0%)

### Detailed Prompt for Claude Code

```
TASK: Fine-tune LLaMA-2-7B language decoder for structured JSON generation in ADC (Adaptive Document Contract) format with constrained decoding.

CONTEXT:
You are working on SAP_LLM at /Users/ajithkumarr/Desktop/SAP_LLM. The language decoder (sap_llm/models/language_decoder.py) currently loads base meta-llama/Llama-2-7b-hf, which cannot generate proper SAP document JSON. You need to fine-tune it for:
1. ADC JSON generation with schema compliance
2. Constrained decoding to ensure valid JSON
3. Multi-field extraction with confidence scoring

OBJECTIVE:
Train the language decoder to achieve:
- ≥92% extraction F1 score across 180+ fields
- ≥99% schema compliance (valid JSON, required fields present)
- ≥90% field-level accuracy
- <1.5s generation time per document

REQUIREMENTS:
1. Fine-tune LLaMA-2-7B on 1M+ documents with ground truth ADC JSON
2. Implement constrained decoding (FSM-based) to enforce JSON schema
3. Use LoRA/QLoRA for parameter-efficient fine-tuning
4. Apply INT8 quantization for production deployment
5. Integrate with vision encoder outputs (visual features)
6. Validate on all 13 document types and 180+ fields

TECHNICAL SPECIFICATIONS:

1. Hardware Requirements:
   - 4-8x NVIDIA A100 80GB or H100 80GB
   - 512GB RAM
   - 10TB NVMe SSD

2. Training Configuration:
   ```yaml
   # configs/training/language_decoder.yaml
   model:
     base: meta-llama/Llama-2-7b-hf
     parameters: 7_000_000_000
     quantization: int8  # Load in 8-bit for training
     lora:
       enabled: true
       rank: 64
       alpha: 128
       dropout: 0.05
       target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
     
   training:
     batch_size: 4  # Per GPU (limited by 7B model size)
     gradient_accumulation_steps: 16
     effective_batch_size: 512  # 4 * 16 * 8 GPUs
     learning_rate: 2e-4  # Higher for LoRA
     warmup_steps: 500
     max_steps: 30000
     weight_decay: 0.01
     max_grad_norm: 0.3
     
   optimization:
     optimizer: adamw_8bit  # Memory-efficient
     scheduler: cosine_with_warmup
     mixed_precision: fp16
     distributed_backend: deepspeed
     deepspeed_stage: 2
     
   data:
     train_dir: data/processed/train/
     val_dir: data/processed/val/
     test_dir: data/processed/test/
     max_input_length: 1024  # OCR text
     max_output_length: 512  # Generated JSON
     
   constrained_decoding:
     enabled: true
     method: fsm  # Finite State Machine
     enforce_json_schema: true
     beam_size: 1  # Greedy for speed
   ```

3. Training Data Format:
   ```json
   {
     "input": {
       "ocr_text": "PURCHASE ORDER\\nPO Number: 4500012345\\nDate: 18.11.2024\\nSupplier: Acme Corp\\n...",
       "document_type": "PURCHASE_ORDER",
       "subtype": "STANDARD_PO",
       "visual_features": [0.123, -0.456, ...]  // 768-dim from vision encoder
     },
     "output": {
       "po_number": "4500012345",
       "po_date": "2024-11-18",
       "supplier_name": "Acme Corp",
       "supplier_address": "123 Main St, City, State 12345",
       "company_code": "1000",
       "purchasing_organization": "1000",
       "purchasing_group": "001",
       "currency": "USD",
       "total_amount": 15000.00,
       "line_items": [
         {
           "item": "00010",
           "material": "MAT-001",
           "description": "Laptop Computer",
           "quantity": 10,
           "unit": "EA",
           "unit_price": 1500.00,
           "net_amount": 15000.00
         }
       ]
     },
     "schema": {
       "type": "object",
       "required": ["po_number", "po_date", "supplier_name", "total_amount"],
       "properties": {
         "po_number": {"type": "string", "pattern": "^[0-9]{10}$"},
         "po_date": {"type": "string", "format": "date"},
         ...
       }
     }
   }
   ```

4. Prompt Engineering:
   ```python
   def create_extraction_prompt(ocr_text, doc_type, schema):
       prompt = f"""<s>[INST] <<SYS>>
   You are an expert SAP document extraction system. Extract structured data from the provided document and output ONLY valid JSON matching the schema. Do not include explanations or markdown formatting.
   <</SYS>>

   Document Type: {doc_type}
   
   OCR Text:
   {ocr_text[:1500]}
   
   Required JSON Schema:
   {json.dumps(schema, indent=2)}
   
   Extract all fields and output ONLY the JSON object: [/INST]
   """
       return prompt
   ```

5. LoRA Fine-Tuning:
   ```python
   from peft import LoraConfig, get_peft_model, TaskType
   
   lora_config = LoraConfig(
       task_type=TaskType.CAUSAL_LM,
       r=64,  # Rank
       lora_alpha=128,
       lora_dropout=0.05,
       target_modules=[
           "q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj"
       ],
       bias="none",
   )
   
   model = get_peft_model(base_model, lora_config)
   model.print_trainable_parameters()
   # trainable params: 419M || all params: 7B || trainable%: 5.98%
   ```

6. Constrained Decoding Implementation:
   ```python
   # Implement in sap_llm/models/constrained_decoder.py
   from transformers.generation import LogitsProcessor
   
   class JSONSchemaLogitsProcessor(LogitsProcessor):
       """Constrain generation to follow JSON schema."""
       
       def __init__(self, schema, tokenizer):
           self.schema = schema
           self.tokenizer = tokenizer
           self.fsm = self._build_fsm(schema)
       
       def _build_fsm(self, schema):
           # Build finite state machine from JSON schema
           # States: OBJECT_START, KEY, COLON, VALUE, COMMA, OBJECT_END
           # Transitions based on schema constraints
           pass
       
       def __call__(self, input_ids, scores):
           # Mask invalid tokens based on current FSM state
           current_state = self.fsm.get_state(input_ids)
           valid_tokens = self.fsm.get_valid_tokens(current_state)
           mask = torch.ones_like(scores) * -float("inf")
           mask[:, valid_tokens] = 0
           return scores + mask
   ```

7. Evaluation Metrics:
   ```python
   def evaluate_extraction(predictions, ground_truth):
       metrics = {
           "field_accuracy": {},  # Per-field accuracy
           "field_f1": {},  # Per-field F1
           "schema_compliance": 0.0,  # % valid JSON
           "required_fields_present": 0.0,  # % required fields
           "overall_f1": 0.0,
       }
       
       for field in all_fields:
           tp = sum(p[field] == gt[field] for p, gt in zip(predictions, ground_truth) if field in p and field in gt)
           fp = sum(field in p and (field not in gt or p[field] != gt[field]) for p in predictions)
           fn = sum(field in gt and field not in p for gt in ground_truth)
           
           precision = tp / (tp + fp) if (tp + fp) > 0 else 0
           recall = tp / (tp + fn) if (tp + fn) > 0 else 0
           f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
           
           metrics["field_accuracy"][field] = tp / len(predictions)
           metrics["field_f1"][field] = f1
       
       metrics["overall_f1"] = np.mean(list(metrics["field_f1"].values()))
       return metrics
   ```

EXECUTION STEPS:

1. Prepare training data:
   ```bash
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   
   # Convert training data to LLaMA prompt format
   python -m sap_llm.data_pipeline.prepare_llm_training \
     --input-dir data/processed/ \
     --output-dir data/llm_training/ \
     --task extraction \
     --prompt-template llama2
   ```

2. Fine-tune with LoRA:
   ```bash
   torchrun --nproc_per_node=8 \
     -m sap_llm.training.train_language_decoder \
     --config configs/training/language_decoder.yaml \
     --base-model meta-llama/Llama-2-7b-hf \
     --output-dir models/language_decoder/ \
     --use-lora \
     --lora-rank 64 \
     --num-epochs 3 \
     --eval-steps 500 \
     --save-steps 1000 \
     --logging-steps 10
   ```

3. Implement constrained decoding:
   ```bash
   # Add constrained decoder to generation
   python -m sap_llm.models.implement_constrained_decoding \
     --model models/language_decoder/best_lora/ \
     --output models/language_decoder/constrained/
   ```

4. Merge LoRA weights with base model:
   ```bash
   python -m sap_llm.training.merge_lora \
     --base-model meta-llama/Llama-2-7b-hf \
     --lora-weights models/language_decoder/best_lora/ \
     --output models/language_decoder/merged/
   ```

5. Quantize to INT8:
   ```bash
   python -m sap_llm.optimization.quantize \
     --model models/language_decoder/merged/ \
     --output models/language_decoder/llama2_7b_int8/ \
     --quantization-mode int8 \
     --calibration-data data/processed/val/
   ```

6. Evaluate on test set:
   ```bash
   python -m sap_llm.training.evaluate_language_decoder \
     --model models/language_decoder/llama2_7b_int8/ \
     --test-data data/processed/test/ \
     --output results/language_decoder_evaluation.json \
     --use-constrained-decoding
   ```

7. Test with vision encoder integration:
   ```bash
   python -m sap_llm.testing.test_vision_language_integration \
     --vision-encoder models/vision_encoder/vision_encoder.onnx \
     --language-decoder models/language_decoder/llama2_7b_int8/ \
     --test-documents data/processed/test/ \
     --output results/integrated_extraction_test.json
   ```

SUCCESS CRITERIA:
✓ Extraction F1 score ≥92% on test set (180+ fields)
✓ Schema compliance ≥99% (valid JSON, required fields)
✓ Field-level accuracy ≥90% per field
✓ Generation time < 1.5s per document (A10 GPU)
✓ LoRA trainable parameters ~420M (6% of 7B)
✓ Constrained decoding enforces JSON schema 100%
✓ No hallucinations (extracted values must be in OCR text)
✓ Model size ≤4GB (INT8 quantized)

DELIVERABLES:
1. models/language_decoder/best_lora/ - LoRA weights
2. models/language_decoder/merged/ - Merged full model
3. models/language_decoder/llama2_7b_int8/ - Production model (INT8)
4. sap_llm/models/constrained_decoder.py - Constrained decoding implementation
5. results/language_decoder_evaluation.json - Test metrics
6. LANGUAGE_DECODER_TRAINING_REPORT.md - Report with:
   - Training curves (loss over time)
   - Per-field F1 scores (all 180+ fields)
   - Schema compliance rate
   - Hallucination rate analysis
   - Generation time benchmarks
   - Constrained vs unconstrained decoding comparison
   - Error analysis (hardest documents)

NOTES:
- LLaMA-2 requires HuggingFace authentication (accept license)
- LoRA training takes ~72 hours on 8x A100 GPUs (3 epochs)
- Use gradient checkpointing to reduce memory usage
- Constrained decoding adds ~100ms latency but ensures valid JSON
- Monitor generation for off-schema outputs and refine FSM
- Test with all 13 document types to ensure generalization
- Some fields are optional - handle missing fields gracefully

ESTIMATED TIME: 3-4 weeks
- Week 1: Data preparation + LoRA fine-tuning
- Week 2: Constrained decoding implementation
- Week 3: Model merging + quantization + integration testing
- Week 4: Evaluation + optimization + documentation
```

---

## TODO #5: Reasoning Engine Training (Mixtral-8x7B → SAP Routing with RLHF) 🔴 CRITICAL

**Priority**: P0 - BLOCKER  
**Estimated Time**: 4-5 weeks  
**Dependencies**: TODO #1, TODO #2, TODO #3, TODO #4  
**Status**: Not Started (0%)

### Detailed Prompt for Claude Code

```
TASK: Train Mixtral-8x7B reasoning engine for autonomous SAP API routing decisions, business rule application, and exception handling using Reinforcement Learning with Human Feedback (RLHF).

CONTEXT:
You are working on SAP_LLM at /Users/ajithkumarr/Desktop/SAP_LLM. The reasoning engine (sap_llm/models/reasoning_engine.py) loads base mistralai/Mixtral-8x7B-v0.1 which cannot make correct SAP routing decisions. You need to train it using:
1. Supervised Fine-Tuning (SFT) on routing examples
2. Reward Model training for decision quality
3. Reinforcement Learning (PPO) for optimal routing

OBJECTIVE:
Train the reasoning engine to achieve:
- ≥97% routing accuracy (correct SAP API selection)
- 100% valid API endpoint selection (no hallucinated APIs)
- ≥90% confidence calibration (predicted confidence matches actual accuracy)
- Chain-of-thought reasoning with explainability
- <800ms inference time per routing decision

REQUIREMENTS:
1. Collect 200K+ routing decisions from QorSync historical data
2. Implement 3-phase training:
   - Phase 1: Supervised Fine-Tuning on routing examples
   - Phase 2: Reward model training on human preferences
   - Phase 3: PPO-based RLHF
3. Use LoRA for parameter-efficient fine-tuning (47B is huge)
4. INT8 quantization for production
5. Validate on all 13 document types and 400+ SAP APIs

TECHNICAL SPECIFICATIONS:

1. Hardware Requirements:
   - 8x NVIDIA H100 80GB (Mixtral-8x7B requires more memory)
   - 1TB RAM
   - 20TB NVMe SSD

2. Training Configuration:
   ```yaml
   # configs/training/reasoning_engine.yaml
   model:
     base: mistralai/Mixtral-8x7B-v0.1
     parameters: 47_000_000_000  # Total (6B active per token)
     quantization: int8
     lora:
       enabled: true
       rank: 128
       alpha: 256
       dropout: 0.05
       target_modules: [q_proj, k_proj, v_proj, o_proj, w1, w2, w3]
     
   sft:  # Supervised Fine-Tuning
     batch_size: 2  # Per GPU
     gradient_accumulation_steps: 32
     effective_batch_size: 512  # 2 * 32 * 8 GPUs
     learning_rate: 1e-4
     warmup_steps: 500
     max_steps: 20000
     weight_decay: 0.01
     
   reward_model:
     batch_size: 4
     learning_rate: 1e-5
     max_steps: 5000
     pairwise_comparisons: true
     
   rlhf:  # Reinforcement Learning
     algorithm: ppo
     batch_size: 2
     gradient_accumulation_steps: 32
     learning_rate: 5e-6
     kl_penalty: 0.1  # KL divergence from SFT model
     clip_range: 0.2
     num_ppo_epochs: 4
     max_steps: 10000
     
   optimization:
     optimizer: adamw_8bit
     scheduler: cosine_with_warmup
     mixed_precision: fp16
     distributed_backend: deepspeed
     deepspeed_stage: 3  # ZeRO-3 for 47B model
   ```

3. Training Data Collection:
   ```python
   # Collect routing decisions from QorSync production
   # SQL query on PostgreSQL:
   SELECT 
       d.document_id,
       d.document_type,
       d.subtype,
       e.extracted_data,  # ADC JSON
       r.selected_api,
       r.reasoning,
       r.confidence,
       s.success,
       s.sap_response,
       s.error_message
   FROM documents d
   JOIN extractions e ON d.document_id = e.document_id
   JOIN routing_decisions r ON d.document_id = r.document_id
   JOIN sap_integration_results s ON d.document_id = s.document_id
   WHERE s.timestamp >= '2022-01-01'
   ORDER BY s.timestamp DESC
   LIMIT 200000;
   
   # Format:
   {
     "input": {
       "document_type": "PURCHASE_ORDER",
       "subtype": "STANDARD_PO",
       "adc_json": {...},  # Extracted data
       "available_apis": [
         "API_PURCHASEORDER_PROCESS_SRV",
         "API_PURCHASE_ORDER_SRV_V2",
         ...
       ],
       "similar_cases": [...]  # From PMG
     },
     "output": {
       "endpoint": "API_PURCHASEORDER_PROCESS_SRV",
       "method": "POST",
       "entity": "A_PurchaseOrder",
       "confidence": 0.98,
       "reasoning": "This is a standard purchase order with all required fields (PO number, supplier, line items). The API_PURCHASEORDER_PROCESS_SRV is the correct endpoint for creating new purchase orders in SAP S/4HANA. Similar past cases with the same supplier and document structure were successfully processed using this API."
     },
     "human_feedback": {
       "success": true,
       "sap_response_code": 201,
       "corrections": null
     }
   }
   ```

4. Chain-of-Thought Prompting:
   ```python
   def create_routing_prompt(adc_json, doc_type, available_apis, similar_cases):
       prompt = f"""<s>[INST] <<SYS>>
   You are an expert SAP integration architect. Analyze the document and determine the correct SAP API endpoint. Use chain-of-thought reasoning and explain your decision.
   <</SYS>>

   ## Task
   Route this {doc_type} document to the appropriate SAP S/4HANA OData API.

   ## Document Information
   - Type: {doc_type}
   - Supplier: {adc_json.get('supplier_name', 'Unknown')}
   - Company Code: {adc_json.get('company_code', 'Unknown')}
   - Total Amount: {adc_json.get('total_amount', 0)} {adc_json.get('currency', 'USD')}
   - Key Fields: {list(adc_json.keys())}

   ## Extracted Data (ADC)
   {json.dumps(adc_json, indent=2)[:1000]}

   ## Available SAP APIs
   {json.dumps([api['name'] for api in available_apis])}

   ## Similar Past Routing Decisions
   {json.dumps(similar_cases[:3], indent=2)[:800] if similar_cases else 'No similar cases found'}

   ## Instructions
   1. Analyze the document type and subtype
   2. Check if all required fields for SAP integration are present
   3. Consider similar past successful routing decisions
   4. Select the appropriate SAP API endpoint
   5. Explain your reasoning step-by-step
   6. Provide a confidence score (0.0-1.0)

   Output format (JSON only):
   {{
     "endpoint": "API_NAME",
     "method": "POST",
     "entity": "EntityName",
     "confidence": 0.95,
     "reasoning": "Step-by-step explanation of why this endpoint is appropriate"
   }}
   [/INST]
   """
       return prompt
   ```

5. Phase 1: Supervised Fine-Tuning (SFT):
   ```bash
   # Train on 200K routing examples
   torchrun --nproc_per_node=8 \
     -m sap_llm.training.sft_reasoning_engine \
     --config configs/training/reasoning_engine_sft.yaml \
     --base-model mistralai/Mixtral-8x7B-v0.1 \
     --train-data data/routing_training/train.jsonl \
     --val-data data/routing_training/val.jsonl \
     --output-dir models/reasoning_engine/sft/ \
     --use-lora \
     --lora-rank 128 \
     --num-epochs 2 \
     --eval-steps 500 \
     --save-steps 1000
   ```

6. Phase 2: Reward Model Training:
   ```python
   # Train reward model on pairwise comparisons
   # For each document, collect:
   # - Chosen routing (successful SAP integration)
   # - Rejected routing (failed SAP integration or suboptimal)
   
   reward_training_data = [
       {
           "prompt": routing_prompt,
           "chosen": successful_routing_json,
           "rejected": failed_routing_json,
       },
       ...
   ]
   
   # Use sap_llm/training/rlhf_trainer.py
   from sap_llm.training.rlhf_trainer import RewardModelTrainer
   
   reward_trainer = RewardModelTrainer(
       base_model="models/reasoning_engine/sft/best_lora/",
       train_data=reward_training_data,
       output_dir="models/reasoning_engine/reward_model/",
   )
   
   reward_trainer.train()
   ```

7. Phase 3: RLHF with PPO:
   ```bash
   # Reinforcement learning with Proximal Policy Optimization
   torchrun --nproc_per_node=8 \
     -m sap_llm.training.rlhf_ppo \
     --config configs/training/reasoning_engine_rlhf.yaml \
     --policy-model models/reasoning_engine/sft/best_lora/ \
     --reward-model models/reasoning_engine/reward_model/best/ \
     --output-dir models/reasoning_engine/rlhf/ \
     --num-steps 10000 \
     --kl-penalty 0.1 \
     --clip-range 0.2
   ```

8. Evaluation Metrics:
   ```python
   def evaluate_routing(predictions, ground_truth):
       metrics = {
           "routing_accuracy": 0.0,  # Correct API selected
           "api_selection_validity": 0.0,  # API exists (no hallucination)
           "confidence_calibration": 0.0,  # Predicted confidence vs actual
           "reasoning_quality": 0.0,  # Human evaluation of reasoning
           "inference_latency": 0.0,  # P95 latency
       }
       
       correct = sum(p['endpoint'] == gt['endpoint'] for p, gt in zip(predictions, ground_truth))
       metrics["routing_accuracy"] = correct / len(predictions)
       
       valid_apis = sum(p['endpoint'] in all_sap_apis for p in predictions)
       metrics["api_selection_validity"] = valid_apis / len(predictions)
       
       # Confidence calibration (Expected Calibration Error)
       confidence_buckets = {i/10: [] for i in range(11)}
       for p, gt in zip(predictions, ground_truth):
           bucket = round(p['confidence'] * 10) / 10
           confidence_buckets[bucket].append(1 if p['endpoint'] == gt['endpoint'] else 0)
       
       ece = 0
       for conf, accuracies in confidence_buckets.items():
           if accuracies:
               avg_accuracy = np.mean(accuracies)
               ece += abs(conf - avg_accuracy) * len(accuracies)
       metrics["confidence_calibration"] = 1 - (ece / len(predictions))
       
       return metrics
   ```

EXECUTION STEPS:

1. Collect routing training data:
   ```bash
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   
   # Extract routing decisions from QorSync PostgreSQL
   python -m sap_llm.data_pipeline.collect_routing_data \
     --db-uri $QORSYNC_DB_URI \
     --output data/routing_training/ \
     --min-examples-per-api 100 \
     --date-range 2022-01-01:2025-11-18
   ```

2. Phase 1 - Supervised Fine-Tuning:
   ```bash
   torchrun --nproc_per_node=8 \
     -m sap_llm.training.sft_reasoning_engine \
     --config configs/training/reasoning_engine_sft.yaml \
     --base-model mistralai/Mixtral-8x7B-v0.1 \
     --train-data data/routing_training/train.jsonl \
     --val-data data/routing_training/val.jsonl \
     --output-dir models/reasoning_engine/sft/ \
     --use-lora --lora-rank 128 --num-epochs 2
   ```

3. Phase 2 - Train Reward Model:
   ```bash
   torchrun --nproc_per_node=8 \
     -m sap_llm.training.train_reward_model \
     --config configs/training/reward_model.yaml \
     --base-model models/reasoning_engine/sft/best_lora/ \
     --train-data data/routing_training/reward_train.jsonl \
     --val-data data/routing_training/reward_val.jsonl \
     --output-dir models/reasoning_engine/reward_model/
   ```

4. Phase 3 - RLHF with PPO:
   ```bash
   torchrun --nproc_per_node=8 \
     -m sap_llm.training.rlhf_ppo \
     --config configs/training/reasoning_engine_rlhf.yaml \
     --policy-model models/reasoning_engine/sft/best_lora/ \
     --reward-model models/reasoning_engine/reward_model/best/ \
     --output-dir models/reasoning_engine/rlhf/ \
     --num-steps 10000
   ```

5. Merge LoRA + Quantize:
   ```bash
   python -m sap_llm.training.merge_lora \
     --base-model mistralai/Mixtral-8x7B-v0.1 \
     --lora-weights models/reasoning_engine/rlhf/best_lora/ \
     --output models/reasoning_engine/merged/
   
   python -m sap_llm.optimization.quantize \
     --model models/reasoning_engine/merged/ \
     --output models/reasoning_engine/mixtral_8x7b_int8/ \
     --quantization-mode int8
   ```

6. Evaluate on test set:
   ```bash
   python -m sap_llm.training.evaluate_reasoning_engine \
     --model models/reasoning_engine/mixtral_8x7b_int8/ \
     --test-data data/routing_training/test.jsonl \
     --sap-apis data/knowledge_base/api_list.json \
     --output results/reasoning_engine_evaluation.json
   ```

SUCCESS CRITERIA:
✓ Routing accuracy ≥97% on test set
✓ API selection validity 100% (no hallucinated APIs)
✓ Confidence calibration ECE ≤0.05
✓ Chain-of-thought reasoning present in 100% of outputs
✓ Inference latency P95 < 800ms
✓ No KL divergence explosion during RLHF (KL < 5.0)
✓ Reward model achieves ≥75% pairwise accuracy
✓ Model size ≤25GB (INT8 quantized)

DELIVERABLES:
1. models/reasoning_engine/sft/ - SFT model
2. models/reasoning_engine/reward_model/ - Reward model
3. models/reasoning_engine/rlhf/ - RLHF-trained model
4. models/reasoning_engine/mixtral_8x7b_int8/ - Production model
5. results/reasoning_engine_evaluation.json - Test metrics
6. REASONING_ENGINE_TRAINING_REPORT.md - Report with:
   - SFT training curves
   - Reward model pairwise accuracy
   - RLHF training curves (reward, KL divergence)
   - Routing accuracy per document type
   - Confidence calibration plot
   - Chain-of-thought reasoning examples
   - Failure analysis (incorrect routings)
   - Inference latency benchmarks

NOTES:
- Mixtral-8x7B requires HuggingFace authentication
- SFT takes ~96 hours on 8x H100 GPUs (2 epochs)
- Reward model training takes ~24 hours
- RLHF PPO takes ~120 hours (10K steps)
- Total training time: ~10 days of continuous GPU usage
- Monitor KL divergence during RLHF (should stay < 5.0)
- Use DeepSpeed ZeRO-3 to fit 47B model in GPU memory
- Reward model should be smaller (use 7B or 13B variant)
- Test with all 400+ SAP APIs to ensure coverage

ESTIMATED TIME: 4-5 weeks
- Week 1: Data collection + SFT training
- Week 2: Reward model training + validation
- Week 3-4: RLHF PPO training
- Week 5: Evaluation + quantization + documentation
```

---

## TODO #6: Populate Process Memory Graph with Historical Data 🟡 HIGH PRIORITY

**Priority**: P1 - Important for Auto-Learning  
**Estimated Time**: 2-3 weeks  
**Dependencies**: TODO #1 (some overlap), TODO #2  
**Status**: Not Started (0%)

### Detailed Prompt for Claude Code

```
TASK: Ingest 100K+ historical documents into the Process Memory Graph (PMG) to enable continuous learning, context retrieval, and similarity-based decision making.

CONTEXT:
You are working on SAP_LLM at /Users/ajithkumarr/Desktop/SAP_LLM. The PMG infrastructure (Cosmos DB Gremlin + FAISS) is ready, but the graph is EMPTY. You need to populate it with historical documents, routing decisions, exceptions, and their relationships to enable auto-learning capabilities.

OBJECTIVE:
1. Ingest 100K+ documents from QorSync production database
2. Generate 768-dim embeddings for all documents
3. Build graph relationships (Document → Extraction → Routing → SAP Response)
4. Create FAISS index for fast similarity search
5. Validate PMG queries (<100ms latency)

REQUIREMENTS:
1. Extract documents from QorSync PostgreSQL (last 2 years)
2. Generate embeddings using sentence-transformers
3. Store in Cosmos DB Gremlin as graph
4. Build FAISS HNSW index
5. Implement Merkle tree versioning
6. Validate graph structure and queries

TECHNICAL SPECIFICATIONS:

1. Data Sources:
   - QorSync PostgreSQL:
     * documents table: document_id, type, image_path, ocr_text
     * extractions table: extracted_fields, confidence
     * routing_decisions table: selected_api, reasoning
     * sap_integration_results table: success, response_code, error
     * exceptions table: exception_type, severity, resolution
   
2. Graph Schema (7 vertex types):
   ```python
   # Vertex Types
   - Document: id, type, subtype, ocr_text, embedding[768]
   - Extraction: id, fields_json, confidence, timestamp
   - Rule: id, condition, action, confidence, created_date
   - Exception: id, category, severity, message, resolution
   - RoutingDecision: id, api_endpoint, reasoning, confidence
   - SAPResponse: id, status_code, success, response_json
   - Feedback: id, source, rating, comment, timestamp
   
   # Edge Types
   - Document --hasExtraction--> Extraction
   - Extraction --triggersRouting--> RoutingDecision
   - RoutingDecision --sendsToSAP--> SAPResponse
   - Document --causesException--> Exception
   - Exception --resolvedBy--> Rule
   - Document --similarTo--> Document (weight = cosine similarity)
   - Document --hasFeedback--> Feedback
   ```

3. Embedding Generation:
   ```python
   from sentence_transformers import SentenceTransformer
   
   model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
   # Generates 768-dim embeddings
   
   def generate_document_embedding(doc):
       # Combine multiple fields for rich embedding
       text = f"""
       Document Type: {doc['type']}
       Subtype: {doc['subtype']}
       Supplier: {doc.get('supplier_name', '')}
       Key Fields: {', '.join(doc.get('extracted_fields', {}).keys())}
       OCR Text: {doc['ocr_text'][:500]}
       """
       embedding = model.encode(text, normalize_embeddings=True)
       return embedding.tolist()  # 768-dim
   ```

4. Cosmos DB Gremlin Ingestion:
   ```python
   from sap_llm.pmg.graph_client import ProcessMemoryGraph
   
   pmg = ProcessMemoryGraph(config)
   
   async def ingest_document(doc):
       # Create Document vertex
       doc_vertex = await pmg.add_vertex(
           label="Document",
           properties={
               "id": doc["document_id"],
               "type": doc["type"],
               "subtype": doc["subtype"],
               "ocr_text": doc["ocr_text"][:1000],  # Truncate
               "embedding": doc["embedding"],  # 768-dim
               "timestamp": doc["created_date"],
           }
       )
       
       # Create Extraction vertex
       extraction_vertex = await pmg.add_vertex(
           label="Extraction",
           properties={
               "id": doc["extraction_id"],
               "fields_json": json.dumps(doc["extracted_fields"]),
               "confidence": doc["extraction_confidence"],
               "timestamp": doc["extraction_timestamp"],
           }
       )
       
       # Create edge
       await pmg.add_edge(
           from_vertex=doc_vertex,
           to_vertex=extraction_vertex,
           label="hasExtraction",
       )
       
       # Similar for RoutingDecision, SAPResponse, Exception, etc.
   ```

5. FAISS Index Building:
   ```python
   import faiss
   import numpy as np
   
   # Collect all embeddings
   embeddings = []  # List of 768-dim vectors
   doc_ids = []
   
   for doc in documents:
       embeddings.append(doc["embedding"])
       doc_ids.append(doc["document_id"])
   
   embeddings_np = np.array(embeddings).astype('float32')
   
   # Build HNSW index
   dimension = 768
   index = faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
   index.hnsw.efConstruction = 40
   index.hnsw.efSearch = 16
   
   index.add(embeddings_np)
   
   # Save index
   faiss.write_index(index, "data/pmg/faiss_index.bin")
   
   # Save doc_id mapping
   with open("data/pmg/doc_id_mapping.json", "w") as f:
       json.dump({"ids": doc_ids}, f)
   ```

6. Merkle Tree Versioning:
   ```python
   from sap_llm.pmg.merkle_tree import MerkleTree
   
   merkle = MerkleTree()
   
   # Add each document as a leaf
   for doc in documents:
       leaf_data = f"{doc['document_id']}:{doc['type']}:{doc['timestamp']}"
       merkle.add_leaf(leaf_data)
   
   # Build tree
   merkle.build_tree()
   
   # Get root hash (snapshot)
   root_hash = merkle.get_root_hash()
   
   # Store in Cosmos DB
   await pmg.add_vertex(
       label="PMGSnapshot",
       properties={
           "id": f"snapshot_{datetime.now().isoformat()}",
           "root_hash": root_hash,
           "document_count": len(documents),
           "timestamp": datetime.now().isoformat(),
       }
   )
   ```

7. Similarity Search:
   ```python
   def find_similar_documents(query_embedding, top_k=5):
       query_np = np.array([query_embedding]).astype('float32')
       
       # Search FAISS index
       distances, indices = index.search(query_np, top_k)
       
       # Get document IDs
       similar_doc_ids = [doc_ids[idx] for idx in indices[0]]
       similarities = [1 - dist for dist in distances[0]]  # Convert distance to similarity
       
       return list(zip(similar_doc_ids, similarities))
   ```

EXECUTION STEPS:

1. Set up environment:
   ```bash
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   
   export COSMOS_ENDPOINT="https://your-cosmos.documents.azure.com:443/"
   export COSMOS_KEY="your_cosmos_key"
   export QORSYNC_DB_URI="postgresql://user:password@host:5432/qorsync"
   ```

2. Extract data from QorSync:
   ```bash
   python -m sap_llm.pmg.extract_historical_data \
     --db-uri $QORSYNC_DB_URI \
     --output data/pmg/historical_data/ \
     --date-range 2022-01-01:2025-11-18 \
     --min-confidence 0.7
   ```

3. Generate embeddings:
   ```bash
   python -m sap_llm.pmg.generate_embeddings \
     --input data/pmg/historical_data/ \
     --output data/pmg/embeddings/ \
     --model sentence-transformers/all-mpnet-base-v2 \
     --batch-size 64
   ```

4. Ingest into Cosmos DB:
   ```bash
   python -m sap_llm.pmg.ingest_to_cosmos \
     --input data/pmg/embeddings/ \
     --cosmos-endpoint $COSMOS_ENDPOINT \
     --cosmos-key $COSMOS_KEY \
     --database sap_llm_pmg \
     --graph pmg \
     --batch-size 100
   ```

5. Build FAISS index:
   ```bash
   python -m sap_llm.pmg.build_faiss_index \
     --embeddings data/pmg/embeddings/ \
     --output data/pmg/faiss_index.bin \
     --index-type hnsw \
     --m-param 32
   ```

6. Create Merkle tree snapshot:
   ```bash
   python -m sap_llm.pmg.create_merkle_snapshot \
     --cosmos-endpoint $COSMOS_ENDPOINT \
     --cosmos-key $COSMOS_KEY \
     --database sap_llm_pmg \
     --output data/pmg/merkle_snapshot.json
   ```

7. Validate PMG:
   ```bash
   python -m sap_llm.pmg.validate \
     --cosmos-endpoint $COSMOS_ENDPOINT \
     --cosmos-key $COSMOS_KEY \
     --faiss-index data/pmg/faiss_index.bin \
     --output results/pmg_validation.json
   ```

SUCCESS CRITERIA:
✓ 100K+ documents ingested into Cosmos DB Gremlin
✓ All documents have 768-dim embeddings
✓ Graph relationships created (7 vertex types, 9 edge types)
✓ FAISS index built and saved
✓ Merkle tree snapshot created
✓ Similarity search P95 latency < 100ms
✓ Context retrieval returns relevant documents (P@5 > 85%)
✓ Graph traversal queries complete in < 200ms
✓ No orphan vertices (all connected to graph)

DELIVERABLES:
1. Cosmos DB populated with 100K+ documents
2. data/pmg/faiss_index.bin - FAISS index
3. data/pmg/merkle_snapshot.json - Merkle tree snapshot
4. results/pmg_validation.json - Validation results
5. PMG_INGESTION_REPORT.md - Report with:
   - Documents ingested per document type
   - Graph statistics (vertices, edges, density)
   - Embedding quality metrics
   - Similarity search validation
   - Query latency benchmarks
   - Known issues and resolutions

NOTES:
- Cosmos DB Gremlin has cost implications (RU/s consumption)
- Use batch operations to minimize RU/s consumption
- Embedding generation takes ~24 hours for 100K documents on CPU
- Use GPU for faster embedding generation (Tesla T4: ~4 hours)
- FAISS index requires ~300MB for 100K 768-dim vectors
- Merkle tree snapshot allows versioning and rollback

ESTIMATED TIME: 2-3 weeks
- Week 1: Data extraction + embedding generation
- Week 2: Cosmos DB ingestion + FAISS index building
- Week 3: Validation + Merkle tree + documentation
```

---

## TODO #7: End-to-End Accuracy Validation on Hold-Out Test Set 🟡 HIGH PRIORITY

**Priority**: P1 - Critical for Production Confidence  
**Estimated Time**: 2-3 weeks  
**Dependencies**: TODO #3, TODO #4, TODO #5 (all models trained)  
**Status**: Not Started (0%)

### Detailed Prompt for Claude Code

```
TASK: Perform comprehensive end-to-end accuracy validation of the complete SAP_LLM pipeline on a hold-out test set to measure real-world performance against targets.

CONTEXT:
You are working on SAP_LLM at /Users/ajithkumarr/Desktop/SAP_LLM. All 3 models (Vision Encoder, Language Decoder, Reasoning Engine) have been trained. Now you need to validate the COMPLETE pipeline (8 stages) on a hold-out test set to measure actual accuracy and identify failure modes.

OBJECTIVE:
Validate against targets:
- ≥95% document classification accuracy
- ≥92% field extraction F1 score
- ≥97% routing accuracy
- ≥85% touchless rate (no human intervention)
- P95 latency ≤1.5s end-to-end
- Cost per document <$0.005

REQUIREMENTS:
1. Use hold-out test set (150K documents from TODO #1)
2. Run complete 8-stage pipeline
3. Measure all accuracy metrics
4. Perform error analysis and failure mode identification
5. Generate comprehensive validation report
6. Create confusion matrices and calibration plots
7. Benchmark latency and cost

TECHNICAL SPECIFICATIONS:

1. Test Set:
   - Location: data/processed/test/
   - Size: 150,000 documents
   - Distribution:
     * 13 document types (balanced ~11.5K each)
     * 5 languages (EN, DE, FR, ES, IT)
     * Multiple suppliers, company codes, currencies
   - Ground truth:
     * Document type and subtype
     * 180+ extracted fields
     * Correct SAP API endpoint

2. Metrics to Measure:
   ```python
   metrics = {
       # Classification (Stage 3)
       "classification_accuracy": 0.0,  # Target: ≥95%
       "classification_f1_macro": 0.0,
       "classification_confusion_matrix": [],
       
       # Subtype Classification (Stage 4)
       "subtype_accuracy": 0.0,  # Target: ≥92%
       "subtype_f1_macro": 0.0,
       
       # Extraction (Stage 5)
       "extraction_f1_overall": 0.0,  # Target: ≥92%
       "extraction_f1_per_field": {},  # All 180+ fields
       "field_completeness": 0.0,  # % required fields extracted
       
       # Quality Check (Stage 6)
       "overall_confidence_avg": 0.0,
       "confidence_calibration_ece": 0.0,  # Expected Calibration Error
       
       # Validation (Stage 7)
       "business_rules_passed": 0.0,  # % passing validation
       "tolerance_check_passed": 0.0,
       
       # Routing (Stage 8)
       "routing_accuracy": 0.0,  # Target: ≥97%
       "api_selection_validity": 0.0,  # No hallucinated APIs
       
       # End-to-End
       "touchless_rate": 0.0,  # Target: ≥85% (no exceptions)
       "end_to_end_latency_p50": 0.0,
       "end_to_end_latency_p95": 0.0,  # Target: ≤1.5s
       "cost_per_document": 0.0,  # Target: <$0.005
   }
   ```

3. Test Harness:
   ```python
   from sap_llm.models import UnifiedExtractorModel
   from sap_llm.stages import *
   from sap_llm.utils.metrics import compute_all_metrics
   
   async def evaluate_document(doc_path, ground_truth):
       # Load document
       image = Image.open(doc_path)
       
       # Stage 1: Inbox
       inbox_result = inbox_stage.process(image)
       
       # Stage 2: Preprocessing
       preprocess_result = preprocessing_stage.process(inbox_result)
       
       # Stage 3: Classification
       classification_result = classification_stage.process(preprocess_result)
       
       # Stage 4: Type Identifier
       type_id_result = type_identifier_stage.process(classification_result)
       
       # Stage 5: Extraction
       extraction_result = extraction_stage.process(type_id_result)
       
       # Stage 6: Quality Check
       quality_result = quality_check_stage.process(extraction_result)
       
       # Stage 7: Validation
       validation_result = validation_stage.process(quality_result)
       
       # Stage 8: Routing
       routing_result = routing_stage.process(validation_result)
       
       # Compare with ground truth
       results = {
           "document_id": ground_truth["id"],
           "classification_correct": classification_result["type"] == ground_truth["type"],
           "subtype_correct": type_id_result["subtype"] == ground_truth["subtype"],
           "extracted_fields": extraction_result["extracted_data"],
           "routing_correct": routing_result["endpoint"] == ground_truth["api"],
           "exceptions": validation_result.get("exceptions", []),
           "latency": routing_result["total_latency"],
       }
       
       return results
   
   async def evaluate_test_set(test_dir):
       results = []
       
       for doc_file in tqdm(os.listdir(test_dir)):
           doc_path = os.path.join(test_dir, doc_file)
           ground_truth = load_ground_truth(doc_file)
           
           result = await evaluate_document(doc_path, ground_truth)
           results.append(result)
       
       # Compute metrics
       metrics = compute_all_metrics(results)
       return metrics
   ```

4. Error Analysis:
   ```python
   def analyze_errors(results):
       # Find all incorrectly classified documents
       misclassified = [r for r in results if not r["classification_correct"]]
       
       # Analyze by document type
       errors_by_type = {}
       for r in misclassified:
           doc_type = r["ground_truth"]["type"]
           if doc_type not in errors_by_type:
               errors_by_type[doc_type] = []
           errors_by_type[doc_type].append(r)
       
       # Find hardest documents (low confidence + incorrect)
       hardest = sorted(
           [r for r in results if not r["classification_correct"]],
           key=lambda x: x.get("confidence", 1.0)
       )[:100]
       
       return {
           "total_errors": len(misclassified),
           "errors_by_type": errors_by_type,
           "hardest_documents": hardest,
       }
   ```

5. Visualization:
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
   
   def create_visualizations(results, metrics):
       # Confusion matrix
       y_true = [r["ground_truth"]["type"] for r in results]
       y_pred = [r["predicted_type"] for r in results]
       cm = confusion_matrix(y_true, y_pred)
       ConfusionMatrixDisplay(cm, display_labels=DOCUMENT_TYPES).plot()
       plt.savefig("results/confusion_matrix.png")
       
       # Calibration plot
       confidence_bins = np.linspace(0, 1, 11)
       bin_accs = []
       bin_confs = []
       for i in range(len(confidence_bins) - 1):
           bin_results = [r for r in results if confidence_bins[i] <= r["confidence"] < confidence_bins[i+1]]
           if bin_results:
               bin_accs.append(np.mean([r["classification_correct"] for r in bin_results]))
               bin_confs.append(np.mean([r["confidence"] for r in bin_results]))
       
       plt.figure()
       plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
       plt.plot(bin_confs, bin_accs, 'o-', label='Model Calibration')
       plt.xlabel('Confidence')
       plt.ylabel('Accuracy')
       plt.legend()
       plt.savefig("results/calibration_plot.png")
       
       # Latency distribution
       plt.figure()
       latencies = [r["latency"] for r in results]
       plt.hist(latencies, bins=50)
       plt.axvline(np.percentile(latencies, 95), color='r', linestyle='--', label='P95')
       plt.xlabel('Latency (seconds)')
       plt.ylabel('Count')
       plt.legend()
       plt.savefig("results/latency_distribution.png")
   ```

EXECUTION STEPS:

1. Prepare test environment:
   ```bash
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   
   # Verify models are trained
   ls -lh models/vision_encoder/vision_encoder.onnx
   ls -lh models/language_decoder/llama2_7b_int8/
   ls -lh models/reasoning_engine/mixtral_8x7b_int8/
   
   # Verify test data
   ls -lh data/processed/test/ | wc -l  # Should show ~150K files
   ```

2. Run end-to-end validation:
   ```bash
   python -m sap_llm.testing.validate_end_to_end \
     --test-data data/processed/test/ \
     --vision-encoder models/vision_encoder/vision_encoder.onnx \
     --language-decoder models/language_decoder/llama2_7b_int8/ \
     --reasoning-engine models/reasoning_engine/mixtral_8x7b_int8/ \
     --output results/end_to_end_validation.json \
     --num-workers 8 \
     --batch-size 32
   ```

3. Generate detailed metrics:
   ```bash
   python -m sap_llm.testing.compute_metrics \
     --results results/end_to_end_validation.json \
     --output results/detailed_metrics.json \
     --generate-visualizations
   ```

4. Perform error analysis:
   ```bash
   python -m sap_llm.testing.analyze_errors \
     --results results/end_to_end_validation.json \
     --output results/error_analysis.json \
     --export-hardest-examples results/hardest_examples/
   ```

5. Benchmark latency:
   ```bash
   python -m sap_llm.testing.benchmark_latency \
     --test-data data/processed/test/ \
     --num-samples 1000 \
     --output results/latency_benchmarks.json
   ```

6. Calculate cost per document:
   ```bash
   python -m sap_llm.testing.calculate_costs \
     --results results/end_to_end_validation.json \
     --gpu-hourly-rate 3.0 \
     --output results/cost_analysis.json
   ```

SUCCESS CRITERIA:
✓ Classification accuracy ≥95% on 150K test documents
✓ Extraction F1 score ≥92% across all 180+ fields
✓ Routing accuracy ≥97%
✓ Touchless rate ≥85% (documents with no exceptions)
✓ End-to-end latency P95 ≤1.5s
✓ Cost per document <$0.005
✓ Confusion matrix generated for all document types
✓ Calibration plot shows ECE <0.05
✓ Error analysis identifies top failure modes
✓ All metrics documented and validated

DELIVERABLES:
1. results/end_to_end_validation.json - Complete results
2. results/detailed_metrics.json - All accuracy metrics
3. results/confusion_matrix.png - Classification confusion matrix
4. results/calibration_plot.png - Confidence calibration
5. results/latency_distribution.png - Latency histogram
6. results/error_analysis.json - Failure mode analysis
7. results/hardest_examples/ - Top 100 hardest documents
8. results/cost_analysis.json - Cost breakdown
9. END_TO_END_VALIDATION_REPORT.md - Comprehensive report with:
   - Executive summary (pass/fail for each target)
   - Accuracy metrics per document type
   - Per-field extraction performance
   - Latency breakdown by stage
   - Cost analysis and projections
   - Top 10 failure modes with examples
   - Recommendations for improvement
   - Production readiness assessment

NOTES:
- Validation takes ~24 hours on 2x A10 GPUs for 150K documents
- Use parallel processing to speed up (8 workers)
- Monitor GPU memory usage during validation
- Save intermediate results every 10K documents (checkpoint)
- Generate visualizations after completion
- Compare results with targets - document any gaps
- If targets not met, identify specific areas for improvement

ESTIMATED TIME: 2-3 weeks
- Week 1: Run validation + initial metrics
- Week 2: Error analysis + visualization
- Week 3: Cost analysis + documentation + recommendations
```

---

## TODO #8: Load Testing & Performance Optimization 🟡 HIGH PRIORITY

**Priority**: P1 - Critical for Production Scaling  
**Estimated Time**: 1-2 weeks  
**Dependencies**: TODO #7 (validation complete)  
**Status**: Not Started (0%)

### Detailed Prompt for Claude Code

```
TASK: Perform comprehensive load testing of the SAP_LLM system and optimize performance to achieve production throughput targets (5K docs/hour per GPU).

CONTEXT:
You are working on SAP_LLM at /Users/ajithkumarr/Desktop/SAP_LLM. The models are trained and validated. Now you need to test the system under production load, identify bottlenecks, and optimize for maximum throughput while maintaining accuracy.

OBJECTIVE:
Achieve production targets:
- 5,000+ documents/hour per GPU node
- P95 latency ≤1.5s end-to-end
- 95%+ GPU utilization
- <2GB memory per document
- Auto-scaling from 1-10 nodes
- Zero downtime during deployments

REQUIREMENTS:
1. Set up load testing infrastructure (Locust or K6)
2. Test with realistic document distribution
3. Measure throughput, latency, resource utilization
4. Identify bottlenecks in all 8 stages
5. Optimize critical paths
6. Implement caching strategies
7. Test auto-scaling behavior
8. Validate under sustained load (24+ hours)

TECHNICAL SPECIFICATIONS:

1. Load Testing Setup:
   ```python
   # Use Locust for load testing
   from locust import HttpUser, task, between
   import random
   
   class SAPLLMUser(HttpUser):
       wait_time = between(1, 3)
       
       @task(10)
       def process_purchase_order(self):
           # Upload and process PO
           with open(f"test_documents/po_{random.randint(1, 1000)}.pdf", "rb") as f:
               self.client.post("/upload-document", files={"file": f})
       
       @task(8)
       def process_invoice(self):
           # Upload and process invoice
           with open(f"test_documents/invoice_{random.randint(1, 1000)}.pdf", "rb") as f:
               self.client.post("/upload-document", files={"file": f})
       
       @task(2)
       def check_status(self):
           # Check job status
           job_id = random.choice(self.job_ids)
           self.client.get(f"/jobs/{job_id}")
   ```

2. Load Test Scenarios:
   ```yaml
   # configs/load_tests/scenarios.yaml
   scenarios:
     # Scenario 1: Normal Load
     normal_load:
       users: 50
       spawn_rate: 5
       duration: 3600  # 1 hour
       document_distribution:
         PURCHASE_ORDER: 0.40
         SUPPLIER_INVOICE: 0.35
         GOODS_RECEIPT: 0.15
         OTHER: 0.10
     
     # Scenario 2: Peak Load
     peak_load:
       users: 200
       spawn_rate: 20
       duration: 1800  # 30 minutes
       document_distribution:
         PURCHASE_ORDER: 0.40
         SUPPLIER_INVOICE: 0.35
         GOODS_RECEIPT: 0.15
         OTHER: 0.10
     
     # Scenario 3: Sustained Load
     sustained_load:
       users: 100
       spawn_rate: 10
       duration: 86400  # 24 hours
       document_distribution:
         PURCHASE_ORDER: 0.40
         SUPPLIER_INVOICE: 0.35
         GOODS_RECEIPT: 0.15
         OTHER: 0.10
     
     # Scenario 4: Burst Load
     burst_load:
       users: 500
       spawn_rate: 100
       duration: 300  # 5 minutes
       document_distribution:
         PURCHASE_ORDER: 0.40
         SUPPLIER_INVOICE: 0.35
         GOODS_RECEIPT: 0.15
         OTHER: 0.10
   ```

3. Metrics to Collect:
   ```python
   metrics = {
       # Throughput
       "documents_per_hour": 0,
       "documents_per_second": 0,
       "peak_throughput": 0,
       
       # Latency
       "latency_p50": 0.0,
       "latency_p95": 0.0,
       "latency_p99": 0.0,
       "latency_per_stage": {},
       
       # Resource Utilization
       "gpu_utilization_avg": 0.0,
       "gpu_memory_usage_avg": 0.0,
       "cpu_utilization_avg": 0.0,
       "memory_usage_avg": 0.0,
       
       # Errors
       "error_rate": 0.0,
       "timeout_rate": 0.0,
       "retry_rate": 0.0,
       
       # Auto-Scaling
       "scale_up_events": 0,
       "scale_down_events": 0,
       "scale_up_latency": 0.0,
       "scale_down_latency": 0.0,
   }
   ```

4. Performance Optimization Strategies:
   
   a) Batch Processing:
   ```python
   # Implement batching in preprocessing stage
   class BatchProcessor:
       def __init__(self, batch_size=32, max_wait_time=0.1):
           self.batch_size = batch_size
           self.max_wait_time = max_wait_time
           self.queue = []
       
       async def process_batch(self, documents):
           # Process multiple documents in parallel
           results = await asyncio.gather(*[
               self.process_single(doc) for doc in documents
           ])
           return results
   ```
   
   b) Caching Strategy:
   ```python
   # Implement multi-tier caching
   class CacheManager:
       def __init__(self):
           self.l1_cache = {}  # In-memory (LRU, 1GB)
           self.l2_cache = redis_client  # Redis (100GB)
       
       def get(self, key):
           # Check L1 first
           if key in self.l1_cache:
               return self.l1_cache[key]
           
           # Check L2
           value = self.l2_cache.get(key)
           if value:
               self.l1_cache[key] = value  # Promote to L1
           
           return value
   ```
   
   c) Model Optimization:
   ```bash
   # Use TensorRT for inference acceleration
   python -m sap_llm.optimization.tensorrt_optimize \
     --model models/vision_encoder/vision_encoder.onnx \
     --output models/vision_encoder/vision_encoder.trt \
     --precision fp16
   
   # Enable dynamic batching
   python -m sap_llm.optimization.enable_dynamic_batching \
     --max-batch-size 32 \
     --max-wait-time 100  # ms
   ```
   
   d) Database Optimization:
   ```python
   # Optimize PMG queries
   # - Use batch queries instead of individual
   # - Implement connection pooling
   # - Add indices on frequently queried fields
   
   # Cosmos DB optimization
   await pmg.create_index("Document", ["type", "timestamp"])
   await pmg.create_index("Extraction", ["confidence"])
   ```

5. Auto-Scaling Configuration:
   ```yaml
   # deployments/kubernetes/hpa.yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: sap-llm-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: sap-llm-api
     minReplicas: 1
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
     - type: Resource
       resource:
         name: memory
         target:
           type: Utilization
           averageUtilization: 80
     - type: Pods
       pods:
         metric:
           name: documents_per_second
         target:
           type: AverageValue
           averageValue: "1.5"  # 5400 docs/hour / 3600 = 1.5/sec
     behavior:
       scaleUp:
         stabilizationWindowSeconds: 60
         policies:
         - type: Percent
           value: 100
           periodSeconds: 60
       scaleDown:
         stabilizationWindowSeconds: 300
         policies:
         - type: Pods
           value: 1
           periodSeconds: 60
   ```

EXECUTION STEPS:

1. Set up load testing environment:
   ```bash
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   
   # Install Locust
   pip install locust
   
   # Prepare test documents (1000 samples per type)
   python -m sap_llm.testing.prepare_load_test_data \
     --output test_documents/ \
     --num-samples 1000 \
     --document-types all
   ```

2. Run baseline load test:
   ```bash
   # Start SAP_LLM API (1 node, 2x A10 GPUs)
   kubectl scale deployment sap-llm-api --replicas=1 -n sap-llm
   
   # Run normal load test
   locust -f tests/load/locust_file.py \
     --host http://sap-llm-api.sap-llm.svc.cluster.local:8000 \
     --users 50 \
     --spawn-rate 5 \
     --run-time 1h \
     --html results/load_test_baseline.html
   ```

3. Profile for bottlenecks:
   ```bash
   # Use py-spy for CPU profiling
   py-spy record -o results/profile.svg --pid $(pgrep -f "uvicorn sap_llm.api.server")
   
   # Use NVIDIA Nsight for GPU profiling
   nsys profile -o results/gpu_profile \
     python -m sap_llm.api.server
   ```

4. Apply optimizations:
   ```bash
   # Enable TensorRT
   python -m sap_llm.optimization.apply_optimizations \
     --enable-tensorrt \
     --enable-dynamic-batching \
     --enable-caching \
     --cache-size 10GB
   ```

5. Run peak load test:
   ```bash
   locust -f tests/load/locust_file.py \
     --host http://sap-llm-api.sap-llm.svc.cluster.local:8000 \
     --users 200 \
     --spawn-rate 20 \
     --run-time 30m \
     --html results/load_test_peak.html
   ```

6. Test auto-scaling:
   ```bash
   # Enable HPA
   kubectl apply -f deployments/kubernetes/hpa.yaml -n sap-llm
   
   # Run burst load test
   locust -f tests/load/locust_file.py \
     --host http://sap-llm-api.sap-llm.svc.cluster.local:8000 \
     --users 500 \
     --spawn-rate 100 \
     --run-time 5m \
     --html results/load_test_burst.html
   
   # Monitor scaling
   watch kubectl get hpa sap-llm-hpa -n sap-llm
   ```

7. Run sustained load test (24 hours):
   ```bash
   locust -f tests/load/locust_file.py \
     --host http://sap-llm-api.sap-llm.svc.cluster.local:8000 \
     --users 100 \
     --spawn-rate 10 \
     --run-time 24h \
     --html results/load_test_sustained.html
   ```

SUCCESS CRITERIA:
✓ Throughput ≥5,000 documents/hour per GPU node
✓ P95 latency ≤1.5s under normal load
✓ P95 latency ≤2.0s under peak load
✓ GPU utilization ≥95% under load
✓ Memory usage <2GB per document
✓ Error rate <0.1% under normal load
✓ Error rate <1% under peak load
✓ Auto-scaling works correctly (1-10 nodes)
✓ Scale-up latency <2 minutes
✓ No performance degradation over 24 hours
✓ Zero downtime during rolling updates

DELIVERABLES:
1. results/load_test_baseline.html - Baseline performance
2. results/load_test_peak.html - Peak load results
3. results/load_test_sustained.html - 24-hour test results
4. results/profile.svg - CPU profiling flamegraph
5. results/gpu_profile.qdrep - GPU profiling report
6. results/optimization_comparison.json - Before/after metrics
7. LOAD_TESTING_REPORT.md - Report with:
   - Throughput analysis (docs/hour vs target)
   - Latency breakdown by stage
   - Resource utilization graphs
   - Bottleneck identification
   - Optimization results (before/after)
   - Auto-scaling behavior analysis
   - Sustained load stability
   - Recommendations for production

NOTES:
- Load testing requires production-like environment (Kubernetes cluster)
- Use staging environment to avoid impacting production
- Monitor costs during testing (GPU hours can be expensive)
- TensorRT optimization provides 2-3x speedup
- Dynamic batching improves throughput by 30-40%
- Caching reduces duplicate processing by 20-30%
- Auto-scaling should handle 10x traffic spikes

ESTIMATED TIME: 1-2 weeks
- Week 1: Baseline testing + Optimization + Peak testing
- Week 2: Auto-scaling validation + Sustained testing + Documentation
```

---

## TODO #9: Web Search API Configuration & Integration Testing 🟢 NICE-TO-HAVE

**Priority**: P2 - Enhances Features  
**Estimated Time**: 3-5 days  
**Dependencies**: None (can run in parallel)  
**Status**: Not Started (0%)

### Detailed Prompt for Claude Code

```
TASK: Configure all web search provider API keys and test the complete web search integration for entity enrichment and knowledge base updates.

CONTEXT:
You are working on SAP_LLM at /Users/ajithkumarr/Desktop/SAP_LLM. The web search infrastructure (sap_llm/web_search/*.py) is complete but not configured with API keys. You need to set up all 4 providers (Tavily, Google, Bing, DuckDuckGo) and test the complete integration.

OBJECTIVE:
1. Obtain API keys for all search providers
2. Configure environment variables
3. Test each provider independently
4. Test failover behavior
5. Validate entity enrichment
6. Measure cache hit rates
7. Test cost optimization
8. Integrate with pipeline

REQUIREMENTS:
1. Sign up for API keys (Tavily AI, Google Custom Search, Bing Search)
2. Configure in .env file
3. Test search functionality for each provider
4. Validate failover when provider fails
5. Test entity enrichment (vendors, products, tax rates)
6. Measure performance and cost
7. Integrate with extraction stage
8. Validate end-to-end workflow

TECHNICAL SPECIFICATIONS:

1. API Key Setup:
   
   a) Tavily AI (Primary):
   - Sign up: https://tavily.com
   - Free tier: 1,000 requests/month
   - Paid: $0.001 per request
   - Best for: Real-time web search with AI-powered results
   
   b) Google Custom Search (Secondary):
   - Sign up: https://developers.google.com/custom-search
   - Free tier: 100 requests/day
   - Paid: $5 per 1,000 requests
   - Best for: Reliable, high-quality results
   
   c) Bing Search API (Tertiary):
   - Sign up: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
   - Free tier: 3 requests/second, 1,000 requests/month
   - Paid: $5 per 1,000 requests
   - Best for: Fallback when others fail
   
   d) DuckDuckGo (Fallback - Free):
   - No API key required
   - Rate limited: ~30 requests/minute
   - Best for: Offline mode and emergency fallback

2. Environment Configuration:
   ```bash
   # Add to .env file
   TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxxxxxxxxx
   GOOGLE_SEARCH_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxx
   GOOGLE_SEARCH_CX=xxxxxxxxxxxxxxxxxxxx
   BING_SEARCH_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   
   # Web search config
   WEB_SEARCH_ENABLED=true
   WEB_SEARCH_OFFLINE_MODE=false
   WEB_SEARCH_CACHE_TTL=86400  # 24 hours
   WEB_SEARCH_MAX_RESULTS=5
   ```

3. Provider Testing:
   ```python
   from sap_llm.web_search import WebSearchEngine
   
   # Initialize search engine
   search_engine = WebSearchEngine(config)
   
   # Test Tavily
   results = await search_engine.search(
       query="Acme Corporation headquarters",
       mode=SearchMode.WEB,
       provider="tavily"
   )
   assert len(results) > 0
   print(f"Tavily: {len(results)} results")
   
   # Test Google
   results = await search_engine.search(
       query="Acme Corporation headquarters",
       mode=SearchMode.WEB,
       provider="google"
   )
   assert len(results) > 0
   print(f"Google: {len(results)} results")
   
   # Test Bing
   results = await search_engine.search(
       query="Acme Corporation headquarters",
       mode=SearchMode.WEB,
       provider="bing"
   )
   assert len(results) > 0
   print(f"Bing: {len(results)} results")
   
   # Test DuckDuckGo
   results = await search_engine.search(
       query="Acme Corporation headquarters",
       mode=SearchMode.WEB,
       provider="duckduckgo"
   )
   assert len(results) > 0
   print(f"DuckDuckGo: {len(results)} results")
   ```

4. Failover Testing:
   ```python
   # Test automatic failover
   # Simulate Tavily failure by using invalid key
   original_key = os.environ["TAVILY_API_KEY"]
   os.environ["TAVILY_API_KEY"] = "invalid_key"
   
   results = await search_engine.search(
       query="Test query",
       mode=SearchMode.WEB
   )
   
   # Should automatically fall back to Google
   assert len(results) > 0
   assert search_engine.last_used_provider == "google"
   
   # Restore key
   os.environ["TAVILY_API_KEY"] = original_key
   ```

5. Entity Enrichment Testing:
   ```python
   from sap_llm.web_search import EntityEnrichment
   
   enrichment = EntityEnrichment(search_engine)
   
   # Test vendor enrichment
   vendor_info = await enrichment.enrich_entity(
       entity_name="Acme Corporation",
       entity_type="vendor",
       context="Supplier for purchase order"
   )
   
   assert "address" in vendor_info
   assert "tax_id" in vendor_info or "vat_number" in vendor_info
   print(f"Vendor info: {vendor_info}")
   
   # Test product enrichment
   product_info = await enrichment.enrich_entity(
       entity_name="Dell Latitude 7420",
       entity_type="product",
       context="Laptop computer"
   )
   
   assert "description" in product_info
   assert "category" in product_info
   print(f"Product info: {product_info}")
   
   # Test tax rate validation
   tax_rate = await enrichment.validate_tax_rate(
       country="US",
       state="California",
       product_category="Electronics"
   )
   
   assert 0.0 <= tax_rate <= 0.15
   print(f"Tax rate for CA Electronics: {tax_rate}%")
   ```

6. Cache Testing:
   ```python
   # Test cache functionality
   query = "Microsoft Corporation headquarters"
   
   # First request (cache miss)
   start_time = time.time()
   results1 = await search_engine.search(query)
   first_request_time = time.time() - start_time
   
   # Second request (cache hit)
   start_time = time.time()
   results2 = await search_engine.search(query)
   second_request_time = time.time() - start_time
   
   assert results1 == results2
   assert second_request_time < first_request_time / 10  # Should be 10x faster
   
   # Check cache stats
   stats = search_engine.get_stats()
   assert stats["cache_hit_rate"] > 0.0
   print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
   ```

7. Integration with Pipeline:
   ```python
   # Test web search in extraction stage
   from sap_llm.stages import ExtractionStage
   
   extraction_stage = ExtractionStage(config, web_search=search_engine)
   
   # Process document with missing supplier info
   document = {
       "ocr_text": "PO #12345\nSupplier: Acme Corp\n...",
       "extracted_data": {
           "supplier_name": "Acme Corp",
           "supplier_address": None,  # Missing
           "supplier_tax_id": None,  # Missing
       }
   }
   
   # Enrichment should fill missing fields
   result = await extraction_stage.enrich_with_web_search(document)
   
   assert result["extracted_data"]["supplier_address"] is not None
   assert result["extracted_data"]["supplier_tax_id"] is not None
   print("Web search enrichment successful!")
   ```

EXECUTION STEPS:

1. Obtain API keys:
   ```bash
   # Tavily AI
   # 1. Visit https://tavily.com
   # 2. Sign up for account
   # 3. Get API key from dashboard
   
   # Google Custom Search
   # 1. Visit https://developers.google.com/custom-search/v1/overview
   # 2. Create project in Google Cloud Console
   # 3. Enable Custom Search API
   # 4. Create credentials (API key)
   # 5. Create Custom Search Engine and get CX ID
   
   # Bing Search API
   # 1. Visit https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
   # 2. Create Azure account if needed
   # 3. Create Bing Search resource
   # 4. Get API key from Keys and Endpoint
   ```

2. Configure environment:
   ```bash
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   
   # Add keys to .env
   echo "TAVILY_API_KEY=your_tavily_key" >> .env
   echo "GOOGLE_SEARCH_API_KEY=your_google_key" >> .env
   echo "GOOGLE_SEARCH_CX=your_custom_search_engine_id" >> .env
   echo "BING_SEARCH_API_KEY=your_bing_key" >> .env
   ```

3. Test providers:
   ```bash
   # Test each provider independently
   python -m sap_llm.web_search.test_providers \
     --test-all \
     --output results/web_search_provider_tests.json
   ```

4. Test failover:
   ```bash
   python -m sap_llm.web_search.test_failover \
     --simulate-failures \
     --output results/web_search_failover_tests.json
   ```

5. Test entity enrichment:
   ```bash
   python -m sap_llm.web_search.test_entity_enrichment \
     --test-cases examples/web_search/test_entities.json \
     --output results/entity_enrichment_tests.json
   ```

6. Measure cache performance:
   ```bash
   python -m sap_llm.web_search.test_cache \
     --num-queries 1000 \
     --cache-enabled \
     --output results/cache_performance.json
   ```

7. Integration test:
   ```bash
   python -m sap_llm.testing.test_web_search_integration \
     --test-documents data/processed/test/sample_100/ \
     --output results/web_search_integration.json
   ```

SUCCESS CRITERIA:
✓ All 4 providers configured and working
✓ Failover works correctly (automatic fallback)
✓ Entity enrichment ≥80% success rate
✓ Cache hit rate ≥60% after warmup
✓ Average search latency <200ms (with cache)
✓ Cost per enrichment <$0.001
✓ Integration with pipeline works end-to-end
✓ No API rate limit violations

DELIVERABLES:
1. .env file with all API keys configured
2. results/web_search_provider_tests.json - Provider test results
3. results/web_search_failover_tests.json - Failover validation
4. results/entity_enrichment_tests.json - Enrichment accuracy
5. results/cache_performance.json - Cache metrics
6. results/web_search_integration.json - End-to-end test
7. WEB_SEARCH_CONFIGURATION_REPORT.md - Report with:
   - Provider comparison (speed, accuracy, cost)
   - Failover behavior validation
   - Entity enrichment success rate
   - Cache performance analysis
   - Cost projections for production
   - Integration validation
   - Recommendations for optimization

NOTES:
- Free tiers have strict rate limits - use carefully during testing
- Tavily AI is preferred for best results
- Google has highest accuracy but costs more
- DuckDuckGo is free but less reliable
- Cache significantly reduces API costs (60-80% reduction)
- Entity enrichment improves extraction accuracy by 5-10%
- Production cost: ~$0.0005 per document with 70% cache hit rate

ESTIMATED TIME: 3-5 days
- Day 1: Obtain API keys + Configure environment
- Day 2: Test providers + Failover testing
- Day 3: Entity enrichment testing
- Day 4: Cache performance + Integration
- Day 5: Documentation
```

---

## TODO #10: Production Deployment & Monitoring Setup 🟡 HIGH PRIORITY

**Priority**: P1 - Required for Go-Live  
**Estimated Time**: 1 week  
**Dependencies**: TODO #7 (validation), TODO #8 (load testing)  
**Status**: Not Started (0%)

### Detailed Prompt for Claude Code

```
TASK: Deploy SAP_LLM to production Kubernetes cluster with complete monitoring, alerting, logging, and disaster recovery capabilities.

CONTEXT:
You are working on SAP_LLM at /Users/ajithkumarr/Desktop/SAP_LLM. All models are trained and validated. Load testing is complete. Now you need to deploy to production with full observability and operational readiness.

OBJECTIVE:
1. Deploy to production Kubernetes cluster
2. Set up Prometheus metrics and Grafana dashboards
3. Configure alerting rules (PagerDuty/Slack)
4. Implement distributed tracing (Jaeger)
5. Set up centralized logging (ELK/Loki)
6. Configure backup and disaster recovery
7. Implement blue-green deployment
8. Validate production readiness

REQUIREMENTS:
1. Production-ready Kubernetes deployment
2. High availability (3+ replicas, multi-AZ)
3. Comprehensive monitoring (Prometheus + Grafana)
4. Alerting (critical, warning, info)
5. Distributed tracing (OpenTelemetry + Jaeger)
6. Centralized logging (Elasticsearch/Loki)
7. Backup strategy (models, databases, configs)
8. Disaster recovery plan (RTO < 1hr, RPO < 5min)
9. Security hardening (network policies, RBAC)
10. Documentation and runbooks

TECHNICAL SPECIFICATIONS:

1. Production Kubernetes Deployment:
   ```yaml
   # deployments/kubernetes/production/deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: sap-llm-api
     namespace: sap-llm-prod
     labels:
       app: sap-llm
       environment: production
   spec:
     replicas: 3  # HA
     strategy:
       type: RollingUpdate
       rollingUpdate:
         maxSurge: 1
         maxUnavailable: 0  # Zero downtime
     selector:
       matchLabels:
         app: sap-llm
     template:
       metadata:
         labels:
           app: sap-llm
           version: v1.0.0
         annotations:
           prometheus.io/scrape: "true"
           prometheus.io/port: "8000"
           prometheus.io/path: "/metrics"
       spec:
         affinity:
           podAntiAffinity:  # Spread across nodes
             requiredDuringSchedulingIgnoredDuringExecution:
             - labelSelector:
                 matchExpressions:
                 - key: app
                   operator: In
                   values:
                   - sap-llm
               topologyKey: "kubernetes.io/hostname"
         nodeSelector:
           node-type: gpu  # GPU nodes only
         tolerations:
         - key: nvidia.com/gpu
           operator: Exists
           effect: NoSchedule
         containers:
         - name: sap-llm-api
           image: sap-llm:v1.0.0-prod
           imagePullPolicy: IfNotPresent
           ports:
           - containerPort: 8000
             name: http
             protocol: TCP
           resources:
             requests:
               memory: "32Gi"
               cpu: "8"
               nvidia.com/gpu: "2"
             limits:
               memory: "64Gi"
               cpu: "16"
               nvidia.com/gpu: "2"
           env:
           - name: ENVIRONMENT
             value: "production"
           - name: LOG_LEVEL
             value: "INFO"
           - name: COSMOS_ENDPOINT
             valueFrom:
               secretKeyRef:
                 name: sap-llm-secrets
                 key: cosmos-endpoint
           - name: COSMOS_KEY
             valueFrom:
               secretKeyRef:
                 name: sap-llm-secrets
                 key: cosmos-key
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 120
             periodSeconds: 30
             timeoutSeconds: 10
             failureThreshold: 3
           readinessProbe:
             httpGet:
               path: /health/ready
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
             timeoutSeconds: 5
             failureThreshold: 3
           volumeMounts:
           - name: models
             mountPath: /models
             readOnly: true
           - name: config
             mountPath: /config
             readOnly: true
         volumes:
         - name: models
           persistentVolumeClaim:
             claimName: sap-llm-models-pvc
         - name: config
           configMap:
             name: sap-llm-config
   ```

2. Monitoring Setup (Prometheus + Grafana):
   ```yaml
   # deployments/monitoring/servicemonitor.yaml
   apiVersion: monitoring.coreos.com/v1
   kind: ServiceMonitor
   metadata:
     name: sap-llm-monitor
     namespace: sap-llm-prod
   spec:
     selector:
       matchLabels:
         app: sap-llm
     endpoints:
     - port: http
       path: /metrics
       interval: 30s
       scrapeTimeout: 10s
   ```
   
   Grafana Dashboards:
   - Dashboard 1: Overview (throughput, latency, errors)
   - Dashboard 2: Model Performance (accuracy, confidence, drift)
   - Dashboard 3: Resource Utilization (GPU, CPU, memory)
   - Dashboard 4: Pipeline Stages (latency per stage)
   - Dashboard 5: Business Metrics (touchless rate, cost, SLA)

3. Alerting Rules:
   ```yaml
   # configs/alerting_rules_prod.yml
   groups:
   - name: sap_llm_critical
     interval: 30s
     rules:
     # High Error Rate
     - alert: HighErrorRate
       expr: rate(sap_llm_requests_failed_total[5m]) > 0.05
       for: 5m
       labels:
         severity: critical
         team: ml-ops
       annotations:
         summary: "High error rate detected"
         description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"
         runbook: "https://docs.qorsync.com/runbooks/high-error-rate"
     
     # High Latency
     - alert: HighLatency
       expr: histogram_quantile(0.95, rate(sap_llm_request_duration_seconds_bucket[5m])) > 2.0
       for: 10m
       labels:
         severity: critical
         team: ml-ops
       annotations:
         summary: "High latency detected"
         description: "P95 latency is {{ $value }}s (threshold: 2.0s)"
         runbook: "https://docs.qorsync.com/runbooks/high-latency"
     
     # Low Throughput
     - alert: LowThroughput
       expr: rate(sap_llm_documents_processed_total[10m]) < 1.0
       for: 15m
       labels:
         severity: warning
         team: ml-ops
       annotations:
         summary: "Low throughput detected"
         description: "Processing {{ $value }} docs/sec (threshold: 1.0/sec)"
         runbook: "https://docs.qorsync.com/runbooks/low-throughput"
     
     # Model Accuracy Degradation
     - alert: ModelAccuracyDegradation
       expr: sap_llm_classification_accuracy < 0.90
       for: 1h
       labels:
         severity: warning
         team: ml-ops
       annotations:
         summary: "Model accuracy degraded"
         description: "Classification accuracy is {{ $value | humanizePercentage }} (threshold: 90%)"
         runbook: "https://docs.qorsync.com/runbooks/accuracy-degradation"
     
     # GPU Out of Memory
     - alert: GPUOutOfMemory
       expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
       for: 5m
       labels:
         severity: critical
         team: ml-ops
       annotations:
         summary: "GPU memory critically high"
         description: "GPU memory usage is {{ $value | humanizePercentage }}"
         runbook: "https://docs.qorsync.com/runbooks/gpu-oom"
     
     # PMG Connection Failed
     - alert: PMGConnectionFailed
       expr: sap_llm_pmg_connection_status == 0
       for: 5m
       labels:
         severity: critical
         team: ml-ops
       annotations:
         summary: "Cannot connect to Process Memory Graph"
         description: "PMG connection has been down for 5 minutes"
         runbook: "https://docs.qorsync.com/runbooks/pmg-connection"
   ```

4. Distributed Tracing (Jaeger):
   ```python
   # Already implemented in sap_llm/monitoring/tracing.py
   # Deploy Jaeger
   kubectl apply -f deployments/monitoring/jaeger.yaml -n sap-llm-prod
   
   # Verify tracing
   # Traces should appear in Jaeger UI: http://jaeger.sap-llm-prod.svc.cluster.local:16686
   ```

5. Centralized Logging (Loki):
   ```yaml
   # deployments/monitoring/loki.yaml
   apiVersion: v1
   kind: ConfigMap
   metadata:
     name: promtail-config
     namespace: sap-llm-prod
   data:
     promtail.yaml: |
       server:
         http_listen_port: 9080
       positions:
         filename: /tmp/positions.yaml
       clients:
         - url: http://loki:3100/loki/api/v1/push
       scrape_configs:
         - job_name: sap-llm-logs
           kubernetes_sd_configs:
             - role: pod
               namespaces:
                 names:
                   - sap-llm-prod
           relabel_configs:
             - source_labels: [__meta_kubernetes_pod_label_app]
               target_label: app
             - source_labels: [__meta_kubernetes_pod_name]
               target_label: pod
             - source_labels: [__meta_kubernetes_namespace]
               target_label: namespace
           pipeline_stages:
             - json:
                 expressions:
                   level: level
                   timestamp: timestamp
                   message: message
             - labels:
                 level:
   ```

6. Backup Strategy:
   ```bash
   # Backup script: scripts/backup_production.sh
   #!/bin/bash
   
   BACKUP_DIR="/backups/sap-llm/$(date +%Y%m%d_%H%M%S)"
   mkdir -p $BACKUP_DIR
   
   # Backup models
   echo "Backing up models..."
   aws s3 sync /models/ s3://sap-llm-backups/models/$BACKUP_DIR/
   
   # Backup Cosmos DB PMG
   echo "Backing up Cosmos DB PMG..."
   az cosmosdb sql database throughput show \
     --account-name sap-llm-pmg \
     --name pmg \
     --resource-group sap-llm-rg \
     > $BACKUP_DIR/cosmos_throughput.json
   
   # Backup MongoDB Knowledge Base
   echo "Backing up MongoDB..."
   mongodump --uri="$MONGO_URI" --out=$BACKUP_DIR/mongodb/
   aws s3 sync $BACKUP_DIR/mongodb/ s3://sap-llm-backups/mongodb/$BACKUP_DIR/
   
   # Backup Redis cache (snapshots)
   echo "Backing up Redis..."
   redis-cli BGSAVE
   cp /var/lib/redis/dump.rdb $BACKUP_DIR/redis_dump.rdb
   aws s3 cp $BACKUP_DIR/redis_dump.rdb s3://sap-llm-backups/redis/$BACKUP_DIR/
   
   # Backup Kubernetes configs
   echo "Backing up Kubernetes configs..."
   kubectl get all -n sap-llm-prod -o yaml > $BACKUP_DIR/k8s_resources.yaml
   kubectl get secrets -n sap-llm-prod -o yaml > $BACKUP_DIR/k8s_secrets.yaml
   kubectl get configmaps -n sap-llm-prod -o yaml > $BACKUP_DIR/k8s_configmaps.yaml
   
   # Upload to S3
   aws s3 sync $BACKUP_DIR/ s3://sap-llm-backups/full-backup/$BACKUP_DIR/
   
   echo "Backup complete: $BACKUP_DIR"
   ```
   
   Schedule via CronJob:
   ```yaml
   apiVersion: batch/v1
   kind: CronJob
   metadata:
     name: sap-llm-backup
     namespace: sap-llm-prod
   spec:
     schedule: "0 2 * * *"  # Daily at 2 AM
     jobTemplate:
       spec:
         template:
           spec:
             containers:
             - name: backup
               image: sap-llm-backup:latest
               command: ["/scripts/backup_production.sh"]
             restartPolicy: OnFailure
   ```

7. Disaster Recovery Plan:
   ```bash
   # Restore script: scripts/restore_production.sh
   #!/bin/bash
   
   BACKUP_DATE=$1  # Format: 20251118_020000
   BACKUP_DIR="/backups/sap-llm/$BACKUP_DATE"
   
   echo "Restoring from backup: $BACKUP_DATE"
   
   # Download from S3
   aws s3 sync s3://sap-llm-backups/full-backup/$BACKUP_DIR/ $BACKUP_DIR/
   
   # Restore models
   aws s3 sync s3://sap-llm-backups/models/$BACKUP_DIR/ /models/
   
   # Restore MongoDB
   mongorestore --uri="$MONGO_URI" $BACKUP_DIR/mongodb/
   
   # Restore Redis
   redis-cli FLUSHALL
   cp $BACKUP_DIR/redis_dump.rdb /var/lib/redis/dump.rdb
   redis-cli SHUTDOWN
   systemctl start redis
   
   # Restore Kubernetes
   kubectl apply -f $BACKUP_DIR/k8s_resources.yaml -n sap-llm-prod
   kubectl apply -f $BACKUP_DIR/k8s_secrets.yaml -n sap-llm-prod
   kubectl apply -f $BACKUP_DIR/k8s_configmaps.yaml -n sap-llm-prod
   
   echo "Restore complete!"
   echo "RTO achieved: $(date)"
   ```

8. Blue-Green Deployment:
   ```bash
   # Deploy new version (green)
   kubectl apply -f deployments/kubernetes/production/deployment-v1.1.0.yaml -n sap-llm-prod
   
   # Wait for green to be ready
   kubectl wait --for=condition=available deployment/sap-llm-api-v1.1.0 -n sap-llm-prod
   
   # Run smoke tests on green
   python -m sap_llm.testing.smoke_tests --target green --env production
   
   # Switch traffic to green
   kubectl patch service sap-llm-api -n sap-llm-prod \
     -p '{"spec":{"selector":{"version":"v1.1.0"}}}'
   
   # Monitor for 15 minutes
   sleep 900
   
   # If no issues, delete blue
   kubectl delete deployment sap-llm-api-v1.0.0 -n sap-llm-prod
   
   # If issues, rollback
   # kubectl patch service sap-llm-api -n sap-llm-prod \
   #   -p '{"spec":{"selector":{"version":"v1.0.0"}}}'
   ```

EXECUTION STEPS:

1. Prepare production environment:
   ```bash
   cd /Users/ajithkumarr/Desktop/SAP_LLM
   
   # Create production namespace
   kubectl create namespace sap-llm-prod
   
   # Create secrets
   kubectl create secret generic sap-llm-secrets \
     --from-literal=cosmos-endpoint=$COSMOS_ENDPOINT \
     --from-literal=cosmos-key=$COSMOS_KEY \
     --from-literal=mongo-uri=$MONGO_URI \
     --from-literal=redis-password=$REDIS_PASSWORD \
     -n sap-llm-prod
   
   # Create configmap
   kubectl create configmap sap-llm-config \
     --from-file=configs/default_config.yaml \
     --from-file=configs/document_types.yaml \
     -n sap-llm-prod
   ```

2. Deploy models to persistent storage:
   ```bash
   # Create PVC
   kubectl apply -f deployments/kubernetes/production/pvc.yaml -n sap-llm-prod
   
   # Upload models
   kubectl run upload-models --image=alpine --restart=Never -n sap-llm-prod \
     --overrides='
       {
         "spec": {
           "containers": [{
             "name": "upload",
             "image": "alpine",
             "command": ["sleep", "3600"],
             "volumeMounts": [{
               "name": "models",
               "mountPath": "/models"
             }]
           }],
           "volumes": [{
             "name": "models",
             "persistentVolumeClaim": {
               "claimName": "sap-llm-models-pvc"
             }
           }]
         }
       }'
   
   kubectl cp models/ sap-llm-prod/upload-models:/models/
   kubectl delete pod upload-models -n sap-llm-prod
   ```

3. Deploy SAP_LLM:
   ```bash
   kubectl apply -f deployments/kubernetes/production/ -n sap-llm-prod
   
   # Verify deployment
   kubectl get pods -n sap-llm-prod
   kubectl logs -f deployment/sap-llm-api -n sap-llm-prod
   ```

4. Deploy monitoring stack:
   ```bash
   # Deploy Prometheus
   kubectl apply -f deployments/monitoring/prometheus.yaml -n sap-llm-prod
   
   # Deploy Grafana
   kubectl apply -f deployments/monitoring/grafana.yaml -n sap-llm-prod
   
   # Import dashboards
   for dashboard in deployments/monitoring/dashboards/*.json; do
     curl -X POST http://grafana.sap-llm-prod.svc.cluster.local:3000/api/dashboards/db \
       -H "Content-Type: application/json" \
       -d @$dashboard
   done
   
   # Deploy Jaeger
   kubectl apply -f deployments/monitoring/jaeger.yaml -n sap-llm-prod
   
   # Deploy Loki + Promtail
   kubectl apply -f deployments/monitoring/loki.yaml -n sap-llm-prod
   kubectl apply -f deployments/monitoring/promtail.yaml -n sap-llm-prod
   ```

5. Configure alerting:
   ```bash
   # Deploy AlertManager
   kubectl apply -f deployments/monitoring/alertmanager.yaml -n sap-llm-prod
   
   # Configure PagerDuty integration
   kubectl create secret generic alertmanager-pagerduty \
     --from-literal=service-key=$PAGERDUTY_SERVICE_KEY \
     -n sap-llm-prod
   
   # Apply alerting rules
   kubectl apply -f configs/alerting_rules_prod.yml -n sap-llm-prod
   ```

6. Set up backups:
   ```bash
   # Deploy backup CronJob
   kubectl apply -f deployments/kubernetes/production/backup-cronjob.yaml -n sap-llm-prod
   
   # Test backup manually
   kubectl create job --from=cronjob/sap-llm-backup sap-llm-backup-test -n sap-llm-prod
   kubectl logs job/sap-llm-backup-test -n sap-llm-prod
   ```

7. Run production smoke tests:
   ```bash
   python -m sap_llm.testing.smoke_tests \
     --env production \
     --api-url https://api.sap-llm.production.qorsync.com \
     --num-tests 100 \
     --output results/production_smoke_tests.json
   ```

8. Validate monitoring:
   ```bash
   # Check Prometheus targets
   curl http://prometheus.sap-llm-prod.svc.cluster.local:9090/api/v1/targets
   
   # Check Grafana dashboards
   open http://grafana.sap-llm-prod.svc.cluster.local:3000
   
   # Check Jaeger traces
   open http://jaeger.sap-llm-prod.svc.cluster.local:16686
   
   # Check logs in Loki
   curl -G http://loki.sap-llm-prod.svc.cluster.local:3100/loki/api/v1/query \
     --data-urlencode 'query={app="sap-llm"}' | jq
   ```

SUCCESS CRITERIA:
✓ All pods running and healthy (3 replicas)
✓ Load balancer configured and accessible
✓ Prometheus collecting metrics (>50 metrics)
✓ Grafana dashboards displaying data (5 dashboards)
✓ Alerting rules active (>15 rules)
✓ Distributed tracing working (Jaeger)
✓ Centralized logging working (Loki)
✓ Backups running successfully (daily)
✓ Disaster recovery tested (RTO < 1hr)
✓ Blue-green deployment tested
✓ Zero downtime during rolling update
✓ Production smoke tests passing (100/100)

DELIVERABLES:
1. Production Kubernetes cluster with SAP_LLM deployed
2. Prometheus + Grafana monitoring stack
3. Jaeger distributed tracing
4. Loki centralized logging
5. AlertManager with PagerDuty integration
6. Automated backup system
7. Disaster recovery runbook
8. PRODUCTION_DEPLOYMENT_REPORT.md - Report with:
   - Deployment architecture diagram
   - Monitoring dashboard screenshots
   - Alerting rule documentation
   - Backup and recovery procedures
   - Incident response playbook
   - SLA commitments
   - Operational runbooks
   - Post-deployment validation results

NOTES:
- Production deployment requires approval from stakeholders
- Schedule deployment during maintenance window
- Have rollback plan ready
- Monitor closely for first 24 hours
- Run blue-green deployment for zero downtime
- Test disaster recovery in staging first
- Document all operational procedures
- Train operations team on monitoring and alerting

ESTIMATED TIME: 1 week
- Day 1-2: Deploy to production + Monitoring setup
- Day 3: Alerting + Logging configuration
- Day 4: Backup + Disaster recovery testing
- Day 5: Blue-green deployment testing
- Day 6-7: Smoke tests + Validation + Documentation
```

---

## Summary

This document contains 10 detailed TODO tasks to complete SAP_LLM production readiness:

1. ✅ **Training Data Collection** (6-8 weeks) - Collect 1M+ documents
2. ✅ **SAP Knowledge Base Population** (4-6 weeks) - Scrape 400+ APIs
3. ✅ **Vision Encoder Training** (3-4 weeks) - Fine-tune LayoutLMv3
4. ✅ **Language Decoder Training** (3-4 weeks) - Fine-tune LLaMA-2-7B
5. ✅ **Reasoning Engine Training** (4-5 weeks) - Fine-tune Mixtral-8x7B with RLHF
6. ✅ **PMG Population** (2-3 weeks) - Ingest 100K+ historical documents
7. ✅ **End-to-End Validation** (2-3 weeks) - Validate on 150K test set
8. ✅ **Load Testing & Optimization** (1-2 weeks) - Achieve 5K docs/hour
9. ✅ **Web Search Configuration** (3-5 days) - Configure all providers
10. ✅ **Production Deployment** (1 week) - Deploy with full monitoring

**Total Estimated Time**: 22-32 weeks (5.5-8 months)

**Critical Path**: TODO #1 → #3 → #4 → #5 → #7 → #8 → #10

**Parallel Execution**: TODO #2, #6, and #9 can run in parallel with others

---

**Next Steps**:
1. Prioritize TODO #1 (Training Data Collection) - starts immediately
2. Start TODO #2 (SAP Knowledge Base) in parallel
3. Once data is ready, begin model training (TODO #3, #4, #5)
4. After training, proceed with validation and deployment

**Success Metrics**:
- Classification Accuracy: ≥95%
- Extraction F1 Score: ≥92%
- Routing Accuracy: ≥97%
- Touchless Rate: ≥85%
- End-to-End Latency P95: ≤1.5s
- Throughput: ≥5K docs/hour per GPU
- Cost per Document: <$0.005

**Investment Required**: $350K (cloud infrastructure + personnel)

**ROI**: Break-even at 31,858 documents (~6.4 hours at 5K/hour)

---

**Document Version**: 1.0  
**Last Updated**: November 18, 2025  
**Status**: Ready for Execution