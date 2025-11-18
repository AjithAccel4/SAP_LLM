# SAP_LLM Development Plan - Complete From-Scratch Implementation

## Project Scope

Develop a custom, enterprise-grade Large Language Model (SAP_LLM) specifically trained for SAP document processing that replaces all third-party LLM dependencies (OpenAI GPT-4, Claude, etc.) with a self-hosted solution handling all 8 pipeline stages autonomously.

## PHASE 1: Foundation & Data Infrastructure (Months 1-3)

### 1.1 Data Collection & Corpus Building

**Training Data Requirements:**

- 50M+ tokens minimum for baseline model (target: 100B+ tokens for production)
- 1M+ annotated business documents across all types
- SAP-specific knowledge base (400+ API documentation, S/4HANA processes)

**Data Sources:**

1. **Existing QorSync Data:**

                        - Extract all processed documents from PostgreSQL (processed_documents table)
                        - Extract Neo4j classification patterns (244 relationships, 36 PO types, 16 Invoice types)
                        - MongoDB extraction results and audit trails
                        - Current schema: `services/enhanced-document-processor/po_si_fields_schema.json`

2. **SAP Knowledge Base:**

                        - SAP Business Accelerator Hub API documentation (400+ APIs)
                        - S/4HANA Cloud OData V2/V4 endpoint specifications
                        - SAP IDoc formats (INVOIC02, ORDERS05, DESADV, ORDRSP)
                        - SAP Business Process documentation (Lead-to-Cash, Source-to-Pay)
                        - Current SAP config: `services/router/config/sap_production_config.py`

3. **Synthetic Data Generation:**

                        - Generate 500K+ synthetic documents using current templates
                        - Use existing PO/Invoice generators for training data augmentation
                        - Create adversarial examples for robustness training

4. **Public Business Document Datasets:**

                        - SROIE (Scanned Receipt OCR and Information Extraction)
                        - CORD (Consolidated Receipt Dataset) - 11K receipts
                        - RVL-CDIP (Tobacco Industry documents) - 400K documents
                        - FUNSD (Form Understanding in Noisy Scanned Documents)

**Annotation Requirements:**

- Label all 50+ document types (Purchase Orders, Invoices, Receipts, etc.)
- Annotate 35+ PO subtypes (STANDARD, BLANKET, CONSIGNMENT, DROP_SHIP, etc.)
- Annotate 15+ Invoice subtypes (SERVICE, RECURRING, MILESTONE, CREDIT_MEMO)
- Field-level annotations for 200+ business fields
- Confidence scores for each extraction

**Implementation:**

```python
# New service: services/sap-llm/data_pipeline/corpus_builder.py
class CorpusBuilder:
    """Build training corpus from all QorSync data sources"""
    
    async def extract_qorsync_corpus(self):
        """Extract training data from existing system"""
        # PostgreSQL: processed_documents
        # Neo4j: classification patterns
        # MongoDB: extraction results
        
    async def scrape_sap_documentation(self):
        """Scrape SAP Business Accelerator Hub"""
        # API specifications, endpoint docs, business processes
        
    async def generate_synthetic_documents(self, count: int):
        """Generate synthetic training examples"""
        # Use existing document templates
        
    async def annotate_corpus(self, corpus: List[Document]):
        """Create training annotations"""
        # Document type, subtype, fields, confidence
```

### 1.2 Model Architecture Selection

**Recommended Architecture: Transformer-based Multi-Modal Model**

**Option A: Document-Specific Transformer (Recommended)**

- Base: LayoutLMv3 architecture (Microsoft)
- Parameters: 7B-13B (optimal for document understanding)
- Modalities: Text + Layout + Visual (for tables, stamps, signatures)
- Context window: 8K tokens (covers most business documents)

**Option B: Llama-Based Custom Architecture**

- Base: Llama 3.1 architecture (Meta)
- Parameters: 8B-13B (cost-effective, high performance)
- Fine-tuning: LoRA/QLoRA for efficient training
- Context window: 8K-32K tokens

**SAP_LLM Specific Layers:**

1. **Document Understanding Layer:**

                        - Multi-modal encoder (text + visual + layout)
                        - Table detection and extraction
                        - Signature/stamp recognition
                        - Handwriting recognition

2. **Classification Layer:**

                        - 50+ document type classification head
                        - 35+ PO subtype classification head
                        - 15+ Invoice subtype classification head
                        - Confidence scoring network

3. **Extraction Layer:**

                        - Named Entity Recognition (NER) for business fields
                        - Key-value pair extraction
                        - Line item detection and parsing
                        - Hierarchical data extraction

4. **Validation Layer:**

                        - Business rule validation
                        - Cross-field consistency checks
                        - Format validation (dates, amounts, IDs)

5. **SAP Integration Layer:**

                        - SAP API schema mapping
                        - OData payload generation
                        - IDoc format generation

**Implementation:**

```python
# services/sap-llm/model/architecture.py
class SAPLLMArchitecture:
    """Custom transformer architecture for SAP document processing"""
    
    def __init__(self):
        self.document_encoder = DocumentUnderstandingEncoder()
        self.classification_head = MultiTaskClassificationHead()
        self.extraction_head = FieldExtractionHead()
        self.validation_head = ValidationHead()
        self.sap_integration_head = SAPIntegrationHead()
        
    def forward(self, document_inputs):
        """Process document through all layers"""
        # Encode document (text + visual + layout)
        embeddings = self.document_encoder(document_inputs)
        
        # Multi-task processing
        doc_type = self.classification_head(embeddings)
        extracted_fields = self.extraction_head(embeddings)
        validation_results = self.validation_head(extracted_fields)
        sap_payload = self.sap_integration_head(extracted_fields)
        
        return {
            'document_type': doc_type,
            'fields': extracted_fields,
            'validation': validation_results,
            'sap_payload': sap_payload
        }
```

### 1.3 Training Infrastructure Setup

**Compute Requirements:**

For 7B-13B parameter model training:

- GPUs: 8x NVIDIA H100 (80GB) or 16x A100 (80GB)
- Training time: 2-4 weeks for full training
- Estimated cost: $150K-$300K (cloud) or $500K (on-premise hardware)

**Infrastructure Options:**

**Option A: Cloud (Recommended for MVP):**

- AWS EC2 P5 instances (H100 GPUs)
- Google Cloud A3 instances (H100 GPUs)
- Azure NC H100 v5 series

**Option B: On-Premise:**

- Purchase 8x H100 servers ($40K each = $320K)
- High-speed interconnect (NVLink, InfiniBand)
- Power/cooling infrastructure

**Option C: Hybrid (Recommended for Production):**

- Initial training on cloud
- Fine-tuning and inference on-premise
- Use QLoRA for efficient fine-tuning

**Training Framework:**

- PyTorch 2.0+ with FSDP (Fully Sharded Data Parallel)
- Hugging Face Transformers + Accelerate
- DeepSpeed ZeRO-3 for memory optimization
- Weights & Biases for experiment tracking

**Implementation:**

```python
# services/sap-llm/training/trainer.py
import torch
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator

class SAPLLMTrainer:
    """Distributed training for SAP_LLM"""
    
    def __init__(self, model, train_dataset, eval_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.training_args = TrainingArguments(
            output_dir="./sap-llm-checkpoints",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-5,
            warmup_steps=1000,
            logging_steps=100,
            save_steps=5000,
            eval_steps=1000,
            fp16=True,  # Mixed precision training
            fsdp="full_shard",  # Fully sharded data parallel
            fsdp_transformer_layer_cls_to_wrap="LayoutLMv3Layer"
        )
        
    async def train(self):
        """Execute distributed training"""
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset
        )
        
        trainer.train()
```

## PHASE 2: Model Training & Optimization (Months 4-6)

### 2.1 Pre-training Stage

**Objective:** Learn general business document understanding

**Training Strategy:**

1. **Masked Language Modeling (MLM):** Predict masked tokens in documents
2. **Masked Layout Modeling (MLM):** Predict masked layout elements
3. **Masked Visual Modeling (MVM):** Predict masked image patches

**Pre-training Corpus:**

- 1M+ unlabeled business documents
- SAP documentation (100M+ tokens)
- General business text (invoices, orders, contracts)

**Training Configuration:**

- Batch size: 256 (across 8 GPUs)
- Learning rate: 2e-5 with warmup
- Training time: 2-3 weeks
- Checkpoint every 10K steps

### 2.2 Fine-tuning for QorSync Pipeline Stages

**Stage-Specific Fine-Tuning:**

**Stage 1: Inbox (Document Reception)**

- Task: Document format detection, metadata extraction
- Dataset: 100K+ documents with file metadata
- Metrics: Format accuracy 99%+

**Stage 2: Preprocessing (Text Extraction)**

- Task: OCR quality assessment, text cleaning
- Dataset: 50K+ scanned/image documents
- Metrics: OCR accuracy 98%+, text quality 95%+

**Stage 3: Classification (Document Type)**

- Task: 50+ document type classification
- Dataset: 500K+ labeled documents (current Neo4j + new data)
- Metrics: Classification accuracy 99%+
- Reference: Current implementation in `services/classifier/`

**Stage 4: Type Identifier (PO/Invoice Subtype)**

- Task: 35+ PO subtypes, 15+ Invoice subtypes
- Dataset: 300K+ documents with subtype labels
- Metrics: Subtype accuracy 99%+
- Reference: `po-type-identifier/src/config/po_type_config.py`

**Stage 5: Extraction (Field Extraction)**

- Task: 200+ business field extraction
- Dataset: 1M+ annotated documents
- Metrics: Field accuracy 95%+, F1 score 0.92+
- Reference: `services/enhanced-document-processor/ultimate_extraction.py`

**Stage 6: Quality Check (Data Validation)**

- Task: Field completeness, format validation
- Dataset: 200K+ validation cases
- Metrics: Validation accuracy 98%+

**Stage 7: Validation (Business Rules)**

- Task: Business logic validation
- Dataset: 100K+ validation scenarios
- Metrics: Rule application accuracy 99%+
- Reference: `services/rules/`

**Stage 8: Routing (SAP Integration)**

- Task: SAP payload generation, API selection
- Dataset: 200K+ SAP transactions
- Metrics: Payload accuracy 99%+, API selection 100%
- Reference: `services/router/enhanced_sap_integration_service.py`

**Multi-Task Learning Configuration:**

```python
# services/sap-llm/training/multitask_trainer.py
class MultiTaskTrainer:
    """Train SAP_LLM on all 8 pipeline stages simultaneously"""
    
    def __init__(self):
        self.tasks = {
            'stage1_inbox': InboxTask(),
            'stage2_preprocessing': PreprocessingTask(),
            'stage3_classification': ClassificationTask(),
            'stage4_type_identifier': TypeIdentifierTask(),
            'stage5_extraction': ExtractionTask(),
            'stage6_quality_check': QualityCheckTask(),
            'stage7_validation': ValidationTask(),
            'stage8_routing': RoutingTask()
        }
        
    async def train_multitask(self, model, datasets):
        """Train with task-specific losses"""
        total_loss = 0
        
        for task_name, task in self.tasks.items():
            task_loss = task.compute_loss(model, datasets[task_name])
            total_loss += task_loss * task.weight
            
        return total_loss
```

### 2.3 Reinforcement Learning with Human Feedback (RLHF)

**Objective:** Align model outputs with human preferences and business requirements

**Implementation:**

1. **Reward Model Training:**

                        - Collect human feedback on model outputs (10K+ examples)
                        - Train reward model to predict human preferences

2. **Proximal Policy Optimization (PPO):**

                        - Fine-tune SAP_LLM using PPO algorithm
                        - Maximize reward while staying close to supervised policy

3. **Human-in-the-Loop:**

                        - Continuous feedback collection
                        - Model updates based on production performance

## PHASE 3: Process Memory Graph (PMG) Integration (Months 7-9)

### 3.1 PMG Architecture for SAP_LLM

**Whitepaper Specification:** Cosmos DB Gremlin API with versioning, Merkle hashing, as-of queries

**Practical Implementation:** Neo4j + Redis + PostgreSQL (existing infrastructure)

**PMG Components:**

1. **Document Memory:**

                        - Store every processed document with 768-dim embeddings
                        - Version history for document evolution
                        - Similarity links for context retrieval

2. **Processing Context Memory:**

                        - Store all AI decisions (classification, extraction, validation)
                        - Store confidence scores and reasoning traces
                        - Store SAP integration results

3. **Rule Memory:**

                        - Store business rules and validation logic
                        - Version rules with Merkle tree for audit trail
                        - Track rule evolution over time

4. **Feedback Memory:**

                        - Store human corrections and feedback
                        - Store SAP system responses (success/failure)
                        - Store exception patterns for self-healing

**Implementation:**

```python
# services/sap-llm/pmg/graph_memory.py
from neo4j import AsyncGraphDatabase
import hashlib
import json

class ProcessMemoryGraph:
    """PMG implementation using Neo4j + embeddings"""
    
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            "bolt://neo4j:7687",
            auth=("neo4j", "qorsync_password")
        )
        self.embedding_dim = 768
        
    async def store_processing_context(self, context: Dict):
        """Store complete processing trace"""
        async with self.driver.session() as session:
            query = """
            CREATE (d:Document {
                id: $doc_id,
                filename: $filename,
                doc_type: $doc_type,
                embedding: $embedding,
                merkle_hash: $merkle_hash,
                created_at: datetime(),
                version: $version
            })
            
            // Link to classification decision
            CREATE (c:ClassificationDecision {
                doc_type: $doc_type,
                subtype: $subtype,
                confidence: $confidence,
                reasoning: $reasoning
            })
            CREATE (d)-[:CLASSIFIED_BY]->(c)
            
            // Link to extraction results
            CREATE (e:ExtractionResult {
                fields: $extracted_fields,
                confidence_scores: $field_confidences
            })
            CREATE (d)-[:EXTRACTED_TO]->(e)
            
            // Link to SAP payload
            CREATE (s:SAPPayload {
                api_endpoint: $sap_endpoint,
                payload: $sap_payload,
                response: $sap_response
            })
            CREATE (d)-[:ROUTED_TO]->(s)
            """
            
            # Compute Merkle hash for version control
            merkle_hash = self._compute_merkle_hash(context)
            
            await session.run(query, {
                'doc_id': context['document_id'],
                'filename': context['filename'],
                'doc_type': context['document_type'],
                'embedding': context['embedding'],
                'merkle_hash': merkle_hash,
                'version': context.get('version', 1),
                'subtype': context['subtype'],
                'confidence': context['confidence'],
                'reasoning': context.get('reasoning', ''),
                'extracted_fields': json.dumps(context['fields']),
                'field_confidences': json.dumps(context['field_confidences']),
                'sap_endpoint': context['sap_endpoint'],
                'sap_payload': json.dumps(context['sap_payload']),
                'sap_response': json.dumps(context.get('sap_response', {}))
            })
    
    async def retrieve_similar_contexts(self, query_embedding: List[float], top_k: int = 5):
        """Find similar processing contexts for learning"""
        async with self.driver.session() as session:
            query = """
            MATCH (d:Document)
            WITH d, gds.similarity.cosine(d.embedding, $query_embedding) AS similarity
            WHERE similarity > 0.85
            ORDER BY similarity DESC
            LIMIT $top_k
            
            MATCH (d)-[:CLASSIFIED_BY]->(c:ClassificationDecision)
            MATCH (d)-[:EXTRACTED_TO]->(e:ExtractionResult)
            OPTIONAL MATCH (d)-[:ROUTED_TO]->(s:SAPPayload)
            
            RETURN d, c, e, s, similarity
            """
            
            results = await session.run(query, {
                'query_embedding': query_embedding,
                'top_k': top_k
            })
            
            return [dict(record) async for record in results]
    
    def _compute_merkle_hash(self, context: Dict) -> str:
        """Compute Merkle hash for versioning"""
        context_json = json.dumps(context, sort_keys=True)
        return hashlib.sha256(context_json.encode()).hexdigest()
```

### 3.2 Context-Aware Processing

**SAP_LLM + PMG Integration:**

```python
# services/sap-llm/inference/context_aware_processor.py
class ContextAwareProcessor:
    """Use PMG to enhance SAP_LLM predictions"""
    
    def __init__(self, sap_llm_model, pmg: ProcessMemoryGraph):
        self.model = sap_llm_model
        self.pmg = pmg
        
    async def process_document_with_context(self, document: Document):
        """Process document using historical context"""
        
        # Step 1: Generate document embedding
        embedding = await self.model.encode(document.text)
        
        # Step 2: Retrieve similar historical contexts
        similar_contexts = await self.pmg.retrieve_similar_contexts(
            embedding, top_k=5
        )
        
        # Step 3: Augment model input with historical context
        context_prompt = self._build_context_prompt(similar_contexts)
        
        # Step 4: Run SAP_LLM with context
        result = await self.model.process_with_context(
            document=document,
            historical_context=context_prompt
        )
        
        # Step 5: Store new processing context in PMG
        await self.pmg.store_processing_context({
            'document_id': document.id,
            'filename': document.filename,
            'embedding': embedding,
            'document_type': result['document_type'],
            'subtype': result['subtype'],
            'confidence': result['confidence'],
            'fields': result['extracted_fields'],
            'field_confidences': result['field_confidences'],
            'sap_endpoint': result['sap_endpoint'],
            'sap_payload': result['sap_payload']
        })
        
        return result
```

## PHASE 4: Self-Healing Workflow Loop (SHWL) (Months 10-12)

### 4.1 SHWL Architecture

**Whitepaper 5-Phase Cycle:** Detect → Cluster → Explain → Review → Apply

**Implementation:**

```python
# services/sap-llm/shwl/self_healing_loop.py
class SelfHealingWorkflowLoop:
    """Autonomous improvement system"""
    
    def __init__(self, sap_llm_model, pmg: ProcessMemoryGraph):
        self.model = sap_llm_model
        self.pmg = pmg
        
    # PHASE 1: DETECT
    async def detect_anomalies(self) -> List[Anomaly]:
        """Detect processing failures and low-confidence predictions"""
        
        # Query PMG for recent low-confidence results
        query = """
        MATCH (d:Document)-[:CLASSIFIED_BY]->(c:ClassificationDecision)
        WHERE c.confidence < 0.7 OR c.failed = true
        AND datetime(d.created_at) > datetime() - duration({days: 7})
        RETURN d, c
        """
        
        anomalies = []
        async with self.pmg.driver.session() as session:
            results = await session.run(query)
            async for record in results:
                anomalies.append(Anomaly(
                    document_id=record['d']['id'],
                    reason='low_confidence',
                    confidence=record['c']['confidence'],
                    document_type=record['c']['doc_type']
                ))
        
        # Also detect SAP integration failures
        sap_failures = await self._detect_sap_failures()
        anomalies.extend(sap_failures)
        
        return anomalies
    
    # PHASE 2: CLUSTER
    async def cluster_patterns(self, anomalies: List[Anomaly]) -> List[PatternCluster]:
        """Group similar anomalies into patterns"""
        
        # Extract embeddings for all anomaly documents
        embeddings = []
        for anomaly in anomalies:
            doc = await self.pmg.get_document(anomaly.document_id)
            embedding = doc['embedding']
            embeddings.append(embedding)
        
        # Cluster using DBSCAN or K-Means
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.15, min_samples=3).fit(embeddings)
        
        # Group anomalies by cluster
        clusters = {}
        for idx, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(anomalies[idx])
        
        return [PatternCluster(id=k, anomalies=v) for k, v in clusters.items()]
    
    # PHASE 3: EXPLAIN
    async def explain_root_causes(self, clusters: List[PatternCluster]) -> List[RootCauseAnalysis]:
        """Use SAP_LLM to explain why patterns fail"""
        
        analyses = []
        for cluster in clusters:
            # Sample representative documents from cluster
            sample_docs = cluster.anomalies[:5]
            
            # Use SAP_LLM to analyze failure pattern
            explanation_prompt = f"""
            Analyze the following {len(sample_docs)} failed document processing cases:
            
            {self._format_anomalies(sample_docs)}
            
            Provide:
                                                      1. Root cause of failure
                                                      2. Common patterns across failures
                                                      3. Recommended fix/rule to prevent future failures
            """
            
            explanation = await self.model.generate_explanation(explanation_prompt)
            
            analyses.append(RootCauseAnalysis(
                cluster_id=cluster.id,
                root_cause=explanation['root_cause'],
                common_patterns=explanation['patterns'],
                recommended_fix=explanation['fix']
            ))
        
        return analyses
    
    # PHASE 4: REVIEW
    async def review_proposed_changes(self, analyses: List[RootCauseAnalysis]) -> List[ApprovedChange]:
        """Governance gate: Review and approve changes"""
        
        approved_changes = []
        for analysis in analyses:
            # Automatically approve low-risk changes
            if analysis.risk_score < 0.3:
                approved_changes.append(ApprovedChange(
                    cluster_id=analysis.cluster_id,
                    change_type='rule_update',
                    change_spec=analysis.recommended_fix,
                    approval_status='auto_approved'
                ))
            else:
                # Queue for human review
                await self._queue_for_human_review(analysis)
        
        return approved_changes
    
    # PHASE 5: APPLY
    async def apply_improvements(self, changes: List[ApprovedChange]) -> ApplicationResult:
        """Deploy improvements to production"""
        
        results = []
        for change in changes:
            if change.change_type == 'rule_update':
                # Update validation rules
                await self._update_validation_rule(change.change_spec)
            
            elif change.change_type == 'model_retrain':
                # Trigger fine-tuning on failed examples
                await self._trigger_model_finetuning(change.change_spec)
            
            elif change.change_type == 'classification_pattern':
                # Add new classification pattern to Neo4j
                await self._add_classification_pattern(change.change_spec)
            
            results.append(f"Applied change {change.cluster_id}")
        
        return ApplicationResult(success=True, applied_changes=results)
    
    async def run_continuous_loop(self):
        """Run SHWL continuously"""
        while True:
            logger.info("🔄 Starting SHWL cycle")
            
            # Phase 1: Detect
            anomalies = await self.detect_anomalies()
            logger.info(f"Detected {len(anomalies)} anomalies")
            
            if len(anomalies) > 0:
                # Phase 2: Cluster
                clusters = await self.cluster_patterns(anomalies)
                logger.info(f"Clustered into {len(clusters)} patterns")
                
                # Phase 3: Explain
                analyses = await self.explain_root_causes(clusters)
                
                # Phase 4: Review
                approved = await self.review_proposed_changes(analyses)
                
                # Phase 5: Apply
                if len(approved) > 0:
                    await self.apply_improvements(approved)
            
            # Sleep for 1 hour before next cycle
            await asyncio.sleep(3600)
```

### 4.2 Continuous Learning

**Fine-tuning from Production Data:**

```python
# services/sap-llm/training/continuous_learner.py
class ContinuousLearner:
    """Continuously fine-tune SAP_LLM from production data"""
    
    async def collect_training_examples(self, days: int = 7) -> List[TrainingExample]:
        """Collect recent corrections and feedback"""
        
        # Get human corrections from PMG
        corrections = await self.pmg.get_human_corrections(days=days)
        
        # Get successful extractions with high confidence
        successful = await self.pmg.get_successful_extractions(
            days=days, 
            min_confidence=0.95
        )
        
        return corrections + successful
    
    async def fine_tune_model(self, training_examples: List[TrainingExample]):
        """Fine-tune model on new examples"""
        
        # Use LoRA for efficient fine-tuning
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        
        peft_model = get_peft_model(self.model, lora_config)
        
        # Fine-tune for 1 epoch
        trainer = Trainer(
            model=peft_model,
            train_dataset=training_examples,
            args=TrainingArguments(
                output_dir="./sap-llm-finetuned",
                num_train_epochs=1,
                per_device_train_batch_size=4,
                learning_rate=5e-5
            )
        )
        
        trainer.train()
        
        # Merge LoRA weights back into base model
        merged_model = peft_model.merge_and_unload()
        
        return merged_model
```

## PHASE 5: Agentic Process Orchestration Protocol (APOP) (Months 13-15)

### 5.1 APOP Architecture

**Whitepaper Specification:** CloudEvents-compatible messaging with self-routing

**Implementation:**

```python
# services/sap-llm/apop/orchestrator.py
from cloudevents.http import CloudEvent
import asyncio

class AgenticProcessOrchestrator:
    """APOP implementation for autonomous workflow execution"""
    
    def __init__(self, sap_llm_model, pmg: ProcessMemoryGraph):
        self.model = sap_llm_model
        self.pmg = pmg
        self.agents = {
            'classification_agent': ClassificationAgent(sap_llm_model),
            'extraction_agent': ExtractionAgent(sap_llm_model),
            'validation_agent': ValidationAgent(sap_llm_model),
            'routing_agent': RoutingAgent(sap_llm_model)
        }
    
    async def process_document_autonomously(self, document: Document) -> ProcessingResult:
        """Let agents decide next actions autonomously"""
        
        # Create initial CloudEvent
        event = CloudEvent({
            "type": "com.qorsync.document.received",
            "source": "inbox",
            "data": {
                "document_id": document.id,
                "filename": document.filename,
                "content": document.text,
                "next_action": "classify"  # Self-routing hint
            }
        })
        
        # Let agents process autonomously
        while event.data.get('next_action') != 'complete':
            next_action = event.data['next_action']
            
            if next_action == 'classify':
                event = await self.agents['classification_agent'].process(event)
            
            elif next_action == 'extract':
                event = await self.agents['extraction_agent'].process(event)
            
            elif next_action == 'validate':
                event = await self.agents['validation_agent'].process(event)
            
            elif next_action == 'route':
                event = await self.agents['routing_agent'].process(event)
            
            else:
                raise ValueError(f"Unknown action: {next_action}")
        
        return ProcessingResult(event.data)


class ClassificationAgent:
    """Autonomous classification agent"""
    
    def __init__(self, sap_llm_model):
        self.model = sap_llm_model
    
    async def process(self, event: CloudEvent) -> CloudEvent:
        """Classify document and decide next action"""
        
        document_text = event.data['content']
        
        # Use SAP_LLM to classify
        classification = await self.model.classify(document_text)
        
        # Store in event
        event.data['document_type'] = classification['doc_type']
        event.data['subtype'] = classification['subtype']
        event.data['confidence'] = classification['confidence']
        
        # Decide next action based on confidence
        if classification['confidence'] > 0.9:
            event.data['next_action'] = 'extract'
        else:
            event.data['next_action'] = 'human_review'
        
        # Emit new CloudEvent
        return CloudEvent({
            "type": "com.qorsync.document.classified",
            "source": "classification_agent",
            "data": event.data
        })


class ExtractionAgent:
    """Autonomous extraction agent"""
    
    async def process(self, event: CloudEvent) -> CloudEvent:
        """Extract fields and decide next action"""
        
        document_text = event.data['content']
        document_type = event.data['document_type']
        
        # Use SAP_LLM to extract fields
        extraction = await self.model.extract_fields(
            text=document_text,
            doc_type=document_type
        )
        
        event.data['extracted_fields'] = extraction['fields']
        event.data['field_confidences'] = extraction['confidences']
        
        # Decide next action
        avg_confidence = sum(extraction['confidences'].values()) / len(extraction['confidences'])
        
        if avg_confidence > 0.85:
            event.data['next_action'] = 'validate'
        else:
            event.data['next_action'] = 'human_review'
        
        return CloudEvent({
            "type": "com.qorsync.document.extracted",
            "source": "extraction_agent",
            "data": event.data
        })
```

### 5.2 Agent Communication via Kafka + CloudEvents

**Replace existing Kafka topics with CloudEvents:**

```python
# services/sap-llm/apop/cloudevents_kafka.py
from cloudevents.http import to_structured
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

class CloudEventsKafkaPublisher:
    """Publish CloudEvents to Kafka"""
    
    async def publish(self, event: CloudEvent, topic: str):
        """Publish CloudEvent to Kafka topic"""
        
        # Convert CloudEvent to structured JSON
        event_json = to_structured(event)
        
        producer = AIOKafkaProducer(
            bootstrap_servers='kafka:9092'
        )
        
        await producer.start()
        await producer.send(topic, event_json)
        await producer.stop()
```

## PHASE 6: Deployment & Integration (Months 16-18)

### 6.1 Model Serving Infrastructure

**Inference Server Options:**

**Option A: FastAPI + Triton Inference Server (Recommended)**

- FastAPI for REST API
- NVIDIA Triton for optimized inference
- Support for TensorRT optimization
- Batch inference for throughput

**Option B: vLLM (For LLM-based architecture)**

- High-throughput LLM serving
- Continuous batching
- PagedAttention for memory efficiency

**Implementation:**

```python
# services/sap-llm/inference/server.py
from fastapi import FastAPI, UploadFile, File
import torch
from transformers import AutoModel, AutoTokenizer

app = FastAPI(title="SAP_LLM Inference Server")

# Load model once at startup
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model = AutoModel.from_pretrained("./sap-llm-final-checkpoint")
    tokenizer = AutoTokenizer.from_pretrained("./sap-llm-final-checkpoint")
    
    # Move to GPU
    model = model.to("cuda")
    model.eval()

@app.post("/process-document")
async def process_document(file: UploadFile = File(...)):
    """Process document through all 8 stages"""
    
    # Read file
    content = await file.read()
    text = extract_text(content)
    
    # Stage 1: Inbox (metadata extraction)
    metadata = extract_metadata(content)
    
    # Stage 2: Preprocessing (OCR if needed)
    if needs_ocr(content):
        text = await ocr_text(content)
    
    # Stages 3-8: Use SAP_LLM
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
    
    # Parse outputs for each stage
    classification = parse_classification(outputs)
    extraction = parse_extraction(outputs)
    validation = parse_validation(outputs)
    sap_payload = generate_sap_payload(extraction)
    
    return {
        "document_type": classification['doc_type'],
        "subtype": classification['subtype'],
        "extracted_fields": extraction,
        "validation_results": validation,
        "sap_payload": sap_payload
    }
```

### 6.2 Replace Existing Services

**Migration Strategy:**

1. **Deploy SAP_LLM inference server** as new service:

                        - `services/sap-llm-inference/` (port 8020)

2. **Create adapter layer** to gradually migrate:

                        - Start with Stage 5 (Extraction) - replace Claude/GPT-4
                        - Then Stage 3 (Classification) - replace Neo4j + AI
                        - Then Stage 4 (Type Identifier) - replace PO identifier
                        - Then Stage 7 (Validation) - replace rules engine
                        - Finally Stage 8 (Routing) - replace SAP integration logic

3. **A/B testing**:

                        - Route 10% of traffic to SAP_LLM
                        - Compare accuracy and latency
                        - Gradually increase to 100%

4. **Remove third-party dependencies**:

                        - Remove OpenAI API calls from `ultimate_extraction.py`
                        - Remove Claude API calls from `enhanced_processor.py`
                        - Keep fallback for 6 months during transition

### 6.3 Docker Compose Integration

**Add SAP_LLM service to docker-compose:**

```yaml
# compose/docker-compose-enhanced.yml
services:
  sap-llm-inference:
    build:
      context: ../services/sap-llm
      dockerfile: Dockerfile
    networks:
                           - QorSync_net
    ports:
                           - "8020:8020"
    environment:
                           - MODEL_PATH=/models/sap-llm-final
                           - CUDA_VISIBLE_DEVICES=0
                           - BATCH_SIZE=8
                           - MAX_TOKENS=4096
    volumes:
                           - ../models/sap-llm:/models/sap-llm-final
    deploy:
      resources:
        reservations:
          devices:
                                                      - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
                           - neo4j
                           - redis
                           - postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8020/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## PHASE 7: Testing & Validation (Months 19-21)

### 7.1 Comprehensive Testing

**Test Suites:**

1. **Unit Tests:**

                        - Each pipeline stage (8 stages × 100 tests = 800 unit tests)
                        - PMG operations (50 tests)
                        - SHWL cycle (30 tests)
                        - APOP agent behaviors (40 tests)

2. **Integration Tests:**

                        - End-to-end document processing (200 test documents)
                        - SAP API integration (50 test scenarios)
                        - Multi-document batch processing (10 batches)

3. **Accuracy Tests:**

                        - Classification: 10K test documents → 99%+ accuracy target
                        - Extraction: 5K test documents → 95%+ field accuracy target
                        - Validation: 3K test scenarios → 98%+ accuracy target

4. **Performance Tests:**

                        - Latency: Average processing time < 10s per document
                        - Throughput: 100+ documents/minute per GPU
                        - Memory: < 16GB GPU memory per request

**Implementation:**

```python
# tests/test_sap_llm_pipeline.py
import pytest

@pytest.mark.parametrize("document_type", ["PO", "INVOICE", "RECEIPT"])
async def test_end_to_end_processing(document_type):
    """Test complete pipeline for each document type"""
    
    # Load test document
    test_doc = load_test_document(document_type)
    
    # Process with SAP_LLM
    result = await sap_llm_inference.process_document(test_doc)
    
    # Validate results
    assert result['document_type'] == document_type
    assert result['confidence'] > 0.9
    assert len(result['extracted_fields']) > 10
    assert result['validation_results']['passed'] == True
    assert result['sap_payload'] is not None

@pytest.mark.benchmark
async def test_processing_latency():
    """Benchmark processing latency"""
    
    documents = load_benchmark_documents(100)
    
    start = time.time()
    results = await asyncio.gather(*[
        sap_llm_inference.process_document(doc) 
        for doc in documents
    ])
    elapsed = time.time() - start
    
    avg_latency = elapsed / len(documents)
    assert avg_latency < 10.0  # Less than 10 seconds per document
```

### 7.2 Production Readiness Checklist

- [ ] Model achieves 99%+ classification accuracy on test set
- [ ] Model achieves 95%+ extraction accuracy on test set
- [ ] Average processing latency < 10 seconds
- [ ] Throughput > 100 documents/minute per GPU
- [ ] PMG stores all processing contexts correctly
- [ ] SHWL detects and fixes anomalies automatically
- [ ] APOP agents orchestrate workflows autonomously
- [ ] All 8 pipeline stages functional without third-party APIs
- [ ] Docker deployment tested and stable
- [ ] Integration with existing QorSync system complete
- [ ] A/B testing shows equal or better performance vs. GPT-4/Claude
- [ ] Cost per document < $0.10 (vs. $0.80 with third-party APIs)

## Cost Estimation & ROI

### Development Costs

| Phase | Duration | Resources | Cost |

|-------|----------|-----------|------|

| Data Collection | 3 months | 2 engineers | $60K |

| Model Training | 3 months | 8x H100 GPUs + 2 ML engineers | $200K |

| PMG Integration | 3 months | 2 engineers | $60K |

| SHWL Development | 3 months | 2 engineers | $60K |

| APOP Development | 3 months | 2 engineers | $60K |

| Deployment & Testing | 3 months | 3 engineers | $90K |

| **TOTAL** | **21 months** | | **$530K** |

### Ongoing Costs

| Item | Monthly Cost | Annual Cost |

|------|--------------|-------------|

| GPU Infrastructure (4x H100) | $10K | $120K |

| Fine-tuning & Retraining | $2K | $24K |

| Storage (models + PMG) | $1K | $12K |

| Engineering Maintenance | $15K | $180K |

| **TOTAL** | **$28K/month** | **$336K/year** |

### ROI Analysis

**Current System (with GPT-4/Claude):**

- Cost per document: $0.80
- Monthly volume: 50K documents
- Monthly cost: $40K
- Annual cost: $480K

**SAP_LLM System:**

- Cost per document: $0.08 (10x reduction)
- Monthly volume: 50K documents
- Monthly cost: $4K (inference only)
- Annual cost: $48K + $336K infrastructure = $384K

**Net Savings:** $480K - $384K = **$96K/year**

**Payback Period:** $530K / $96K = **5.5 years**

**Additional Benefits (not quantified):**

- No third-party API dependency risk
- Complete data privacy (no data sent to OpenAI/Anthropic)
- Customization and control
- Self-improving system (SHWL)
- Competitive advantage

## Success Metrics

### Technical Metrics

- Classification accuracy: 99%+
- Extraction accuracy: 95%+
- Average latency: < 10 seconds
- Throughput: 100+ docs/minute/GPU
- Model size: 7B-13B parameters
- GPU memory: < 16GB per request

### Business Metrics

- Cost per document: < $0.10
- Processing accuracy vs. GPT-4: Equal or better
- System uptime: 99.9%+
- SHWL fix rate: 80%+ of anomalies auto-fixed
- Customer satisfaction: 4.5+ / 5.0

### Operational Metrics

- Model retraining frequency: Weekly
- SHWL cycle frequency: Hourly
- PMG storage growth: < 100GB/month
- Human review rate: < 5% of documents

## Risk Mitigation

### Technical Risks

1. **Model doesn't achieve target accuracy**

                        - Mitigation: Keep GPT-4/Claude fallback for 12 months
                        - Fallback plan: Use ensemble of SAP_LLM + GPT-4

2. **Training costs exceed budget**

                        - Mitigation: Use QLoRA for efficient training
                        - Fallback plan: Start with smaller 7B model

3. **Inference latency too high**

                        - Mitigation: Use TensorRT optimization, batch inference
                        - Fallback plan: Deploy multiple GPU instances

### Business Risks

1. **Long payback period (5.5 years)**

                        - Mitigation: Focus on competitive advantage, not just cost savings
                        - Additional value: Data privacy, customization, independence

2. **Resource intensive (21 months)**

                        - Mitigation: Phased rollout, start with highest-value stages
                        - Prioritize: Extraction (Stage 5) → Classification (Stage 3) → Others

## Conclusion

Developing SAP_LLM from scratch is a **major investment** requiring:

- **$530K** initial development cost
- **21 months** development timeline
- **Specialized ML engineering team** (2-3 engineers)
- **Significant GPU infrastructure** (8x H100 for training, 4x H100 for production)

However, it provides:

- **Complete independence** from third-party LLM APIs
- **Full data privacy** (no data sent to external providers)
- **Self-improving system** via SHWL
- **Customization** for SAP-specific workflows
- **Long-term cost savings** after 5.5 year payback period

**Recommendation:**

For immediate ROI, consider **hybrid approach**:

1. **Phase 1 (Months 1-6):** Fine-tune open-source model (Llama 3.1 8B) on QorSync data → 70% cost reduction, 6-month timeline, $150K cost
2. **Phase 2 (Months 7-12):** Implement PMG + SHWL on fine-tuned model → Self-improving system
3. **Phase 3 (Months 13-21):** Full custom training if Phase 1-2 proves successful

This phased approach reduces risk while delivering immediate value.