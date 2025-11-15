# Orchestrator PMG Integration - Summary of Changes

## Overview
Fixed two TODO items in `sap_llm/apop/orchestrator.py` to integrate Process Memory Graph (PMG) for intelligent workflow orchestration and status tracking.

## Changes Made

### 1. TODO Item #1 (Line 175): Query PMG for Similar Workflows

**File:** `/home/user/SAP_LLM/sap_llm/apop/orchestrator.py`

**Method:** `_get_pmg_suggested_action(self, envelope: APOPEnvelope) -> Optional[str]`

**Implementation:**
- Extracts document metadata from envelope (doc_type, supplier_id, company_code)
- Queries PMG for similar successful routing decisions using `pmg.get_similar_routing()`
- Analyzes historical patterns using statistical analysis (Counter)
- Suggests most common next action based on historical data
- Includes comprehensive error handling with fallback to None

**New Helper Method:** `_endpoint_to_action(self, endpoint: str) -> Optional[str]`
- Converts API endpoints to action hints
- Maps endpoints like `/api/extract` to `extract.fields`
- Supports: extract, classify, validate, quality, route, ocr

**Key Features:**
- Intelligent routing based on historical workflow patterns
- Graceful degradation when PMG data is unavailable
- Detailed logging for debugging and monitoring
- Statistical analysis to find most common action patterns

---

### 2. TODO Item #2 (Line 269): Query PMG for Workflow Status

**File:** `/home/user/SAP_LLM/sap_llm/apop/orchestrator.py`

**Method:** `get_workflow_status(self, correlation_id: str) -> Dict[str, Any]`

**Implementation:**
- Queries PMG for complete workflow history by correlation_id
- Retrieves workflow steps, routing decisions, responses, and exceptions
- Determines workflow status (active, completed, failed, not_found, error)
- Calculates steps completed and identifies current step
- Checks for exceptions and success indicators
- Provides comprehensive error handling with detailed fallback responses

**Return Structure:**
```python
{
    "correlation_id": str,
    "status": str,  # active, completed, failed, not_found, unknown, error
    "steps_completed": int,
    "current_step": Optional[str],
    "steps": List[Dict],
    "exceptions": List[Dict],
    "success": Optional[bool],
    "document": Dict,
    "error": Optional[str]  # Only on error
}
```

**Key Features:**
- Complete workflow visibility
- Step-by-step tracking
- Exception detection
- Success/failure determination
- Graceful handling when PMG is unavailable

---

### 3. PMG Extension: Workflow Query Methods

**File:** `/home/user/SAP_LLM/sap_llm/pmg/graph_client.py`

Added two new methods to support workflow status queries:

#### Method 1: `get_workflow_by_correlation_id(self, correlation_id: str) -> Optional[Dict[str, Any]]`
- Retrieves complete workflow history from graph database
- Uses Gremlin graph traversal to fetch:
  - Document vertex
  - Connected routing decisions
  - SAP responses
  - Raised exceptions
- Returns structured workflow data with all relationships

**Gremlin Query:**
```gremlin
g.V().hasLabel('Document')
  .has('id', '{correlation_id}')
  .project('document', 'routing_decisions', 'responses', 'exceptions')
  .by(valueMap())
  .by(out('ROUTED_TO').valueMap().fold())
  .by(out('ROUTED_TO').out('GOT_RESPONSE').valueMap().fold())
  .by(out('RAISED_EXCEPTION').valueMap().fold())
```

#### Method 2: `get_workflow_steps(self, correlation_id: str) -> List[Dict[str, Any]]`
- Extracts ordered workflow steps from workflow history
- Builds chronological step list from routing decisions
- Includes: step number, endpoint, timestamp, confidence, status

**Step Structure:**
```python
{
    "step_number": int,
    "endpoint": str,
    "timestamp": str,
    "confidence": float,
    "status": "completed"
}
```

---

## Integration Architecture

### Workflow Orchestration Priority
The orchestrator now uses a 4-tier priority system for determining next actions:

1. **Explicit Hint** - `next_action_hint` in envelope (highest priority)
2. **PMG Patterns** - Historical workflow data from similar documents
3. **Default Flow** - Pre-configured routing rules
4. **Agent Capabilities** - Agent subscriptions (fallback)

### Error Handling Strategy
All PMG integrations include multi-layer error handling:

1. **Null Checks** - Validates PMG availability before queries
2. **Try-Catch Blocks** - Comprehensive exception handling
3. **Fallback Mechanisms** - Graceful degradation when PMG unavailable
4. **Detailed Logging** - Debug, info, warning, and error logs
5. **Mock Mode Support** - Works without Cosmos DB connection

---

## Files Modified

1. **`/home/user/SAP_LLM/sap_llm/apop/orchestrator.py`**
   - Enhanced `_get_pmg_suggested_action()` method (~96 lines)
   - Added `_endpoint_to_action()` helper method (~29 lines)
   - Implemented `get_workflow_status()` method (~115 lines)
   - Total: ~240 lines of new/modified code

2. **`/home/user/SAP_LLM/sap_llm/pmg/graph_client.py`**
   - Added `get_workflow_by_correlation_id()` method (~59 lines)
   - Added `get_workflow_steps()` method (~42 lines)
   - Total: ~101 lines of new code

---

## Testing & Verification

### Code Validation
- ✅ Python syntax validation passed (`py_compile`)
- ✅ No TODO items remaining in orchestrator.py
- ✅ All imports properly configured
- ✅ Type hints properly defined

### Verification Script
Created `/home/user/SAP_LLM/verify_orchestrator_pmg_integration.py`:
- Tests PMG suggested action functionality
- Tests workflow status query functionality
- Tests next action determination priority
- Tests endpoint to action mapping
- Runs in mock mode (no Cosmos DB required)

### Mock Mode Testing
All functionality works in mock mode:
- PMG gracefully returns empty results
- Orchestrator falls back to default flow
- No errors or crashes when PMG unavailable

---

## Benefits & Features

### Intelligent Orchestration
- **Historical Learning** - Learns from past successful workflows
- **Pattern Recognition** - Identifies common routing patterns
- **Supplier-Specific** - Adapts to supplier-specific workflows
- **Company-Specific** - Handles company code variations

### Workflow Visibility
- **Complete Tracking** - Full workflow history by correlation_id
- **Step-by-Step** - Chronological step progression
- **Exception Monitoring** - Tracks all exceptions raised
- **Success Metrics** - Determines workflow success/failure

### Production Readiness
- **Error Resilience** - Comprehensive error handling
- **Graceful Degradation** - Works without PMG when needed
- **Detailed Logging** - Full observability for debugging
- **Mock Mode Support** - Easy development and testing

---

## Usage Examples

### Example 1: PMG-Guided Orchestration
```python
from sap_llm.apop.orchestrator import AgenticOrchestrator
from sap_llm.apop.envelope import create_envelope
from sap_llm.pmg.graph_client import ProcessMemoryGraph

# Initialize with PMG
pmg = ProcessMemoryGraph(
    endpoint="wss://...",
    key="..."
)
orchestrator = AgenticOrchestrator(pmg=pmg)

# Create envelope
envelope = create_envelope(
    source="classifier",
    event_type="classify.done",
    data={
        "doc_type": "SUPPLIER_INVOICE",
        "supplier_id": "VENDOR_001",
        "company_code": "1000"
    }
)

# Process - PMG will suggest next action based on history
envelopes = await orchestrator.process_envelope(envelope)
```

### Example 2: Workflow Status Tracking
```python
# Get workflow status
status = orchestrator.get_workflow_status("correlation-id-123")

print(f"Status: {status['status']}")
print(f"Steps: {status['steps_completed']}")
print(f"Current: {status['current_step']}")
print(f"Success: {status['success']}")

# Check for exceptions
if status['exceptions']:
    print(f"Exceptions: {len(status['exceptions'])}")
```

---

## Performance Considerations

- **Query Limits** - PMG queries limited to 20 similar workflows
- **Caching Opportunity** - Consider caching similar routing results
- **Graph Traversal** - Optimized Gremlin queries with projections
- **Mock Mode** - Zero overhead when PMG unavailable

---

## Future Enhancements

Potential improvements for future iterations:

1. **Caching Layer** - Cache PMG query results to reduce database calls
2. **Confidence Scoring** - Add confidence scores to PMG suggestions
3. **A/B Testing** - Compare PMG suggestions vs default flow
4. **Learning Feedback** - Update PMG with orchestration outcomes
5. **Multi-Tenant Support** - Tenant-specific workflow patterns
6. **Analytics Dashboard** - Visualize workflow patterns and trends

---

## Conclusion

Both TODO items have been successfully implemented with:
- ✅ PMG integration for similar workflow queries
- ✅ Workflow status and history retrieval
- ✅ Proper error handling and fallbacks
- ✅ Enhanced workflow optimization with historical data
- ✅ Production-ready code with comprehensive logging
- ✅ Full mock mode support for development

The orchestrator now leverages PMG's historical data to make intelligent routing decisions while maintaining robustness through comprehensive error handling and fallback mechanisms.
