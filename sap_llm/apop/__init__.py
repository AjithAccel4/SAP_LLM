"""
APOP - Agentic Process Orchestration Protocol

CloudEvents-compliant envelope and routing system for autonomous agent orchestration.

Features:
- CloudEvents 1.0 compliance
- ECDSA signature verification
- Agent-to-agent communication
- Service Bus integration
- Distributed tracing (W3C Trace Context)
"""

from sap_llm.apop.envelope import APOPEnvelope, create_envelope
from sap_llm.apop.orchestrator import AgenticOrchestrator
from sap_llm.apop.agent import BaseAgent
from sap_llm.apop.signature import sign_envelope, verify_signature

__all__ = [
    "APOPEnvelope",
    "create_envelope",
    "AgenticOrchestrator",
    "BaseAgent",
    "sign_envelope",
    "verify_signature",
]
