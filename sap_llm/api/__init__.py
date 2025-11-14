"""
FastAPI Server for SAP_LLM

Production-ready REST API with:
- Document processing endpoints
- Health checks
- Authentication
- Rate limiting
- WebSocket support for real-time updates
"""

from sap_llm.api.server import app, create_app

__all__ = ["app", "create_app"]
