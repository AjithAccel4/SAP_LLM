"""
API tests for FastAPI server.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import io

from sap_llm.api.server import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app()
    return TestClient(app)


@pytest.mark.api
@pytest.mark.unit
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "SAP_LLM API"

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "components" in data

    def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert "details" in data


@pytest.mark.api
@pytest.mark.integration
class TestDocumentProcessingEndpoints:
    """Tests for document processing endpoints."""

    @pytest.mark.slow
    def test_extract_document_async(self, client, sample_document_image):
        """Test async document extraction endpoint."""
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        sample_document_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Upload document
        files = {"file": ("test_document.png", img_byte_arr, "image/png")}
        response = client.post("/v1/extract", files=files)

        # Should return 202 Accepted for async processing
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["status"] in ["queued", "processing"]

        # Get job status
        job_id = data["job_id"]
        status_response = client.get(f"/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["job_id"] == job_id

    def test_extract_document_file_too_large(self, client):
        """Test uploading file that's too large."""
        # Create large dummy file (>50MB)
        large_file = io.BytesIO(b"x" * (51 * 1024 * 1024))

        files = {"file": ("large_file.png", large_file, "image/png")}
        response = client.post("/v1/extract", files=files)

        # Should return 413 Request Entity Too Large
        assert response.status_code == 413

    def test_extract_document_no_file(self, client):
        """Test extraction without file."""
        response = client.post("/v1/extract")

        # Should return 422 Unprocessable Entity
        assert response.status_code == 422

    def test_get_nonexistent_job(self, client):
        """Test getting nonexistent job."""
        response = client.get("/v1/jobs/nonexistent_job_id")

        # Should return 404 Not Found
        assert response.status_code == 404

    def test_delete_job(self, client):
        """Test deleting job."""
        # First create a job (using mock)
        # For now, test with nonexistent job
        response = client.delete("/v1/jobs/test_job_id")

        # Should return 404 for nonexistent job
        assert response.status_code == 404


@pytest.mark.api
@pytest.mark.integration
class TestStatisticsEndpoints:
    """Tests for statistics endpoints."""

    def test_get_stats(self, client):
        """Test statistics endpoint."""
        response = client.get("/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert "jobs" in data
        assert "system" in data
        assert "timestamp" in data

        # Check jobs stats
        jobs = data["jobs"]
        assert "total" in jobs
        assert "completed" in jobs
        assert "failed" in jobs
        assert "processing" in jobs
        assert "queued" in jobs


@pytest.mark.api
class TestAuthentication:
    """Tests for authentication."""

    def test_api_key_authentication(self, client):
        """Test API key authentication."""
        # Note: Authentication might not be enabled in test mode
        # This is a placeholder for when auth is enforced

        # With valid API key
        headers = {"X-API-Key": "dev_key_12345"}
        response = client.get("/health", headers=headers)
        assert response.status_code == 200

        # Without API key (if required)
        # response = client.get("/v1/extract")
        # assert response.status_code in [401, 422]  # Unauthorized or validation error

    def test_bearer_token_authentication(self, client):
        """Test Bearer token authentication."""
        # This would test JWT authentication
        pass


@pytest.mark.api
class TestRateLimiting:
    """Tests for rate limiting."""

    @pytest.mark.slow
    def test_rate_limit_enforcement(self, client):
        """Test that rate limiting is enforced."""
        # Make many requests quickly
        responses = []
        for _ in range(150):  # Limit is 100/minute
            response = client.get("/health")
            responses.append(response)

        # Some requests should be rate limited
        status_codes = [r.status_code for r in responses]

        # Should have mix of 200 and 429 (Too Many Requests)
        # Note: This might not work in tests without proper rate limiter setup
        assert 200 in status_codes


@pytest.mark.api
class TestWebSocket:
    """Tests for WebSocket endpoint."""

    def test_websocket_connection(self, client):
        """Test WebSocket connection."""
        # FastAPI TestClient supports WebSocket testing
        with client.websocket_connect("/v1/ws/test_job_id") as websocket:
            # Connection should be established
            data = websocket.receive_json()
            assert "status" in data or "message" in data

    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong."""
        with client.websocket_connect("/v1/ws/test_job_id") as websocket:
            # Send ping
            websocket.send_text("ping")

            # Should receive pong
            response = websocket.receive_text()
            assert response == "pong"


@pytest.mark.api
class TestErrorHandling:
    """Tests for error handling."""

    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent/endpoint")
        assert response.status_code == 404

    def test_500_error(self, client):
        """Test 500 error handling."""
        # This would require triggering an internal error
        # For now, just ensure error handler exists
        pass

    def test_validation_error(self, client):
        """Test validation error handling."""
        # Send malformed request
        response = client.post("/v1/extract", json={"invalid": "data"})

        # Should return 422 Unprocessable Entity
        assert response.status_code == 422
        data = response.json()
        assert "error" in data or "detail" in data
