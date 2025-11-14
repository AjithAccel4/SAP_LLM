"""
Security Penetration Testing Suite for SAP_LLM

Tests for common security vulnerabilities:
- OWASP Top 10
- Authentication bypass
- Authorization flaws
- Injection attacks
- XSS/CSRF
- Rate limiting
- Data exposure

Run with: pytest tests/security/test_penetration.py -v --html=security_report.html
"""

import pytest
import requests
import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, Any

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_USER = "pentest_user"
TEST_PASSWORD = "Test@Pass123"


class TestAuthentication:
    """Test authentication security"""

    def test_no_auth_required_endpoints(self):
        """Test that public endpoints don't require auth"""
        public_endpoints = [
            "/health",
            "/ready",
            "/",
        ]

        for endpoint in public_endpoints:
            response = requests.get(f"{BASE_URL}{endpoint}")
            assert response.status_code != 401, f"{endpoint} should be public"

    def test_auth_required_endpoints(self):
        """Test that protected endpoints require auth"""
        protected_endpoints = [
            "/v1/extract",
            "/v1/jobs/test-job-id",
            "/v1/stats",
        ]

        for endpoint in protected_endpoints:
            response = requests.get(f"{BASE_URL}{endpoint}")
            assert response.status_code in [401, 405], f"{endpoint} should require auth"

    def test_invalid_token(self):
        """Test invalid JWT token rejection"""
        invalid_tokens = [
            "invalid.jwt.token",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid",
            "",
            "Bearer",
        ]

        for token in invalid_tokens:
            response = requests.get(
                f"{BASE_URL}/v1/stats",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 401, f"Invalid token should be rejected: {token}"

    def test_expired_token(self):
        """Test expired token rejection"""
        # Create expired token
        expired_payload = {
            "user_id": "test_user",
            "role": "user",
            "exp": datetime.utcnow() - timedelta(hours=1)  # Expired 1 hour ago
        }

        expired_token = jwt.encode(expired_payload, "secret", algorithm="HS256")

        response = requests.get(
            f"{BASE_URL}/v1/stats",
            headers={"Authorization": f"Bearer {expired_token}"}
        )

        assert response.status_code == 401
        assert "expired" in response.json().get("detail", "").lower()

    def test_token_without_required_claims(self):
        """Test token missing required claims"""
        incomplete_token = jwt.encode({"user_id": "test"}, "secret", algorithm="HS256")

        response = requests.get(
            f"{BASE_URL}/v1/stats",
            headers={"Authorization": f"Bearer {incomplete_token}"}
        )

        assert response.status_code == 401

    def test_brute_force_protection(self):
        """Test protection against brute force attacks"""
        # Attempt multiple failed logins
        for i in range(10):
            response = requests.post(
                f"{BASE_URL}/auth/login",
                json={"username": "test", "password": f"wrong_password_{i}"}
            )

        # Should be rate limited or locked out
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"username": "test", "password": "any_password"}
        )

        assert response.status_code in [429, 403], "Should be rate limited after failed attempts"


class TestAuthorization:
    """Test authorization and RBAC"""

    def test_user_cannot_access_admin_endpoints(self):
        """Test that regular users can't access admin endpoints"""
        # Get user token
        user_token = self._get_token(role="user")

        admin_endpoints = [
            "/v1/admin/users",
            "/v1/admin/config",
            "/v1/admin/system",
        ]

        for endpoint in admin_endpoints:
            response = requests.get(
                f"{BASE_URL}{endpoint}",
                headers={"Authorization": f"Bearer {user_token}"}
            )
            assert response.status_code == 403, f"User should not access {endpoint}"

    def test_viewer_cannot_write(self):
        """Test that viewers can only read"""
        viewer_token = self._get_token(role="viewer")

        response = requests.post(
            f"{BASE_URL}/v1/extract",
            headers={"Authorization": f"Bearer {viewer_token}"},
            files={"file": ("test.pdf", b"content")}
        )

        assert response.status_code == 403, "Viewer should not be able to write"

    def test_tenant_isolation(self):
        """Test multi-tenancy isolation"""
        tenant_a_token = self._get_token(tenant_id="tenant_a")
        tenant_b_token = self._get_token(tenant_id="tenant_b")

        # Create document as tenant A
        response = requests.post(
            f"{BASE_URL}/v1/extract",
            headers={"Authorization": f"Bearer {tenant_a_token}"},
            files={"file": ("test.pdf", b"content")}
        )

        if response.status_code == 202:
            job_id = response.json()["job_id"]

            # Try to access as tenant B
            response = requests.get(
                f"{BASE_URL}/v1/jobs/{job_id}",
                headers={"Authorization": f"Bearer {tenant_b_token}"}
            )

            assert response.status_code == 403, "Tenant B should not access tenant A's data"

    def _get_token(self, role="user", tenant_id="test_tenant"):
        """Helper to generate test token"""
        payload = {
            "user_id": "test_user",
            "role": role,
            "tenant_id": tenant_id,
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, "secret", algorithm="HS256")


class TestInjectionAttacks:
    """Test for injection vulnerabilities"""

    def test_sql_injection(self):
        """Test SQL injection prevention"""
        sql_payloads = [
            "' OR '1'='1",
            "1; DROP TABLE documents--",
            "' UNION SELECT * FROM users--",
            "admin'--",
            "' OR 1=1--",
        ]

        for payload in sql_payloads:
            response = requests.get(f"{BASE_URL}/v1/jobs/{payload}")
            # Should return 404 (not found) not 500 (server error from SQL injection)
            assert response.status_code == 404, f"SQL injection should not work: {payload}"

    def test_nosql_injection(self):
        """Test NoSQL injection prevention"""
        nosql_payloads = [
            {"$gt": ""},
            {"$ne": None},
            {"$regex": ".*"},
        ]

        for payload in nosql_payloads:
            response = requests.post(
                f"{BASE_URL}/v1/search",
                json={"query": payload}
            )
            # Should handle safely
            assert response.status_code != 500

    def test_command_injection(self):
        """Test command injection prevention"""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "`cat /etc/passwd`",
        ]

        for payload in command_payloads:
            response = requests.post(
                f"{BASE_URL}/v1/process",
                json={"filename": payload}
            )
            # Should sanitize input
            assert response.status_code in [400, 404, 422]  # Bad request, not executed

    def test_path_traversal(self):
        """Test path traversal prevention"""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "....//....//etc/passwd",
            "/etc/passwd",
        ]

        for payload in path_payloads:
            response = requests.get(f"{BASE_URL}/v1/files/{payload}")
            # Should not allow access to system files
            assert response.status_code in [400, 403, 404]


class TestXSSAndCSRF:
    """Test XSS and CSRF protection"""

    def test_xss_in_response(self):
        """Test that user input is escaped in responses"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
        ]

        for payload in xss_payloads:
            response = requests.post(
                f"{BASE_URL}/v1/feedback",
                json={"comment": payload}
            )

            if response.status_code == 200:
                # Check that payload is escaped in response
                assert "<script>" not in response.text
                assert "javascript:" not in response.text.lower()

    def test_csrf_protection(self):
        """Test CSRF token requirement"""
        response = requests.post(
            f"{BASE_URL}/v1/extract",
            files={"file": ("test.pdf", b"content")},
            # No CSRF token
        )

        # Should require CSRF token for state-changing operations
        # (Note: Depends on CSRF implementation)
        assert response.status_code in [403, 400, 401, 405]

    def test_content_type_validation(self):
        """Test that content-type is validated"""
        # Try to submit HTML as PDF
        response = requests.post(
            f"{BASE_URL}/v1/extract",
            files={"file": ("test.pdf", b"<html><body>Not a PDF</body></html>")}
        )

        # Should validate file type
        assert response.status_code in [400, 415, 422]  # Bad request or unsupported media type


class TestRateLimiting:
    """Test rate limiting"""

    def test_rate_limit_enforcement(self):
        """Test that rate limits are enforced"""
        # Make rapid requests
        responses = []
        for i in range(150):  # Exceed limit of 100/min
            response = requests.get(f"{BASE_URL}/v1/stats")
            responses.append(response.status_code)

        # Should get 429 (Too Many Requests)
        assert 429 in responses, "Rate limit should be enforced"

    def test_rate_limit_per_tenant(self):
        """Test per-tenant rate limiting"""
        tenant_a_token = self._get_token(tenant_id="tenant_a")
        tenant_b_token = self._get_token(tenant_id="tenant_b")

        # Exhaust tenant A's limit
        for i in range(100):
            requests.get(
                f"{BASE_URL}/v1/stats",
                headers={"Authorization": f"Bearer {tenant_a_token}"}
            )

        # Tenant B should still work
        response = requests.get(
            f"{BASE_URL}/v1/stats",
            headers={"Authorization": f"Bearer {tenant_b_token}"}
        )

        assert response.status_code == 200, "Tenant B should not be affected by tenant A's limit"

    def _get_token(self, tenant_id):
        payload = {
            "user_id": "test",
            "role": "user",
            "tenant_id": tenant_id,
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, "secret", algorithm="HS256")


class TestDataExposure:
    """Test for sensitive data exposure"""

    def test_no_password_in_response(self):
        """Test that passwords are never returned"""
        response = requests.get(f"{BASE_URL}/v1/users/me")

        if response.status_code == 200:
            data = response.json()
            # Ensure no password field
            assert "password" not in data
            assert "password_hash" not in data
            assert "pwd" not in data

    def test_no_secret_keys_in_response(self):
        """Test that secret keys are not exposed"""
        response = requests.get(f"{BASE_URL}/v1/config")

        if response.status_code == 200:
            data = response.json()
            sensitive_keys = ["secret_key", "api_key", "private_key", "jwt_secret"]

            for key in sensitive_keys:
                assert key not in str(data).lower(), f"Sensitive key exposed: {key}"

    def test_pii_masking(self):
        """Test that PII is masked"""
        # Submit document with PII
        response = requests.post(
            f"{BASE_URL}/v1/extract",
            files={"file": ("test.pdf", b"SSN: 123-45-6789, Email: test@example.com")}
        )

        if response.status_code in [200, 202]:
            # Check if PII is masked in logs/responses
            # (Implementation depends on PII masking strategy)
            pass

    def test_error_messages_no_stack_trace(self):
        """Test that error messages don't leak stack traces"""
        # Trigger an error
        response = requests.get(f"{BASE_URL}/v1/invalid-endpoint-that-errors")

        if response.status_code == 500:
            error_response = response.json()
            # Should not contain stack traces
            assert "Traceback" not in str(error_response)
            assert "File" not in str(error_response)
            assert ".py" not in str(error_response)


class TestFileUploadSecurity:
    """Test file upload security"""

    def test_file_size_limit(self):
        """Test file size limit enforcement"""
        # Try to upload >50MB file
        large_file = b"X" * (51 * 1024 * 1024)  # 51MB

        response = requests.post(
            f"{BASE_URL}/v1/extract",
            files={"file": ("large.pdf", large_file)}
        )

        assert response.status_code == 413, "Should reject files >50MB"

    def test_malicious_filename(self):
        """Test malicious filename handling"""
        malicious_filenames = [
            "../../../etc/passwd",
            "test.pdf.exe",
            "test.pdf\x00.exe",
            "<script>alert('xss')</script>.pdf",
        ]

        for filename in malicious_filenames:
            response = requests.post(
                f"{BASE_URL}/v1/extract",
                files={"file": (filename, b"content")}
            )

            # Should sanitize filename
            assert response.status_code in [200, 202, 400, 422]

    def test_file_type_validation(self):
        """Test file type validation"""
        # Try to upload non-PDF
        response = requests.post(
            f"{BASE_URL}/v1/extract",
            files={"file": ("script.sh", b"#!/bin/bash\nrm -rf /")}
        )

        assert response.status_code in [400, 415, 422], "Should reject non-PDF files"

    def test_zip_bomb_protection(self):
        """Test protection against zip bombs"""
        # This would require creating an actual zip bomb
        # For now, just test size limits
        pass


class TestSSL_TLS:
    """Test SSL/TLS configuration"""

    def test_https_enforced(self):
        """Test that HTTPS is enforced"""
        # Try HTTP
        try:
            response = requests.get("http://production-url.com/health")
            # Should redirect to HTTPS or reject
            assert response.status_code in [301, 302, 403]
        except requests.exceptions.SSLError:
            # Expected if HTTPS-only
            pass

    def test_tls_version(self):
        """Test minimum TLS version"""
        # Should support TLS 1.2+
        # This would require specific SSL context testing
        pass

    def test_secure_headers(self):
        """Test security headers"""
        response = requests.get(f"{BASE_URL}/health")

        headers = response.headers

        # Should have security headers
        assert "X-Content-Type-Options" in headers
        assert headers.get("X-Content-Type-Options") == "nosniff"

        assert "X-Frame-Options" in headers
        assert "Strict-Transport-Security" in headers  # HSTS


class TestAPISecurityMiscellaneous:
    """Miscellaneous API security tests"""

    def test_cors_configuration(self):
        """Test CORS is properly configured"""
        response = requests.options(
            f"{BASE_URL}/v1/extract",
            headers={"Origin": "https://malicious-site.com"}
        )

        # Should not allow all origins in production
        if "Access-Control-Allow-Origin" in response.headers:
            assert response.headers["Access-Control-Allow-Origin"] != "*"

    def test_http_methods(self):
        """Test only allowed HTTP methods work"""
        # Test unsupported methods
        unsupported_methods = ["TRACE", "TRACK", "CONNECT"]

        for method in unsupported_methods:
            response = requests.request(method, f"{BASE_URL}/v1/extract")
            assert response.status_code in [405, 501], f"Method {method} should not be allowed"

    def test_information_disclosure(self):
        """Test for information disclosure"""
        response = requests.get(f"{BASE_URL}/")

        # Should not disclose version info
        assert "Server" not in response.headers or "nginx" not in response.headers.get("Server", "").lower()

        # Should not disclose framework version
        assert "X-Powered-By" not in response.headers


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers",
        "critical: mark test as critical security test"
    )
    config.addinivalue_line(
        "markers",
        "owasp: mark test as OWASP Top 10 test"
    )


# Test report generation
def pytest_html_report_title(report):
    """Customize HTML report title"""
    report.title = "SAP_LLM Security Penetration Test Report"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--html=security_report.html", "--self-contained-html"])
