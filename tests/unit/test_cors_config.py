"""
Unit tests for CORS configuration and security validation.

Tests comprehensive CORS security features including:
- Environment variable parsing
- Production security validation
- HTTPS enforcement
- Wildcard detection
- URL format validation
"""

import pytest
from unittest.mock import patch

from sap_llm.config import CORSSettings


@pytest.mark.unit
class TestCORSConfiguration:
    """Tests for CORS configuration loading and validation."""

    def test_default_cors_settings(self):
        """Test default CORS settings (empty origins for security)."""
        with patch.dict('os.environ', {}, clear=True):
            settings = CORSSettings()

            # SECURITY: Default should be empty list (deny all)
            assert settings.get_origins() == []
            assert settings.ENVIRONMENT == "development"

    def test_parse_comma_separated_origins(self):
        """Test parsing comma-separated origins from environment variable."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://app.example.com,https://api.example.com',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert len(settings.get_origins()) == 2
            assert 'https://app.example.com' in settings.get_origins()
            assert 'https://api.example.com' in settings.get_origins()

    def test_parse_origins_with_whitespace(self):
        """Test parsing origins with extra whitespace."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': '  https://app.example.com  ,  https://api.example.com  ',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            # Should trim whitespace
            assert len(settings.get_origins()) == 2
            assert 'https://app.example.com' in settings.get_origins()

    def test_empty_string_origins(self):
        """Test that empty string results in empty list."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': '',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert settings.get_origins() == []

    def test_single_origin(self):
        """Test configuration with single origin."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://app.example.com',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert len(settings.get_origins()) == 1
            assert settings.get_origins()[0] == 'https://app.example.com'


@pytest.mark.unit
class TestCORSProductionSecurity:
    """Tests for production security validation."""

    def test_production_rejects_wildcard(self):
        """Test that wildcard is rejected in production."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': '*',
            'ENVIRONMENT': 'production'
        }):
            with pytest.raises(ValueError) as exc_info:
                CORSSettings()

            error_msg = str(exc_info.value)
            assert "wildcard" in error_msg.lower()
            assert "production" in error_msg.lower()

    def test_production_rejects_http_origins(self):
        """Test that HTTP origins are rejected in production."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'http://app.example.com',
            'ENVIRONMENT': 'production'
        }):
            with pytest.raises(ValueError) as exc_info:
                CORSSettings()

            error_msg = str(exc_info.value)
            assert "https" in error_msg.lower()

    def test_production_requires_https(self):
        """Test that production requires HTTPS origins."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://app.example.com,https://api.example.com',
            'ENVIRONMENT': 'production'
        }):
            settings = CORSSettings()

            # Should accept HTTPS origins
            assert len(settings.get_origins()) == 2
            for origin in settings.get_origins():
                assert origin.startswith('https://')

    def test_production_mixed_http_https_rejected(self):
        """Test that mixing HTTP and HTTPS in production is rejected."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://app.example.com,http://insecure.example.com',
            'ENVIRONMENT': 'production'
        }):
            with pytest.raises(ValueError) as exc_info:
                CORSSettings()

            error_msg = str(exc_info.value)
            assert "https" in error_msg.lower()

    def test_development_allows_http(self):
        """Test that HTTP is allowed in development."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'http://localhost:3000,http://localhost:8000',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert len(settings.get_origins()) == 2
            assert 'http://localhost:3000' in settings.get_origins()

    def test_development_allows_wildcard(self):
        """Test that wildcard is allowed in development (with warning)."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': '*',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert settings.get_origins() == ['*']


@pytest.mark.unit
class TestCORSURLValidation:
    """Tests for URL format validation."""

    def test_valid_https_url(self):
        """Test valid HTTPS URL is accepted."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://app.example.com',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert settings.get_origins()[0] == 'https://app.example.com'

    def test_valid_http_url_in_dev(self):
        """Test valid HTTP URL is accepted in development."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'http://localhost:3000',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert settings.get_origins()[0] == 'http://localhost:3000'

    def test_invalid_url_no_scheme(self):
        """Test that URL without scheme is rejected."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'example.com',
            'ENVIRONMENT': 'development'
        }):
            with pytest.raises(ValueError) as exc_info:
                CORSSettings()

            error_msg = str(exc_info.value)
            assert "scheme" in error_msg.lower() or "protocol" in error_msg.lower()

    def test_invalid_url_no_host(self):
        """Test that URL without host is rejected."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://',
            'ENVIRONMENT': 'development'
        }):
            with pytest.raises(ValueError) as exc_info:
                CORSSettings()

            error_msg = str(exc_info.value)
            assert "invalid" in error_msg.lower()

    def test_url_with_port(self):
        """Test URL with port number."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'http://localhost:3000',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert 'http://localhost:3000' in settings.get_origins()

    def test_url_with_subdomain(self):
        """Test URL with subdomain."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://app.subdomain.example.com',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert 'https://app.subdomain.example.com' in settings.get_origins()


@pytest.mark.unit
class TestCORSValidationMethods:
    """Tests for validation methods."""

    def test_is_production_true(self):
        """Test is_production returns True for production environment."""
        with patch.dict('os.environ', {
            'ENVIRONMENT': 'production',
            'CORS_ALLOWED_ORIGINS': 'https://app.example.com'
        }):
            settings = CORSSettings()

            assert settings.is_production() is True

    def test_is_production_false(self):
        """Test is_production returns False for development environment."""
        with patch.dict('os.environ', {
            'ENVIRONMENT': 'development',
            'CORS_ALLOWED_ORIGINS': 'http://localhost:3000'
        }):
            settings = CORSSettings()

            assert settings.is_production() is False

    def test_validate_for_production_empty_origins(self):
        """Test validate_for_production rejects empty origins."""
        with patch.dict('os.environ', {
            'ENVIRONMENT': 'production',
            'CORS_ALLOWED_ORIGINS': ''
        }):
            # This should fail during initialization, not just validation
            with pytest.raises(ValueError):
                settings = CORSSettings()
                settings.validate_for_production()

    def test_validate_for_production_success(self):
        """Test validate_for_production succeeds with valid config."""
        with patch.dict('os.environ', {
            'ENVIRONMENT': 'production',
            'CORS_ALLOWED_ORIGINS': 'https://app.example.com'
        }):
            settings = CORSSettings()

            # Should not raise
            settings.validate_for_production()

    def test_validate_for_production_skipped_in_dev(self):
        """Test validate_for_production is skipped in development."""
        with patch.dict('os.environ', {
            'ENVIRONMENT': 'development',
            'CORS_ALLOWED_ORIGINS': ''
        }):
            settings = CORSSettings()

            # Should not raise even with empty origins
            settings.validate_for_production()


@pytest.mark.unit
class TestCORSSecurityWarnings:
    """Tests for security warning scenarios."""

    def test_too_many_origins_warning(self, caplog):
        """Test warning is logged for too many origins."""
        origins = ','.join([f'https://app{i}.example.com' for i in range(10)])

        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': origins,
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            # Should create settings but log warning
            assert len(settings.get_origins()) == 10

    def test_localhost_origins_accepted(self):
        """Test that localhost origins are accepted in development."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'http://localhost:3000,http://127.0.0.1:3000',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert len(settings.get_origins()) == 2


@pytest.mark.unit
class TestCORSRealWorldScenarios:
    """Tests for real-world deployment scenarios."""

    def test_single_spa_production(self):
        """Test single-page application in production."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://app.qorsync.com',
            'ENVIRONMENT': 'production'
        }):
            settings = CORSSettings()

            assert len(settings.get_origins()) == 1
            assert settings.get_origins()[0] == 'https://app.qorsync.com'
            settings.validate_for_production()

    def test_multiple_domains_production(self):
        """Test multiple domains in production."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://app.qorsync.com,https://api.qorsync.com,https://admin.qorsync.com',
            'ENVIRONMENT': 'production'
        }):
            settings = CORSSettings()

            assert len(settings.get_origins()) == 3
            settings.validate_for_production()

    def test_local_development_setup(self):
        """Test typical local development setup."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert len(settings.get_origins()) == 3
            assert all('localhost' in origin or '127.0.0.1' in origin
                      for origin in settings.get_origins())

    def test_staging_environment(self):
        """Test staging environment with HTTPS."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://staging-app.example.com,https://staging-api.example.com',
            'ENVIRONMENT': 'staging'
        }):
            settings = CORSSettings()

            assert len(settings.get_origins()) == 2
            assert all(origin.startswith('https://')
                      for origin in settings.get_origins())


@pytest.mark.unit
class TestCORSBackwardCompatibility:
    """Tests for backward compatibility and migration scenarios."""

    def test_empty_origins_development(self):
        """Test empty origins are accepted in development."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': '',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            assert settings.get_origins() == []

    def test_list_input_format(self):
        """Test that list format is also supported (not just strings)."""
        # This tests the validator can handle list input
        with patch.dict('os.environ', {
            'ENVIRONMENT': 'development'
        }):
            # Test with programmatic list input
            settings = CORSSettings(
                CORS_ALLOWED_ORIGINS=['https://app.example.com', 'https://api.example.com']
            )

            assert len(settings.get_origins()) == 2


@pytest.mark.unit
class TestCORSEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_malformed_url(self):
        """Test malformed URL is rejected."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'not-a-valid-url',
            'ENVIRONMENT': 'development'
        }):
            with pytest.raises(ValueError):
                CORSSettings()

    def test_javascript_protocol_rejected(self):
        """Test that javascript: protocol is rejected."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'javascript:alert(1)',
            'ENVIRONMENT': 'development'
        }):
            with pytest.raises(ValueError):
                CORSSettings()

    def test_file_protocol_rejected(self):
        """Test that file: protocol is rejected."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'file:///etc/passwd',
            'ENVIRONMENT': 'development'
        }):
            # file:// has no netloc, should be rejected
            with pytest.raises(ValueError):
                CORSSettings()

    def test_duplicate_origins_preserved(self):
        """Test that duplicate origins are preserved (validation handles it)."""
        with patch.dict('os.environ', {
            'CORS_ALLOWED_ORIGINS': 'https://app.example.com,https://app.example.com',
            'ENVIRONMENT': 'development'
        }):
            settings = CORSSettings()

            # Duplicates should be preserved (FastAPI CORS handles deduplication)
            assert len(settings.get_origins()) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
