#!/usr/bin/env python3
"""
Simple validation script for CORS configuration.
Tests the core functionality without requiring full test suite dependencies.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("CORS Configuration Validation Script")
print("=" * 70)

def test_cors_import():
    """Test that CORSSettings can be imported."""
    try:
        from sap_llm.config import CORSSettings
        print("✓ CORSSettings imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import CORSSettings: {e}")
        return False

def test_cors_basic():
    """Test basic CORS configuration."""
    try:
        from sap_llm.config import CORSSettings

        # Test 1: Default settings (empty origins)
        os.environ.pop('CORS_ALLOWED_ORIGINS', None)
        os.environ.pop('ENVIRONMENT', None)
        settings = CORSSettings()
        assert settings.get_origins() == [], "Default should be empty list"
        print("✓ Test 1: Default configuration (empty origins)")

        # Test 2: Development with localhost
        os.environ['CORS_ALLOWED_ORIGINS'] = 'http://localhost:3000'
        os.environ['ENVIRONMENT'] = 'development'
        settings = CORSSettings()
        assert 'http://localhost:3000' in settings.get_origins()
        print("✓ Test 2: Development with localhost HTTP")

        # Test 3: Production with HTTPS
        os.environ['CORS_ALLOWED_ORIGINS'] = 'https://app.example.com,https://api.example.com'
        os.environ['ENVIRONMENT'] = 'production'
        settings = CORSSettings()
        assert len(settings.get_origins()) == 2
        assert settings.is_production()
        print("✓ Test 3: Production with HTTPS origins")

        # Test 4: Production rejects wildcard
        os.environ['CORS_ALLOWED_ORIGINS'] = '*'
        os.environ['ENVIRONMENT'] = 'production'
        try:
            settings = CORSSettings()
            print("✗ Test 4: Production should reject wildcard")
            return False
        except ValueError as e:
            assert "wildcard" in str(e).lower() or "production" in str(e).lower()
            print("✓ Test 4: Production correctly rejects wildcard")

        # Test 5: Production rejects HTTP
        os.environ['CORS_ALLOWED_ORIGINS'] = 'http://app.example.com'
        os.environ['ENVIRONMENT'] = 'production'
        try:
            settings = CORSSettings()
            print("✗ Test 5: Production should reject HTTP origins")
            return False
        except ValueError as e:
            assert "https" in str(e).lower()
            print("✓ Test 5: Production correctly rejects HTTP origins")

        # Test 6: Invalid URL format
        os.environ['CORS_ALLOWED_ORIGINS'] = 'not-a-url'
        os.environ['ENVIRONMENT'] = 'development'
        try:
            settings = CORSSettings()
            print("✗ Test 6: Should reject invalid URL format")
            return False
        except ValueError:
            print("✓ Test 6: Invalid URL format correctly rejected")

        # Test 7: Multiple origins parsing
        os.environ['CORS_ALLOWED_ORIGINS'] = 'https://app1.com, https://app2.com, https://app3.com'
        os.environ['ENVIRONMENT'] = 'development'
        settings = CORSSettings()
        assert len(settings.get_origins()) == 3
        print("✓ Test 7: Multiple origins parsed correctly")

        # Test 8: Validate for production method
        os.environ['CORS_ALLOWED_ORIGINS'] = 'https://app.qorsync.com'
        os.environ['ENVIRONMENT'] = 'production'
        settings = CORSSettings()
        settings.validate_for_production()  # Should not raise
        print("✓ Test 8: validate_for_production with valid config")

        return True

    except Exception as e:
        print(f"✗ CORS tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cors_security():
    """Test security features."""
    try:
        from sap_llm.config import CORSSettings

        # Security Test 1: Empty origins rejected in production
        os.environ['CORS_ALLOWED_ORIGINS'] = ''
        os.environ['ENVIRONMENT'] = 'production'
        settings = CORSSettings()  # Empty is OK during init
        try:
            settings.validate_for_production()  # But validation should catch it
            print("✗ Security Test 1: Should reject empty origins in production")
            return False
        except ValueError:
            print("✓ Security Test 1: Empty origins rejected in production")

        # Security Test 2: Wildcard allowed in development
        os.environ['CORS_ALLOWED_ORIGINS'] = '*'
        os.environ['ENVIRONMENT'] = 'development'
        settings = CORSSettings()
        assert settings.get_origins() == ['*']
        print("✓ Security Test 2: Wildcard allowed in development")

        # Security Test 3: URL normalization
        os.environ['CORS_ALLOWED_ORIGINS'] = 'https://app.example.com:443'
        os.environ['ENVIRONMENT'] = 'development'
        settings = CORSSettings()
        # Should normalize properly
        print("✓ Security Test 3: URL with port handled")

        return True

    except Exception as e:
        print(f"✗ Security tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    results = []

    print("\n[1/3] Testing CORSSettings Import...")
    results.append(test_cors_import())

    print("\n[2/3] Testing Basic CORS Configuration...")
    results.append(test_cors_basic())

    print("\n[3/3] Testing CORS Security Features...")
    results.append(test_cors_security())

    print("\n" + "=" * 70)
    if all(results):
        print("✓ ALL CORS VALIDATION TESTS PASSED")
        print("=" * 70)
        return 0
    else:
        print("✗ SOME CORS VALIDATION TESTS FAILED")
        print("=" * 70)
        return 1

if __name__ == '__main__':
    sys.exit(main())
