# SAP_LLM Testing Guide

**Version:** 1.0
**Last Updated:** 2025-11-19
**Target Coverage:** ≥90%
**Current Coverage:** 1.09% → **Target: 90%**

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Writing Tests](#writing-tests)
5. [Coverage Requirements](#coverage-requirements)
6. [CI/CD Integration](#cicd-integration)
7. [Best Practices](#best-practices)
8. [Common Patterns](#common-patterns)

---

## Overview

SAP_LLM uses **pytest** as the testing framework with comprehensive coverage measurement via **pytest-cov**. The project requires a minimum of 90% code coverage for production readiness.

### Test Categories

- **Unit Tests** (`@pytest.mark.unit`): Test individual functions and classes in isolation
- **Integration Tests** (`@pytest.mark.integration`): Test component interactions
- **Performance Tests** (`@pytest.mark.performance`): Benchmark critical paths
- **Load Tests** (`@pytest.mark.load`): Test under high load
- **Chaos Tests** (`@pytest.mark.chaos`): Test resilience and error handling

---

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── test_config.py              # Configuration tests
├── test_utils.py               # Utility function tests
├── test_api.py                 # API endpoint tests
├── test_stages.py              # Pipeline stage tests
├── fixtures/                   # Test data and mocks
│   ├── __init__.py
│   ├── mock_data.py
│   └── sample_documents.py
├── unit/                       # Unit tests
│   ├── test_pmg.py
│   ├── test_shwl.py
│   ├── test_apop.py
│   └── test_models.py
├── integration/                # Integration tests
│   ├── test_end_to_end.py
│   └── test_full_pipeline_real.py
├── performance/                # Performance tests
│   ├── test_latency.py
│   ├── test_throughput.py
│   └── test_memory.py
├── load/                       # Load tests
│   └── test_api.py
└── chaos/                      # Chaos engineering tests
    └── test_chaos_engineering.py
```

---

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests
pytest -m integration

# Performance tests
pytest -m performance

# Skip slow tests
pytest -m "not slow"

# Run specific test file
pytest tests/test_utils.py

# Run specific test class
pytest tests/test_utils.py::TestHash

# Run specific test method
pytest tests/test_utils.py::TestHash::test_hash_string_sha256
```

### Run with Coverage

```bash
# Full coverage report
pytest --cov=sap_llm --cov-report=html --cov-report=term-missing

# Coverage for specific module
pytest tests/test_utils.py --cov=sap_llm.utils --cov-report=term

# Check if coverage meets threshold (90%)
pytest --cov=sap_llm --cov-fail-under=90
```

### Parallel Testing

```bash
# Run tests in parallel (faster)
pytest -n auto

# Run with 4 workers
pytest -n 4
```

### Debugging Tests

```bash
# Verbose output
pytest -v

# Very verbose (show all output)
pytest -vv

# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Stop on first failure
pytest -x

# Run failed tests from last run
pytest --lf
```

---

## Writing Tests

### Basic Test Structure

```python
import pytest
from sap_llm.module import function_to_test


@pytest.mark.unit
class TestModuleName:
    """Tests for module functionality."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing test data."""
        return {"field": "value"}

    def test_normal_case(self, sample_data):
        """Test normal operation."""
        result = function_to_test(sample_data)
        assert result is not None
        assert result["status"] == "success"

    def test_edge_case_empty_input(self):
        """Test edge case with empty input."""
        with pytest.raises(ValueError, match="Input cannot be empty"):
            function_to_test({})

    @pytest.mark.parametrize("input,expected", [
        ("value1", "result1"),
        ("value2", "result2"),
        ("value3", "result3"),
    ])
    def test_multiple_scenarios(self, input, expected):
        """Test multiple input scenarios."""
        assert function_to_test(input) == expected
```

### Testing Exceptions

```python
def test_invalid_input_raises_error():
    """Test that invalid input raises ValueError."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_to_test("invalid")
```

### Testing Async Functions

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test asynchronous function."""
    result = await async_function()
    assert result is not None
```

### Mocking External Dependencies

```python
from unittest.mock import Mock, patch


def test_with_mock(mocker):
    """Test with mocked dependency."""
    # Mock external service
    mock_service = mocker.patch('sap_llm.module.ExternalService')
    mock_service.return_value.call.return_value = "mocked_response"

    result = function_that_uses_service()
    assert result == "mocked_response"
    mock_service.return_value.call.assert_called_once()
```

### Testing File Operations

```python
def test_file_processing(temp_dir):
    """Test file processing with temporary directory."""
    # Create test file
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")

    # Test file processing
    result = process_file(str(test_file))
    assert result is not None
```

---

## Coverage Requirements

### Overall Target: ≥90%

| Component | Target Coverage | Priority |
|-----------|----------------|----------|
| Core Pipeline Stages | 95% | Critical |
| Business Logic (APOP) | 95% | Critical |
| Utils & Config | 95% | High |
| Self-Healing (SHWL) | 90% | High |
| Knowledge Management (PMG) | 90% | High |
| Models | 85% | Medium |
| API Layer | 90% | High |
| Training Pipeline | 80% | Medium |

### Coverage Enforcement

Tests automatically fail if coverage falls below 90%:

```bash
pytest --cov=sap_llm --cov-fail-under=90
```

### Viewing Coverage Reports

After running tests with coverage:

```bash
# View HTML report
open htmlcov/index.html

# View terminal report
coverage report -m
```

---

## CI/CD Integration

### Automatic Coverage Checks

Coverage is automatically checked in CI/CD on:
- Every push to main/develop
- Every pull request
- Manual workflow dispatch

### CI Workflow

`.github/workflows/ci.yml` includes:

1. **Unit Tests** (Multi-Python versions)
   - Python 3.9, 3.10, 3.11
   - Parallel execution
   - Coverage threshold enforcement

2. **Integration Tests**
   - With Redis & MongoDB services
   - Full pipeline testing

3. **Coverage Upload**
   - Automatically uploads to Codecov
   - PR comments with coverage changes

### Coverage Badges

Add to your PR or README:

```markdown
![Coverage](https://codecov.io/gh/qorsync/sap-llm/branch/main/graph/badge.svg)
```

---

## Best Practices

### 1. Test Naming

```python
# ✅ Good: Descriptive, indicates what is tested
def test_hash_string_with_sha256_returns_64_char_hex():
    pass

# ❌ Bad: Vague, doesn't explain test purpose
def test_hash():
    pass
```

### 2. One Assert Per Test (When Possible)

```python
# ✅ Good: Single focused assertion
def test_hash_returns_string():
    result = compute_hash("data")
    assert isinstance(result, str)

def test_hash_returns_correct_length():
    result = compute_hash("data")
    assert len(result) == 64

# ⚠️ Acceptable: Related assertions
def test_hash_output_format():
    result = compute_hash("data")
    assert isinstance(result, str)
    assert len(result) == 64
    assert result.isalnum()
```

### 3. Use Fixtures for Setup

```python
@pytest.fixture
def config():
    """Load test configuration."""
    return load_config("tests/test_config.yaml")

def test_with_config(config):
    """Test using config fixture."""
    assert config.system.environment == "test"
```

### 4. Test Edge Cases

Always test:
- Empty inputs
- None values
- Maximum/minimum values
- Invalid types
- Error conditions

```python
def test_edge_cases():
    # Empty input
    with pytest.raises(ValueError):
        process("")

    # None input
    with pytest.raises(TypeError):
        process(None)

    # Invalid type
    with pytest.raises(TypeError):
        process(12345)
```

### 5. Use Parametrize for Multiple Cases

```python
@pytest.mark.parametrize("algorithm,expected_length", [
    ("sha256", 64),
    ("sha384", 96),
    ("sha512", 128),
])
def test_hash_with_algorithms(algorithm, expected_length):
    result = compute_hash("data", algorithm=algorithm)
    assert len(result) == expected_length
```

### 6. Mock External Services

Never make real API calls or database connections in unit tests:

```python
@pytest.fixture
def mock_api(mocker):
    """Mock external API."""
    return mocker.patch('sap_llm.api.client.ApiClient')

def test_with_mocked_api(mock_api):
    mock_api.return_value.get.return_value = {"status": "ok"}
    result = fetch_data()
    assert result["status"] == "ok"
```

### 7. Test Both Success and Failure Paths

```python
def test_success_path():
    """Test successful operation."""
    result = operation("valid_input")
    assert result["success"] is True

def test_failure_path():
    """Test operation failure."""
    with pytest.raises(OperationError):
        operation("invalid_input")
```

---

## Common Patterns

### Pattern 1: Testing Data Transformations

```python
@pytest.mark.unit
class TestDataTransformation:
    """Test data transformation functions."""

    @pytest.fixture
    def input_data(self):
        return {"raw": "data"}

    @pytest.fixture
    def expected_output(self):
        return {"processed": "data"}

    def test_transform_success(self, input_data, expected_output):
        result = transform(input_data)
        assert result == expected_output

    def test_transform_preserves_structure(self, input_data):
        result = transform(input_data)
        assert isinstance(result, dict)
        assert "processed" in result
```

### Pattern 2: Testing Async Operations

```python
@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations."""

    async def test_async_fetch(self):
        result = await async_fetch_data()
        assert result is not None

    async def test_async_timeout(self):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=1.0)
```

### Pattern 3: Testing with Databases

```python
@pytest.mark.integration
class TestDatabaseOperations:
    """Test database operations."""

    @pytest.fixture(autouse=True)
    async def setup_db(self):
        """Set up test database."""
        await init_test_db()
        yield
        await cleanup_test_db()

    async def test_insert(self):
        doc_id = await insert_document({"data": "test"})
        assert doc_id is not None

    async def test_query(self):
        await insert_document({"name": "test"})
        results = await query_documents({"name": "test"})
        assert len(results) == 1
```

### Pattern 4: Testing Configuration

```python
@pytest.mark.unit
class TestConfiguration:
    """Test configuration loading and validation."""

    def test_load_default_config(self):
        config = load_config()
        assert config is not None
        assert config.system.environment in ["development", "staging", "production"]

    def test_invalid_config_raises_error(self):
        with pytest.raises(ValueError):
            load_config("invalid_config.yaml")

    def test_env_var_substitution(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "test_value")
        config = load_config()
        assert "test_value" in str(config)
```

### Pattern 5: Testing Security

```python
@pytest.mark.unit
class TestSecurity:
    """Test security features."""

    def test_rejects_insecure_algorithm(self):
        """Test that insecure algorithms are rejected."""
        with pytest.raises(ValueError, match="Insecure hash algorithm"):
            compute_hash("data", algorithm="md5")

    def test_validates_input_sanitization(self):
        """Test that input is properly sanitized."""
        malicious_input = "<script>alert('xss')</script>"
        result = sanitize_input(malicious_input)
        assert "<script>" not in result
```

---

## Test Performance

### Target Metrics

- **Unit Tests**: < 5 minutes total
- **Integration Tests**: < 15 minutes total
- **Individual Test**: < 1 second (mark slow tests with `@pytest.mark.slow`)

### Optimizing Test Speed

```python
# Use pytest-xdist for parallel execution
pytest -n auto

# Skip slow tests during development
pytest -m "not slow"

# Use smaller test datasets
@pytest.fixture
def small_dataset():
    return generate_test_data(size=100)  # Not 10000!
```

---

## Troubleshooting

### Tests Fail Locally But Pass in CI

- Check Python version: `python --version`
- Ensure dependencies are up to date: `pip install -r requirements.txt -e .[dev]`
- Clear pytest cache: `pytest --cache-clear`

### Coverage Not Reaching 90%

1. Identify uncovered lines:
   ```bash
   coverage report -m
   ```

2. View HTML report:
   ```bash
   open htmlcov/index.html
   ```

3. Write tests for uncovered code

### Tests Are Slow

- Use `pytest --durations=10` to identify slow tests
- Mark slow tests: `@pytest.mark.slow`
- Use mocks instead of real services
- Run in parallel: `pytest -n auto`

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Project Coverage Baseline Report](../COVERAGE_BASELINE_REPORT.md)

---

## Next Steps

To achieve 90% coverage:

1. **High Priority** (Weeks 1-2):
   - Utils modules (✅ 95% achieved)
   - Config module (✅ 95% achieved)
   - Core pipeline stages
   - APOP business logic

2. **Medium Priority** (Weeks 3-4):
   - PMG (Process Memory Graph)
   - SHWL (Self-Healing Workflow Loop)
   - API endpoints

3. **Lower Priority** (Week 5+):
   - Model inference (use mocks extensively)
   - Training pipelines
   - Performance optimizations

**Estimated Timeline**: 5-7 weeks to reach 90% coverage
**Estimated New Tests Needed**: 220-305 additional tests

---

*For questions or issues, please open a GitHub issue or contact the development team.*
