# Contributing to SAP_LLM

First off, thank you for considering contributing to SAP_LLM! It's people like you that make SAP_LLM such a great tool.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How Can I Contribute?](#how-can-i-contribute)
4. [Development Process](#development-process)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation Guidelines](#documentation-guidelines)
8. [Commit Message Guidelines](#commit-message-guidelines)
9. [Pull Request Process](#pull-request-process)
10. [Community](#community)

---

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to conduct@qorsync.com.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behavior includes:**
- The use of sexualized language or imagery
- Trolling, insulting/derogatory comments, and personal attacks
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

---

## Getting Started

### Prerequisites

Before you begin, ensure you have:
- Python 3.10 or higher installed
- Git installed and configured
- A GitHub account
- Familiarity with Python and machine learning concepts

### Setting Up Your Development Environment

1. **Fork the Repository**

   Visit https://github.com/qorsync/sap-llm and click "Fork"

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/sap-llm.git
   cd sap-llm
   ```

3. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/qorsync/sap-llm.git
   ```

4. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate  # Windows
   ```

5. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

6. **Verify Installation**
   ```bash
   pytest
   python scripts/health_check.py
   ```

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, files, etc.)
- **Describe the behavior you observed and expected**
- **Include screenshots if applicable**
- **Provide system information** (OS, Python version, GPU, etc.)

**Bug Report Template:**

```markdown
## Bug Description
A clear and concise description of the bug.

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Screenshots
If applicable, add screenshots.

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.10.12]
- SAP_LLM Version: [e.g., 1.0.0]
- GPU: [e.g., NVIDIA A10]
- CUDA Version: [e.g., 11.8]

## Additional Context
Add any other context about the problem.
```

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Use a clear and descriptive title**
- **Provide a detailed description of the suggested enhancement**
- **Explain why this enhancement would be useful**
- **List examples of how the enhancement would be used**
- **Describe alternatives you've considered**

**Enhancement Template:**

```markdown
## Enhancement Description
A clear description of the enhancement.

## Use Case
Why would this enhancement be useful? Who would benefit?

## Proposed Solution
How should this work?

## Alternatives Considered
What other solutions did you consider?

## Additional Context
Any mockups, diagrams, or examples.
```

### Your First Code Contribution

Not sure where to start? Look for issues labeled:
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `documentation` - Documentation improvements

### Pull Requests

We actively welcome your pull requests! Follow these steps:

1. Fork the repo and create your branch from `main`
2. Make your changes
3. Add tests if applicable
4. Ensure tests pass
5. Update documentation
6. Submit a pull request

---

## Development Process

### 1. Create a Feature Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create a new branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch Naming Conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes
- `perf/` - Performance improvements

### 2. Make Your Changes

- Follow the [coding standards](#coding-standards)
- Write or update tests
- Update documentation
- Keep commits atomic and focused

### 3. Test Your Changes

```bash
# Run linters
black sap_llm/
ruff check sap_llm/
mypy sap_llm/

# Run tests
pytest

# Run tests with coverage
pytest --cov=sap_llm --cov-report=html

# Run integration tests
pytest tests/integration/

# Run performance tests
pytest tests/performance/ -v
```

### 4. Commit Your Changes

Follow the [commit message guidelines](#commit-message-guidelines)

```bash
git add .
git commit -m "feat: add new document type support for credit memos"
```

### 5. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 6. Create a Pull Request

Go to GitHub and create a pull request from your branch to `qorsync/sap-llm:main`

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Use double quotes for strings
- **Imports**: Organized in groups (stdlib, third-party, local)

### Code Formatting

We use automated formatting tools:

```bash
# Format code with Black
black sap_llm/

# Check code with Ruff
ruff check sap_llm/

# Type checking with mypy
mypy sap_llm/
```

These run automatically on commit if you installed pre-commit hooks.

### Type Hints

All functions must have type hints:

```python
from typing import List, Dict, Optional

def process_document(
    file_path: str,
    document_type: Optional[str] = None,
    priority: str = "normal"
) -> Dict[str, Any]:
    """
    Process a document through the pipeline.

    Args:
        file_path: Path to the document file
        document_type: Type of document (auto-detect if None)
        priority: Processing priority (low, normal, high)

    Returns:
        Dictionary containing extracted data and metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        ValidationError: If document type is invalid
    """
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def extract_fields(image: np.ndarray, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extract fields from document image.

    Extracts structured data from a document image according to the
    provided schema. Uses the unified AI model for field extraction.

    Args:
        image: Document image as numpy array (H, W, 3)
        schema: Field extraction schema with field definitions

    Returns:
        Dictionary mapping field names to extracted values. Each value
        includes the extracted text and confidence score.

    Raises:
        ValueError: If image is invalid or schema is malformed
        RuntimeError: If model inference fails

    Example:
        >>> image = load_image("invoice.pdf")
        >>> schema = get_schema("invoice")
        >>> fields = extract_fields(image, schema)
        >>> print(fields["invoice_number"]["value"])
        "INV-2025-001234"
    """
    pass
```

### Code Organization

```python
# 1. Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# 2. Third-party imports
import torch
import numpy as np
from transformers import AutoModel

# 3. Local imports
from sap_llm.config import get_config
from sap_llm.utils import validate_input
from sap_llm.models import UnifiedExtractorModel
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `ExtractionStage`)
- **Functions/Methods**: `snake_case` (e.g., `process_document`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_FILE_SIZE`)
- **Private**: Prefix with `_` (e.g., `_internal_helper`)

### Error Handling

```python
# Use specific exceptions
class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass

class ExtractionError(DocumentProcessingError):
    """Raised when field extraction fails."""
    pass

# Provide context in error messages
try:
    result = extract_field(document, field_name)
except ExtractionError as e:
    logger.error(
        f"Failed to extract field '{field_name}' from document {document.id}",
        exc_info=True,
        extra={"document_id": document.id, "field": field_name}
    )
    raise
```

### Async Code

```python
# Use async/await for I/O operations
async def process_document_async(file_path: str) -> Dict[str, Any]:
    """Process document asynchronously."""
    # Await async operations
    content = await read_file_async(file_path)
    result = await model.infer_async(content)
    return result

# Use asyncio.gather for parallel operations
results = await asyncio.gather(
    process_document_async("doc1.pdf"),
    process_document_async("doc2.pdf"),
    process_document_async("doc3.pdf"),
)
```

---

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                  # Unit tests (fast, isolated)
│   ├── test_stages/
│   ├── test_models/
│   └── test_utils/
├── integration/           # Integration tests (slower, multiple components)
│   ├── test_pipeline/
│   └── test_api/
├── performance/           # Performance and load tests
│   └── test_throughput/
└── conftest.py           # Shared fixtures
```

### Writing Unit Tests

```python
import pytest
from unittest.mock import Mock, AsyncMock
from sap_llm.stages.extraction import ExtractionStage

@pytest.fixture
def mock_model():
    """Mock AI model for testing."""
    model = Mock()
    model.extract = AsyncMock(return_value={
        "invoice_number": "INV-123",
        "total": 1000.00
    })
    return model

@pytest.fixture
def extraction_stage(mock_model):
    """Create extraction stage with mocked dependencies."""
    return ExtractionStage(
        model=mock_model,
        cache=Mock(),
        pmg=Mock()
    )

@pytest.mark.asyncio
async def test_extraction_success(extraction_stage):
    """Test successful field extraction."""
    # Arrange
    context = create_test_context()

    # Act
    result = await extraction_stage.process(context)

    # Assert
    assert "invoice_number" in result.extracted_fields
    assert result.extracted_fields["invoice_number"] == "INV-123"
    assert result.confidence > 0.8

@pytest.mark.asyncio
async def test_extraction_failure(extraction_stage):
    """Test extraction failure handling."""
    # Arrange
    extraction_stage.model.extract.side_effect = ExtractionError("Model failed")
    context = create_test_context()

    # Act & Assert
    with pytest.raises(ExtractionError):
        await extraction_stage.process(context)
```

### Test Coverage Requirements

- **Minimum coverage**: 80%
- **New code**: 90%+ coverage required
- **Critical paths**: 100% coverage required

```bash
# Check coverage
pytest --cov=sap_llm --cov-report=term --cov-fail-under=80
```

### Integration Tests

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline():
    """Test complete pipeline end-to-end."""
    # This test uses real components
    config = initialize("configs/test_config.yaml")
    pipeline = Pipeline(config)

    result = await pipeline.process(
        file_path="tests/data/sample_invoice.pdf",
        document_type="invoice"
    )

    assert result["status"] == "completed"
    assert result["confidence"] > 0.85
```

### Performance Tests

```python
@pytest.mark.performance
def test_throughput(benchmark):
    """Benchmark document processing throughput."""

    def process_batch():
        return model.process_batch(sample_documents)

    result = benchmark(process_batch)

    # Assert performance targets
    assert result.stats.mean < 0.1  # 100ms average
    assert result.stats.stddev < 0.05  # Low variance
```

---

## Documentation Guidelines

### Documentation Types

1. **Code Documentation** - Docstrings in code
2. **User Documentation** - `docs/USER_GUIDE.md`
3. **Developer Documentation** - `docs/DEVELOPER_GUIDE.md`
4. **API Documentation** - `docs/API_DOCUMENTATION.md`
5. **Architecture Documentation** - `docs/ARCHITECTURE.md`

### Writing Documentation

- **Be clear and concise**
- **Use examples**
- **Include code snippets**
- **Keep it up to date**
- **Use proper markdown formatting**

### Example Documentation

````markdown
## Processing Documents

SAP_LLM can process various document types including invoices and purchase orders.

### Basic Usage

Here's how to process a document:

```python
from sap_llm import process_document

result = process_document(
    file_path="invoice.pdf",
    document_type="invoice"
)

print(result["extracted_fields"])
```

### Advanced Options

You can specify additional options:

```python
result = process_document(
    file_path="invoice.pdf",
    document_type="invoice",
    priority="high",
    extract_tables=True,
    confidence_threshold=0.90
)
```

**Parameters:**
- `file_path` (str): Path to the document file
- `document_type` (str): Type of document (auto-detect if omitted)
- `priority` (str): Processing priority - "low", "normal", "high"
- `extract_tables` (bool): Whether to extract table data
- `confidence_threshold` (float): Minimum confidence score (0.0-1.0)

**Returns:**
- `dict`: Extraction results with fields and metadata
````

---

## Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Test additions or changes
- `chore`: Build process or tooling changes
- `ci`: CI/CD changes

### Examples

```bash
# Feature
feat(extraction): add support for credit memo documents

Implements field extraction for credit memo document type including
header fields, line items, and totals. Adds validation rules for
credit amount limits.

Closes #123

# Bug Fix
fix(api): resolve rate limiting bypass issue

Previously, rate limits could be bypassed by using multiple API keys.
This fix ensures rate limits are enforced per user, not per API key.

Fixes #456

# Documentation
docs(readme): update installation instructions

Clarifies Python version requirements and adds troubleshooting section
for common installation issues on Windows.

# Breaking Change
feat(api)!: change authentication to JWT-only

BREAKING CHANGE: Session-based authentication is no longer supported.
All clients must migrate to JWT tokens. See MIGRATION.md for details.
```

### Commit Message Rules

1. Use the imperative mood ("Add feature" not "Added feature")
2. Don't capitalize the first letter of the subject
3. No period at the end of the subject
4. Limit subject line to 72 characters
5. Wrap body at 72 characters
6. Use body to explain what and why, not how
7. Reference issues and PRs in the footer

---

## Pull Request Process

### Before Submitting

- [ ] Tests pass locally
- [ ] Code is formatted (black, ruff)
- [ ] Type checking passes (mypy)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)
- [ ] Commits follow commit message guidelines

### PR Template

When creating a PR, use this template:

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Related Issues
Fixes #(issue number)
Closes #(issue number)

## How Has This Been Tested?
Describe the tests you ran to verify your changes.

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

**Test Configuration**:
* Python version: 3.10.12
* OS: Ubuntu 22.04
* GPU: NVIDIA A10

## Checklist
- [ ] My code follows the project's code style
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Additional Notes
Any additional information or context.
```

### Review Process

1. **Automated Checks**: CI/CD runs tests, linting, and security scans
2. **Code Review**: At least one maintainer reviews the code
3. **Changes Requested**: Address feedback and push updates
4. **Approval**: Maintainer approves the PR
5. **Merge**: PR is merged to main branch

### After Merge

- Delete your feature branch
- Update your local repository
- Close related issues

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Slack**: Real-time chat (qorsync.slack.com)
- **Email**: dev@qorsync.com

### Getting Help

- Check [Documentation](docs/)
- Search [GitHub Issues](https://github.com/qorsync/sap-llm/issues)
- Ask in [GitHub Discussions](https://github.com/qorsync/sap-llm/discussions)
- Join our Slack community

### Recognition

Contributors are recognized in:
- CHANGELOG.md
- GitHub contributors page
- Release notes
- Special mentions in documentation

---

## Additional Resources

- [User Guide](docs/USER_GUIDE.md)
- [Developer Guide](docs/DEVELOPER_GUIDE.md)
- [Architecture Documentation](docs/ARCHITECTURE.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

---

## License

By contributing to SAP_LLM, you agree that your contributions will be licensed under the project's proprietary license.

---

## Questions?

Don't hesitate to ask! We're here to help:
- Email: dev@qorsync.com
- Slack: qorsync.slack.com
- GitHub Discussions: https://github.com/qorsync/sap-llm/discussions

**Thank you for contributing to SAP_LLM!** 🎉
