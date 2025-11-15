# SAP_LLM CI/CD Pipeline Guide

## Overview

This document provides a comprehensive guide to the enterprise-grade CI/CD pipeline implemented for SAP_LLM using GitHub Actions.

## 📋 Table of Contents

- [Workflow Overview](#workflow-overview)
- [Workflow Details](#workflow-details)
- [Setup Requirements](#setup-requirements)
- [Usage Guide](#usage-guide)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🔄 Workflow Overview

The SAP_LLM CI/CD pipeline consists of 5 comprehensive workflows:

### 1. **ci.yml** - Continuous Integration
**Triggers:** Push/PR to main, develop, and feature branches
**Purpose:** Validate code quality, run tests, build artifacts
**Duration:** ~30-45 minutes

**Jobs:**
- Code Quality Checks (Black, Ruff, MyPy, Pylint, Flake8)
- Multi-Version Unit Tests (Python 3.9, 3.10, 3.11)
- Integration Tests (with Redis, MongoDB services)
- Docker Image Building
- Artifact Generation
- CI Success Validation

### 2. **cd.yml** - Continuous Deployment
**Triggers:** Release tags (v*.*.*)
**Purpose:** Deploy to staging/production environments
**Duration:** ~45-60 minutes

**Jobs:**
- Production Image Building (multi-arch)
- SBOM Generation
- Staging Deployment
- Smoke Tests (Staging)
- Production Deployment (Canary)
- Smoke Tests (Production)
- Automatic Rollback on Failure

### 3. **test.yml** - Comprehensive Testing
**Triggers:** Push/PR, Daily Schedule (2 AM UTC), Manual
**Purpose:** Run comprehensive test suites
**Duration:** ~60-90 minutes

**Jobs:**
- Unit Tests (Matrix: Python versions × Test groups)
- Integration Tests (API, PMG, APOP, SHWL, Knowledge Base)
- Performance Benchmarks
- Security Tests
- End-to-End Tests
- Coverage Reporting
- Nightly Test Reports

### 4. **security.yml** - Security Scanning
**Triggers:** Push/PR, Daily Schedule (3 AM UTC), Manual
**Purpose:** Comprehensive security analysis
**Duration:** ~30-40 minutes

**Jobs:**
- Dependency Scanning (Safety, pip-audit)
- Code Security Analysis (Bandit, Semgrep)
- Container Scanning (Trivy, Grype)
- Secret Scanning (Gitleaks, TruffleHog)
- CodeQL Analysis
- IaC Scanning (Checkov)
- License Compliance

### 5. **release.yml** - Release Automation
**Triggers:** Version tags (v*.*.*)
**Purpose:** Automate release process
**Duration:** ~20-30 minutes

**Jobs:**
- Version Validation
- Artifact Building
- Docker Image Publishing
- Release Tests
- Release Notes Generation
- GitHub Release Creation
- PyPI Publishing (optional)
- Documentation Updates
- Release Announcement

## 📊 Workflow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Code Push / PR                          │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
    ┌──────┐    ┌──────┐    ┌──────────┐
    │  CI  │    │ Test │    │ Security │
    └───┬──┘    └───┬──┘    └────┬─────┘
        │           │            │
        └───────────┼────────────┘
                    │
            ┌───────▼───────┐
            │  All Passed?  │
            └───────┬───────┘
                    │ Yes
        ┌───────────▼───────────┐
        │   Ready for Release   │
        └───────────┬───────────┘
                    │
            ┌───────▼────────┐
            │  Create Tag    │
            └───────┬────────┘
                    │
        ┌───────────▼────────────┐
        │    Release Workflow    │
        └───────────┬────────────┘
                    │
        ┌───────────▼────────────┐
        │  CD Workflow (Deploy)  │
        └────────────────────────┘
```

## 🛠️ Setup Requirements

### Required GitHub Secrets

```yaml
# Container Registry (if using external registry)
REGISTRY_USERNAME: <your-registry-username>
REGISTRY_PASSWORD: <your-registry-password>

# PyPI Publishing (optional)
PYPI_API_TOKEN: <your-pypi-token>

# Codecov (optional but recommended)
CODECOV_TOKEN: <your-codecov-token>

# Cloud Provider (for deployment)
# Azure
AZURE_CREDENTIALS: <azure-service-principal>

# AWS
AWS_ACCESS_KEY_ID: <aws-key>
AWS_SECRET_ACCESS_KEY: <aws-secret>

# GCP
GCP_SA_KEY: <gcp-service-account-key>

# Kubernetes
KUBECONFIG: <kubernetes-config>
```

### Required GitHub Variables

```yaml
# PyPI Publishing Control
PUBLISH_TO_PYPI: 'true'  # or 'false'
```

### GitHub Settings

1. **Branch Protection Rules** (recommended):
   - Require status checks (CI Success)
   - Require review before merging
   - Require linear history
   - Include administrators

2. **Environments** (required for CD):
   - `staging` - staging environment
   - `production` - production environment (with required reviewers)
   - `pypi` - PyPI publishing environment

3. **Security Settings**:
   - Enable Dependabot alerts
   - Enable secret scanning
   - Enable code scanning (CodeQL)

## 📖 Usage Guide

### Running CI/CD

#### Automatic Triggers

1. **Push to main/develop**:
   ```bash
   git push origin main
   # Triggers: CI, Test, Security
   ```

2. **Create Pull Request**:
   ```bash
   gh pr create --base main --head feature/my-feature
   # Triggers: CI, Test, Security
   ```

3. **Create Release**:
   ```bash
   git tag -a v1.2.3 -m "Release version 1.2.3"
   git push origin v1.2.3
   # Triggers: Release, CD
   ```

#### Manual Triggers

1. **Run Tests Manually**:
   ```bash
   gh workflow run test.yml -f test_suite=all
   ```

2. **Run Security Scan**:
   ```bash
   gh workflow run security.yml
   ```

3. **Create Release**:
   ```bash
   gh workflow run release.yml -f version=1.2.3 -f prerelease=false
   ```

4. **Deploy to Environment**:
   ```bash
   gh workflow run cd.yml -f environment=staging
   ```

### Version Management

#### Semantic Versioning

Follow semantic versioning (SemVer):
- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features (backward compatible)
- **Patch** (0.0.X): Bug fixes

#### Pre-releases

- **Alpha**: `v1.2.3-alpha.1`
- **Beta**: `v1.2.3-beta.1`
- **Release Candidate**: `v1.2.3-rc.1`

### Deployment Strategy

#### Staging Deployment
Automatically deployed on:
- Push to `develop` branch
- Successful CI/CD pipeline

#### Production Deployment
Uses **Canary Deployment** strategy:
1. Deploy to 10% of traffic (5 min monitoring)
2. Promote to 50% of traffic (3 min monitoring)
3. Promote to 100% of traffic
4. Automatic rollback on failure

## 🎯 Best Practices

### Code Quality

1. **Run linters locally**:
   ```bash
   black sap_llm/ tests/
   ruff check sap_llm/ tests/
   mypy sap_llm/
   ```

2. **Run tests locally**:
   ```bash
   pytest tests/ -m unit -v
   pytest tests/ -m integration -v
   ```

3. **Check coverage**:
   ```bash
   pytest tests/ --cov=sap_llm --cov-report=html
   ```

### Git Workflow

1. **Feature Development**:
   ```bash
   git checkout -b feature/my-feature
   # Make changes
   git commit -m "feat: Add my feature"
   git push origin feature/my-feature
   # Create PR
   ```

2. **Hotfix**:
   ```bash
   git checkout -b hotfix/critical-bug
   # Fix bug
   git commit -m "fix: Critical bug fix"
   git push origin hotfix/critical-bug
   # Create PR to main
   ```

3. **Release**:
   ```bash
   # Update version in pyproject.toml
   git commit -m "chore: Bump version to 1.2.3"
   git tag -a v1.2.3 -m "Release 1.2.3"
   git push origin main --tags
   ```

### Monitoring

1. **Check Workflow Status**:
   ```bash
   gh run list
   gh run view <run-id>
   ```

2. **View Logs**:
   ```bash
   gh run view <run-id> --log
   ```

3. **Download Artifacts**:
   ```bash
   gh run download <run-id>
   ```

## 🔧 Troubleshooting

### Common Issues

#### 1. CI Fails on Code Quality

**Problem**: Black/Ruff/MyPy failures

**Solution**:
```bash
# Fix formatting
black sap_llm/ tests/

# Auto-fix linting issues
ruff check --fix sap_llm/ tests/

# Check types
mypy sap_llm/ --config-file=pyproject.toml
```

#### 2. Tests Fail Locally but Pass in CI (or vice versa)

**Problem**: Environment differences

**Solution**:
```bash
# Use same Python version as CI
pyenv install 3.11
pyenv local 3.11

# Clean install dependencies
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -e .[dev]
```

#### 3. Docker Build Fails

**Problem**: Build context or dependency issues

**Solution**:
```bash
# Test build locally
docker build -t sap-llm:test .

# Check .dockerignore
cat .dockerignore

# Build with no cache
docker build --no-cache -t sap-llm:test .
```

#### 4. Deployment Fails

**Problem**: Kubernetes/cluster access issues

**Solution**:
- Verify cluster credentials in secrets
- Check namespace exists
- Verify Helm chart syntax:
  ```bash
  helm lint ./helm/sap-llm
  helm template ./helm/sap-llm --debug
  ```

#### 5. Security Scan Failures

**Problem**: Vulnerabilities detected

**Solution**:
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Check specific package
pip show <package-name>

# Update package
pip install --upgrade <package-name>
```

### Getting Help

1. **Check workflow logs**:
   - Navigate to Actions tab in GitHub
   - Select failed workflow
   - Review job logs

2. **Review artifacts**:
   - Download test results
   - Check coverage reports
   - Review security scan outputs

3. **Consult documentation**:
   - GitHub Actions docs
   - Tool-specific documentation
   - This guide

## 📈 Metrics and Reporting

### Code Coverage

- Target: **≥70%** overall coverage
- Unit tests: **≥80%** coverage
- Integration tests: **≥60%** coverage

View coverage reports:
- Codecov dashboard (if configured)
- Workflow artifacts (htmlcov)
- PR comments (automatic)

### Performance Benchmarks

Performance tests track:
- Response time (p50, p95, p99)
- Throughput (requests/second)
- Memory usage
- CPU utilization

Access benchmarks:
- GitHub Actions artifacts
- Workflow summary
- Performance comparison graphs

### Security Metrics

Monitor security findings in:
- GitHub Security tab
- Workflow artifacts
- SARIF reports

## 🚀 Advanced Features

### Caching Strategy

All workflows use aggressive caching:
- **Pip dependencies**: `~/.cache/pip`
- **Docker layers**: GitHub Actions cache
- **Test data**: `~/.cache/torch`, `~/.cache/huggingface`
- **Pre-commit hooks**: `~/.cache/pre-commit`

### Parallel Execution

- Unit tests run in parallel (pytest-xdist)
- Multi-version testing (matrix strategy)
- Independent jobs run concurrently

### Matrix Testing

Test combinations:
- Python versions: 3.9, 3.10, 3.11
- Test groups: core, api, models, stages, utils
- OS: Ubuntu (can add macOS, Windows)

## 📝 Customization

### Adding New Tests

1. Add test marker in `pytest.ini`:
   ```ini
   markers =
       mytest: My custom test marker
   ```

2. Update workflow to include new tests:
   ```yaml
   - name: Run my tests
     run: pytest tests/ -m "mytest" -v
   ```

### Adding New Environments

1. Create environment in GitHub:
   - Settings → Environments → New environment

2. Update CD workflow:
   ```yaml
   deploy-myenv:
     environment:
       name: myenv
       url: https://myenv.example.com
   ```

### Custom Notifications

Add notification steps:
```yaml
- name: Notify Slack
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## 🔐 Security Considerations

1. **Secrets Management**:
   - Never commit secrets
   - Use GitHub Secrets
   - Rotate secrets regularly
   - Use environment protection

2. **Access Control**:
   - Limit workflow permissions
   - Use GITHUB_TOKEN with minimal scope
   - Require approvals for production

3. **Dependency Security**:
   - Regular dependency updates
   - Automated security scans
   - Pin dependency versions

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Deployment Guide](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Helm Charts](https://helm.sh/docs/)
- [CodeQL Documentation](https://codeql.github.com/docs/)

## 🤝 Contributing

To improve the CI/CD pipeline:

1. Test changes in a feature branch
2. Update this documentation
3. Create PR with detailed description
4. Ensure all workflows pass

## 📄 License

This CI/CD pipeline configuration is part of the SAP_LLM project.

---

**Last Updated**: 2025-11-14
**Version**: 1.0.0
**Maintainer**: QorSync AI Team
